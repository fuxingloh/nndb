//! In-memory vector-search server. Loads the base vectors into RAM once and
//! serves nearest-neighbor queries over HTTP/JSON — the interface layer a real
//! cluster serves through.
//!
//! Serving model: each request is one query, searched single-threaded. A
//! semaphore caps concurrent searches at the core count, so excess requests
//! *queue* — that queuing is the user-facing latency we want to measure under
//! load. The CPU-bound search runs on `spawn_blocking` so it never stalls the
//! async reactor handling other connections.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use vector_search::{fvecs, search};

#[derive(Parser)]
#[command(about = "In-memory vector-search HTTP server (SIFT1M)")]
struct Args {
    /// Directory holding sift_base.fvecs
    #[arg(long, default_value = "data/sift")]
    data: PathBuf,

    /// Address to bind
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,

    /// Max concurrent searches (0 = number of CPU cores)
    #[arg(long, default_value_t = 0)]
    max_concurrency: usize,
}

struct AppState {
    base: fvecs::Vectors,
    /// Bounds in-flight CPU-bound searches so load translates into queuing.
    sem: Semaphore,
}

#[derive(Deserialize)]
struct SearchReq {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
}
fn default_k() -> usize {
    10
}

#[derive(Serialize)]
struct SearchResp {
    ids: Vec<u32>,
    /// Pure search compute time on the server (excludes HTTP/queue).
    compute_us: u128,
}

#[derive(Serialize)]
struct Info {
    n_base: usize,
    dim: usize,
    max_concurrency: usize,
}

async fn health() -> &'static str {
    "ok"
}

async fn info(State(state): State<Arc<AppState>>) -> Json<Info> {
    Json(Info {
        n_base: state.base.len(),
        dim: state.base.dim,
        max_concurrency: state.sem.available_permits(),
    })
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchReq>,
) -> Result<Json<SearchResp>, (StatusCode, String)> {
    if req.vector.len() != state.base.dim {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("expected {}-dim vector, got {}", state.base.dim, req.vector.len()),
        ));
    }
    if req.k == 0 || req.k > state.base.len() {
        return Err((StatusCode::BAD_REQUEST, "k out of range".into()));
    }

    // Acquire a permit *before* spawning: requests beyond core count wait here,
    // and that wait is exactly the queuing component of user-facing latency.
    let _permit = state
        .sem
        .acquire()
        .await
        .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "shutting down".into()))?;

    let state2 = state.clone();
    let ids = tokio::task::spawn_blocking(move || {
        let t = Instant::now();
        let ids = search::knn(&state2.base, &req.vector, req.k);
        (ids, t.elapsed().as_micros())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(SearchResp {
        ids: ids.0,
        compute_us: ids.1,
    }))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let base = fvecs::read_fvecs(args.data.join("sift_base.fvecs"))?;
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max_conc = if args.max_concurrency == 0 {
        cores
    } else {
        args.max_concurrency
    };

    println!(
        "loaded {} x {} dim; serving on http://{} (max_concurrency={})",
        base.len(),
        base.dim,
        args.addr,
        max_conc
    );

    let state = Arc::new(AppState {
        base,
        sem: Semaphore::new(max_conc),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/search", post(search_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    axum::serve(listener, app).await
}
