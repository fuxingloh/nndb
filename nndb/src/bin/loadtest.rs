//! Closed-loop load generator for the vector-search server. Holds a fixed
//! number of concurrent in-flight requests (the "production traffic" knob) and
//! measures **user-facing latency** — client-side, end-to-end, including HTTP
//! and any server-side queuing.
//!
//! It also reads the server's reported compute time per request, so we can
//! split user-facing latency into (search compute) + (interface/queue overhead).

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use serde::{Deserialize, Serialize};
use nndb::{eval, fvecs};

#[derive(Parser)]
#[command(about = "Concurrent load generator measuring user-facing search latency")]
struct Args {
    /// Server search endpoint
    #[arg(long, default_value = "http://127.0.0.1:8080/search")]
    url: String,

    /// Directory holding <prefix>_query.fvecs (source of realistic queries)
    #[arg(long, default_value = "data/sift")]
    data: PathBuf,

    /// Dataset file prefix (e.g. "sift", "cohere")
    #[arg(long, default_value = "sift")]
    prefix: String,

    /// Number of concurrent in-flight requests
    #[arg(long, default_value_t = 8)]
    concurrency: usize,

    /// Total requests to send (after warmup)
    #[arg(long, default_value_t = 2000)]
    requests: usize,

    /// Neighbors per query
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Warmup requests (not measured)
    #[arg(long, default_value_t = 100)]
    warmup: usize,

    /// Label for the record
    #[arg(long, default_value = "brute-force")]
    label: String,

    /// Emit a single JSON result line
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
struct ReqBody<'a> {
    vector: &'a [f32],
    k: usize,
}

#[derive(Deserialize)]
struct RespBody {
    #[allow(dead_code)]
    ids: Vec<u32>,
    compute_us: u128,
}

/// One measured request: (end-to-end client ms, server compute ms).
async fn one_request(
    client: &reqwest::Client,
    url: &str,
    vector: &[f32],
    k: usize,
) -> Result<(f64, f64), String> {
    let t = Instant::now();
    let resp = client
        .post(url)
        .json(&ReqBody { vector, k })
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("status {}", resp.status()));
    }
    let body: RespBody = resp.json().await.map_err(|e| e.to_string())?;
    let client_ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok((client_ms, body.compute_us as f64 / 1000.0))
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let qfile = format!("{}_query.fvecs", args.prefix);
    let queries = fvecs::read_fvecs(args.data.join(&qfile))
        .unwrap_or_else(|_| panic!("read {qfile}"));
    let nq = queries.len();
    let queries = Arc::new(queries);
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(args.concurrency)
        .build()
        .expect("build client");

    // Warmup: pay first-connection + page-in costs before measuring.
    for i in 0..args.warmup {
        let _ = one_request(&client, &args.url, queries.row(i % nq), args.k).await;
    }

    let counter = Arc::new(AtomicUsize::new(0));
    let total = args.requests;

    let start = Instant::now();
    let mut handles = Vec::with_capacity(args.concurrency);
    for _ in 0..args.concurrency {
        let client = client.clone();
        let url = args.url.clone();
        let queries = queries.clone();
        let counter = counter.clone();
        let k = args.k;
        handles.push(tokio::spawn(async move {
            let mut samples: Vec<(f64, f64)> = Vec::new();
            let mut errors = 0usize;
            loop {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    break;
                }
                match one_request(&client, &url, queries.row(i % nq), k).await {
                    Ok(s) => samples.push(s),
                    Err(_) => errors += 1,
                }
            }
            (samples, errors)
        }));
    }

    let mut client_ms = Vec::with_capacity(total);
    let mut server_ms = Vec::with_capacity(total);
    let mut errors = 0usize;
    for h in handles {
        let (samples, e) = h.await.unwrap();
        errors += e;
        for (c, s) in samples {
            client_ms.push(c);
            server_ms.push(s);
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let done = client_ms.len();

    // Interface/queue overhead = end-to-end minus pure compute, per request.
    let mut overhead: Vec<f64> = client_ms
        .iter()
        .zip(&server_ms)
        .map(|(c, s)| (c - s).max(0.0))
        .collect();

    client_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    server_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    overhead.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let qps = done as f64 / wall;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let cl_mean = mean(&client_ms);
    let cl_p50 = eval::percentile(&client_ms, 50.0);
    let cl_p95 = eval::percentile(&client_ms, 95.0);
    let cl_p99 = eval::percentile(&client_ms, 99.0);
    let cl_max = *client_ms.last().unwrap_or(&f64::NAN);
    let sv_p50 = eval::percentile(&server_ms, 50.0);
    let sv_p99 = eval::percentile(&server_ms, 99.0);
    let ov_mean = mean(&overhead);
    let ov_p50 = eval::percentile(&overhead, 50.0);

    if args.json {
        println!(
            concat!(
                "{{\"label\":\"{}\",\"transport\":\"http\",\"concurrency\":{},\"requests\":{},\"errors\":{},",
                "\"qps\":{:.1},\"client_latency_ms\":{{\"mean\":{:.3},\"p50\":{:.3},\"p95\":{:.3},\"p99\":{:.3},\"max\":{:.3}}},",
                "\"server_compute_ms\":{{\"p50\":{:.3},\"p99\":{:.3}}},",
                "\"interface_overhead_ms\":{{\"mean\":{:.3},\"p50\":{:.3}}}}}"
            ),
            args.label, args.concurrency, done, errors,
            qps, cl_mean, cl_p50, cl_p95, cl_p99, cl_max,
            sv_p50, sv_p99, ov_mean, ov_p50,
        );
    } else {
        println!("approach:    {}", args.label);
        println!("transport:   HTTP/JSON  ->  {}", args.url);
        println!("concurrency: {}  ({} requests, {} errors)", args.concurrency, done, errors);
        println!();
        println!("throughput:");
        println!("  QPS:       {:.1}", qps);
        println!();
        println!("user-facing latency (client-side, end-to-end):");
        println!("  mean:      {:.2} ms", cl_mean);
        println!("  p50:       {:.2} ms", cl_p50);
        println!("  p95:       {:.2} ms", cl_p95);
        println!("  p99:       {:.2} ms", cl_p99);
        println!("  max:       {:.2} ms", cl_max);
        println!();
        println!("breakdown:");
        println!("  server compute p50/p99:   {:.2} / {:.2} ms", sv_p50, sv_p99);
        println!("  interface+queue overhead: {:.2} ms mean, {:.2} ms p50", ov_mean, ov_p50);
    }
}
