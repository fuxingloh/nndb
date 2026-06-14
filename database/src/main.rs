//! Baseline benchmark: load SIFT1M into memory, run exact brute-force KNN, and
//! report recall@k + QPS against the ground truth.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use vector_search::{eval, fvecs, search};

#[derive(Parser)]
#[command(about = "In-memory exact vector search baseline (SIFT1M / .fvecs)")]
struct Args {
    /// Directory holding sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs
    #[arg(long, default_value = "data/sift")]
    data: PathBuf,

    /// Number of nearest neighbors to retrieve
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Number of queries to run (0 = all queries in the file)
    #[arg(long, default_value_t = 1000)]
    queries: usize,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let base_path = args.data.join("sift_base.fvecs");
    let query_path = args.data.join("sift_query.fvecs");
    let gt_path = args.data.join("sift_groundtruth.ivecs");

    // --- Load (timed) -------------------------------------------------------
    let t = Instant::now();
    let base = fvecs::read_fvecs(&base_path)?;
    let mut queries = fvecs::read_fvecs(&query_path)?;
    let gt = fvecs::read_ivecs(&gt_path)?;
    let load_secs = t.elapsed().as_secs_f64();

    if base.dim != queries.dim {
        eprintln!(
            "dimension mismatch: base={} query={}",
            base.dim, queries.dim
        );
        std::process::exit(1);
    }

    // Optionally run a subset of queries for fast iteration.
    let n_queries = if args.queries == 0 {
        queries.len()
    } else {
        args.queries.min(queries.len())
    };
    queries.data.truncate(n_queries * queries.dim);

    let mem_mb = (base.data.len() * std::mem::size_of::<f32>()) as f64 / (1 << 20) as f64;

    println!("dataset:   {}", args.data.display());
    println!(
        "base:      {} vectors x {} dim  ({:.0} MB in memory)",
        base.len(),
        base.dim,
        mem_mb
    );
    println!("queries:   {} (of {})", n_queries, gt.len());
    println!("k:         {}", args.k);
    println!("load time: {:.2}s", load_secs);
    println!();

    // --- Search (timed) -----------------------------------------------------
    let t = Instant::now();
    let found = search::knn_batch(&base, &queries, args.k);
    let search_secs = t.elapsed().as_secs_f64();

    // --- Report -------------------------------------------------------------
    let recall = eval::recall_at_k(&found, &gt, args.k);
    let qps = n_queries as f64 / search_secs;
    let amortized_ms = search_secs / n_queries as f64 * 1000.0;
    let threads = rayon::current_num_threads();

    println!("recall@{}:  {:.4}", args.k, recall);
    println!("search:    {:.2}s total ({} threads)", search_secs, threads);
    println!("QPS:       {:.1}", qps);
    println!("latency:   {:.3} ms/query (amortized wall-clock)", amortized_ms);
    println!(
        "           ~{:.1} ms/query single-thread-equivalent (full scan cost)",
        amortized_ms * threads as f64
    );

    Ok(())
}
