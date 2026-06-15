//! Baseline benchmark: load SIFT1M into memory, run exact brute-force KNN, and
//! report the three axes we track per improvement — recall, throughput (QPS),
//! latency distribution, and memory.
//!
//! Throughput and latency are measured in *separate passes* on purpose:
//!   - throughput: all queries fanned across cores (saturated batch) -> QPS
//!   - latency:    one query at a time, timed individually -> p50/p95/p99
//! Reporting 1/QPS as "latency" would hide the tail; these numbers genuinely
//! differ once the search is parallelized.

use std::hint::black_box;
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

    /// Number of searches in the throughput (QPS) pass
    #[arg(long, default_value_t = 1000)]
    queries: usize,

    /// Number of single-query timings in the latency pass (sequential, slower)
    #[arg(long, default_value_t = 200)]
    latency_queries: usize,

    /// Query tile size: reuse each base vector across this many queries (1 = per-query)
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Name of the index/approach being measured (recorded with results)
    #[arg(long, default_value = "brute-force")]
    label: String,

    /// Emit a single JSON result line instead of human-readable output
    #[arg(long)]
    json: bool,
}

/// Peak resident set size of this process, in bytes.
fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage only writes into the zeroed struct we pass.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0;
        }
        let maxrss = usage.ru_maxrss as u64;
        // macOS reports ru_maxrss in bytes; Linux in kilobytes.
        #[cfg(target_os = "macos")]
        {
            maxrss
        }
        #[cfg(not(target_os = "macos"))]
        {
            maxrss * 1024
        }
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let base = fvecs::read_fvecs(args.data.join("sift_base.fvecs"))?;
    let queries_all = fvecs::read_fvecs(args.data.join("sift_query.fvecs"))?;
    let gt = fvecs::read_ivecs(args.data.join("sift_groundtruth.ivecs"))?;

    if base.dim != queries_all.dim {
        eprintln!("dimension mismatch: base={} query={}", base.dim, queries_all.dim);
        std::process::exit(1);
    }

    let n_qps = clamp_count(args.queries, queries_all.len());
    let n_lat = clamp_count(args.latency_queries, queries_all.len());

    // Throughput slice (parallel batch).
    let mut qps_set = fvecs::Vectors {
        data: queries_all.data[..n_qps * queries_all.dim].to_vec(),
        dim: queries_all.dim,
    };

    // --- Throughput pass: saturate all cores with a batch of searches -------
    let t = Instant::now();
    let found = if args.batch > 1 {
        search::knn_batch_tiled(&base, &qps_set, args.k, args.batch)
    } else {
        search::knn_batch(&base, &qps_set, args.k)
    };
    let qps_secs = t.elapsed().as_secs_f64();
    let qps = n_qps as f64 / qps_secs;
    let recall = eval::recall_at_k(&found, &gt, args.k);
    qps_set.data.clear(); // free the throughput slice before the latency pass

    // --- Latency pass: time each query on its own, build the distribution ---
    let mut lat_ms: Vec<f64> = Vec::with_capacity(n_lat);
    for q in 0..n_lat {
        let t = Instant::now();
        let r = search::knn(&base, queries_all.row(q), args.k);
        lat_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        black_box(r); // keep the search from being optimized away
    }
    lat_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = lat_ms.iter().sum::<f64>() / lat_ms.len() as f64;
    let p50 = eval::percentile(&lat_ms, 50.0);
    let p95 = eval::percentile(&lat_ms, 95.0);
    let p99 = eval::percentile(&lat_ms, 99.0);

    // --- Memory -------------------------------------------------------------
    let index_bytes = base.data.len() * std::mem::size_of::<f32>();
    let rss = peak_rss_bytes();
    let threads = rayon::current_num_threads();

    if args.json {
        // Single line; benchmark/run.sh enriches it with date + commit.
        println!(
            concat!(
                "{{\"label\":\"{}\",\"dataset\":\"{}\",\"n_base\":{},\"dim\":{},",
                "\"k\":{},\"batch\":{},\"recall_at_k\":{:.4},\"qps\":{:.1},\"qps_queries\":{},\"threads\":{},",
                "\"latency_ms\":{{\"mean\":{:.3},\"p50\":{:.3},\"p95\":{:.3},\"p99\":{:.3},\"samples\":{}}},",
                "\"memory_bytes\":{{\"index\":{},\"peak_rss\":{}}}}}"
            ),
            args.label,
            args.data.display(),
            base.len(),
            base.dim,
            args.k,
            args.batch,
            recall,
            qps,
            n_qps,
            threads,
            mean,
            p50,
            p95,
            p99,
            n_lat,
            index_bytes,
            rss,
        );
    } else {
        let mb = |b: usize| b as f64 / (1u64 << 20) as f64;
        println!("approach:   {}", args.label);
        println!("dataset:    {}", args.data.display());
        println!("base:       {} vectors x {} dim", base.len(), base.dim);
        println!();
        println!("recall@{}:   {:.4}", args.k, recall);
        println!();
        println!("throughput ({} searches, {} threads, batch={}):", n_qps, threads, args.batch);
        println!("  QPS:      {:.1}", qps);
        println!();
        println!("latency ({} single queries, sequential):", n_lat);
        println!("  mean:     {:.2} ms", mean);
        println!("  p50:      {:.2} ms", p50);
        println!("  p95:      {:.2} ms", p95);
        println!("  p99:      {:.2} ms", p99);
        println!();
        println!("memory:");
        println!("  index:    {:.0} MB (raw vectors)", mb(index_bytes));
        println!("  peak RSS: {:.0} MB (process)", mb(rss as usize));
    }

    Ok(())
}

/// `0` means "use all available"; otherwise cap at what's in the file.
fn clamp_count(requested: usize, available: usize) -> usize {
    if requested == 0 {
        available
    } else {
        requested.min(available)
    }
}
