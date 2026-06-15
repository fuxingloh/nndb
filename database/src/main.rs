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

    /// Use only the first N base vectors (0 = all). Shrinks the working set so a
    /// sweep can find the cache->DRAM crossover. Recall is N/A when subsetting
    /// (ground truth references the full base).
    #[arg(long, default_value_t = 0)]
    base_subset: usize,

    /// Repeat the throughput pass this many times; report median (1st run is
    /// warmup and discarded when reps>1). Gives variance instead of a single shot.
    #[arg(long, default_value_t = 1)]
    reps: usize,

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

    let mut base = fvecs::read_fvecs(args.data.join("sift_base.fvecs"))?;
    let queries_all = fvecs::read_fvecs(args.data.join("sift_query.fvecs"))?;
    let gt = fvecs::read_ivecs(args.data.join("sift_groundtruth.ivecs"))?;

    if base.dim != queries_all.dim {
        eprintln!("dimension mismatch: base={} query={}", base.dim, queries_all.dim);
        std::process::exit(1);
    }

    // Optionally shrink the base to change the working-set size (cache vs DRAM).
    let base_full = base.len();
    if args.base_subset > 0 && args.base_subset < base_full {
        base.data.truncate(args.base_subset * base.dim);
    }
    let recall_valid = base.len() == base_full; // GT references the full base

    let n_qps = clamp_count(args.queries, queries_all.len());
    let n_lat = clamp_count(args.latency_queries, queries_all.len());

    // Throughput slice (parallel batch).
    let mut qps_set = fvecs::Vectors {
        data: queries_all.data[..n_qps * queries_all.dim].to_vec(),
        dim: queries_all.dim,
    };

    // --- Throughput pass: repeat R times, discard warmup, take median -------
    let reps = args.reps.max(1);
    let run = |qs: &fvecs::Vectors| {
        if args.batch > 1 {
            search::knn_batch_tiled(&base, qs, args.k, args.batch)
        } else {
            search::knn_batch(&base, qs, args.k)
        }
    };
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    let mut found = Vec::new();
    for r in 0..reps {
        let t = Instant::now();
        found = run(&qps_set);
        let dt = t.elapsed().as_secs_f64();
        if !(reps > 1 && r == 0) {
            times.push(dt); // first run is warmup when reps>1
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_secs = times[times.len() / 2];
    let t_mean = times.iter().sum::<f64>() / times.len() as f64;
    let t_var = times.iter().map(|t| (t - t_mean).powi(2)).sum::<f64>() / times.len() as f64;
    let cv = if t_mean > 0.0 { t_var.sqrt() / t_mean } else { 0.0 };

    let qps = n_qps as f64 / median_secs;
    let qps_min = n_qps as f64 / times[times.len() - 1]; // slowest run
    let qps_max = n_qps as f64 / times[0]; // fastest run
    // The bound-detection metric: time per distance, normalized over working-set
    // size. Flat across cache->DRAM = compute-bound; rises = memory-bound.
    let total_distances = n_qps as f64 * base.len() as f64;
    let ns_per_distance = median_secs / total_distances * 1e9;
    let recall = if recall_valid {
        eval::recall_at_k(&found, &gt, args.k)
    } else {
        -1.0
    };
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
                "\"k\":{},\"batch\":{},\"reps\":{},\"recall_at_k\":{:.4},\"recall_valid\":{},",
                "\"qps\":{:.1},\"qps_min\":{:.1},\"qps_max\":{:.1},\"qps_cv\":{:.4},",
                "\"ns_per_distance\":{:.4},\"qps_queries\":{},\"threads\":{},",
                "\"latency_ms\":{{\"mean\":{:.3},\"p50\":{:.3},\"p95\":{:.3},\"p99\":{:.3},\"samples\":{}}},",
                "\"memory_bytes\":{{\"index\":{},\"peak_rss\":{}}}}}"
            ),
            args.label,
            args.data.display(),
            base.len(),
            base.dim,
            args.k,
            args.batch,
            reps,
            recall,
            recall_valid,
            qps,
            qps_min,
            qps_max,
            cv,
            ns_per_distance,
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
        println!(
            "base:       {} vectors x {} dim  ({:.1} MB working set)",
            base.len(),
            base.dim,
            mb(index_bytes)
        );
        println!();
        if recall_valid {
            println!("recall@{}:   {:.4}", args.k, recall);
        } else {
            println!("recall@{}:   n/a (base subset)", args.k);
        }
        println!();
        println!(
            "throughput ({} searches, {} threads, batch={}, reps={}):",
            n_qps, threads, args.batch, reps
        );
        println!("  QPS:      {:.1}  (min {:.1}, max {:.1}, CV {:.1}%)", qps, qps_min, qps_max, cv * 100.0);
        println!("  ns/dist:  {:.4}  <- bound detector: flat across sizes = compute-bound", ns_per_distance);
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
