//! crossover.rs — within-cell engine bake-off: the binary funnel vs HNSW *inside
//! one IVF cell*.
//!
//! Question (history/042 follow-up): if a node is one IVF cell, should the
//! within-cell search be our rotated-binary funnel (scan ALL N at popcount speed,
//! then rerank) or an HNSW graph built over the cell (visit ~O(log N) candidates,
//! but random-access + approximate)? They are *competitors* for the same job, so
//! we measure both on recall / latency / QPS and find the cell size N where the
//! winner flips.
//!
//! HNSW's advantage is asymptotic (sub-linear visits) and its cost is random
//! memory access; the funnel is O(N) with a tiny sequential constant. So we expect
//! a crossover in N, and we expect it to move with dimensionality (high-D makes
//! each HNSW distance expensive while the funnel's 1-bit compare is dim/64 words).
//! Both SIFT (128-D) and Cohere (1024-D) are swept so the dimension effect shows.
//!
//! Fair fight: HNSW is the `hnsw_rs` crate (a real, parallel HNSW), DistL2 — the
//! same metric our exact GT and rerank use (for unit-norm Cohere, L2 rank == cosine
//! rank). The funnel is its best known config (rotate×2, tiled QPS). Each method
//! is measured at ITS best: funnel QPS uses tiling (its genuine throughput edge),
//! HNSW QPS uses per-query parallelism (it cannot tile a shared scan).

use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

use clap::Parser;
use hnsw_rs::prelude::*;
use rayon::prelude::*;

use vector_search::fvecs::{self, Vectors};
use vector_search::quant::{self, hamming, rerank, QuantBinary, Rotation};
use vector_search::search;

#[derive(Parser)]
#[command(about = "Within-cell binary funnel vs HNSW-in-cell: recall/latency/QPS crossover")]
struct Args {
    #[arg(long, default_value = "data/cohere")]
    data: std::path::PathBuf,
    #[arg(long, default_value = "cohere")]
    prefix: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    /// cell sizes (N) to sweep
    #[arg(long, value_delimiter = ',', default_value = "1000,5000,20000,50000,100000")]
    cells: Vec<usize>,
    /// funnel rerank widths C to sweep
    #[arg(long, value_delimiter = ',', default_value = "20,50,100,200,500")]
    funnel_c: Vec<usize>,
    /// HNSW ef_search values to sweep
    #[arg(long, value_delimiter = ',', default_value = "10,20,40,80,160")]
    hnsw_ef: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    rotate: usize,
    /// funnel tile for the QPS measurement
    #[arg(long, default_value_t = 8)]
    tile: usize,
    #[arg(long, default_value_t = 16)]
    hnsw_m: usize,
    #[arg(long, default_value_t = 200)]
    hnsw_efc: usize,
    /// timed repetitions for latency/QPS (median/best taken)
    #[arg(long, default_value_t = 3)]
    reps: usize,
}

fn sub_vectors(v: &Vectors, n: usize) -> Vectors {
    let n = n.min(v.len());
    Vectors { data: v.data[..n * v.dim].to_vec(), dim: v.dim }
}

fn recall(found: &[Vec<u32>], truth: &[Vec<u32>], k: usize) -> f64 {
    let mut s = 0.0;
    for (f, t) in found.iter().zip(truth.iter()) {
        let tset: HashSet<u32> = t.iter().take(k).copied().collect();
        let hit = f.iter().take(k).filter(|x| tset.contains(x)).count();
        s += hit as f64 / k as f64;
    }
    s / found.len() as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Single-query funnel: scan all N at Hamming speed, keep top-C, rerank exactly.
fn funnel_one(bbase: &QuantBinary, fbase: &Vectors, qbin: &[u64], fq: &[f32], k: usize, c: usize) -> Vec<u32> {
    let n = bbase.len();
    let c = c.max(k);
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(c + 1);
    for i in 0..n {
        let h = hamming(qbin, bbase.row(i));
        if heap.len() < c {
            heap.push((h, i as u32));
        } else if h < heap.peek().unwrap().0 {
            heap.pop();
            heap.push((h, i as u32));
        }
    }
    let cands: Vec<u32> = heap.into_iter().map(|(_, i)| i).collect();
    rerank(fbase, fq, &cands, k)
}

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let base = fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?;
    let queries_all = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let nq = if args.queries == 0 { queries_all.len() } else { args.queries.min(queries_all.len()) };
    let qsub = sub_vectors(&queries_all, nq);
    let dim = base.dim;
    eprintln!(
        "dataset={p} dim={dim} base={} queries={nq} k={} rotate={} tile={} | hnsw M={} efc={}",
        base.len(), args.k, args.rotate, args.tile, args.hnsw_m, args.hnsw_efc
    );

    let mut cell_json: Vec<String> = Vec::new();

    for &ncell in &args.cells {
        if ncell > base.len() {
            eprintln!("skip N={ncell} (> base {})", base.len());
            continue;
        }
        let cell = sub_vectors(&base, ncell);
        let n = cell.len();
        eprintln!("\n===== cell N={n} =====");

        // ---- exact ground truth within the cell ----
        let t = Instant::now();
        let gt: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|q| search::knn(&cell, qsub.row(q), args.k)).collect();
        let gt_ms = t.elapsed().as_secs_f64() * 1e3;

        // ================= FUNNEL =================
        let r_before = rss_kb();
        let rot = Rotation::new(dim, args.rotate, 0xC0FFEE);
        let tb = Instant::now();
        let bbase = QuantBinary::from_f32_rotated(&cell, &rot, 0);
        let bqueries = QuantBinary::from_f32_rotated(&qsub, &rot, 0);
        let funnel_build_ms = tb.elapsed().as_secs_f64() * 1e3;
        let funnel_mem_mb = (r_before.max(rss_kb()) - r_before) as f64 / 1024.0;
        let qbins: Vec<Vec<u64>> = (0..nq).map(|q| quant::binarize_query_rotated(qsub.row(q), &rot, 0)).collect();

        let mut funnel_pts: Vec<String> = Vec::new();
        for &c in &args.funnel_c {
            // recall + single-query latency (untiled, one query at a time)
            let mut lat_us: Vec<f64> = Vec::with_capacity(nq);
            let mut res: Vec<Vec<u32>> = Vec::with_capacity(nq);
            for q in 0..nq {
                let t = Instant::now();
                let r = funnel_one(&bbase, &cell, &qbins[q], qsub.row(q), args.k, c);
                lat_us.push(t.elapsed().as_secs_f64() * 1e6);
                res.push(r);
            }
            let rec = recall(&res, &gt, args.k);
            let lat_p50 = median(lat_us);
            // throughput: tiled funnel (its real QPS edge), best of reps
            let mut best_qps = 0.0;
            for _ in 0..args.reps {
                let t = Instant::now();
                let _ = quant::knn_binary_funnel_tiled(&bbase, &bqueries, &cell, &qsub, args.k, c, args.tile, false, false);
                let qps = nq as f64 / t.elapsed().as_secs_f64();
                if qps > best_qps {
                    best_qps = qps;
                }
            }
            eprintln!("  funnel C={c:<4} recall={rec:.4} lat_p50={lat_p50:>8.1}us qps={best_qps:>9.0}");
            funnel_pts.push(format!(
                "{{\"c\":{c},\"recall\":{rec:.4},\"lat_us_p50\":{lat_p50:.1},\"qps\":{best_qps:.0}}}"
            ));
        }

        // ================= HNSW =================
        let r_before = rss_kb();
        let tb = Instant::now();
        let hnsw = Hnsw::<f32, DistL2>::new(args.hnsw_m, n, 16, args.hnsw_efc, DistL2 {});
        let data: Vec<(Vec<f32>, usize)> = (0..n).map(|i| (cell.row(i).to_vec(), i)).collect();
        let data_ref: Vec<(&Vec<f32>, usize)> = data.iter().map(|(v, i)| (v, *i)).collect();
        hnsw.parallel_insert(&data_ref);
        let hnsw_build_ms = tb.elapsed().as_secs_f64() * 1e3;
        let hnsw_mem_mb = (rss_kb().max(r_before) - r_before) as f64 / 1024.0;
        drop(data); // the graph keeps its own copy

        let qvecs: Vec<Vec<f32>> = (0..nq).map(|q| qsub.row(q).to_vec()).collect();
        let mut hnsw_pts: Vec<String> = Vec::new();
        for &ef in &args.hnsw_ef {
            let ef = ef.max(args.k);
            // recall + single-query latency
            let mut lat_us: Vec<f64> = Vec::with_capacity(nq);
            let mut res: Vec<Vec<u32>> = Vec::with_capacity(nq);
            for q in 0..nq {
                let t = Instant::now();
                let nbrs = hnsw.search(&qvecs[q], args.k, ef);
                lat_us.push(t.elapsed().as_secs_f64() * 1e6);
                res.push(nbrs.iter().map(|nb| nb.d_id as u32).collect());
            }
            let rec = recall(&res, &gt, args.k);
            let lat_p50 = median(lat_us);
            // throughput: per-query parallel (HNSW cannot tile), best of reps
            let mut best_qps = 0.0;
            for _ in 0..args.reps {
                let t = Instant::now();
                let _: Vec<Vec<u32>> = (0..nq)
                    .into_par_iter()
                    .map(|q| hnsw.search(&qvecs[q], args.k, ef).iter().map(|nb| nb.d_id as u32).collect())
                    .collect();
                let qps = nq as f64 / t.elapsed().as_secs_f64();
                if qps > best_qps {
                    best_qps = qps;
                }
            }
            eprintln!("  hnsw  ef={ef:<4} recall={rec:.4} lat_p50={lat_p50:>8.1}us qps={best_qps:>9.0}");
            hnsw_pts.push(format!(
                "{{\"ef\":{ef},\"recall\":{rec:.4},\"lat_us_p50\":{lat_p50:.1},\"qps\":{best_qps:.0}}}"
            ));
        }

        cell_json.push(format!(
            "{{\"n\":{n},\"gt_ms\":{gt_ms:.1},\
             \"funnel\":{{\"build_ms\":{funnel_build_ms:.1},\"mem_mb\":{funnel_mem_mb:.1},\"points\":[{}]}},\
             \"hnsw\":{{\"build_ms\":{hnsw_build_ms:.1},\"mem_mb\":{hnsw_mem_mb:.1},\"points\":[{}]}}}}",
            funnel_pts.join(","),
            hnsw_pts.join(",")
        ));
    }

    println!(
        "{{\"dataset\":\"{p}\",\"dim\":{dim},\"k\":{},\"queries\":{nq},\"rotate\":{},\"tile\":{},\
         \"hnsw\":{{\"m\":{},\"efc\":{}}},\"cells\":[{}]}}",
        args.k,
        args.rotate,
        args.tile,
        args.hnsw_m,
        args.hnsw_efc,
        cell_json.join(",")
    );
    Ok(())
}
