//! cascade.rs — Matryoshka-over-BITS: a 64-bit sketch tier in front of the 256-bit funnel.
//!
//! 069 made the 256-bit scan ~3x cheaper and the stream (320 MB at 10M) came back as
//! the co-limiter. The Matryoshka idea applied to the *code* instead of the embedding:
//! after rotation every bit is an independent random hyperplane, so word 0 of each
//! 4-word code is itself a valid 64-bit sketch. Pack those words CONTIGUOUSLY (80 MB —
//! crucial: word 0 read in-place from the 32 B rows would still touch every 64 B cache
//! line, i.e. the full 320 MB) and run a 3-tier funnel:
//!
//!   A. 64-bit Hamming over the packed sketches -> top-C1 (histogram threshold select,
//!      two passes, NO per-doc dists buffer — the 013 counting trick without its
//!      bandwidth tax; 64-bit Hamming ∈ [0,64] so the histogram is 65 bins)
//!   B. 256-bit Hamming on the C1 survivors (random 32 B gathers) -> top-C2
//!   C. exact f32 rerank of C2 -> top-k
//!
//! vs the shipped 2-tier funnel (256-bit scan -> C -> f32). Same recall target, less
//! streamed bytes (80/T + gathers vs 320/T MB) and 1/4 the bulk popcounts.
//! Ground truth from <prefix>_groundtruth.ivecs; emits one JSON line per config.

use std::collections::BinaryHeap;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use nndb::fvecs::{self, Vectors};
use nndb::quant::{self, hamming, rerank, QuantBinary};

#[derive(Parser)]
#[command(about = "3-tier Matryoshka-bits cascade (64-bit sketch -> 256-bit -> f32) vs 2-tier funnel")]
struct Args {
    #[arg(long, default_value = "data/snowflake")]
    data: std::path::PathBuf,
    #[arg(long, default_value = "arctic256")]
    prefix: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 2000)]
    queries: usize,
    /// sketch-tier shortlist widths to sweep (comma-sep)
    #[arg(long, value_delimiter = ',', default_value = "25000,50000,100000,200000")]
    c1: Vec<usize>,
    /// 256-bit-tier shortlist width (= the 2-tier funnel's C)
    #[arg(long, default_value_t = 2000)]
    c2: usize,
    #[arg(long, default_value_t = 8)]
    tile: usize,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    #[arg(long, default_value_t = 2)]
    rotate: usize,
    /// sketch width in u64 words (1 = 64-bit, 2 = 128-bit)
    #[arg(long, default_value_t = 1)]
    sketch_words: usize,
}

/// Tier A+B+C for one tile of queries. `sketch` is the packed word-0 array.
#[allow(clippy::too_many_arguments)]
fn cascade_tiled(
    sketch: &[u64],
    sw: usize,
    bbase: &QuantBinary,
    fbase: &Vectors,
    bq: &QuantBinary,
    fq: &Vectors,
    k: usize,
    c1: usize,
    c2: usize,
    tile: usize,
) -> Vec<Vec<u32>> {
    let n = sketch.len() / sw;
    let nq = bq.len();
    let mut results: Vec<Vec<u32>> = (0..nq).map(|_| Vec::new()).collect();
    results.par_chunks_mut(tile).enumerate().for_each(|(ci, chunk)| {
        let q0 = ci * tile;
        let t = chunk.len();
        let qs: Vec<Vec<u64>> = (0..t).map(|j| bq.row(q0 + j)[..sw].to_vec()).collect();
        let maxh = sw * 64;

        // --- tier A pass 1: histogram of 64-bit Hamming per lane (65 bins, no dists buf)
        let mut hist = vec![vec![0u32; maxh + 1]; t];
        for i in 0..n {
            let s = &sketch[i * sw..(i + 1) * sw];
            for j in 0..t {
                hist[j][hamming(&qs[j], s) as usize] += 1;
            }
        }
        // per-lane threshold: smallest thr with cumulative count >= c1
        let mut thr = vec![0u32; t];
        let mut strict_under = vec![0usize; t]; // docs with h < thr (all kept)
        for j in 0..t {
            let mut acc = 0usize;
            let mut b = 0usize;
            while b <= maxh && acc + (hist[j][b] as usize) < c1 {
                acc += hist[j][b] as usize;
                b += 1;
            }
            thr[j] = b as u32;
            strict_under[j] = acc;
        }
        // --- tier A pass 2: collect ids (all h<thr; ties at h==thr up to the c1 budget)
        let mut cands: Vec<Vec<u32>> = (0..t).map(|_| Vec::with_capacity(c1 + 8)).collect();
        let mut tie_budget: Vec<usize> = (0..t).map(|j| c1 - strict_under[j]).collect();
        for i in 0..n {
            let s = &sketch[i * sw..(i + 1) * sw];
            for j in 0..t {
                let h = hamming(&qs[j], s);
                if h < thr[j] {
                    cands[j].push(i as u32);
                } else if h == thr[j] && tie_budget[j] > 0 {
                    cands[j].push(i as u32);
                    tie_budget[j] -= 1;
                }
            }
        }
        let _ = n;
        // --- tier B: full 256-bit Hamming on survivors -> top-c2, then tier C rerank
        for j in 0..t {
            let qrow = bq.row(q0 + j);
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(c2 + 1);
            for &id in &cands[j] {
                let h = hamming(qrow, bbase.row(id as usize));
                if heap.len() < c2 {
                    heap.push((h, id));
                } else if h < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push((h, id));
                }
            }
            let short: Vec<u32> = heap.into_iter().map(|(_, i)| i).collect();
            chunk[j] = rerank(fbase, fq.row(q0 + j), &short, k);
        }
    });
    results
}

fn recall(found: &[Vec<u32>], gt: &[Vec<u32>], k: usize) -> f64 {
    let mut s = 0.0;
    for (f, t) in found.iter().zip(gt.iter()) {
        let set: std::collections::HashSet<u32> = t.iter().take(k).copied().collect();
        s += f.iter().take(k).filter(|x| set.contains(x)).count() as f64 / k as f64;
    }
    s / found.len() as f64
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let base = fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?;
    let queries_all = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let gt_all = fvecs::read_ivecs(args.data.join(format!("{p}_groundtruth.ivecs")))?;
    let nq = args.queries.min(queries_all.len());
    let qsub = Vectors { data: queries_all.data[..nq * queries_all.dim].to_vec(), dim: queries_all.dim };
    let gt: Vec<Vec<u32>> = (0..nq).map(|i| gt_all.row(i).iter().map(|&x| x as u32).collect()).collect();

    // codes: same recipe as the shipped funnel (rotate x2 + residual), full 256 bits
    let dim = base.dim;
    let rot = quant::Rotation::new(dim, args.rotate, 0x5EED);
    let cent = quant::centroid(&base);
    let bbase = QuantBinary::from_f32_rotated(&quant::subtract_centroid(&base, &cent), &rot, dim);
    let bq = QuantBinary::from_f32_rotated(&quant::subtract_centroid(&qsub, &cent), &rot, dim);

    // packed word-0 sketch array (contiguous 8 B/doc — the whole point)
    let sw = args.sketch_words.max(1);
    let sketch: Vec<u64> = (0..bbase.len()).flat_map(|i| bbase.row(i)[..sw].to_vec()).collect();
    eprintln!("n={} dim={dim} nq={nq} sketch={}MB codes={}MB", base.len(), sketch.len() * 8 / 1_000_000, bbase.data.len() * 8 / 1_000_000);

    // baseline: shipped 2-tier tiled funnel, same harness/box/run
    let mut base_qps = 0.0f64;
    let mut base_rec = 0.0f64;
    for rep in 0..args.reps {
        let t0 = Instant::now();
        let found = quant::knn_binary_funnel_tiled(&bbase, &bq, &base, &qsub, args.k, args.c2, args.tile);
        let dt = t0.elapsed().as_secs_f64();
        if rep > 0 || args.reps == 1 {
            base_qps = base_qps.max(nq as f64 / dt);
        }
        base_rec = recall(&found, &gt, args.k);
    }
    println!(
        "{{\"config\":\"funnel-2tier\",\"c\":{},\"tile\":{},\"recall\":{base_rec:.4},\"qps\":{base_qps:.1}}}",
        args.c2, args.tile
    );

    for &c1 in &args.c1 {
        let mut qps = 0.0f64;
        let mut rec = 0.0f64;
        for rep in 0..args.reps {
            let t0 = Instant::now();
            let found = cascade_tiled(&sketch, sw, &bbase, &base, &bq, &qsub, args.k, c1, args.c2, args.tile);
            let dt = t0.elapsed().as_secs_f64();
            if rep > 0 || args.reps == 1 {
                qps = qps.max(nq as f64 / dt);
            }
            rec = recall(&found, &gt, args.k);
        }
        eprintln!("  c1={c1:<7} recall={rec:.4}  qps={qps:.1}  (baseline {base_rec:.4} @ {base_qps:.1})");
        println!(
            "{{\"config\":\"cascade-3tier\",\"sketch_bits\":{},\"c1\":{c1},\"c2\":{},\"tile\":{},\"recall\":{rec:.4},\"qps\":{qps:.1}}}",
            sw * 64, args.c2, args.tile
        );
    }
    Ok(())
}
