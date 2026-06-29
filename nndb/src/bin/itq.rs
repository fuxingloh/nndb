//! itq.rs — Iterative Quantization (Gong & Lazebnik 2011): a *learned* rotation that
//! minimizes binarization error, vs our *random* (FWHT) rotation. Question: does a
//! learned rotation beat random for the binary funnel's recall at a fixed bit budget?
//!
//! We test at b=256 bits (where rotation matters most, 046/051). The b-dim projection is
//! the first b dims of the FWHT-rotated vector (a sensible random projection, = our
//! current "rotated prefix" 027). ITQ then learns a b×b rotation R on top, minimizing
//! ‖sign(VR) − VR‖. Baseline = R=identity (the current random-rotation codes).
//!
//! ITQ fit (on a sample S, m×b): iterate  B = sign(S·R);  M = Sᵀ·B (b×b);  SVD(M)=UΣWᵀ;
//! R = U·Wᵀ  (orthogonal Procrustes). R is DENSE (b×b) → applying it costs O(b²)/vector
//! (vs FWHT's O(b log b)); we measure recall and note that cost.

use std::collections::BinaryHeap;

use clap::Parser;
use nalgebra::DMatrix;
use rayon::prelude::*;

use nndb::fvecs::{self, Vectors};
use nndb::quant::{self, hamming, rerank, QuantBinary, Rotation};
use nndb::search;

#[derive(Parser)]
#[command(about = "ITQ learned rotation vs random rotation for binary codes")]
struct Args {
    #[arg(long, default_value = "data/cohere")]
    data: std::path::PathBuf,
    #[arg(long, default_value = "cohere")]
    prefix: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    #[arg(long, default_value_t = 1000000)]
    n: usize,
    /// code bits b (must be ≤ dim)
    #[arg(long, default_value_t = 256)]
    bits: usize,
    #[arg(long, default_value_t = 50000)]
    train: usize,
    #[arg(long, default_value_t = 30)]
    iters: usize,
    #[arg(long, value_delimiter = ',', default_value = "200,1000")]
    rerank: Vec<usize>,
}

fn sub(v: &Vectors, n: usize) -> Vectors {
    let n = n.min(v.len());
    Vectors { data: v.data[..n * v.dim].to_vec(), dim: v.dim }
}

/// Project every row through the FWHT rotation and keep the first `b` dims → V (n×b).
fn project(v: &Vectors, rot: &Rotation, b: usize) -> Vec<f32> {
    let dim = v.dim;
    let mut out = vec![0f32; v.len() * b];
    out.par_chunks_mut(b).enumerate().for_each(|(i, o)| {
        let r = rot.apply(v.row(i));
        o.copy_from_slice(&r[..b]);
    });
    out
}

/// y = V·R  (n×b · b×b), parallel over rows.
fn apply_rot(vbig: &[f32], r: &[f32], n: usize, b: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * b];
    out.par_chunks_mut(b).enumerate().for_each(|(p, o)| {
        let v = &vbig[p * b..(p + 1) * b];
        for j in 0..b {
            let mut acc = 0f32;
            for i in 0..b {
                acc += v[i] * r[i * b + j];
            }
            o[j] = acc;
        }
    });
    out
}

/// Pack sign bits of an n×b row-major matrix into QuantBinary (b bits/vec).
fn binarize(mat: &[f32], n: usize, b: usize) -> QuantBinary {
    let words = b.div_ceil(64);
    let mut data = vec![0u64; n * words];
    data.par_chunks_mut(words).enumerate().for_each(|(p, out)| {
        let row = &mat[p * b..(p + 1) * b];
        for d in 0..b {
            if row[d] > 0.0 {
                out[d / 64] |= 1u64 << (d % 64);
            }
        }
    });
    QuantBinary { data, words, dim: b }
}

/// ITQ: learn b×b orthogonal R minimizing ‖sign(S·R) − S·R‖ on sample S (m×b).
fn fit_itq(sample: &[f32], m: usize, b: usize, iters: usize) -> Vec<f32> {
    // init R = identity (deterministic; ITQ converges from it fine here)
    let mut r = vec![0f32; b * b];
    for i in 0..b {
        r[i * b + i] = 1.0;
    }
    for _ in 0..iters {
        let sr = apply_rot(sample, &r, m, b); // m×b
                                              // M = Sᵀ·B  (b×b), B = sign(SR)
        let mtx: Vec<f32> = (0..b * b)
            .into_par_iter()
            .map(|idx| {
                let i = idx / b; // S column
                let j = idx % b; // B column
                let mut acc = 0f32;
                for p in 0..m {
                    let bsign = if sr[p * b + j] > 0.0 { 1.0 } else { -1.0 };
                    acc += sample[p * b + i] * bsign;
                }
                acc
            })
            .collect();
        let dm = DMatrix::from_row_slice(b, b, &mtx);
        let svd = dm.svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let rnew = u * vt; // R = U·Wᵀ
        for i in 0..b {
            for j in 0..b {
                r[i * b + j] = rnew[(i, j)];
            }
        }
    }
    r
}

fn recall(found: &[Vec<u32>], gt: &[Vec<u32>], k: usize) -> f64 {
    let mut s = 0.0;
    for (f, t) in found.iter().zip(gt.iter()) {
        let set: std::collections::HashSet<u32> = t.iter().take(k).copied().collect();
        s += f.iter().take(k).filter(|x| set.contains(x)).count() as f64 / k as f64;
    }
    s / found.len() as f64
}

fn funnel(codes: &QuantBinary, qcodes: &QuantBinary, base: &Vectors, q: &Vectors, k: usize, c: usize, nq: usize) -> Vec<Vec<u32>> {
    let want = c.max(k);
    (0..nq)
        .into_par_iter()
        .map(|qi| {
            let qb = qcodes.row(qi);
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(want + 1);
            for i in 0..codes.len() {
                let h = hamming(qb, codes.row(i));
                if heap.len() < want {
                    heap.push((h, i as u32));
                } else if h < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push((h, i as u32));
                }
            }
            let cands: Vec<u32> = heap.into_iter().map(|(_, i)| i).collect();
            rerank(base, q.row(qi), &cands, k)
        })
        .collect()
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let base = sub(&fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?, args.n);
    let queries = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let nq = if args.queries == 0 { queries.len() } else { args.queries.min(queries.len()) };
    let qsub = sub(&queries, nq);
    let dim = base.dim;
    let b = args.bits;
    eprintln!("n={} dim={dim} nq={nq} bits={b} train={} iters={}", base.len(), args.train, args.iters);

    let gt: Vec<Vec<u32>> =
        (0..nq).into_par_iter().map(|q| search::knn(&base, qsub.row(q), args.k)).collect();

    // b-dim projection (FWHT rotate → first b dims), for both base and queries.
    let rot = Rotation::new(dim, 2, 0x5EED);
    let vbase = project(&base, &rot, b);
    let vq = project(&qsub, &rot, b);
    let _ = quant::binarize_query; // (kept import path consistent)

    // --- baseline: random rotation = R identity (just the FWHT prefix sign bits) ---
    let rnd_base = binarize(&vbase, base.len(), b);
    let rnd_q = binarize(&vq, nq, b);

    // --- ITQ: learn R on a sample, apply, binarize ---
    let m = args.train.min(base.len());
    let r = fit_itq(&vbase[..m * b], m, b, args.iters);
    let itq_base = binarize(&apply_rot(&vbase, &r, base.len(), b), base.len(), b);
    let itq_q = binarize(&apply_rot(&vq, &r, nq, b), nq, b);

    let mut pts = Vec::new();
    for &c in &args.rerank {
        let rnd = recall(&funnel(&rnd_base, &rnd_q, &base, &qsub, args.k, c, nq), &gt, args.k);
        let itq = recall(&funnel(&itq_base, &itq_q, &base, &qsub, args.k, c, nq), &gt, args.k);
        eprintln!("  C={c:<5} random={rnd:.4}  ITQ={itq:.4}  delta={:+.4}", itq - rnd);
        pts.push(format!("{{\"c\":{c},\"random\":{rnd:.4},\"itq\":{itq:.4}}}"));
    }
    println!(
        "{{\"dataset\":\"{p}\",\"n\":{},\"bits\":{b},\"nq\":{nq},\"train\":{m},\"iters\":{},\"points\":[{}]}}",
        base.len(),
        args.iters,
        pts.join(",")
    );
    Ok(())
}
