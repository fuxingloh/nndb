//! funnel3.rs — Task #1 (experiment B): the 3-tier funnel.
//!   binary scan (RAM) → top-C1 by Hamming
//!     → PQ-ADC re-rank those C1 (RAM, M-byte codes) → top-C2
//!       → exact f32 rerank the C2 survivors → top-k
//!
//! The point is the DISK regime (052): the exact rerank is C random SSD reads, so the
//! metric is **exact reads** (= C2 for 3-tier, = C1 for the 2-tier binary→exact baseline).
//! Question: does PQ-pruning (a better-than-Hamming estimate, in RAM) let C2 ≪ C1 at the
//! same recall → far fewer SSD reads? PQ-ADC here runs only over the C1 candidates (cheap),
//! never the whole base.

use std::collections::{BinaryHeap, HashSet};

use clap::Parser;
use rayon::prelude::*;

use nndb::fvecs::{self, Vectors};
use nndb::quant::{self, hamming, QuantBinary, Rotation};
use nndb::search::{self, l2_sq};

#[derive(Parser)]
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
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 100000)]
    train: usize,
    #[arg(long, value_delimiter = ',', default_value = "1000,2000")]
    c1: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "32,64,128,256,500")]
    c2: Vec<usize>,
}

fn sub(v: &Vectors, n: usize) -> Vectors {
    let n = n.min(v.len());
    Vectors { data: v.data[..n * v.dim].to_vec(), dim: v.dim }
}
fn recall(found: &[Vec<u32>], gt: &[Vec<u32>], k: usize) -> f64 {
    let mut s = 0.0;
    for (f, t) in found.iter().zip(gt.iter()) {
        let set: HashSet<u32> = t.iter().take(k).copied().collect();
        s += f.iter().take(k).filter(|x| set.contains(x)).count() as f64 / k as f64;
    }
    s / found.len() as f64
}
fn kmeans(data: &[f32], n: usize, dim: usize, k: usize, iters: usize) -> Vec<f32> {
    let mut c = vec![0f32; k * dim];
    let stride = (n / k).max(1);
    for j in 0..k {
        c[j * dim..(j + 1) * dim].copy_from_slice(&data[((j * stride) % n) * dim..((j * stride) % n) * dim + dim]);
    }
    for _ in 0..iters {
        let asn: Vec<u32> = (0..n).into_par_iter().map(|p| {
            let pv = &data[p * dim..(p + 1) * dim];
            let (mut b, mut bd) = (0u32, f32::INFINITY);
            for j in 0..k { let d = l2_sq(pv, &c[j * dim..(j + 1) * dim]); if d < bd { bd = d; b = j as u32; } }
            b
        }).collect();
        let mut sum = vec![0f64; k * dim]; let mut cnt = vec![0u32; k];
        for p in 0..n { let j = asn[p] as usize; cnt[j] += 1; let pv = &data[p * dim..(p + 1) * dim];
            for d in 0..dim { sum[j * dim + d] += pv[d] as f64; } }
        for j in 0..k { if cnt[j] == 0 { continue; } let inv = 1.0 / cnt[j] as f64;
            for d in 0..dim { c[j * dim + d] = (sum[j * dim + d] * inv) as f32; } }
    }
    c
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let base = sub(&fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?, args.n);
    let queries = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let nq = if args.queries == 0 { queries.len() } else { args.queries.min(queries.len()) };
    let qsub = sub(&queries, nq);
    let dim = base.dim;
    let (m, sub_d, kk) = (args.m, dim / args.m, 256usize);
    eprintln!("n={} dim={dim} nq={nq} M={m} (sub={sub_d})", base.len());

    let gt: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|q| search::knn(&base, qsub.row(q), args.k)).collect();

    // stage-1 binary codes (rotation + residual — the best codes)
    let rot = Rotation::new(dim, 2, 0x5EED);
    let cen = quant::centroid(&base);
    let bbase = QuantBinary::from_f32_rotated(&quant::subtract_centroid(&base, &cen), &rot, 0);
    let qb: Vec<Vec<u64>> = (0..nq).map(|q| {
        let r: Vec<f32> = qsub.row(q).iter().zip(&cen).map(|(a, b)| a - b).collect();
        quant::binarize_query_rotated(&r, &rot, 0)
    }).collect();

    // PQ codebooks (trained on residual? use raw base for PQ — exact rerank is on raw anyway)
    let train = sub(&base, args.train.min(base.len()));
    let books: Vec<Vec<f32>> = (0..m).into_par_iter().map(|s| {
        let mut sd = vec![0f32; train.len() * sub_d];
        for pp in 0..train.len() { sd[pp * sub_d..(pp + 1) * sub_d].copy_from_slice(&train.row(pp)[s * sub_d..(s + 1) * sub_d]); }
        kmeans(&sd, train.len(), sub_d, kk, 12)
    }).collect();
    let mut codes = vec![0u8; base.len() * m];
    codes.par_chunks_mut(m).enumerate().for_each(|(pp, out)| {
        let row = base.row(pp);
        for s in 0..m { let sv = &row[s * sub_d..(s + 1) * sub_d]; let bk = &books[s];
            let (mut b, mut bd) = (0u8, f32::INFINITY);
            for cc in 0..kk { let d = l2_sq(sv, &bk[cc * sub_d..(cc + 1) * sub_d]); if d < bd { bd = d; b = cc as u8; } }
            out[s] = b; }
    });

    let mut pts = Vec::new();
    for &c1 in &args.c1 {
        // 2-tier baseline: binary top-c1 → exact rerank all c1 (exact reads = c1)
        let base2: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|q| {
            let cands = scan_topc(&bbase, &qb[q], c1);
            exact_topk(&base, qsub.row(q), &cands, args.k)
        }).collect();
        let r2 = recall(&base2, &gt, args.k);
        eprintln!("\nC1={c1}: 2-tier (binary->exact, {c1} reads) recall={r2:.4}");
        pts.push(format!("{{\"c1\":{c1},\"c2\":{c1},\"tier\":2,\"recall\":{r2:.4},\"exact_reads\":{c1}}}"));
        for &c2 in &args.c2 {
            if c2 >= c1 { continue; }
            let res: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|q| {
                let cands = scan_topc(&bbase, &qb[q], c1);
                // PQ-ADC table for this query
                let qrow = qsub.row(q);
                let mut tab = vec![0f32; m * kk];
                for s in 0..m { let qs = &qrow[s * sub_d..(s + 1) * sub_d]; let bk = &books[s];
                    for cc in 0..kk { tab[s * kk + cc] = l2_sq(qs, &bk[cc * sub_d..(cc + 1) * sub_d]); } }
                // re-rank c1 candidates by PQ distance → top c2
                let mut scored: Vec<(f32, u32)> = cands.iter().map(|&id| {
                    let code = &codes[id as usize * m..(id as usize + 1) * m];
                    let mut d = 0f32; for s in 0..m { d += tab[s * kk + code[s] as usize]; }
                    (d, id)
                }).collect();
                scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let c2cands: Vec<u32> = scored.into_iter().take(c2).map(|(_, i)| i).collect();
                exact_topk(&base, qsub.row(q), &c2cands, args.k)
            }).collect();
            let r3 = recall(&res, &gt, args.k);
            eprintln!("  3-tier C1={c1} C2={c2:<4} recall={r3:.4}  exact_reads={c2} ({:.1}x fewer)", c1 as f64 / c2 as f64);
            pts.push(format!("{{\"c1\":{c1},\"c2\":{c2},\"tier\":3,\"recall\":{r3:.4},\"exact_reads\":{c2}}}"));
        }
    }
    println!("{{\"dataset\":\"{p}\",\"n\":{},\"m\":{m},\"nq\":{nq},\"points\":[{}]}}", base.len(), pts.join(","));
    Ok(())
}

fn scan_topc(codes: &QuantBinary, qb: &[u64], want: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(want + 1);
    for i in 0..codes.len() {
        let h = hamming(qb, codes.row(i));
        if heap.len() < want { heap.push((h, i as u32)); }
        else if h < heap.peek().unwrap().0 { heap.pop(); heap.push((h, i as u32)); }
    }
    heap.into_iter().map(|(_, i)| i).collect()
}
fn exact_topk(base: &Vectors, q: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
    let mut s: Vec<(f32, u32)> = cands.iter().map(|&id| (l2_sq(q, base.row(id as usize)), id)).collect();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    s.into_iter().take(k).map(|(_, i)| i).collect()
}
