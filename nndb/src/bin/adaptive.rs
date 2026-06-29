//! adaptive.rs — direction 5: query-adaptive funnel width + a per-query certificate.
//!
//! Fixed C wastes rerank on "easy" queries (top-k clearly Hamming-separated) and
//! under-serves "hard" ones (many near-tied candidates). Adaptive C keys off a
//! stage-1 signal — the Hamming margin — and spends rerank only where it's needed:
//!   C_q = #{candidates with hamming <= hamming[k-1] + margin}
//! Easy query → tight Hamming cluster → small C_q. Hard query → many ties → large C_q.
//!
//! Two results:
//!  (1) adaptive beats fixed on MEAN C at matched MEAN recall (less rerank work for the
//!      same recall). C is cheap vs the scan in RAM, but in the disk-resident regime
//!      (history 045) each reranked vector is an SSD read (~0.4 ms), so mean-C IS the
//!      latency — adaptive cuts it directly.
//!  (2) certificate: a per-query, computable-without-truth signal (the Hamming gap
//!      hamming[C]-hamming[k-1]) that PREDICTS per-query recall → a miss-risk bound.

use std::collections::{BinaryHeap, HashSet};

use clap::Parser;
use rayon::prelude::*;

use nndb::fvecs::{self, Vectors};
use nndb::quant::{self, hamming, rerank, QuantBinary, Rotation};
use nndb::search;

#[derive(Parser)]
#[command(about = "Query-adaptive funnel width (Hamming-margin) + per-query certificate")]
struct Args {
    #[arg(long, default_value = "data/cohere")]
    data: std::path::PathBuf,
    #[arg(long, default_value = "cohere")]
    prefix: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    #[arg(long, default_value_t = 100000)]
    cell: usize,
    #[arg(long, default_value_t = 2)]
    rotate: usize,
    /// top-M Hamming candidates collected per query (M >= max C / adaptive cap)
    #[arg(long, default_value_t = 2000)]
    m: usize,
    #[arg(long, value_delimiter = ',', default_value = "50,100,200,500,1000")]
    fixed_c: Vec<usize>,
    /// adaptive Hamming margins to sweep (candidates within hamming[k-1]+margin)
    #[arg(long, value_delimiter = ',', default_value = "0,2,4,8,16,32,64")]
    margins: Vec<u32>,
}

fn sub_vectors(v: &Vectors, n: usize) -> Vectors {
    let n = n.min(v.len());
    Vectors { data: v.data[..n * v.dim].to_vec(), dim: v.dim }
}

fn recall_one(found: &[u32], truth: &[u32], k: usize) -> f64 {
    let t: HashSet<u32> = truth.iter().take(k).copied().collect();
    found.iter().take(k).filter(|x| t.contains(x)).count() as f64 / k as f64
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let base = fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?;
    let queries = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let nq = if args.queries == 0 { queries.len() } else { args.queries.min(queries.len()) };
    let cell = sub_vectors(&base, args.cell);
    let n = cell.len();
    let dim = cell.dim;
    let k = args.k;
    eprintln!("cell N={n} dim={dim} nq={nq} k={k} M={} rotate={}", args.m, args.rotate);

    let gt: Vec<Vec<u32>> =
        (0..nq).into_par_iter().map(|q| search::knn(&cell, queries.row(q), k)).collect();

    let rot = Rotation::new(dim, args.rotate, 0xC0FFEE);
    let codes = QuantBinary::from_f32_rotated(&cell, &rot, 0);

    // Per query: top-M (hamming,id) ascending. This is the shared stage-1 output that
    // both fixed and adaptive funnels consume; rerank is exact on the raw cell vectors.
    let cand: Vec<Vec<(u32, u32)>> = (0..nq)
        .into_par_iter()
        .map(|q| {
            let qb = quant::binarize_query_rotated(queries.row(q), &rot, 0);
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(args.m + 1);
            for i in 0..n {
                let h = hamming(&qb, codes.row(i));
                if heap.len() < args.m {
                    heap.push((h, i as u32));
                } else if h < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push((h, i as u32));
                }
            }
            let mut v = heap.into_vec();
            v.sort_unstable();
            v
        })
        .collect();

    let rerank_topc = |q: usize, c: usize| -> Vec<u32> {
        let ids: Vec<u32> = cand[q].iter().take(c).map(|&(_, i)| i).collect();
        rerank(&cell, queries.row(q), &ids, k)
    };

    // ---- fixed C ----
    let mut fixed_json = Vec::new();
    eprintln!("\n-- fixed C --");
    for &c in &args.fixed_c {
        let rec: f64 = (0..nq)
            .into_par_iter()
            .map(|q| recall_one(&rerank_topc(q, c), &gt[q], k))
            .sum::<f64>()
            / nq as f64;
        eprintln!("  C={c:<5} mean_recall={rec:.4} mean_C={c}");
        fixed_json.push(format!("{{\"c\":{c},\"mean_recall\":{rec:.4},\"mean_c\":{c}.0}}"));
    }

    // ---- adaptive C (Hamming margin) ----
    let mut adapt_json = Vec::new();
    eprintln!("\n-- adaptive C (margin) --");
    for &m in &args.margins {
        let (sumrec, sumc) = (0..nq)
            .into_par_iter()
            .map(|q| {
                let hk = cand[q][k - 1].0;
                let thresh = hk + m;
                let cq = cand[q].partition_point(|&(h, _)| h <= thresh).max(k);
                (recall_one(&rerank_topc(q, cq), &gt[q], k), cq as f64)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let mean_rec = sumrec / nq as f64;
        let mean_c = sumc / nq as f64;
        eprintln!("  margin={m:<4} mean_recall={mean_rec:.4} mean_C={mean_c:.1}");
        adapt_json.push(format!(
            "{{\"margin\":{m},\"mean_recall\":{mean_rec:.4},\"mean_c\":{mean_c:.1}}}"
        ));
    }

    // ---- certificate: does the Hamming gap predict per-query recall? ----
    // At a fixed C, gap_q = hamming[C-1] - hamming[k-1] (headroom of the boundary above
    // the k-th). Bucket queries by gap, report recall per bucket. Monotone => the gap is
    // a valid per-query miss-risk certificate (computable without ground truth).
    let cert_c = 100usize.min(args.m - 1);
    let mut buckets: Vec<(u32, u32, f64, usize)> =
        vec![(0, 0, 0.0, 0), (1, 2, 0.0, 0), (3, 6, 0.0, 0), (7, 14, 0.0, 0), (15, u32::MAX, 0.0, 0)];
    for q in 0..nq {
        let gap = cand[q][cert_c - 1].0.saturating_sub(cand[q][k - 1].0);
        let r = recall_one(&rerank_topc(q, cert_c), &gt[q], k);
        for b in buckets.iter_mut() {
            if gap >= b.0 && gap <= b.1 {
                b.2 += r;
                b.3 += 1;
                break;
            }
        }
    }
    eprintln!("\n-- certificate (gap at C={cert_c} vs recall) --");
    let mut cert_json = Vec::new();
    for b in &buckets {
        let mr = if b.3 > 0 { b.2 / b.3 as f64 } else { 0.0 };
        let hi = if b.1 == u32::MAX { "inf".to_string() } else { b.1.to_string() };
        eprintln!("  gap[{}-{}] n={:<5} mean_recall={mr:.4}", b.0, hi, b.3);
        cert_json.push(format!(
            "{{\"gap_lo\":{},\"gap_hi\":\"{}\",\"n\":{},\"mean_recall\":{mr:.4}}}",
            b.0, hi, b.3
        ));
    }

    println!(
        "{{\"dataset\":\"{p}\",\"n\":{n},\"dim\":{dim},\"nq\":{nq},\"k\":{k},\"rotate\":{},\"cert_c\":{cert_c},\
         \"fixed\":[{}],\"adaptive\":[{}],\"certificate\":[{}]}}",
        args.rotate,
        fixed_json.join(","),
        adapt_json.join(","),
        cert_json.join(",")
    );
    Ok(())
}
