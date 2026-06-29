//! cell.rs — direction 7: within-cell size sweep + residual encoding (042 seam).
//!
//! Two within-cell questions from history/042:
//!   (1) how does the funnel's recall move with cell size N (at fixed bit budget / C)?
//!   (2) does subtracting the cell centroid (residual encoding) before rotation+binary
//!       lift recall vs raw? Hypothesis: inside one IVF cell, vectors cluster around
//!       the centroid, so raw sign-bits are dominated by the shared DC direction and
//!       carry little within-cell information; residuals (centered) should make the
//!       sign-bits encode the actual within-cell variation -> higher stage-1 recall
//!       at the SAME bits.
//!
//! Residual affects ONLY stage-1 selection (which candidates are scanned). Rerank and
//! ground truth are always exact L2 on the RAW vectors, so this isolates the residual's
//! effect on the binary scan's recall. Reports bytes/query (deterministic DRAM traffic:
//! N codes scanned + C f32 reranked) so every point carries the bandwidth currency.

use std::collections::{BinaryHeap, HashSet};

use clap::Parser;
use rayon::prelude::*;

use nndb::fvecs::{self, Vectors};
use nndb::quant::{self, hamming, rerank, QuantBinary, Rotation};
use nndb::search;

#[derive(Parser)]
#[command(about = "Within-cell size sweep + residual (centroid-subtracted) binary funnel")]
struct Args {
    #[arg(long, default_value = "data/cohere")]
    data: std::path::PathBuf,
    #[arg(long, default_value = "cohere")]
    prefix: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    #[arg(long, value_delimiter = ',', default_value = "1000,5000,20000,50000,100000")]
    cells: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "50,100,200,500")]
    funnel_c: Vec<usize>,
    #[arg(long, default_value_t = 2)]
    rotate: usize,
    /// scan bits per vector (0 = full dim)
    #[arg(long, default_value_t = 0)]
    bits: usize,
}

fn sub_vectors(v: &Vectors, n: usize) -> Vectors {
    let n = n.min(v.len());
    Vectors { data: v.data[..n * v.dim].to_vec(), dim: v.dim }
}

fn centroid(cell: &Vectors) -> Vec<f32> {
    let dim = cell.dim;
    let mut c = vec![0f64; dim];
    for i in 0..cell.len() {
        let r = cell.row(i);
        for d in 0..dim {
            c[d] += r[d] as f64;
        }
    }
    let n = cell.len() as f64;
    c.iter().map(|x| (x / n) as f32).collect()
}

fn subtract(v: &Vectors, c: &[f32]) -> Vectors {
    let dim = v.dim;
    let mut data = vec![0f32; v.data.len()];
    data.par_chunks_mut(dim).enumerate().for_each(|(i, out)| {
        let r = v.row(i);
        for d in 0..dim {
            out[d] = r[d] - c[d];
        }
    });
    Vectors { data, dim }
}

fn recall(found: &[Vec<u32>], truth: &[Vec<u32>], k: usize) -> f64 {
    let mut s = 0.0;
    for (f, t) in found.iter().zip(truth.iter()) {
        let tset: HashSet<u32> = t.iter().take(k).copied().collect();
        s += f.iter().take(k).filter(|x| tset.contains(x)).count() as f64 / k as f64;
    }
    s / found.len() as f64
}

fn scan_topc(codes: &QuantBinary, qb: &[u64], want: usize) -> Vec<u32> {
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
    heap.into_iter().map(|(_, i)| i).collect()
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let p = &args.prefix;
    let base = fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?;
    let queries = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let nq = if args.queries == 0 { queries.len() } else { args.queries.min(queries.len()) };
    let qsub = sub_vectors(&queries, nq);
    let dim = base.dim;
    let bits_eff = if args.bits == 0 { dim } else { args.bits };
    let code_bytes = bits_eff.div_ceil(64) * 8;
    let (kk, rr) = (args.k, args.rotate);
    eprintln!("dataset={p} dim={dim} nq={nq} k={kk} rotate={rr} bits={bits_eff} code_bytes={code_bytes}");

    let mut cells_json: Vec<String> = Vec::new();
    for &ncell in &args.cells {
        if ncell > base.len() {
            continue;
        }
        let cell = sub_vectors(&base, ncell);
        let n = cell.len();
        let gt: Vec<Vec<u32>> =
            (0..nq).into_par_iter().map(|q| search::knn(&cell, qsub.row(q), args.k)).collect();
        let cen = centroid(&cell);
        let rescell = subtract(&cell, &cen);
        let rot = Rotation::new(dim, args.rotate, 0xC0FFEE);

        let raw_codes = QuantBinary::from_f32_rotated(&cell, &rot, args.bits);
        let res_codes = QuantBinary::from_f32_rotated(&rescell, &rot, args.bits);
        let raw_qb: Vec<Vec<u64>> =
            (0..nq).map(|q| quant::binarize_query_rotated(qsub.row(q), &rot, args.bits)).collect();
        let res_qb: Vec<Vec<u64>> = (0..nq)
            .map(|q| {
                let rq: Vec<f32> = qsub.row(q).iter().zip(cen.iter()).map(|(a, b)| a - b).collect();
                quant::binarize_query_rotated(&rq, &rot, args.bits)
            })
            .collect();

        eprintln!("\n===== cell N={n} =====");
        let mut pts: Vec<String> = Vec::new();
        for (vname, codes, qbs) in
            [("raw", &raw_codes, &raw_qb), ("residual", &res_codes, &res_qb)]
        {
            for &c in &args.funnel_c {
                let want = c.max(args.k);
                let res: Vec<Vec<u32>> = (0..nq)
                    .into_par_iter()
                    .map(|q| {
                        let cands = scan_topc(codes, &qbs[q], want);
                        rerank(&cell, qsub.row(q), &cands, args.k)
                    })
                    .collect();
                let rec = recall(&res, &gt, args.k);
                let bytes_q = n * code_bytes + c * dim * 4;
                eprintln!("  {vname:<8} C={c:<4} recall={rec:.4} bytes/q={bytes_q}");
                pts.push(format!(
                    "{{\"variant\":\"{vname}\",\"c\":{c},\"recall\":{rec:.4},\"bytes_per_query\":{bytes_q}}}"
                ));
            }
        }
        cells_json.push(format!("{{\"n\":{n},\"points\":[{}]}}", pts.join(",")));
    }

    println!(
        "{{\"dataset\":\"{p}\",\"dim\":{dim},\"k\":{},\"nq\":{nq},\"rotate\":{},\"bits\":{bits_eff},\"code_bytes\":{code_bytes},\"cells\":[{}]}}",
        args.k,
        args.rotate,
        cells_json.join(",")
    );
    Ok(())
}
