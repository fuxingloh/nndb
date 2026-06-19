//! pq.rs — Product Quantization (and OPQ via a learned rotation) for the within-cell
//! search, the one quantization family the binary funnel never tried (044's open branch).
//!
//! PQ splits each vector into M subvectors, k-means-clusters each subspace (K=256 → 1
//! byte/subspace), and stores each vector as M bytes. Search uses ADC (asymmetric
//! distance computation): per query, precompute an M×K table of subvector→centroid
//! distances, then each doc's distance is M table lookups summed. So a doc is M bytes
//! and M lookups — vs the binary funnel's dim/8 bytes and dim/64 popcounts.
//!
//! Question: can PQ at ~16 B/vec (8× fewer bytes than 128-bit binary) match the funnel's
//! recall? Fewer bytes = more QPS at the bandwidth wall (049/050). Caveat (011): the ADC
//! gather must not lose to autovectorized popcount.

use std::collections::BinaryHeap;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use vector_search::fvecs::{self, Vectors};
use vector_search::search::{self, l2_sq};

#[derive(Parser)]
#[command(about = "Product Quantization (PQ) within-cell search")]
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
    /// number of subspaces M (bytes per vector). dim must be divisible by M.
    #[arg(long, value_delimiter = ',', default_value = "8,16,32,64")]
    m: Vec<usize>,
    /// k-means training sample size
    #[arg(long, default_value_t = 100000)]
    train: usize,
    /// k-means iterations
    #[arg(long, default_value_t = 12)]
    iters: usize,
    /// rerank width C (0 = pure ADC top-k, no rerank)
    #[arg(long, value_delimiter = ',', default_value = "0,200,1000")]
    rerank: Vec<usize>,
}

fn sub(v: &Vectors, n: usize) -> Vectors {
    let n = n.min(v.len());
    Vectors { data: v.data[..n * v.dim].to_vec(), dim: v.dim }
}

/// k-means (Lloyd) on `dim`-D points, returns k*dim centroids. Deterministic init
/// (evenly-strided points), empty clusters reseeded to a strided point.
fn kmeans(data: &[f32], npts: usize, dim: usize, k: usize, iters: usize) -> Vec<f32> {
    let mut cent = vec![0f32; k * dim];
    let stride = (npts / k).max(1);
    for c in 0..k {
        let src = (c * stride) % npts;
        cent[c * dim..(c + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
    }
    for _ in 0..iters {
        // assign (parallel) → for each point, nearest centroid
        let assign: Vec<u32> = (0..npts)
            .into_par_iter()
            .map(|p| {
                let pv = &data[p * dim..(p + 1) * dim];
                let mut best = 0u32;
                let mut bd = f32::INFINITY;
                for c in 0..k {
                    let d = l2_sq(pv, &cent[c * dim..(c + 1) * dim]);
                    if d < bd {
                        bd = d;
                        best = c as u32;
                    }
                }
                best
            })
            .collect();
        // update
        let mut sums = vec![0f64; k * dim];
        let mut counts = vec![0u32; k];
        for p in 0..npts {
            let c = assign[p] as usize;
            counts[c] += 1;
            let pv = &data[p * dim..(p + 1) * dim];
            let s = &mut sums[c * dim..(c + 1) * dim];
            for d in 0..dim {
                s[d] += pv[d] as f64;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let src = (c * stride) % npts;
                cent[c * dim..(c + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f64;
                for d in 0..dim {
                    cent[c * dim + d] = (sums[c * dim + d] * inv) as f32;
                }
            }
        }
    }
    cent
}

struct Pq {
    m: usize,
    sub: usize,
    k: usize,
    books: Vec<Vec<f32>>, // m × (k*sub)
    codes: Vec<u8>,       // n × m
    n: usize,
}

impl Pq {
    fn train_encode(base: &Vectors, train: &Vectors, m: usize, k: usize, iters: usize) -> Self {
        let dim = base.dim;
        let sub = dim / m;
        // fit each subspace's codebook in parallel
        let books: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|s| {
                // gather subvectors for subspace s (contiguous copy)
                let mut subdata = vec![0f32; train.len() * sub];
                for p in 0..train.len() {
                    let row = train.row(p);
                    subdata[p * sub..(p + 1) * sub].copy_from_slice(&row[s * sub..(s + 1) * sub]);
                }
                kmeans(&subdata, train.len(), sub, k, iters)
            })
            .collect();
        // encode the full base
        let n = base.len();
        let mut codes = vec![0u8; n * m];
        codes.par_chunks_mut(m).enumerate().for_each(|(p, out)| {
            let row = base.row(p);
            for s in 0..m {
                let sv = &row[s * sub..(s + 1) * sub];
                let book = &books[s];
                let mut best = 0u8;
                let mut bd = f32::INFINITY;
                for c in 0..k {
                    let d = l2_sq(sv, &book[c * sub..(c + 1) * sub]);
                    if d < bd {
                        bd = d;
                        best = c as u8;
                    }
                }
                out[s] = best;
            }
        });
        Pq { m, sub, k, books, codes, n }
    }

    /// ADC: build per-subspace query→centroid distance tables, then top-want by summed
    /// lookups. Returns candidate ids (ascending ADC distance).
    fn adc_topc(&self, q: &[f32], want: usize) -> Vec<u32> {
        // table[s*k + c] = l2(q_sub_s, centroid[s][c])
        let mut table = vec![0f32; self.m * self.k];
        for s in 0..self.m {
            let qs = &q[s * self.sub..(s + 1) * self.sub];
            let book = &self.books[s];
            for c in 0..self.k {
                table[s * self.k + c] = l2_sq(qs, &book[c * self.sub..(c + 1) * self.sub]);
            }
        }
        let mut heap: BinaryHeap<(ordf::OrdF, u32)> = BinaryHeap::with_capacity(want + 1);
        for p in 0..self.n {
            let code = &self.codes[p * self.m..(p + 1) * self.m];
            let mut d = 0f32;
            for s in 0..self.m {
                d += table[s * self.k + code[s] as usize];
            }
            let df = ordf::OrdF(d);
            if heap.len() < want {
                heap.push((df, p as u32));
            } else if df < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((df, p as u32));
            }
        }
        let mut v: Vec<(ordf::OrdF, u32)> = heap.into_iter().collect();
        v.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        v.into_iter().map(|(_, i)| i).collect()
    }

    fn mem_bytes(&self) -> usize {
        self.n * self.m
    }
}

mod ordf {
    #[derive(Clone, Copy, PartialEq)]
    pub struct OrdF(pub f32);
    impl Eq for OrdF {}
    impl PartialOrd for OrdF {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for OrdF {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&o.0)
        }
    }
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
    let base = sub(&fvecs::read_fvecs(args.data.join(format!("{p}_base.fvecs")))?, args.n);
    let queries = fvecs::read_fvecs(args.data.join(format!("{p}_query.fvecs")))?;
    let nq = if args.queries == 0 { queries.len() } else { args.queries.min(queries.len()) };
    let qsub = sub(&queries, nq);
    let dim = base.dim;
    let train = sub(&base, args.train.min(base.len()));
    eprintln!("n={} dim={dim} nq={nq} train={} M={:?}", base.len(), train.len(), args.m);

    let gt: Vec<Vec<u32>> =
        (0..nq).into_par_iter().map(|q| search::knn(&base, qsub.row(q), args.k)).collect();

    let mut pts = Vec::new();
    for &m in &args.m {
        if dim % m != 0 {
            eprintln!("skip M={m} (dim {dim} not divisible)");
            continue;
        }
        let tb = Instant::now();
        let pq = Pq::train_encode(&base, &train, m, 256, args.iters);
        let build_s = tb.elapsed().as_secs_f64();
        eprintln!("\n=== M={m} ({m} B/vec, {:.0} MB) build={build_s:.1}s ===", pq.mem_bytes() as f64 / 1e6);
        for &c in &args.rerank {
            let want = if c == 0 { args.k } else { c.max(args.k) };
            let t = Instant::now();
            let res: Vec<Vec<u32>> = (0..nq)
                .into_par_iter()
                .map(|q| {
                    let cands = pq.adc_topc(qsub.row(q), want);
                    if c == 0 {
                        cands.into_iter().take(args.k).collect()
                    } else {
                        // exact rerank on raw vectors
                        let mut s: Vec<(f32, u32)> =
                            cands.iter().map(|&id| (l2_sq(qsub.row(q), base.row(id as usize)), id)).collect();
                        s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        s.into_iter().take(args.k).map(|(_, i)| i).collect()
                    }
                })
                .collect();
            let secs = t.elapsed().as_secs_f64();
            let rec = recall(&res, &gt, args.k);
            let qps = nq as f64 / secs;
            eprintln!("  C={c:<5} recall={rec:.4} qps={qps:>8.0} bytes/vec={m}");
            pts.push(format!(
                "{{\"m\":{m},\"bytes_per_vec\":{m},\"c\":{c},\"recall\":{rec:.4},\"qps\":{qps:.0}}}"
            ));
        }
    }
    println!(
        "{{\"dataset\":\"{p}\",\"n\":{},\"dim\":{dim},\"nq\":{nq},\"k\":{},\"points\":[{}]}}",
        base.len(),
        args.k,
        pts.join(",")
    );
    Ok(())
}
