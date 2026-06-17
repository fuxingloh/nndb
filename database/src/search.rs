//! Exact brute-force k-nearest-neighbor search.
//!
//! This is the correctness oracle and speed baseline. It scans every base
//! vector for every query, so recall is always 100% by construction — its job
//! is to give us the QPS/latency floor that approximate indexes (HNSW, IVF, …)
//! must beat while staying close to that recall.

use crate::fvecs::Vectors;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Squared L2 distance. We rank by squared distance because the square root is
/// monotonic — it doesn't change neighbor ordering and saves a sqrt per compare.
///
/// Written to let the autovectorizer use the **full SIMD width and FMA**, fixing
/// the two defects the naive `.map().sum()` had (confirmed in disassembly: the
/// reduction de-vectorized to per-lane scalar adds, and no `fmla`/`vfmadd`):
///   1. `ACC` **independent accumulators** (a flat array) — the sum is no longer
///      one serial dependency chain, so the scheduler keeps many FMAs in flight.
///      `ACC = 32` maps to 2 AVX-512 `zmm` chains (16 f32 each) on x86, or 8
///      NEON chains on aarch64 — both fill the width and hide FMA latency.
///   2. `mul_add` — each step is a fused multiply-add (one `fmla` / `vfmadd`).
///
/// Build with `-C target-cpu=native` (see `.cargo/config.toml`) so this uses the
/// widest SIMD the host has and hardware FMA. Reassociating the sum is bit-exact
/// for SIFT's integer-valued vectors, so recall is unchanged.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const ACC: usize = 32;
    let mut acc = [0.0f32; ACC];
    let n = a.len();
    let blocks = n / ACC;

    for blk in 0..blocks {
        let off = blk * ACC;
        for l in 0..ACC {
            let d = a[off + l] - b[off + l];
            acc[l] = d.mul_add(d, acc[l]); // fused multiply-add into lane l
        }
    }

    // Tail for dims not divisible by ACC (e.g. tiny test vectors).
    let mut tail = 0.0f32;
    for i in (blocks * ACC)..n {
        let d = a[i] - b[i];
        tail = d.mul_add(d, tail);
    }

    // Tree-reduce the accumulators: parallel adds, log-depth chain (not a
    // serial 32-add chain), and each round is itself a vector add.
    let mut w = ACC;
    while w > 1 {
        w /= 2;
        for l in 0..w {
            acc[l] += acc[l + w];
        }
    }
    acc[0] + tail
}

/// Total-order wrapper so f32 distances can live in a `BinaryHeap`.
#[derive(Clone, Copy, PartialEq)]
struct Dist(f32);
impl Eq for Dist {}
impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Dist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Indices of the `k` nearest base vectors to `query`, ascending by distance.
///
/// Keeps a bounded max-heap of size `k`: the root is the current k-th best, so
/// a candidate only enters if it beats it. That makes selection O(n log k)
/// instead of sorting all n distances.
pub fn knn(base: &Vectors, query: &[f32], k: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<(Dist, u32)> = BinaryHeap::with_capacity(k + 1);
    for i in 0..base.len() {
        let d = Dist(l2_sq(query, base.row(i)));
        if heap.len() < k {
            heap.push((d, i as u32));
        } else if d < heap.peek().unwrap().0 {
            heap.pop();
            heap.push((d, i as u32));
        }
    }
    // into_sorted_vec yields ascending order (nearest first).
    heap.into_sorted_vec().into_iter().map(|(_, i)| i).collect()
}

/// Run `knn` for every query row in parallel across CPU cores.
pub fn knn_batch(base: &Vectors, queries: &Vectors, k: usize) -> Vec<Vec<u32>> {
    (0..queries.len())
        .into_par_iter()
        .map(|q| knn(base, queries.row(q), k))
        .collect()
}

/// Bandwidth-amortizing KNN. Same exact result as `knn_batch`, but instead of
/// re-streaming the whole base once per query, it processes queries in tiles of
/// `tile`: for each base vector loaded, it computes the distance to all queries
/// in the tile while that vector is hot in cache. The base is streamed once per
/// tile (Q/tile streams) instead of once per query (Q streams), so the DRAM
/// traffic — the bandwidth-bound scan's binding constraint — drops by ~`tile`.
///
/// The tile must be small enough that its queries stay resident in L1 while the
/// base streams past them; ~16–32 is the sweet spot (fits L1, and enough to flip
/// the scan from memory-bound to compute-bound on typical hardware).
pub fn knn_batch_tiled(base: &Vectors, queries: &Vectors, k: usize, tile: usize) -> Vec<Vec<u32>> {
    let tile = tile.max(1);
    let nq = queries.len();

    let starts: Vec<usize> = (0..nq).step_by(tile).collect();
    let per_tile: Vec<Vec<Vec<u32>>> = starts
        .into_par_iter()
        .map(|start| {
            let end = (start + tile).min(nq);
            let b = end - start;
            let mut heaps: Vec<BinaryHeap<(Dist, u32)>> =
                (0..b).map(|_| BinaryHeap::with_capacity(k + 1)).collect();

            // Stream the base once; reuse each row across the whole query tile.
            for n in 0..base.len() {
                let row = base.row(n);
                for (qi, q) in (start..end).enumerate() {
                    let d = Dist(l2_sq(row, queries.row(q)));
                    let h = &mut heaps[qi];
                    if h.len() < k {
                        h.push((d, n as u32));
                    } else if d < h.peek().unwrap().0 {
                        h.pop();
                        h.push((d, n as u32));
                    }
                }
            }

            heaps
                .into_iter()
                .map(|h| h.into_sorted_vec().into_iter().map(|(_, i)| i).collect())
                .collect()
        })
        .collect();

    per_tile.into_iter().flatten().collect()
}

/// ADSampling distance-comparison KNN (Gao & Long, SIGMOD 2023). Inputs MUST be
/// random-rotated (e.g. via `quant::Rotation`) so that the partial squared
/// distance over the first `i` dims is an unbiased estimate of the full distance
/// (Johnson–Lindenstrauss). Distance is accumulated incrementally in batches of
/// `delta`; after each batch, if the partial distance already exceeds the current
/// k-th best by the confidence margin, the candidate is pruned without finishing —
/// saving most of the D-dim work for the (many) far candidates. Because the
/// rotation preserves L2 exactly, survivors get the exact distance, so results
/// match exact KNN up to the (tiny) probabilistic pruning slack set by `eps0`.
pub fn knn_adsampling(rbase: &Vectors, rq: &[f32], k: usize, eps0: f32, delta: usize) -> Vec<u32> {
    let d = rbase.dim;
    let dd = delta.max(1);
    let mut heap: BinaryHeap<(Dist, u32)> = BinaryHeap::with_capacity(k + 1);
    for n in 0..rbase.len() {
        let o = rbase.row(n);
        if heap.len() < k {
            // No threshold yet — must compute the full distance.
            let mut res = 0f32;
            for j in 0..d {
                let df = rq[j] - o[j];
                res += df * df;
            }
            heap.push((Dist(res), n as u32));
            continue;
        }
        let thresh = heap.peek().unwrap().0 .0; // current k-th squared distance
        let mut res = 0f32;
        let mut i = 0usize;
        let mut pruned = false;
        while i < d {
            let end = (i + dd).min(d);
            for j in i..end {
                let df = rq[j] - o[j];
                res += df * df;
            }
            i = end;
            if i < d {
                // ratio(D,i) = (i/D)·(1 + eps0/√i)²  — the ADSampling bound.
                let fi = i as f32;
                let t = 1.0 + eps0 / fi.sqrt();
                let ratio = (fi / d as f32) * t * t;
                if res >= thresh * ratio {
                    pruned = true;
                    break;
                }
            }
        }
        if !pruned && res < thresh {
            heap.pop();
            heap.push((Dist(res), n as u32));
        }
    }
    heap.into_sorted_vec().into_iter().map(|(_, i)| i).collect()
}

/// Parallel ADSampling KNN over a (rotated) query set.
pub fn knn_adsampling_batch(
    rbase: &Vectors,
    rqueries: &Vectors,
    k: usize,
    eps0: f32,
    delta: usize,
) -> Vec<Vec<u32>> {
    (0..rqueries.len())
        .into_par_iter()
        .map(|q| knn_adsampling(rbase, rqueries.row(q), k, eps0, delta))
        .collect()
}

/// PDX (Kuffo & Boncz, SIGMOD 2025): a **dimension-major** block layout. Vectors
/// are grouped into blocks of `block`; within a block the data is stored
/// transposed — all vectors' dim 0, then all vectors' dim 1, … So computing
/// distances for a whole block is a loop over dimensions whose inner loop is over
/// vectors (multiple-vectors-at-a-time), which autovectorizes cleanly and needs no
/// per-vector horizontal reduction. The basis for fast pruned scans (033).
pub struct PdxBase {
    data: Vec<f32>,
    n: usize,
    dim: usize,
    block: usize,
}

impl PdxBase {
    pub fn from_vectors(v: &Vectors, block: usize) -> Self {
        let n = v.len();
        let dim = v.dim;
        let block = block.max(1);
        let mut data = vec![0f32; n * dim];
        let mut off = 0;
        let mut start = 0;
        while start < n {
            let bsz = block.min(n - start);
            for vi in 0..bsz {
                let row = v.row(start + vi);
                for d in 0..dim {
                    data[off + d * bsz + vi] = row[d];
                }
            }
            off += bsz * dim;
            start += bsz;
        }
        PdxBase { data, n, dim, block }
    }
    pub fn len(&self) -> usize {
        self.n
    }
}

/// Exact KNN over the PDX layout — dimension-major, multiple-vectors-at-a-time.
/// Same result as `knn`; the inner per-dimension loop over the block's vectors
/// autovectorizes and avoids the horizontal reduction of row-major distance.
pub fn knn_pdx(pdx: &PdxBase, q: &[f32], k: usize) -> Vec<u32> {
    let dim = pdx.dim;
    let mut heap: BinaryHeap<(Dist, u32)> = BinaryHeap::with_capacity(k + 1);
    let mut partial = vec![0f32; pdx.block];
    let mut off = 0;
    let mut start = 0;
    while start < pdx.n {
        let bsz = pdx.block.min(pdx.n - start);
        let part = &mut partial[..bsz];
        part.iter_mut().for_each(|p| *p = 0.0);
        for d in 0..dim {
            let qd = q[d];
            let col = &pdx.data[off + d * bsz..off + d * bsz + bsz];
            for v in 0..bsz {
                let df = qd - col[v];
                part[v] += df * df;
            }
        }
        for v in 0..bsz {
            let dd = Dist(part[v]);
            let idx = (start + v) as u32;
            if heap.len() < k {
                heap.push((dd, idx));
            } else if dd < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((dd, idx));
            }
        }
        off += bsz * dim;
        start += bsz;
    }
    heap.into_sorted_vec().into_iter().map(|(_, i)| i).collect()
}

pub fn knn_pdx_batch(pdx: &PdxBase, queries: &Vectors, k: usize) -> Vec<Vec<u32>> {
    (0..queries.len())
        .into_par_iter()
        .map(|q| knn_pdx(pdx, queries.row(q), k))
        .collect()
}

/// ADSampling pruning ON the PDX layout (the combination PDX is built for). For
/// each block we accumulate partial squared distances dimension-by-dimension over
/// the *alive* vectors; after each batch we drop vectors whose partial already
/// exceeds the k-th best by the ADSampling margin, so they skip their remaining
/// dims. The dimension-major layout keeps the alive-vector inner loop tight while
/// pruning shrinks the work — recovering the speedup that early termination loses
/// on a row-major layout (031). `q` must be RANDOM-ROTATED.
pub fn knn_pdx_adsampling(pdx: &PdxBase, q: &[f32], k: usize, eps0: f32, delta: usize) -> Vec<u32> {
    let dim = pdx.dim;
    let step = delta.max(1);
    let mut heap: BinaryHeap<(Dist, u32)> = BinaryHeap::with_capacity(k + 1);
    let mut partial = vec![0f32; pdx.block];
    let mut alive: Vec<u32> = Vec::with_capacity(pdx.block);
    let mut off = 0;
    let mut start = 0;
    while start < pdx.n {
        let bsz = pdx.block.min(pdx.n - start);
        partial[..bsz].iter_mut().for_each(|p| *p = 0.0);
        alive.clear();
        alive.extend(0..bsz as u32);
        let full = heap.len() >= k;
        let thr = if full { heap.peek().unwrap().0 .0 } else { f32::INFINITY };
        let mut i = 0;
        while i < dim && !alive.is_empty() {
            let end = (i + step).min(dim);
            for d in i..end {
                let qd = q[d];
                let col = &pdx.data[off + d * bsz..off + d * bsz + bsz];
                for &v in &alive {
                    let vv = v as usize;
                    let df = qd - col[vv];
                    partial[vv] += df * df;
                }
            }
            i = end;
            if full && i < dim {
                let fi = i as f32;
                let t = 1.0 + eps0 / fi.sqrt();
                let bound = thr * (fi / dim as f32) * t * t;
                alive.retain(|&v| partial[v as usize] < bound);
            }
        }
        for &v in &alive {
            let d = Dist(partial[v as usize]);
            let idx = start as u32 + v;
            if heap.len() < k {
                heap.push((d, idx));
            } else if d < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((d, idx));
            }
        }
        off += bsz * dim;
        start += bsz;
    }
    heap.into_sorted_vec().into_iter().map(|(_, i)| i).collect()
}

/// Parallel PDX+ADSampling over a (rotated) query set.
pub fn knn_pdx_adsampling_batch(
    pdx: &PdxBase,
    rqueries: &Vectors,
    k: usize,
    eps0: f32,
    delta: usize,
) -> Vec<Vec<u32>> {
    (0..rqueries.len())
        .into_par_iter()
        .map(|q| knn_pdx_adsampling(pdx, rqueries.row(q), k, eps0, delta))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vecs(rows: &[&[f32]]) -> Vectors {
        let dim = rows[0].len();
        let data = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Vectors { data, dim }
    }

    #[test]
    fn returns_k_nearest_ascending() {
        // 1-D points at 0,1,2,3,4; query at 0 -> nearest are 0,1,2.
        let base = vecs(&[&[0.0], &[4.0], &[1.0], &[3.0], &[2.0]]);
        let got = knn(&base, &[0.0], 3);
        assert_eq!(got, vec![0, 2, 4]); // indices of points 0.0, 1.0, 2.0
    }

    #[test]
    fn pdx_matches_exact() {
        let base = vecs(&[
            &[0.0, 0.0], &[1.0, 1.0], &[2.0, 0.5], &[5.0, 5.0],
            &[0.2, 0.1], &[3.0, 3.0], &[1.5, 0.0], &[4.0, 1.0],
        ]);
        let q = [0.1, 0.1];
        let pdx = PdxBase::from_vectors(&base, 3); // 8 rows, block 3 -> 3,3,2
        for k in 1..=4 {
            assert_eq!(knn_pdx(&pdx, &q, k), knn(&base, &q, k), "k={k}");
        }
    }

    #[test]
    fn pdx_adsampling_conservative_matches_exact() {
        let base = vecs(&[
            &[0.0, 0.0], &[1.0, 1.0], &[2.0, 0.5], &[5.0, 5.0],
            &[0.2, 0.1], &[3.0, 3.0], &[1.5, 0.0], &[4.0, 1.0],
        ]);
        let q = [0.1, 0.1];
        let pdx = PdxBase::from_vectors(&base, 3);
        for k in 1..=4 {
            // eps0 huge => never prunes => exact.
            assert_eq!(knn_pdx_adsampling(&pdx, &q, k, 100.0, 1), knn(&base, &q, k), "k={k}");
        }
    }

    #[test]
    fn adsampling_conservative_matches_exact() {
        // With a huge eps0 the pruning never triggers, so ADSampling must return
        // exactly what brute-force KNN returns.
        let base = vecs(&[
            &[0.0, 0.0], &[1.0, 1.0], &[2.0, 0.5], &[5.0, 5.0],
            &[0.2, 0.1], &[3.0, 3.0], &[1.5, 0.0], &[4.0, 1.0],
        ]);
        let q = [0.1, 0.1];
        for k in 1..=4 {
            assert_eq!(
                knn_adsampling(&base, &q, k, 100.0, 1),
                knn(&base, &q, k),
                "k={k}"
            );
        }
    }

    #[test]
    fn tiled_matches_per_query() {
        // Batched scan must return identical results to the per-query scan.
        let base = vecs(&[
            &[0.0, 0.0], &[1.0, 1.0], &[2.0, 0.5], &[5.0, 5.0],
            &[0.2, 0.1], &[3.0, 3.0], &[1.5, 0.0], &[4.0, 1.0],
        ]);
        let queries = vecs(&[&[0.0, 0.0], &[5.0, 5.0], &[1.4, 0.1], &[3.1, 2.9], &[0.1, 0.1]]);
        for &tile in &[1usize, 2, 3, 5, 100] {
            let plain = knn_batch(&base, &queries, 3);
            let tiled = knn_batch_tiled(&base, &queries, 3, tile);
            assert_eq!(plain, tiled, "tile={tile}");
        }
    }

    #[test]
    fn handles_2d_and_ties() {
        let base = vecs(&[&[0.0, 0.0], &[1.0, 1.0], &[1.0, -1.0], &[5.0, 5.0]]);
        let got = knn(&base, &[0.0, 0.0], 1);
        assert_eq!(got, vec![0]);
        let got2 = knn(&base, &[0.0, 0.0], 3);
        assert_eq!(got2.len(), 3);
        assert_eq!(got2[0], 0);
        assert!(!got2.contains(&3)); // the far point is excluded
    }
}
