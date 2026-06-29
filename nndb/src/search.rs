//! Exact brute-force k-nearest-neighbor search.
//!
//! This is the correctness oracle and recall=1.0 anchor: it scans every base
//! vector for every query, so it is what `--write-ground-truth` uses, what recall
//! is measured against, and what the unit tests check. It is intentionally NOT a
//! throughput contender — the binary funnel (`quant.rs`) is the engine; this just
//! defines exactness.

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
        let base = vecs(&[
            &[0.0, 0.0], &[1.0, 1.0], &[2.0, 0.5], &[5.0, 5.0],
            &[0.2, 0.1], &[3.0, 3.0], &[1.5, 0.0], &[4.0, 1.0],
        ]);
        let q = [0.1, 0.1];
        let got = knn(&base, &q, 3);
        // ascending by squared distance: idx4 (0.2,0.1)=0.01, idx0 (0,0)=0.02, idx1 (1,1)=1.62
        assert_eq!(got, vec![4, 0, 1]);
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
