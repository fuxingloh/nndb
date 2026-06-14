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
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
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
        // 1-D points at 0,1,2,3,4; query at 0 -> nearest are 0,1,2.
        let base = vecs(&[&[0.0], &[4.0], &[1.0], &[3.0], &[2.0]]);
        let got = knn(&base, &[0.0], 3);
        assert_eq!(got, vec![0, 2, 4]); // indices of points 0.0, 1.0, 2.0
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
