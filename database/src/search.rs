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
