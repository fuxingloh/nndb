//! Quality metric for nearest-neighbor search.
//!
//! `recall@k` is the standard ANN-Benchmarks axis: of the `k` neighbors an
//! index returns for a query, how many are in the *true* top-`k`? Averaged over
//! all queries. Exact search scores 1.0; approximate indexes trade recall for
//! speed, and this is what we plot against QPS.

use crate::fvecs::IntVectors;
use std::collections::HashSet;

/// Mean recall@k of `found` against `truth` (ground-truth neighbor indices).
///
/// `found[q]` are the predicted neighbor ids for query `q`; `truth.row(q)` are
/// the true neighbors (at least `k` of them, ordered nearest-first).
pub fn recall_at_k(found: &[Vec<u32>], truth: &IntVectors, k: usize) -> f64 {
    assert!(
        found.len() <= truth.len(),
        "more results than ground-truth rows"
    );
    assert!(truth.dim >= k, "ground truth has fewer than k neighbors");

    let mut total = 0.0;
    for (q, predicted) in found.iter().enumerate() {
        let gold: HashSet<i32> = truth.row(q)[..k].iter().copied().collect();
        let hits = predicted
            .iter()
            .take(k)
            .filter(|&&id| gold.contains(&(id as i32)))
            .count();
        total += hits as f64 / k as f64;
    }
    total / found.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_partial_recall() {
        // 2 queries, ground-truth top-2 neighbors each.
        let truth = IntVectors {
            data: vec![10, 20, 30, 40],
            dim: 2,
        };
        // q0 exact match; q1 gets one of two right.
        let found = vec![vec![10, 20], vec![30, 99]];
        assert!((recall_at_k(&found, &truth, 2) - 0.75).abs() < 1e-9);
        // recall@1 only checks the single nearest: both correct -> 1.0
        assert!((recall_at_k(&found, &truth, 1) - 1.0).abs() < 1e-9);
    }
}
