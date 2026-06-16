//! Quantized representations + search for the two-stage (scan → rerank) funnel.
//!
//! Vectors are unit-normalized at prep time, so ranking by **dot product** ==
//! ranking by cosine. int8 keeps a shared global scale; ordering by the integer
//! dot is monotonic with the float dot, so recall is near-exact for
//! compression-aware embeddings (e.g. Cohere v3).

use crate::fvecs::Vectors;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// int8-quantized vectors: symmetric scalar quant with one global scale.
pub struct QuantI8 {
    pub data: Vec<i8>,
    pub dim: usize,
}

impl QuantI8 {
    /// Quantize f32 vectors to int8 using a single global absmax scale.
    pub fn from_f32(v: &Vectors) -> Self {
        let amax = v.data.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
        let inv = 127.0 / amax;
        let data = v
            .data
            .iter()
            .map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8)
            .collect();
        QuantI8 { data, dim: v.dim }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[i8] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }
}

/// int8 dot product, i32 accumulate. Multiple accumulators for ILP; with
/// `target-cpu=native` LLVM maps this to DotProd (NEON) / VNNI (x86).
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    const ACC: usize = 8;
    let mut acc = [0i32; ACC];
    let n = a.len();
    let blocks = n / ACC;
    for blk in 0..blocks {
        let off = blk * ACC;
        for l in 0..ACC {
            acc[l] += a[off + l] as i32 * b[off + l] as i32;
        }
    }
    let mut s: i32 = acc.iter().sum();
    for i in (blocks * ACC)..n {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

/// Top-k by **largest** int8 dot (cosine). Min-heap of size k keeps the k best.
pub fn knn_i8(base: &QuantI8, query: &[i8], k: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<Reverse<(i32, u32)>> = BinaryHeap::with_capacity(k + 1);
    for i in 0..base.len() {
        let d = dot_i8(query, base.row(i));
        if heap.len() < k {
            heap.push(Reverse((d, i as u32)));
        } else if d > heap.peek().unwrap().0 .0 {
            heap.pop();
            heap.push(Reverse((d, i as u32)));
        }
    }
    let mut v: Vec<(i32, u32)> = heap.into_iter().map(|Reverse(x)| x).collect();
    v.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // descending similarity
    v.into_iter().map(|(_, i)| i).collect()
}

pub fn knn_i8_batch(base: &QuantI8, queries: &QuantI8, k: usize) -> Vec<Vec<u32>> {
    (0..queries.len())
        .into_par_iter()
        .map(|q| knn_i8(base, queries.row(q), k))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i8_matches_exact_on_simple() {
        // Normalized-ish vectors; nearest by dot should match obvious answer.
        let base = Vectors {
            data: vec![
                1.0, 0.0, // a
                0.9, 0.1, // b (closest to query)
                -1.0, 0.0, // c (opposite)
                0.0, 1.0, // d
            ],
            dim: 2,
        };
        let q = Vectors { data: vec![1.0, 0.05], dim: 2 };
        let qb = QuantI8::from_f32(&base);
        let qq = QuantI8::from_f32(&q);
        let got = knn_i8(&qb, qq.row(0), 2);
        assert_eq!(got[0], 0); // [1,0] highest dot with [1,0.05]
        assert!(got.contains(&1));
        assert!(!got.contains(&2)); // opposite excluded
    }
}
