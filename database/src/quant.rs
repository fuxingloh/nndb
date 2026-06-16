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

// ---------------------------------------------------------------------------
// Binary quantization (1 bit/dim = sign) + two-stage rerank.
// ---------------------------------------------------------------------------

/// Sign-bit binary vectors, packed into u64 words. For angular/cosine,
/// agreement of sign bits approximates similarity → rank by *min* Hamming.
pub struct QuantBinary {
    pub data: Vec<u64>,
    pub words: usize, // u64 words per vector = ceil(dim/64)
    pub dim: usize,
}

impl QuantBinary {
    pub fn from_f32(v: &Vectors) -> Self {
        let words = v.dim.div_ceil(64);
        let n = v.len();
        let mut data = vec![0u64; n * words];
        for i in 0..n {
            let row = v.row(i);
            let out = &mut data[i * words..(i + 1) * words];
            for (d, &x) in row.iter().enumerate() {
                if x > 0.0 {
                    out[d / 64] |= 1u64 << (d % 64);
                }
            }
        }
        QuantBinary { data, words, dim: v.dim }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[u64] {
        &self.data[i * self.words..(i + 1) * self.words]
    }
    pub fn len(&self) -> usize {
        self.data.len() / self.words
    }
}

/// Hamming distance (differing sign bits) via XOR + popcount (hardware POPCNT/CNT).
#[inline]
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    let mut d = 0u32;
    for i in 0..a.len() {
        d += (a[i] ^ b[i]).count_ones();
    }
    d
}

/// Top-k by smallest Hamming (most agreeing sign bits).
pub fn knn_binary(base: &QuantBinary, query: &[u64], k: usize) -> Vec<u32> {
    // Max-heap on (hamming, idx): root is the worst kept; keep the k smallest.
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
    for i in 0..base.len() {
        let h = hamming(query, base.row(i));
        if heap.len() < k {
            heap.push((h, i as u32));
        } else if h < heap.peek().unwrap().0 {
            heap.pop();
            heap.push((h, i as u32));
        }
    }
    let mut v = heap.into_vec();
    v.sort_unstable(); // ascending Hamming, then idx
    v.into_iter().map(|(_, i)| i).collect()
}

/// Rerank candidate ids with exact f32 (L2 == cosine for unit vectors) → top-k.
pub fn rerank(fbase: &Vectors, fquery: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = cands
        .iter()
        .map(|&c| (crate::search::l2_sq(fquery, fbase.row(c as usize)), c))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Binary scan only (no rerank), batched.
pub fn knn_binary_batch(base: &QuantBinary, queries: &QuantBinary, k: usize) -> Vec<Vec<u32>> {
    (0..queries.len())
        .into_par_iter()
        .map(|q| knn_binary(base, queries.row(q), k))
        .collect()
}

/// Two-stage: binary scan → top-`c` candidates → exact f32 rerank → top-k.
pub fn knn_binary_rerank_batch(
    bbase: &QuantBinary,
    bqueries: &QuantBinary,
    fbase: &Vectors,
    fqueries: &Vectors,
    k: usize,
    c: usize,
) -> Vec<Vec<u32>> {
    (0..bqueries.len())
        .into_par_iter()
        .map(|q| {
            let cands = knn_binary(bbase, bqueries.row(q), c.max(k));
            rerank(fbase, fqueries.row(q), &cands, k)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_rerank_recovers_exact() {
        // 4 unit-ish vectors; rerank of binary candidates should return exact top-1.
        let base = Vectors {
            data: vec![1.0, 0.1, 0.9, 0.2, -1.0, 0.0, 0.2, 1.0],
            dim: 2,
        };
        let q = Vectors { data: vec![1.0, 0.15], dim: 2 };
        let bb = QuantBinary::from_f32(&base);
        let bq = QuantBinary::from_f32(&q);
        let got = knn_binary_rerank_batch(&bb, &bq, &base, &q, 1, 4);
        assert_eq!(got[0][0], 0); // exact nearest after rerank
    }

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
