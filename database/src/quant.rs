//! Quantized representations + search for the two-stage (scan → rerank) funnel.
//!
//! Vectors are unit-normalized at prep time, so ranking by **dot product** ==
//! ranking by cosine. int8 keeps a shared global scale; ordering by the integer
//! dot is monotonic with the float dot, so recall is near-exact for
//! compression-aware embeddings (e.g. Cohere v3).

use crate::fvecs::Vectors;
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
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

/// Hamming distance (differing sign bits) via XOR + popcount.
/// Keep this loop simple on purpose: `count_ones` over the slice autovectorizes
/// to hardware *vector* popcount (NEON `CNT` on ARM, `VPOPCNTDQ` on AVX-512), so
/// the naive form is already SIMD. Manually splitting the reduction into multiple
/// scalar accumulators (the trick that helps the f32 FMA kernel) *defeats* that
/// autovectorization — measured ~2x slower on M3, see history/012. Don't.
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

// ---------------------------------------------------------------------------
// Asymmetric scoring: full-precision query x binary doc.
// score = sum_i q_i * sign(doc_i) = 2*(sum of q_i where doc bit set) - sum(q).
// Ranking only needs the masked sum (the 2x and -sum(q) are monotonic per query),
// so the query keeps full precision while docs stay 1 bit. Higher stage-1 recall
// than symmetric Hamming (which also throws away the query). Rank by MAX score.
// ---------------------------------------------------------------------------

/// Total-order f32 wrapper for the score heap.
#[derive(Clone, Copy, PartialEq)]
struct ScoreF(f32);
impl Eq for ScoreF {}
impl PartialOrd for ScoreF {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for ScoreF {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Asymmetric score: sum of float-query values at the doc's set sign-bits.
/// (Direct form; the `vpshufb` LUT is a faster way to compute the same value.)
#[inline]
pub fn asym_score(query: &[f32], doc: &[u64]) -> f32 {
    let mut s = 0.0f32;
    for (w, &word) in doc.iter().enumerate() {
        let mut bits = word;
        let base = w * 64;
        while bits != 0 {
            s += query[base + bits.trailing_zeros() as usize];
            bits &= bits - 1; // clear lowest set bit
        }
    }
    s
}

/// Top-k by largest asymmetric score. Dispatches to the LUT kernel when dim is a
/// multiple of 4 (the fast path), else the direct set-bit kernel (small/odd dims).
pub fn knn_asym(bbase: &QuantBinary, query: &[f32], k: usize) -> Vec<u32> {
    if bbase.dim % 4 == 0 {
        knn_asym_lut(bbase, query, k)
    } else {
        knn_asym_setbit(bbase, query, k)
    }
}

/// Direct kernel: iterate set bits + gather. Correct but slow (no precompute).
fn knn_asym_setbit(bbase: &QuantBinary, query: &[f32], k: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<Reverse<(ScoreF, u32)>> = BinaryHeap::with_capacity(k + 1);
    for i in 0..bbase.len() {
        let s = ScoreF(asym_score(query, bbase.row(i)));
        if heap.len() < k {
            heap.push(Reverse((s, i as u32)));
        } else if s > heap.peek().unwrap().0 .0 {
            heap.pop();
            heap.push(Reverse((s, i as u32)));
        }
    }
    let mut v: Vec<(ScoreF, u32)> = heap.into_iter().map(|Reverse(x)| x).collect();
    v.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, i)| i).collect()
}

/// LUT kernel — the Exa trick. Since docs are binary, precompute one table per
/// 4-bit nibble position: table[p][pattern] = masked query-sum for those 4 dims
/// under that sign pattern. Scanning a doc = sum the looked-up partials (one
/// indexed lookup per nibble), instead of per-bit gathers. 1024-d → 256 tables ×
/// 16 = 16 KB (L1-resident). Same score as the direct kernel → same recall.
/// (Full speed wants `vpshufb`/`vpermb` SIMD gather over an int8 table; this is
/// the scalar version that proves the structure.)
fn knn_asym_lut(bbase: &QuantBinary, query: &[f32], k: usize) -> Vec<u32> {
    let nib = bbase.dim / 4;
    let mut lut = vec![0f32; nib * 16];
    for p in 0..nib {
        let (q0, q1, q2, q3) = (query[4 * p], query[4 * p + 1], query[4 * p + 2], query[4 * p + 3]);
        let t = &mut lut[16 * p..16 * p + 16];
        for (pat, e) in t.iter_mut().enumerate() {
            *e = (if pat & 1 != 0 { q0 } else { 0.0 })
                + (if pat & 2 != 0 { q1 } else { 0.0 })
                + (if pat & 4 != 0 { q2 } else { 0.0 })
                + (if pat & 8 != 0 { q3 } else { 0.0 });
        }
    }
    let mut heap: BinaryHeap<Reverse<(ScoreF, u32)>> = BinaryHeap::with_capacity(k + 1);
    for i in 0..bbase.len() {
        let doc = bbase.row(i);
        let mut s = 0f32;
        for p in 0..nib {
            let word = doc[p >> 4]; // 16 nibbles per u64
            let nibble = ((word >> ((p & 15) * 4)) & 0xF) as usize;
            s += lut[16 * p + nibble];
        }
        let sc = ScoreF(s);
        if heap.len() < k {
            heap.push(Reverse((sc, i as u32)));
        } else if sc > heap.peek().unwrap().0 .0 {
            heap.pop();
            heap.push(Reverse((sc, i as u32)));
        }
    }
    let mut v: Vec<(ScoreF, u32)> = heap.into_iter().map(|Reverse(x)| x).collect();
    v.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, i)| i).collect()
}

/// Asymmetric scan, optional f32 rerank (c=0 → take the asym top-k directly).
pub fn knn_asym_rerank_batch(
    bbase: &QuantBinary,
    fbase: &Vectors,
    fqueries: &Vectors,
    k: usize,
    c: usize,
) -> Vec<Vec<u32>> {
    (0..fqueries.len())
        .into_par_iter()
        .map(|q| {
            if c == 0 {
                knn_asym(bbase, fqueries.row(q), k)
            } else {
                let cands = knn_asym(bbase, fqueries.row(q), c.max(k));
                rerank(fbase, fqueries.row(q), &cands, k)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asym_beats_symmetric_ranking() {
        // Query close to vec 0; asym (keeps query magnitude) ranks it first.
        let base = Vectors {
            data: vec![0.9, 0.1, -0.9, 0.1, 0.1, 0.9, -0.1, -0.9],
            dim: 2,
        };
        let q = Vectors { data: vec![1.0, 0.2], dim: 2 };
        let bb = QuantBinary::from_f32(&base);
        let got = knn_asym(&bb, q.row(0), 1);
        // vec 0 = (0.9,0.1) -> bits (1,1); query (1.0,0.2) set-bit sum = 1.2 (max)
        assert_eq!(got[0], 0);
    }

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
