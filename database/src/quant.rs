//! Quantized representations + search for the two-stage (scan → rerank) funnel.
//!
//! Vectors are unit-normalized at prep time, so ranking by **dot product** ==
//! ranking by cosine. int8 keeps a shared global scale; ordering by the integer
//! dot is monotonic with the float dot, so recall is near-exact for
//! compression-aware embeddings (e.g. Cohere v3).

use crate::fvecs::Vectors;
use rayon::prelude::*;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------

/// In-place fast Walsh–Hadamard transform (len must be a power of two). H/√n is
/// orthogonal; we skip the 1/√n scale since sign-binarization is scale-invariant.
fn fwht(a: &mut [f32]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Structured random orthogonal rotation: alternating random ±1 sign-flips and
/// FWHTs (a fast Johnson–Lindenstrauss transform). This is the RaBitQ/ITQ trick —
/// rotating before sign-binarization spreads each vector's information evenly
/// across dimensions so every sign bit carries independent signal, sharply
/// improving binary-code quality. Deterministic (seeded) so base and query use
/// the SAME rotation.
pub struct Rotation {
    signs: Vec<Vec<f32>>,
    pub dim: usize,
}

impl Rotation {
    pub fn new(dim: usize, rounds: usize, seed: u64) -> Self {
        assert!(dim.is_power_of_two(), "rotation requires power-of-two dim, got {dim}");
        let mut s = seed;
        let mut next = || {
            // splitmix64
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let signs = (0..rounds.max(1))
            .map(|_| (0..dim).map(|_| if next() & 1 == 0 { 1.0f32 } else { -1.0 }).collect())
            .collect();
        Rotation { signs, dim }
    }

    /// Rotate one vector (must be `dim`-long) into `out`.
    pub fn apply_into(&self, x: &[f32], out: &mut [f32]) {
        out.copy_from_slice(x);
        for round in &self.signs {
            for (o, s) in out.iter_mut().zip(round.iter()) {
                *o *= *s;
            }
            fwht(out);
        }
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.dim];
        self.apply_into(x, &mut out);
        out
    }
}

/// Apply the rotation to every row of a vector set (parallel), returning a new
/// rotated `Vectors`. Used by ADSampling, which needs random-rotated inputs.
pub fn rotate_vectors(v: &Vectors, rot: &Rotation) -> Vectors {
    assert_eq!(rot.dim, v.dim, "rotation dim must match vector dim");
    let dim = v.dim;
    let mut data = vec![0f32; v.data.len()];
    data.par_chunks_mut(dim)
        .enumerate()
        .for_each(|(i, out)| rot.apply_into(v.row(i), out));
    Vectors { data, dim }
}

/// Mean vector over all rows (the cell centroid). Used for residual encoding:
/// inside one IVF cell, vectors cluster around the centroid, so subtracting it
/// before sign-binarization sharpens the codes (history 046).
pub fn centroid(v: &Vectors) -> Vec<f32> {
    let dim = v.dim;
    let mut c = vec![0f64; dim];
    for i in 0..v.len() {
        let r = v.row(i);
        for d in 0..dim {
            c[d] += r[d] as f64;
        }
    }
    let n = v.len().max(1) as f64;
    c.iter().map(|x| (x / n) as f32).collect()
}

/// Subtract `c` from every row (parallel) → residual vectors. Stage-1 only; rerank
/// and ground truth stay on the raw vectors.
pub fn subtract_centroid(v: &Vectors, c: &[f32]) -> Vectors {
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

    /// Pack only the first `bits` dimensions (a Matryoshka-style prefix). Scanning
    /// fewer words = less bandwidth per doc; recall lost in stage-1 is recovered
    /// by the full-precision rerank. `bits >= dim` is the same as `from_f32`.
    pub fn from_f32_prefix(v: &Vectors, bits: usize) -> Self {
        let dim = bits.min(v.dim);
        let words = dim.div_ceil(64);
        let n = v.len();
        let mut data = vec![0u64; n * words];
        for i in 0..n {
            let row = v.row(i);
            let out = &mut data[i * words..(i + 1) * words];
            for d in 0..dim {
                if row[d] > 0.0 {
                    out[d / 64] |= 1u64 << (d % 64);
                }
            }
        }
        QuantBinary { data, words, dim }
    }

    /// Like `from_f32_prefix` but applies the random rotation before taking the
    /// first `bits` sign bits (RaBitQ/ITQ). Rotation runs per row in parallel.
    pub fn from_f32_rotated(v: &Vectors, rot: &Rotation, bits: usize) -> Self {
        assert_eq!(rot.dim, v.dim, "rotation dim must match vector dim");
        let dim = if bits == 0 { v.dim } else { bits.min(v.dim) };
        let words = dim.div_ceil(64);
        let n = v.len();
        let mut data = vec![0u64; n * words];
        data.par_chunks_mut(words).enumerate().for_each(|(i, out)| {
            let mut rotated = vec![0f32; v.dim];
            rot.apply_into(v.row(i), &mut rotated);
            for d in 0..dim {
                if rotated[d] > 0.0 {
                    out[d / 64] |= 1u64 << (d % 64);
                }
            }
        });
        QuantBinary { data, words, dim }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[u64] {
        &self.data[i * self.words..(i + 1) * self.words]
    }
    pub fn len(&self) -> usize {
        self.data.len() / self.words
    }
}

/// Binarize one query through the rotation, then take the first `bits` sign words.
pub fn binarize_query_rotated(query: &[f32], rot: &Rotation, bits: usize) -> Vec<u64> {
    let rotated = rot.apply(query);
    binarize_query(&rotated, bits)
}

/// Binarize a single query vector's first `bits` dims into packed sign words
/// (matches `from_f32_prefix`'s layout). For the serving path, which binarizes
/// one query at a time. `bits == 0` means full length.
pub fn binarize_query(query: &[f32], bits: usize) -> Vec<u64> {
    let dim = if bits == 0 { query.len() } else { bits.min(query.len()) };
    let words = dim.div_ceil(64);
    let mut out = vec![0u64; words];
    for d in 0..dim {
        if query[d] > 0.0 {
            out[d / 64] |= 1u64 << (d % 64);
        }
    }
    out
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

/// Tiled binary funnel: each doc is loaded once and compared against a tile of
/// `tile` queries before moving on, so the 122 MB binary base streams once per
/// tile instead of once per query — cutting base bandwidth ~tile× on a memory-
/// bound box. Heap selection + f32 rerank; respects prefix bbase (scan-bits).
pub fn knn_binary_funnel_tiled(
    bbase: &QuantBinary,
    bqueries: &QuantBinary,
    fbase: &Vectors,
    fqueries: &Vectors,
    k: usize,
    c: usize,
    tile: usize,
) -> Vec<Vec<u32>> {
    let nq = bqueries.len();
    let want = if c == 0 { k } else { c.max(k) };
    let n = bbase.len();
    let tile = tile.max(1);
    let mut results: Vec<Vec<u32>> = (0..nq).map(|_| Vec::new()).collect();
    results
        .par_chunks_mut(tile)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let q0 = ci * tile;
            let t = chunk.len();
            let qrows: Vec<&[u64]> = (0..t).map(|j| bqueries.row(q0 + j)).collect();
            let mut heaps: Vec<BinaryHeap<(u32, u32)>> =
                (0..t).map(|_| BinaryHeap::with_capacity(want + 1)).collect();
            let mut acc = vec![0u32; t];
            for i in 0..n {
                let doc = bbase.row(i);
                // per-query hamming (autovectorizes to VPOPCNTDQ over words);
                // doc stays hot in L1 across the tile.
                for j in 0..t {
                    acc[j] = hamming(qrows[j], doc);
                }
                for j in 0..t {
                    let h = acc[j];
                    let hp = &mut heaps[j];
                    if hp.len() < want {
                        hp.push((h, i as u32));
                    } else if h < hp.peek().unwrap().0 {
                        hp.pop();
                        hp.push((h, i as u32));
                    }
                }
            }
            for j in 0..t {
                let heap = std::mem::take(&mut heaps[j]);
                if c == 0 {
                    let mut v = heap.into_vec();
                    v.sort_unstable();
                    chunk[j] = v.into_iter().map(|(_, i)| i).collect();
                } else {
                    let cands: Vec<u32> = heap.into_iter().map(|(_, i)| i).collect();
                    chunk[j] = rerank(fbase, fqueries.row(q0 + j), &cands, k);
                }
            }
        });
    results
}

/// Scan ONE query across `shards` rayon tasks (split the doc range), then merge
/// the per-shard top-C. Cuts single-query latency ~linearly — the scan is
/// embarrassingly parallel — at the cost of spending all cores on one request
/// (a latency-vs-throughput trade; do not use under the serving model). Returns
/// up to C candidate ids (unordered) for rerank.
pub fn knn_binary_query_parallel(bbase: &QuantBinary, query: &[u64], c: usize, shards: usize) -> Vec<u32> {
    let n = bbase.len();
    let shards = shards.max(1);
    let chunk = n.div_ceil(shards);
    let partials: Vec<Vec<(u32, u32)>> = (0..shards)
        .into_par_iter()
        .map(|s| {
            let lo = s * chunk;
            let hi = ((s + 1) * chunk).min(n);
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(c + 1);
            for i in lo..hi {
                let h = hamming(query, bbase.row(i));
                if heap.len() < c {
                    heap.push((h, i as u32));
                } else if h < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push((h, i as u32));
                }
            }
            heap.into_vec()
        })
        .collect();
    // global top-C is contained in the union of shard top-Cs
    let mut all: Vec<(u32, u32)> = partials.into_iter().flatten().collect();
    all.sort_unstable();
    all.into_iter().take(c).map(|(_, i)| i).collect()
}


/// Parallel rerank: compute the C candidate L2s across rayon, then serial top-k.
/// For the single-query latency path, where the C rescores are otherwise serial.
pub fn rerank_par(fbase: &Vectors, fquery: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = cands
        .par_iter()
        .map(|&c| (crate::search::l2_sq(fquery, fbase.row(c as usize)), c))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}
/// Selection strategy for the binary scan's top-C step.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BinSel {
    /// Bounded max-heap of size C — O(n log C), branchy.
    Heap,
    /// Counting selection on the bounded integer Hamming distance — O(n),
    /// branch-free histogram + threshold. Candidate order NOT preserved
    /// (fine for the rerank funnel, which rescores and re-sorts).
    Count,
}

/// Top-C smallest Hamming via counting selection. Hamming ∈ [0, dim] is a small
/// bounded integer, so one histogram pass + a threshold scan selects the C
/// nearest in O(n) without a comparison heap. Returns up to C ids (unordered —
/// fine for the rerank funnel, which rescores). Branch-free, so it cuts
/// single-query latency vs the branchy heap; but it touches an extra `dists`
/// buffer per query, which costs aggregate bandwidth in the all-cores batch pass.
pub fn knn_binary_count(base: &QuantBinary, query: &[u64], c: usize) -> Vec<u32> {
    let n = base.len();
    let dim = base.dim;
    let mut dists = vec![0u16; n];
    let mut counts = vec![0u32; dim + 2];
    for i in 0..n {
        let h = hamming(query, base.row(i)) as usize;
        dists[i] = h as u16;
        counts[h] += 1;
    }
    // smallest threshold t whose cumulative count (through t) reaches C
    let mut acc = 0u32;
    let mut t = 0usize;
    while t <= dim && acc + counts[t] < c as u32 {
        acc += counts[t];
        t += 1;
    }
    let mut out = Vec::with_capacity(c);
    for i in 0..n {
        if (dists[i] as usize) < t {
            out.push(i as u32);
        }
    }
    if out.len() < c {
        for i in 0..n {
            if out.len() >= c {
                break;
            }
            if dists[i] as usize == t {
                out.push(i as u32);
            }
        }
    }
    out
}

#[inline]
pub fn knn_binary_sel(base: &QuantBinary, query: &[u64], k: usize, sel: BinSel) -> Vec<u32> {
    match sel {
        BinSel::Heap => knn_binary(base, query, k),
        BinSel::Count => knn_binary_count(base, query, k),
    }
}

/// Two-stage funnel with a selectable top-C strategy. c=0 → scan top-k only.
pub fn knn_binary_funnel_batch(
    bbase: &QuantBinary,
    bqueries: &QuantBinary,
    fbase: &Vectors,
    fqueries: &Vectors,
    k: usize,
    c: usize,
    sel: BinSel,
) -> Vec<Vec<u32>> {
    (0..bqueries.len())
        .into_par_iter()
        .map(|q| {
            if c == 0 {
                knn_binary_sel(bbase, bqueries.row(q), k, sel)
            } else {
                let cands = knn_binary_sel(bbase, bqueries.row(q), c.max(k), sel);
                rerank(fbase, fqueries.row(q), &cands, k)
            }
        })
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
}
