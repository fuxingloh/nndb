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

/// RaBitQ 1-bit code (Gao & Long, SIGMOD 2024): rotated sign bits + a per-vector
/// L1 norm of the rotated vector. The unbiased estimate of ⟨q,o⟩ for unit vectors
/// is ⟨q,code⟩/⟨code,o⟩ = (2·Σ_{set bits} q'_i − Σ q') / ‖o'‖₁, where q' is the
/// rotated query and the √D from the {±1/√D} codebook cancels. The per-vector
/// ‖o'‖₁ is what unbiases the estimate vs plain asymmetric scoring.
pub struct RaBitQ {
    pub bits: QuantBinary, // rotated sign bits
    pub norm1: Vec<f32>,   // ‖rotated vector (first `dim` dims)‖₁ per vector
}

impl RaBitQ {
    pub fn build(v: &Vectors, rot: &Rotation, bits: usize) -> Self {
        assert_eq!(rot.dim, v.dim, "rotation dim must match vector dim");
        let dim = if bits == 0 { v.dim } else { bits.min(v.dim) };
        let words = dim.div_ceil(64);
        let n = v.len();
        let mut data = vec![0u64; n * words];
        let mut norm1 = vec![0f32; n];
        data.par_chunks_mut(words)
            .zip(norm1.par_iter_mut())
            .enumerate()
            .for_each(|(i, (out, nrm))| {
                let mut r = vec![0f32; v.dim];
                rot.apply_into(v.row(i), &mut r);
                let mut s = 0f32;
                for d in 0..dim {
                    let x = r[d];
                    s += x.abs();
                    if x > 0.0 {
                        out[d / 64] |= 1u64 << (d % 64);
                    }
                }
                *nrm = s.max(1e-12);
            });
        RaBitQ { bits: QuantBinary { data, words, dim }, norm1 }
    }
}

/// Top-C by the RaBitQ unbiased estimate. `q_rot` is the ROTATED query.
pub fn knn_rabitq(rb: &RaBitQ, q_rot: &[f32], c: usize) -> Vec<u32> {
    let dim = rb.bits.dim;
    let sumq: f32 = q_rot[..dim].iter().sum();
    let mut heap: BinaryHeap<Reverse<(ScoreF, u32)>> = BinaryHeap::with_capacity(c + 1);
    for i in 0..rb.bits.len() {
        let masked = asym_score(q_rot, rb.bits.row(i)); // Σ q'_i over set bits
        let est = (2.0 * masked - sumq) / rb.norm1[i];
        let s = ScoreF(est);
        if heap.len() < c {
            heap.push(Reverse((s, i as u32)));
        } else if s > heap.peek().unwrap().0 .0 {
            heap.pop();
            heap.push(Reverse((s, i as u32)));
        }
    }
    heap.into_iter().map(|Reverse((_, i))| i).collect()
}

/// RaBitQ funnel: rotate query → estimate top-C → exact f32 rerank → top-k.
pub fn knn_rabitq_rerank_batch(
    rb: &RaBitQ,
    fbase: &Vectors,
    fqueries: &Vectors,
    rot: &Rotation,
    k: usize,
    c: usize,
) -> Vec<Vec<u32>> {
    (0..fqueries.len())
        .into_par_iter()
        .map(|q| {
            let qr = rot.apply(fqueries.row(q));
            let cands = knn_rabitq(rb, &qr, c.max(k));
            if c == 0 {
                cands.into_iter().take(k).collect()
            } else {
                rerank(fbase, fqueries.row(q), &cands, k)
            }
        })
        .collect()
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
    reg: bool,
    pf: bool,
) -> Vec<Vec<u32>> {
    let nq = bqueries.len();
    let want = if c == 0 { k } else { c.max(k) };
    let n = bbase.len();
    let words = bbase.words;
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
                if reg {
                    // doc-word outer: keep each doc word in a register, reused
                    // across the tile (scalar popcount, no VPOPCNTDQ over words).
                    acc.iter_mut().for_each(|a| *a = 0);
                    for w in 0..words {
                        let dw = doc[w];
                        for j in 0..t {
                            acc[j] += (qrows[j][w] ^ dw).count_ones();
                        }
                    }
                } else {
                    // per-query hamming (autovectorizes to VPOPCNTDQ over words);
                    // doc stays hot in L1 across the tile.
                    for j in 0..t {
                        acc[j] = hamming(qrows[j], doc);
                    }
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
                    chunk[j] = if pf {
                        rerank_pf(fbase, fqueries.row(q0 + j), &cands, k)
                    } else {
                        rerank(fbase, fqueries.row(q0 + j), &cands, k)
                    };
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

/// Rerank candidate ids with int8 dot (max == cosine for unit vectors) → top-k.
/// 4× smaller rerank store than f32 and 4× less traffic on the random candidate
/// gather, at a small precision cost vs exact f32 rescoring.
pub fn rerank_i8(i8base: &QuantI8, query: &[i8], cands: &[u32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(i32, u32)> = cands
        .iter()
        .map(|&c| (dot_i8(query, i8base.row(c as usize)), c))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0)); // descending dot
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Two-stage funnel with an int8 rerank tier (instead of f32). c=0 → scan only.
pub fn knn_binary_funnel_i8_batch(
    bbase: &QuantBinary,
    bqueries: &QuantBinary,
    i8base: &QuantI8,
    i8queries: &QuantI8,
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
                rerank_i8(i8base, i8queries.row(q), &cands, k)
            }
        })
        .collect()
}

/// Rerank with software prefetch of upcoming candidate rows. The candidate gather
/// is random-access into the 3.9 GB f32 store; prefetching rows PF ahead hides the
/// initial DRAM latency that the hardware prefetcher (which only streams *within* a
/// row once touched) can't. x86 only; falls back to plain rerank elsewhere.
pub fn rerank_pf(fbase: &Vectors, fquery: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
    const PF: usize = 8;
    let mut scored: Vec<(f32, u32)> = Vec::with_capacity(cands.len());
    for (idx, &c) in cands.iter().enumerate() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if let Some(&nc) = cands.get(idx + PF) {
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                let p = fbase.data.as_ptr().add(nc as usize * fbase.dim) as *const i8;
                _mm_prefetch(p, _MM_HINT_T0);
                _mm_prefetch(p.add(64), _MM_HINT_T0);
            }
        }
        scored.push((crate::search::l2_sq(fquery, fbase.row(c as usize)), c));
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// ADSampling-pruned rerank, STACKED on the binary funnel. The binary scan gives
/// C candidates; here we rescore them with early-terminated exact L2 on the
/// rotated f32 store: a candidate that provably can't beat the current k-th best
/// is dropped without finishing its dims. Lets the funnel use a larger C (more
/// recall) without the rerank cost growing linearly. `rq`/`rfbase` are ROTATED
/// (L2 preserved, partial distances unbiased). eps0 controls pruning aggression.
pub fn rerank_adsampling(
    rfbase: &Vectors,
    rq: &[f32],
    cands: &[u32],
    k: usize,
    eps0: f32,
    delta: usize,
) -> Vec<u32> {
    let d = rfbase.dim;
    let step = delta.max(1);
    let mut heap: BinaryHeap<(ScoreF, u32)> = BinaryHeap::with_capacity(k + 1);
    for &c in cands {
        let o = rfbase.row(c as usize);
        if heap.len() < k {
            let mut res = 0f32;
            for j in 0..d {
                let df = rq[j] - o[j];
                res += df * df;
            }
            heap.push((ScoreF(res), c));
            continue;
        }
        let thr = heap.peek().unwrap().0 .0;
        let mut res = 0f32;
        let mut i = 0usize;
        let mut pruned = false;
        while i < d {
            let end = (i + step).min(d);
            for j in i..end {
                let df = rq[j] - o[j];
                res += df * df;
            }
            i = end;
            if i < d {
                let fi = i as f32;
                let t = 1.0 + eps0 / fi.sqrt();
                if res >= thr * (fi / d as f32) * t * t {
                    pruned = true;
                    break;
                }
            }
        }
        if !pruned && res < thr {
            heap.pop();
            heap.push((ScoreF(res), c));
        }
    }
    let mut v: Vec<(ScoreF, u32)> = heap.into_iter().collect();
    v.sort_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap());
    v.into_iter().map(|(_, i)| i).collect()
}

/// Tiled binary scan (016) + ADSampling-pruned rerank (034) combined. The two
/// optimizations are orthogonal — tiling amortizes scan bandwidth across a query
/// tile, ADSampling trims the rerank — so stacking them should lift QPS at no
/// recall cost (with eps0 conservative). bqueries=rotated binary; rfbase/rqueries
/// =rotated f32.
pub fn knn_binary_funnel_tiled_ads(
    bbase: &QuantBinary,
    bqueries: &QuantBinary,
    rfbase: &Vectors,
    rqueries: &Vectors,
    k: usize,
    c: usize,
    tile: usize,
    eps0: f32,
    delta: usize,
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
            for i in 0..n {
                let doc = bbase.row(i);
                for j in 0..t {
                    let h = hamming(qrows[j], doc);
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
                let cands: Vec<u32> = heap.into_iter().map(|(_, i)| i).collect();
                chunk[j] = if c == 0 {
                    cands.into_iter().take(k).collect()
                } else {
                    rerank_adsampling(rfbase, rqueries.row(q0 + j), &cands, k, eps0, delta)
                };
            }
        });
    results
}

/// Binary funnel with an ADSampling-pruned rerank tier (rotated). bqueries are
/// rotated binary codes; rqueries/rfbase are rotated f32 for the rerank.
pub fn knn_binary_funnel_ads_batch(
    bbase: &QuantBinary,
    bqueries: &QuantBinary,
    rfbase: &Vectors,
    rqueries: &Vectors,
    k: usize,
    c: usize,
    eps0: f32,
    delta: usize,
) -> Vec<Vec<u32>> {
    (0..bqueries.len())
        .into_par_iter()
        .map(|q| {
            let cands = knn_binary(bbase, bqueries.row(q), c.max(k));
            rerank_adsampling(rfbase, rqueries.row(q), &cands, k, eps0, delta)
        })
        .collect()
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

/// Binary scan only (no rerank), batched.
pub fn knn_binary_batch(base: &QuantBinary, queries: &QuantBinary, k: usize) -> Vec<Vec<u32>> {
    (0..queries.len())
        .into_par_iter()
        .map(|q| knn_binary(base, queries.row(q), k))
        .collect()
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
