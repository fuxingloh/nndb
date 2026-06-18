# 050 — SIMD step 1 (safe hints): the binary scan is memory-bound, not popcount-bound

Perf record: [`050-simd-hints-scan.json`](./050-simd-hints-scan.json). Granite box
(Xeon 6975P-C, AVX-512 VPOPCNTDQ). `src/bin/simdbench.rs`. Single-thread Hamming-kernel
microbench, 1024-bit codes, isolating the kernel (no heap; data cache-resident).

## The question

Can we push the binary-scan Hamming kernel with SIMD? Step 1: **safe, stable**
autovectorization *hints* only — before reaching for nightly `core::simd` or unsafe
`std::arch`. The current kernel is a naive `count_ones` loop that already lowers to
VPOPCNTDQ; 012/023 showed hand-restructuring it *loses*. So: is there headroom at all?

## Result

Single-thread Gcmp/s, in-L2 (512 KB, compute-isolated) vs at-scale (25 MB, memory):

| variant | in-L2 | at scale |
|---|---|---|
| baseline `count_ones` | 0.289 | 0.230 |
| `iter.zip().map().sum()` | 0.294 | 0.230 |
| `chunks_exact(8)` | 0.282 | 0.228 |
| **`interleave4` (doc-level ILP)** | **0.357 (+23%)** | 0.233 (~0%) |
| `manual_acc4` (the 012 loser) | 0.092 (−68%) | 0.091 |

1. **Safe hints buy nothing.** `chunks_exact`, iterator fusion → parity with baseline.
   The compiler already emits VPOPCNTDQ; you can't hint past what it already does.
2. **`manual_acc4` is 3× slower — 012/023 reconfirmed on AVX-512.** Hand-splitting the
   popcount into manual accumulators defeats the autovectorizer (forces scalar `popcnt`).
3. **Doc-level ILP wins +23% — but only in cache.** Keeping 4 independent full-width
   Hammings in flight (distinct from 023, which interleaved *queries per word* and went
   scalar) fills the popcount pipeline. At 25 MB it vanishes (0.230→0.233).

## Why the in-cache win doesn't matter at scale

Two things collapse the ILP win at deployment scale:

- **The scan is memory-bandwidth-bound, not popcount-bound.** Single-thread drops
  0.29 (in-L2) → 0.23 (in-DRAM); 047's 8-core aggregate is 1.1 Gcmp/s = **0.14/core**,
  *below* single-thread 0.23 — i.e. 8 cores contend for DRAM bandwidth. The popcount
  units already have idle headroom; feeding them is the limit.
- **The tiled funnel already captures the ILP.** A tile of T queries means T
  independent Hammings per loaded doc — exactly the `interleave4` trick, already in the
  engine (016/038). There's no *additional* compute parallelism to extract.

## Conclusion

For the binary **scan**, SIMD has **~0% headroom at deployment scale**: safe hints are
parity, hand-SIMD loses (012/023), the only compute win (ILP) is in-cache-only and
already captured by tiling, and at scale the wall is DRAM bandwidth. Explicit
`core::simd`/unsafe `vpshufb` on the scan would be effort against memory bandwidth —
not worth it. **The lever at scale is fewer bytes per vector (bits/codes/residual),
not faster popcount.** This closes the parked SIMD question *for the scan*.

(Where SIMD *could* still pay is a different kernel — e.g. the asymmetric LUT (`vpshufb`
gather, history 011) for a recall-per-byte gain — but that's a separate path, not the
symmetric Hamming scan, and was cut as non-competitive scalar.)

## Caveats

- Random codes, kernel-isolated (no heap/selection) — measures raw popcount throughput,
  the quantity SIMD would affect. The full scan adds heap overhead on top (constant).
- Single-thread; the 8-core memory-contention point is cross-referenced from 047.
- `interleave4`'s in-cache win is real but only reachable when the working set fits L2,
  which a 1M+ cell never does (128 MB+).
