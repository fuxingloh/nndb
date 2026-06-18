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

## Scan-sharing flips single-thread to compute-bound (+73%)

The first cut above missed the *scan-sharing* axis (T queries riding one doc-read —
the tiling/carousel model). Single-thread, at-scale (25 MB), as T rises:

| T (riders/doc) | at-scale Gcmp/s |
|---|---|
| 1 | 0.215 (memory-bound) |
| 4 | 0.355 |
| 8 | **0.371 (+73%)** |
| 16–64 | ~0.37 (plateau) |

At T≥8 the at-scale throughput **equals the in-L2 compute ceiling** — the memory tax
is fully amortized and it's now popcount-bound. So scan-sharing *does* push single-
thread QPS up, hard. (The engine already captures this: tiling 016/038, carousel
039–041.)

## But at full chip it's the bandwidth wall, and sharding ≠ throughput

8 cores, N=5M (640 MB > L3), T=16:

| scheme | base DRAM reads | Gcmp/s |
|---|---|---|
| sharded carousel | ×1 (shards) | **1.139** |
| tiled batch (047 model) | ×cores | 1.071 |

They're **equal** — the shared 480 MB L3 absorbs the "redundant" reads, so reading the
base 8× isn't 8× the DRAM traffic. Both saturate at ~1.1 Gcmp/s = **0.14/core**, far
below the 0.33–0.37/core single-thread compute ceiling → the chip is **memory/L3-
bandwidth-bound**. The carousel's sharding gives **no peak-throughput win** over the
tiled batch (its win is latency-under-burst, 041), and faster popcount can't help.

## Conclusion (revised)

- **Scan-sharing pushes single-thread +73%** to the compute ceiling — already captured
  by tiling/carousel.
- **At full chip the scan saturates at the memory/L3 bandwidth wall (~1.1 Gcmp/s)**,
  where sharded == batched and SIMD (faster popcount) has nothing to add — per-core
  (0.14) sits well below the popcount ceiling (0.33).
- Safe hints = parity; hand-SIMD loses (012/023, reconfirmed).
- **The only lever for chip-level scan QPS is fewer bytes per vector** (lower bits +
  residual to hold recall, 046) — that *moves the bandwidth wall*. SIMD pushes on it.

This closes the SIMD question for the scan: not worth `core::simd`/unsafe — chase
bytes, not popcount.

(Where SIMD *could* still pay is a different kernel — e.g. the asymmetric LUT (`vpshufb`
gather, history 011) for a recall-per-byte gain — but that's a separate path, not the
symmetric Hamming scan, and was cut as non-competitive scalar.)

## Caveats

- Random codes, kernel-isolated (no heap/selection) — measures raw popcount throughput,
  the quantity SIMD would affect. The full scan adds heap overhead on top (constant).
- Single-thread; the 8-core memory-contention point is cross-referenced from 047.
- `interleave4`'s in-cache win is real but only reachable when the working set fits L2,
  which a 1M+ cell never does (128 MB+).
