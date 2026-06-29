# 003 — Query batching, and the bandwidth myth

Perf record: [`003-query-batching.json`](./003-query-batching.json)

## What we did

Added `knn_batch_tiled` (`search.rs`): process queries in tiles, and for each base
vector loaded, compute its distance to every query in the tile while it's hot in
cache. The base is streamed **once per tile** instead of once per query, so total
DRAM traffic drops by ~`tile`×. Exact-equivalent to `knn_batch` (unit-tested),
selected via `--batch N`.

The hypothesis (from the 001/002 roofline napkin and the Exa notes): the scan is
bandwidth-bound, so amortizing bytes across queries should give ~10–20× QPS.

## Result: batching did nothing

Batch sweep (1000 queries, k=10, 8 threads):

| batch | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| QPS | ~86 | ~92 | ~87 | ~87 | ~99 | ~99 |
| recall@10 | 0.9994 | … | … | … | … | 0.9994 |

Flat. ~32× less DRAM traffic bought **zero** QPS. The recall stayed exact (the tiling only reorders loops).

The decisive evidence — thread-scaling, batch=1 vs batch=32:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| batch=1 QPS | 20.2 | 39.7 | 72.2 | 98.6 |
| batch=32 QPS | 20.8 | 39.3 | 74.7 | 98.6 |

**Identical at every thread count.** If bytes were the binding constraint, batch=32 (which moves ~32× fewer of them) would pull away at 8 threads. It doesn't.

## Conclusion: we are compute-bound, not bandwidth-bound

This **revises the bandwidth-bound interpretation in 001 and 002.** The napkin roofline (0.5 flop/byte → memory-bound) assumed the kernel runs near peak FLOP/s. It doesn't:

- Effective throughput at 8 threads ≈ 98.5 QPS × 1M × ~383 flop/distance ≈ **~38 GFLOP/s — roughly 7–8% of the chip's ~500 GFLOP/s roofline.** The kernel is the bottleneck, and it's inefficient (the scalar lane-extraction reduction seen in the disassembly — squares vectorized, sum de-vectorized, no FMA).
- 1 thread moves only 488 MB / 49 ms ≈ **~10 GB/s**, far below a single P-core's memory bandwidth → one core is compute-bound, not waiting on RAM.

And the sublinear 4→8 scaling we previously blamed on "memory-bandwidth contention" is actually **P/E core heterogeneity**: the M3 has 4 Performance + 4 Efficiency cores. Threads 1→4 land on P-cores (near-linear: 20→72), threads 5–8 add weak E-cores (+37%). Proof it isn't bandwidth: batch=32 (moving ~1.5 GB/s, nowhere near any wall) shows the *identical* sublinear curve.

## Why batching couldn't help — the two walls coincide

On this machine the compute ceiling (~100 QPS, slow kernel + P/E cores) and the bandwidth ceiling (~49 GB/s ÷ 488 MB ≈ ~100 QPS) sit at roughly the **same height**. Batching removes the bandwidth wall — and exposes the compute wall right behind it at the same ~100 QPS. So:

- **Batching alone** (this entry) → compute binds → no gain.
- A **faster kernel alone** → bandwidth binds at ~100 → also no gain.
- They **unlock each other**: only a faster kernel *and* batching together break ~100. This is the "re-pricing cascade" (Exa concepts.md §L533) made concrete — measured, not assumed.

## Connection to the Exa notes

Exa's binary scan really *is* bandwidth-bound (concepts.md §L527: ~0.25 instr/byte) — because binary quantization + `vpshufb` lookup tables make the compute nearly free, so bytes become the limit. **Our f32 kernel is the opposite regime**: compute is expensive *and* inefficient, so we're compute-bound and bytes don't matter yet. That's precisely *why* Exa quantizes — to drive compute cost low enough to *enter* the bandwidth-bound regime where batching and bytes-per-answer tricks pay off. This experiment shows our engine isn't there yet.

`knn_batch_tiled` is correct and kept — it's the right tool once the kernel is fast enough to actually hit the bandwidth wall. It just isn't the binding constraint today.
