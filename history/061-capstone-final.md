# 061 — Final capstone: the best of the best

The definitive synthesis (001–060), after the PQ/ITQ/OPQ/SIMD-ADC round. Cohere v3
1M × 1024, one 8-core box (Granite, Xeon 6975P-C).

## The production engine (ship this)

**Rotated-residual binary funnel + tiling, served by the carousel.**
- **scan:** 1-bit sign codes, `count_ones`→VPOPCNTDQ, query-tiled (tile=8)
- **codes:** random rotation (026) + residual/centroid-subtraction (046) — both free, both recall
- **rerank:** exact f32 on top-C
- **serving:** carousel, fan-out per load — bounded tail, no cliff

| objective | recall@10 | QPS | p50 |
|---|---|---|---|
| max recall | **0.9986** | 883 | 10.6 ms |
| balanced | 0.9952 | **963** | 8.0 ms |
| predictable tail (p99≈p50) | 0.995 | ~350 | 12 ms |
| serving | 0.998 | no-cliff to ~940 | 12→49 ms |

Footprint: 128 MB codes in RAM (+ f32 for rerank, RAM or SSD). This is the shipped answer.

## The open frontier (the one thing that could beat it)

**SIMD ADC — PQ4 + `core::simd::swizzle_dyn` (=pshufb), safe Rust (059).** This round
*overturned* the long-held "popcount always beats gather": a SIMD LUT makes 4-bit-PQ
ADC **~13× faster than scalar PQ** and **2–9× faster than *untiled* popcount**, because
its **8 B/vec codes (16× smaller)** stay cache-resident while popcount's 128 B/vec hits
the DRAM wall. It's the first technique that could rival the funnel on QPS. Not yet a
clean win (the funnel *tiles*; 4-bit recall needs a rerank) — but the decisive next
build is **pq4-scan (tiled) → exact rerank**, end-to-end recall/QPS.

## The disk answer

**3-tier funnel: binary → PQ-prune (M=64, RAM) → exact (060).** Cuts the exact/SSD
reads **~8× at 0.989 recall** (128 vs 1000), fixing the carousel×disk death-spiral
(052). Stacks with the carousel + disk hybrid (045).

## What's settled (don't retry)
Bit-floor (058, fewer bits ≠ QPS), HNSW-in-cell (043), scalar PQ/OPQ (054/056,
footprint only), ITQ (055, +1.5 pts — residual's +3–9 dominates), bf16/int8 rerank
(037/015), register-tiling (023). All measured, all negative or dominated.

## The two laws that explain everything (049)
- **Throughput:** `QPS ≈ 1.10e9 · cores / N` — compute roofline, no memory cliff (047).
- **Recall:** `miss ≈ e^12.45 · N^0.28 · C^−1.08 · bits^−1.97` — validated to ~1 pt.
- Recall and QPS are **separable axes** — tune one without moving the other.

## The one-line verdict
A 1-bit binary funnel (rotation+residual) served by the carousel does **~0.995 recall @
~960 QPS @ 8 ms** on a single 8-core box — and the only thing that might beat it is
SIMD-ADC PQ4 (the open frontier), while PQ-prune makes the disk regime viable.
