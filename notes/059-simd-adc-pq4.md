# 059 — SIMD ADC (PQ4 + swizzle_dyn): gather isn't dead — it ties/beats popcount

Perf record: [`059-simd-adc-pq4.json`](./059-simd-adc-pq4.json). Granite box (AVX-512,
8 vCPU). `src/bin/pq4simd.rs` (nightly `core::simd`, **no unsafe**). Random codes,
kernel-isolated scan throughput. Task #2 — the question parked in 050/054.

## What we built

4-bit PQ codes (16 centroids/subspace) + an int8 LUT looked up with `swizzle_dyn`
(lowers to `pshufb`/`tbl`) — 16 lookups per instruction, FAISS PQ4 style, in **safe**
Rust. M=16 subspaces = **8 B/vec** (vs the binary code's 128 B).

## Result

| scan | single-thread | 8-core @10M | 8-core @50M |
|---|---|---|---|
| popcount (128 B) | 0.20 | 0.36 | 0.37 |
| scalar PQ (16 B) | 0.14 | — | — |
| **pq4-simd (8 B)** | **1.83** | **0.72** | **0.73** |

- **SIMD ADC is ~13× faster than scalar ADC** (0.14 → 1.83) — the `pshufb` LUT is what
  scalar PQ (054) was missing.
- **It beats untiled popcount: 9× single-thread, ~2× at 8-core scale.** Why: pq4 codes
  are **16× smaller** (8 B vs 128 B), so at scale they stay cache-resident while
  popcount's 128 B/vec saturates DRAM bandwidth.

## What this overturns

054/057 concluded "popcount beats gather, so 1-bit binary is throughput-optimal." That
was true for *scalar* ADC. **With a SIMD LUT (swizzle_dyn), gather is no longer
dominated** — it's competitive-to-faster, mainly via the byte advantage. The parked
question (050/054) resolves as: **SIMD ADC is a real high-QPS contender.**

## Honest caveats (why this isn't yet "pq4 beats the funnel")

1. **The funnel tiles; this popcount doesn't.** Tiling (016) amortizes the 128 B base
   read across a query tile, so the real funnel's effective popcount throughput is well
   above this microbench's 0.20–0.37. A fair **tiled-vs-tiled** comparison is the
   follow-up. (pq4 can tile too — process a tile of queries against one code block.)
2. **4-bit recall is coarse and untested.** M=16 *byte*-PQ was 0.90 (054); 4-bit is
   lower → pq4 needs a rerank tier (funnel-style) to reach 0.99. The end-to-end
   recall/QPS isn't measured here.

## Conclusion

SIMD ADC (PQ4 + `swizzle_dyn`) is **fast and not dominated by popcount** — it closes the
13× gap scalar PQ had and wins on the byte axis at scale. It's the one technique that
could rival the binary funnel on QPS. The decisive next step is tiled-vs-tiled + a recall
pass (pq4-scan → exact rerank). Built with safe `core::simd` (no unsafe), nightly only.
