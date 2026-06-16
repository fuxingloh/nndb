# 006 — FMA + parallel-accumulator kernel; the width experiment

Perf records: [`006-m3-fma-bound.json`](./006-m3-fma-bound.json),
[`006-cascadelake-fma-bound.json`](./006-cascadelake-fma-bound.json),
[`006-cascadelake-fma-batch.json`](./006-cascadelake-fma-batch.json).

## What we did

003–005 proved the scan was compute-bound on the kernel's **serial, FMA-less
reduction** (the `.map().sum()` de-vectorized to per-lane scalar adds; `fmla`/
`vfmadd` = 0). This entry fixes that and tests whether wider SIMD helps.

- **`l2_sq` rewrite:** 32 independent accumulators + `mul_add` + tree-reduce.
  Emits real FMA now (M3 `fmla.4s` 0→2; x86 `vfmadd*ps`). Bit-exact for SIFT's
  integer vectors → recall unchanged (0.9994).
- **`.cargo/config.toml` `target-cpu=native`** so each host builds for its widest
  SIMD + hardware FMA.

## Results — the kernel fix helped both architectures

| | QPS before → after | ns/dist (DRAM) | latency p50 |
|---|---|---|---|
| **M3** (NEON-128) | 98 → **131** (+34%) | 12.6 → **7.2** | 49 → **33 ms** |
| **Cascade Lake** (256-bit FMA) | 50 → **63** (+26%) | 20 → **16** | 140 → ~110 ms |

recall 0.9994 throughout. Bound (working-set sweep):
- **M3:** ratio 1.23× (L2 5.9 → DRAM 7.2 ns) — still compute-bound, but the faster kernel is **starting to approach the memory wall** (was ~flat at 1.18 before).
- **Cascade Lake:** ratio 1.09× — still firmly compute-bound, flat.

Batching still flat on both (Cascade Lake FMA: 63→67 QPS across batch 1→64) — consistent with still-compute-bound.

## The width experiment: should we use 512-bit?

On Cascade Lake, `target-cpu=native` made LLVM emit **256-bit (`ymm`) FMA, not 512-bit (`zmm`)** — deliberately, via its `prefer-256-bit` heuristic (AVX-512 downclock avoidance). We tested forcing 512-bit (`-C target-feature=-prefer-256-bit`, verified `zmm` in the binary):

| Cascade Lake | QPS | ns/dist | CV |
|---|---|---|---|
| 256-bit FMA (default) | **63.1** | 16.0 | — |
| 512-bit FMA (forced) | **42.7** | 23.4 | 0.5% |

**Forcing 512-bit is 32% slower.** Clean signal (CV 0.5%). The AVX-512 downclock costs more than the extra lanes buy — so the compiler's 256-bit choice was optimal, and the answer to "use 512-bit?" on this chip is a measured **no**.

## Conclusions

1. **The real lever was the kernel, exactly as 003–005 predicted.** FMA + parallel accumulators (fixing the serial reduction) gave +26–34% — portably, on both ISAs, recall unchanged. Not SIMD width, not batching, not bandwidth.

2. **SIMD width is *not* a lever here — and can be negative.** The 128-bit M3 is the *fastest* per-distance machine (7.2 ns) — 2.2× faster than the 512-bit-capable Cascade Lake (16 ns). And forcing 512-bit on Cascade Lake made it *worse* (downclock). Wider lanes don't help a kernel that isn't throughput-bound, and on older Intel they hurt. Modern high-IPC cores (M3) beat older wide-SIMD cores.

3. **Still compute-bound — but the M3 is nearing the flip.** Faster kernel → memory latency starting to show (M3 1.23×). To *fully* flip to memory-bound (where batching/quantization finally pay), the kernel must get faster still — which on this hardware means **fewer bytes per distance (quantization)**, not wider f32 SIMD. That's Exa's path (binary + LUT), and it's where the roofline points next.

4. **Rigor:** all numbers are medians of 6 reps, warmup discarded. Cascade Lake CV ≤0.5% (trustworthy). M3 full-base CV was ~22% (laptop thermal/turbo) — its medians hold but single M3 runs swing ±20%; the bound-sweep M3 CVs were 1–5%.

## Caveats

- Virtualized Lightsail → no PMU; bound is via the software working-set detector (005), not hardware top-down. A `.metal` box would give the direct % memory-bound.
- `prefer-256-bit` is Cascade-Lake-specific behavior; newer Intel (Sapphire Rapids) and AMD Zen4/5 have far smaller AVX-512 downclock and may favor 512-bit — untested here.
