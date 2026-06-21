# 005 — SIMD ADC fast-scan & score-aware quantization (FAISS PQ4, ScaNN)

- **FAISS** (Meta): github.com/facebookresearch/faiss. The quantization reference —
  PQ/OPQ/IVFADC, `IndexBinaryFlat`, and crucially the **PQ4 "fast scan"**: 4-bit PQ
  codes + an **in-register lookup table** via SIMD shuffle (`pshufb`/`vpshufb`), 16
  table lookups per instruction. This is the *fast* ADC we never tested (our 054 used
  scalar ADC, which lost to popcount).
- **ScaNN** (Google): github.com/google-research/google-research/tree/master/scann.
  Paper "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"
  (ICML 2020). Two ideas: **score-aware / anisotropic quantization** (codebook loss
  weighted to preserve the *inner-product ranking*, not reconstruction) + the same
  SIMD ADC. Their AVQ beats plain PQ on recall-per-bit.

## Why tracked (relevance to us)

These are the two things that could change our headline verdict:
1. **SIMD ADC could beat popcount on QPS.** We concluded scalar PQ loses to VPOPCNTDQ
   (054, 011) and parked SIMD (050). PQ4/ScaNN's `pshufb` LUT is the proven way to make
   gather-based ADC fast. The safe-Rust route is `core::simd::Simd::swizzle_dyn`, which
   lowers to `pshufb`/`tbl` (nightly, no `unsafe`). The decisive untested experiment.
2. **Score-aware quantization** is a better-codes objective than our rotation+residual
   (which minimize reconstruction, not ranking) — a recall-per-bit frontier we haven't
   tried.

## Related
- Our scalar PQ/OPQ results: [[history 054]], [[history 056]]. SIMD ceiling: [[history 050]].
- RaBitQ ([[002-rabitq]]) — the binary-side equivalent of "better codes".
