# ♫ RaBitQ (and the binary-quantization frontier)

- **Paper (1-bit):** RaBitQ — Gao & Long, SIGMOD 2024. arXiv:2405.12497, DOI 10.1145/3654970
- **Paper (multi-bit / extended):** arXiv:2409.09913 ("Practical and Asymptotically Optimal Quantization…")
- **Code / library:** https://github.com/VectorDB-NTU/RaBitQ-Library (older: github.com/gaoj0017/RaBitQ)
- **Docs:** https://vectordb-ntu.github.io/RaBitQ-Library/
- **Type:** actively evolving method + library (state-of-the-art 1-bit/low-bit vector quantization)

## What it is

A randomized vector quantization that encodes D-dim vectors into ~1 bit/dim with an
**unbiased distance estimator** and a **sharp O(1/√D) error bound** — unlike PQ,
which is biased and bound-free. Two pieces:

1. **Random orthogonal rotation before sign-binarization** (a JL-type transform;
   same core idea as ITQ, Gong & Lazebnik 2011). Spreads variance evenly across
   dims so every sign bit carries independent signal. *This part is free and
   improves any binary code* (we use it in history 026/027/029).
2. **Unbiased estimator:** ⟨q,o⟩ ≈ ⟨q,code⟩/⟨code,o⟩ = (2·Σ_{set bits} q'ᵢ − Σq′)/‖o′‖₁
   in the rotated space, with the per-vector ‖o′‖₁ stored. Far better recall-per-bit
   than symmetric Hamming (history 028). Fast version quantizes the query to a few
   bits and uses bit-sliced popcount / `vpshufb` LUTs (the kernel we've parked).

## Why it's tracked

It's the current SOTA reference for the binary-scan tier this engine implements,
and it's moving: multi-bit extension (2024), a maintained library, and active
comparisons (see below). Our 009 funnel is plain sign-bit binary; RaBitQ is the
upgrade path (rotation already adopted; estimator pending the SIMD kernel).

## Related / competing (the low-bit quant frontier)

- **TurboQuant** (Zandieh et al., Google Research, 2025) — online quantization with
  near-optimal distortion; a RaBitQ alternative. There's an active "RaBitQ vs
  TurboQuant" comparison literature (e.g. arXiv:2604.19528, Milvus blog).
- **SAQ** (arXiv:2509.12086) — code adjustment + dimension segmentation, pushes
  quantization limits further.
- **ITQ** (2011) — the original "rotate before binarizing" insight RaBitQ builds on.
- Ecosystem: FAISS, Milvus implement RaBitQ-style codes.
