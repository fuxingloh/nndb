# 026 — Random rotation before binarization (RaBitQ/ITQ): free recall

Perf record: [`026-rabitq-rotation.json`](./026-rabitq-rotation.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --rotate N`.

## Research → idea

Searched the literature for something that beats plain sign-bit binary. The
standout is **RaBitQ** (Gao & Long, SIGMOD 2024, arXiv:2405.12497): it quantizes to
1 bit/dim but with an unbiased estimator and a sharp error bound, and a big part of
*why* it works is a **random orthogonal rotation applied before sign-binarization**
— the same trick as **ITQ** (Gong & Lazebnik, 2011). Intuition: our naive sign bit
keeps only the sign of each raw coordinate; if variance is concentrated in a few
dims, most bits are near-random. A random rotation spreads the variance evenly so
every bit carries independent signal.

Implemented as a fast structured rotation — alternating random ±1 sign-flips and
**FWHT** (a fast Johnson–Lindenstrauss transform), O(D log D), deterministic seed
so base and query share the rotation. Rerank still uses the original f32.

## Result — recall up at every C, QPS unchanged

| C | no-rot recall | rot×2 recall | Δ |
|---|---|---|---|
| stage-1 (no rerank) | 0.4681 | **0.4925** | +2.4 pts |
| 100  | 0.8865 | **0.9118** | +2.5 pts |
| 500  | 0.9826 | **0.9891** | +0.65 pts |
| 1000 | 0.9943 | **0.9970** | +0.27 pts |

QPS is identical (581.8 vs 584.9 at stage-1): the rotation is baked into the codes
at prep time, and the per-query rotation is a single FWHT (~20K flops, negligible
beside the 1M-doc scan). rot×3 ≈ rot×2 → 2 rounds suffice.

## Conclusions

1. **Rotation is a free recall improvement** — better at every C, no QPS or memory
   cost. The gain is largest at low C (stage-1 +2.4 pts), which is the useful
   regime: it lets you hit a target recall at a *smaller* rerank C, which is where
   throughput is spent. e.g. rot×2 reaches 0.997 at C=1000 vs no-rot's 0.994.
2. **The effect is modest on Cohere because Cohere is already well-conditioned.**
   v3 is compression-aware and fairly isotropic, so there's less variance imbalance
   for the rotation to fix. On raw/un-tuned embeddings the gain would be larger —
   this is exactly the case RaBitQ's error bound is designed to guarantee.
3. **It's the right foundation for the next steps:** spreading info evenly across
   dims should also make *prefix truncation* (014) lose less recall, and it's the
   substrate for RaBitQ's unbiased asymmetric estimator. Both tested next.

## Caveats

- Requires power-of-two dim for the FWHT (Cohere 1024 ✓, SIFT 128 ✓); a general dim
  would pad or use a dense random orthogonal matrix.
- This is the rotation half of RaBitQ; the unbiased estimator (027) is separate.
