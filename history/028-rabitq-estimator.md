# 028 — RaBitQ unbiased estimator: best recall-per-bit (slow scan)

Perf record: [`028-rabitq-estimator.json`](./028-rabitq-estimator.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant rabitq` (rotation auto, 2 rounds).

## Research → implementation

The full RaBitQ method (not just the rotation of 026) replaces Hamming with an
**unbiased inner-product estimator**. In the rotated space, for unit vectors:

  ⟨q, o⟩ ≈ ⟨q', code⟩ / ⟨code, o'⟩ = (2·Σ_{set bits} q'ᵢ − Σ q') / ‖o'‖₁

where q' is the rotated query (kept full precision), the code is the {±1/√D}
sign vector, and **‖o'‖₁ is stored per data vector** — that per-vector factor is
what unbiases the estimate (the √D cancels). Implemented as `RaBitQ::build`
(rotated sign bits + `norm1[]`) and `knn_rabitq` (the estimator).

## Result — far better recall-per-bit, but the scan is slow

| method | stage-1 | C=100 | C=500 | C=1000 | scan QPS |
|---|---|---|---|---|---|
| plain binary | 0.442 | — | — | — | 583 |
| rot-symmetric (026) | 0.463 | 0.899 | 0.988 | 0.998 | ~580 |
| **RaBitQ** | **0.606** | **0.985** | **1.000** | **1.000** | **14.7** |

## Conclusions

1. **RaBitQ has the best stage-1 recall by a wide margin** — 0.606 vs 0.463 for
   rotated-symmetric (+14 pts) and 0.442 for plain binary. The unbiased estimator,
   keeping the query's magnitude and correcting per-vector, is a much sharper
   ranker than sign-vs-sign Hamming. This reproduces the paper's central claim on
   our data.
2. **It needs ~5–10× smaller rerank C for the same recall.** RaBitQ hits 0.985 at
   **C=100**; symmetric needs C≈500–1000 to get there. At billion scale, where the
   f32 rerank tier dominates cost, that smaller C is the whole game.
3. **But as implemented the scan is 40× slower (14.7 vs ~580 QPS)** — the estimator
   is an asymmetric set-bit gather (no popcount), the same kernel wall as 010/011.
   So today RaBitQ is **recall-best, not throughput-best**. RaBitQ's real
   implementation closes this by quantizing the query to a few bits and computing
   the estimate with bit-sliced popcount / `vpshufb` LUTs — the parked SIMD kernel.

## Where this leaves it

- **For max recall / smallest C** (billion-scale, or when the rerank tier is the
  cost): RaBitQ is the method, *if* the fast estimator kernel is built.
- **For throughput today at 1M**: rotated-symmetric Hamming + tiling (free,
  popcount-speed) remains the deployable winner; RaBitQ's recall edge isn't worth
  the 40× scan penalty at this scale.

## Caveats

- 300 queries, reps=1 (recall is deterministic; QPS single-shot but CV 0 here).
- The fast bit-quantized RaBitQ kernel is the same unbuilt SIMD-LUT work parked at
  011; recall here is the upper bound that kernel would deliver at popcount speed.
