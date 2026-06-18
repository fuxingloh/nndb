# 044 — Quantization verdict: the rotated binary funnel is the frontier

Metadata: [`044-quantization-frontier.json`](./044-quantization-frontier.json).
Synthesis entry (no new run) — closes `questions/quantization.md` by collecting the
recall-for-bytes results already measured across `history/008`–`037`.

## The question

Quantization cuts the bytes streamed per query, targeting the memory-bandwidth
bound from `001`/`002`. Which point on the recall↔bytes tradeoff is best for the
within-cell scan — and is shrinking bytes always worth the recall cost?

## What we built and learned

| Quant family | Entries | Verdict |
|---|---|---|
| Scalar int8 (full-precision scan) | `008` | Works, but the binary funnel dominates it on QPS at equal recall. |
| Binary 1-bit (sign) funnel | `009`, `012`, `013`, `016` | **The winner.** Sign-bit Hamming scan → top-C → f32 rerank. Scan runs at the popcount/bandwidth floor (`012`: popcount autovectorizes to VPOPCNTDQ; multi-accumulator *hurts*, −2.2×). |
| Multi-bit prefix codes | `014`, `027` | More bits per dim don't pay their bandwidth vs spending the same budget on rotation. |
| Asymmetric / LUT distance | `010`, `011` | LUT lost once popcount was re-priced (`012`). |
| Rotation (FWHT / SRHT, RaBitQ-style) | `026`, `029` | Random rotation before sign-binarization is the **recall knob** — rescues recall at fixed bit budget, at popcount speed. Orthogonal to the scan kernel. |
| RaBitQ unbiased estimator | `028` | The (2·maskedΣ − Σq)/‖o′‖₁ estimator works and is unbiased, but doesn't beat the plain funnel's QPS. |
| Smaller rerank store (int8 / bf16) | `015`, `037` | **Negative.** Shrinking the rerank vectors doesn't help — we're popcount-bound on the *scan*, not bandwidth-bound on the rerank. `037` (bf16) lost recall for no QPS. |

## Conclusion

The recall-for-bytes question is answered end-to-end: the **rotated binary funnel**
(1-bit sign scan + f32 rerank, with random rotation as the recall dial) is the
Pareto frontier for the within-cell scan. Going *below* 1 bit isn't meaningful here;
going *above* 1 bit (prefix, int8) doesn't pay; shrinking the rerank tier
(int8/bf16) is a dead end because the rerank isn't the bottleneck. This is the same
engine that beat HNSW-in-cell at high dimension in `043`.

## Remaining open branch (untested)

**Product Quantization** (k-means subspace codebooks + ADC table lookup) and
**extended RaBitQ B-bit codes** were never built. PQ is a fundamentally different
mechanism — codebook *reconstruction* via lookup tables, not bit-truncation of the
raw vector — so it isn't covered above. It's the one quantization family worth a
future entry if we revisit this axis. Caution: the LUT-distance result (`011`) is
the bar — table lookups lost to autovectorized popcount once re-priced, so PQ's ADC
must clear that same bar to be worth it.
