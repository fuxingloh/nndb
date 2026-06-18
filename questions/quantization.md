# Quantization — resolved

**Status:** resolved — the recall-for-bytes tradeoff was explored end-to-end
across `history/008`–`037`. Verdict below. One branch (Product Quantization /
extended RaBitQ) remains untested and is the only open item.

**Why it was parked, and why it's now answered:** scalar and product quantization
cut the bytes streamed per query, which targets the memory-bandwidth bound from
`history/001`/`002`. We deferred it to test other within-cell-scan approaches
first — and in doing so we ended up testing most of the quantization axis anyway.

## What we built and learned

| Quant family | Entries | Verdict |
|---|---|---|
| Scalar int8 (full-precision scan) | `008` | Works, but the binary funnel dominates it on QPS at equal recall. |
| Binary 1-bit (sign) funnel | `009`, `012`, `013`, `016` | **The winner.** Sign-bit Hamming scan → top-C → f32 rerank. Scan runs at the popcount/bandwidth floor (`012`: popcount autovectorizes; multi-accumulator hurts). |
| Multi-bit prefix codes | `014`, `027` | More bits per dim don't pay their bandwidth vs spending the same budget on rotation. |
| Asymmetric / LUT distance | `010`, `011` | LUT lost once popcount was re-priced (`012`). |
| Rotation (FWHT / SRHT, RaBitQ-style) | `026`, `029` | Random rotation before sign-binarization is the **recall knob** — rescues recall at fixed bit budget. Orthogonal to the scan kernel. |
| RaBitQ unbiased estimator | `028` | The (2·maskedΣ − Σq)/‖o′‖₁ estimator works and is unbiased, but doesn't beat the plain funnel's QPS. |
| Smaller rerank store (int8 / bf16) | `015`, `037` | **Negative.** Shrinking the rerank vectors doesn't help — we're popcount-bound on the *scan*, not bandwidth-bound on the rerank. `037` (bf16) lost recall for no QPS. |

## Conclusion

The "trade recall for bytes" question is answered: the **rotated binary funnel**
(1-bit sign scan + f32 rerank, with random rotation as the recall dial) is the
Pareto frontier for the within-cell scan. Going *below* 1 bit isn't a thing here,
and going *above* 1 bit (prefix, int8) doesn't pay. Shrinking the rerank tier
(int8/bf16) is a dead end because the rerank isn't the bottleneck.

## Remaining open branch

**Product Quantization** (k-means subspace codebooks + ADC table lookup) and
**extended RaBitQ B-bit codes** were never built. PQ is a fundamentally different
mechanism — codebook *reconstruction* via lookup tables, not bit-truncation of the
raw vector — so it's not covered by anything above. It's the one quantization
family worth a future entry if we revisit this axis. Note the LUT-distance result
(`011`) is a caution: table lookups lost to autovectorized popcount once re-priced,
so PQ's ADC must clear that same bar to be worth it.
