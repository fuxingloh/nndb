# 058 — Bit-floor: fewer scan bits is NOT a QPS lever (negative)

Perf record: [`058-bit-floor.json`](./058-bit-floor.json). Granite box (8 vCPU).
`vsearch --quant binary --rotate 2 --residual --scan-bits N`. Cohere 1M × 1024. Task #3.

## Hypothesis (from 049/050)

The scan cost is ∝ bits, and at scale we're bandwidth-bound — so halving the scan-bits
should ~halve scan time → higher QPS, *if* residual+rotation hold recall.

## Result — it doesn't pan out

| scan-bits | recall C=200 | QPS C=200 | recall C=500 | QPS C=500 |
|---|---|---|---|---|
| 1024 | 0.978 | 1028 | 0.9952 | 968 |
| 768 | 0.941 | 918 | 0.982 | 885 |
| 512 | 0.849 | 1016 | 0.930 | 977 |
| 384 | 0.753 | 1051 | 0.858 | 1014 |
| 256 | 0.588 | 1114 | 0.716 | 1064 |

**4× fewer scan words (1024→256) buys only ~+8% QPS** (1028→1114) — while recall
**collapses** (0.9952 → 0.716 at C=500).

## Why the bandwidth-wall gain didn't materialize

At 1M with C=200–500, the **scan is not the batch bottleneck** — the rerank (C
random f32 gathers × 1024-D) plus heap/selection dominate the per-query cost. So
shrinking the scan 4× barely moves total QPS. And residual can't hold recall at 256-bit
for 1M (far more competition than the 100k cell in 046, where 256-bit residual was ~0.86;
here it's 0.72). So you pay a recall cliff for ~nothing.

## Conclusion

**The lever for QPS is not fewer bits.** The scan is already cheap relative to the
rerank at useful C, so the only ways to raise QPS are (a) make the *per-doc op* faster
than popcount — SIMD ADC, Task #2 — or (b) cut the rerank cost — the PQ-prune tier,
Task #1. Bit-flooring just trades recall for a rounding-error QPS gain. Dead end,
consistent with 051.
