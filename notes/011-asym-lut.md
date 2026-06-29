# 011 — Asymmetric LUT (precompute + indexed lookup)

Perf record: [`011-asym-lut.json`](./011-asym-lut.json). Cohere v3 1M, Granite box.

## What we did

010's asymmetric scan iterated set bits + gathered (slow). Since **docs are binary**, precompute per query a table for each 4-bit nibble position — `table[p][pattern]` = masked query-sum for those 4 dims — and scan a doc with one **indexed lookup per nibble** instead of per-bit gathers. 1024-d → 256 tables × 16 = 16 KB (L1-resident). Same score → same recall. Scalar version (no SIMD gather yet).

## Result

| | recall (no-rr / C=100 / C=200) | QPS |
|---|---|---|
| asym direct (010) | 0.607 / 0.976 / 0.993 | 6.7 |
| **asym LUT (011)** | 0.608 / 0.977 / 0.994 | **14.3** |
| symmetric + rerank C=1000 (009) | 0.994 | 465 |

## Conclusions

1. **Recall identical to 010** — the LUT computes the same score, confirming correctness. Recall-per-C stays excellent (0.994 at C=200 vs symmetric's C=1000).
2. **Scalar LUT is only ~2.1× faster** (6.7 → 14.3 QPS). The 256 nibble lookups/doc are scalar, data-dependent gathers into the table — not vectorized — so still ~16× more work/doc than symmetric's 16 popcounts. Net: still **~33× slower than symmetric+rerank.**
3. **Asymmetric is still not a Pareto win.** Its recall advantage (5× smaller C) is real, but the scalar kernel doesn't close the speed gap. **The win requires the SIMD `vpshufb`/`vpermb` gather over an int8-quantized table** (Exa's actual kernel, 32–64 lookups/instruction) — that's the next entry.

## Practical standing

Best viable today remains **symmetric binary + rerank** (009): recall 0.994 at ~465–585 QPS, 122 MB. Asymmetric beats it on recall-per-C but needs the SIMD LUT to beat it on speed. At billion scale the smaller-C win compounds (less rerank load), making the SIMD LUT worth the effort; at this scale symmetric+rerank is the pragmatic choice.
