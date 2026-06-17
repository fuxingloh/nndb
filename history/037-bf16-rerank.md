# 037 — bf16 rerank store: not a Pareto win

Perf record: [`037-bf16-rerank.json`](./037-bf16-rerank.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --rotate 2 --batch 16 --rerank C [--rerank-store bf16]`.

## The hypothesis

036 pointed at the tiled funnel's rerank being random-gather **bandwidth**-bound:
C cache-cold 4 KB f32 rows per query. So store the rerank docs as **bf16** (top 16
bits of f32 — kept exponent, 7-bit mantissa), halving the gather bytes, query left
full precision (asymmetric). Expectation: QPS up, recall ~unchanged.

## Result — no speed, and recall drops

| C | store | recall@10 | QPS | p50 |
|---|---|---|---|---|
| 1000 | f32 | 0.9960 | 849 | 11.10 ms |
| 1000 | bf16 | 0.9918 | 862 | 11.29 ms |
| 2000 | f32 | 0.9990 | 735 | 12.63 ms |
| 2000 | bf16 | 0.9945 | 728 | 13.34 ms |
| 4000 | f32 | 0.9998 | 580 | 15.47 ms |
| 4000 | bf16 | 0.9953 | 560 | 15.45 ms |

QPS is flat-to-worse (+1.4% / −1% / −3.4%) and recall drops ~0.4 pts. Net negative.

## Why it didn't work

1. **The bytes saved aren't free to use.** Each bf16 element needs a decode
   (`<<16` → f32) before the subtract — that per-element compute offsets the halved
   memory traffic, especially since after rotation+tiling the rerank kernel is
   already efficient.
2. **Rerank isn't the whole cost.** Even in the tiled funnel the binary scan is a
   comparable or larger share, so halving *part* of the rerank's bytes moves the
   total only single digits — and the decode eats that.
3. **7-bit mantissa costs recall** (~0.4 pts) — a real sacrifice, which the goal
   forbids.

## Takeaway

Two routes now exhausted (036 fewer-flops, 037 fewer-bytes): the binary funnel's
rerank can't be made lossless-faster from this angle. Combined with the scan being
at the popcount/bandwidth floor (012/016/023/024), **the funnel is effectively
Pareto-optimal** — the only remaining no-sacrifice lever is exact-equivalent tile
tuning (038). bf16 remains useful purely as a **memory** option (rerank store
3.9 → 1.95 GB) when RAM, not recall, is the constraint.
