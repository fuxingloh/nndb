# 018 — Combined recall↔QPS Pareto (tiling × prefix × C)

Perf record: [`018-combined-pareto.json`](./018-combined-pareto.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --batch 16 --scan-bits N --rerank C`.

## What we did

Compose the two batch levers — tiling (016, free) and prefix truncation (014,
recall-for-bandwidth) — and sweep rerank C, to draw the actual recall-vs-QPS
frontier and pick the best operating point at each recall target. All runs at
tile=16.

## The frontier

| scan-bits | C | recall@10 | QPS | on Pareto |
|---|---|---|---|---|
| 1024 | 2000 | **0.9975** | 731 | ● |
| 1024 | 1000 | **0.9931** | 850 | ● |
| 768  | 2000 | 0.9864 | 762 | |
| 768  | 1000 | **0.9703** | 893 | ● |
| 768  | 4000 | 0.9947 | 588 | |
| 512  | 2000 | **0.9303** | 829 | ● |
| 512  | 4000 | 0.9601 | 626 | |
| 384  | 4000 | 0.8999 | 599 | |

## Conclusions

1. **Tiling is the universal lever; prefix is a low-recall lever.** At high recall
   (≳0.97) the full 1024-bit scan + tiling dominates — 0.993 @ 850, 0.9975 @ 731.
   Truncation only wins *below* ~0.97 (768/C=1000 → 0.970 @ 893), because at high
   recall you must crank C so hard to recover the lost bits that the prefix's
   bandwidth saving is eaten by extra rerank. So: always tile; truncate only when
   you're willing to live under ~0.97 recall for more QPS.
2. **The whole effort beats the 009 baseline on both axes.** Original 009 was
   0.994 @ 585 QPS. With tiling we now hit **0.9975 @ 731** (higher recall *and*
   +25% QPS) or **0.993 @ 850** (+45% QPS at the same recall tier). Recommended
   default: tile=16, full bits, C=1000 → 0.993 @ 850.
3. **Higher C only buys recall at high bits.** 768/C=4000 reaches 0.9947 but at
   588 QPS — dominated by 1024/C=1000 (0.993 @ 850) and 1024/C=2000 (0.9975 @ 731).
   Over-retrieving on a truncated index is the worst of both.

## Caveats

- Batch path (rayon over queries) — these are throughput numbers; single-request
  serving latency is the separate axis (017, and the prefix latency knob next).
- One back-to-back run, CV < 0.3%; spot-level absolutes drift between entries.
