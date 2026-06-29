# 060 — 3-tier funnel (PQ-prune): ~8× fewer disk reads at 0.99 recall

Perf record: [`060-pq-prune-funnel.json`](./060-pq-prune-funnel.json). Granite box.
`src/bin/funnel3.rs`. Cohere 1M × 1024. Task #1 (experiment B).

## The pipeline

`binary scan (RAM) → top-C1 by Hamming → PQ-ADC re-rank those C1 (RAM) → top-C2 →
exact rerank C2`. The metric is **exact reads** (= C2), because in the disk regime
each exact rerank is a random SSD read — and C=1000 of them per query is what
death-spiraled carousel×disk (052). Can a RAM PQ-prune tier cut C2 ≪ C1 at the same
recall?

## Result — yes, if the PQ is accurate enough

2-tier baseline (binary → exact): **0.999 @ 1000 reads**. 3-tier (C1=1000):

| PQ M (bytes/vec) | C2=64 (15.6× fewer) | C2=128 (7.8×) | C2=256 (3.9×) | C2=500 (2×) |
|---|---|---|---|---|
| 16 | 0.571 | 0.732 | 0.872 | 0.970 |
| 32 | 0.818 | 0.920 | 0.975 | 0.996 |
| **64** | 0.955 | **0.989** | 0.997 | 0.999 |

- **M=64 PQ-prune → 0.989 recall at 7.8× fewer reads** (128 vs 1000), 0.955 at 15.6×.
- **M=16 is too coarse** (0.73 @ C2=128) — the prune tier must be an accurate enough
  estimate to keep the true neighbours in its top-C2.

## Why it matters

This **fixes the carousel × disk bottleneck (052)**: there, the unshared per-query
rerank did C=1000 random SSD reads and collapsed under load. Insert a 64 B/vec PQ tier
in RAM (64 MB for 1M — cache-resident) and the disk only sees **~128 reads/query at
0.99 recall** — an ~8× cut, turning the death-spiral into a viable serve. The PQ codes
join the funnel's other RAM-resident parts (128 MB binary codes), with the 4 GB f32 on
SSD touched ~8× less.

## Conclusion

The 3-tier funnel (binary → PQ-prune → exact) is the disk-regime answer: **~8× fewer
SSD reads at ~0.99 recall**, gated on a sufficiently accurate PQ (M≈64). It only helps
on disk (RAM rerank is already cheap, 015/037), and it composes with the carousel
(052) and the disk hybrid (045).

## Caveats
- Reads measured as C2 (the exact-rerank count); a live disk run would confirm the
  latency win, but C2 *is* the SSD-read count.
- M=64 PQ = 64 B/vec extra RAM; still small vs the f32 it saves reading.
- One dataset (Cohere). PQ trained on a 100k sample, K=256.
