# 048 — Query-adaptive funnel width + a per-query certificate

Perf record: [`048-query-adaptive-funnel.json`](./048-query-adaptive-funnel.json).
Granite box (Xeon 6975P-C, 8 vCPU). `src/bin/adaptive.rs`. Cohere v3 cell N=100k,
1000 queries, rotate×2. Direction 5 of the breakout loop.

## The idea

Fixed C wastes rerank on easy queries (top-k clearly Hamming-separated) and
under-serves hard ones (many near-ties). Adaptive C keys off a stage-1 signal — the
Hamming margin — spending rerank only where needed:
`C_q = #{candidates with hamming ≤ hamming[k-1] + margin}`.

A design caveat sets the stakes: **in RAM the scan dominates**, so C barely moves QPS
there. But in the disk-resident regime (045), each reranked vector is an SSD read
(~0.4 ms), so **mean-C *is* the latency** — adaptive cuts it directly. So the headline
metric is mean rerank work (mean C) at matched mean recall.

## Result 1 — adaptive cuts mean rerank ~34% at matched high recall

| approach | mean recall | mean C |
|---|---|---|
| fixed C=200 | 0.9754 | 200 |
| fixed C=500 | 0.9951 | 500 |
| fixed C=1000 | 0.9985 | 1000 |
| **adaptive margin=32** | **0.9939** | **328** |
| adaptive margin=16 | 0.8931 | 64 |
| adaptive margin=64 | 0.9997 | 1753 |

At ~0.994 recall, adaptive needs **mean C=328 vs fixed 500 — ~34% fewer reranks**.
In the disk regime that's a ~34% latency cut (328 vs 500 SSD reads); in RAM it's
marginal (scan-dominated).

**But the simple margin rule overshoots at the extreme-recall tail:** margin=64 hits
0.9997 at mean C=1753 — *worse* than fixed C=1000's 0.9985, because a few pathological
queries blow C_q up to the M=2000 cap and drag the mean. The rule wins in the
high-recall *band* (~0.97–0.995); past that it needs a per-query cap or a relative
(not absolute) margin. Honest: this is a band win, not a global dominance.

## Result 2 — the certificate predicts per-query recall

The certificate is the Hamming gap at the funnel boundary,
`gap_q = hamming[C-1] − hamming[k-1]` (headroom of the boundary above the k-th) —
computable at query time **without** ground truth. Bucketing queries by gap (at C=100):

| gap bucket | #queries | mean recall |
|---|---|---|
| 7–14 | 38 | **0.776** |
| ≥15 | 962 | **0.941** |

Monotone: a tight gap (boundary close to the k-th) flags a high-miss-risk query; a wide
gap certifies low risk. (No query had gap<7 at C=100 — by that depth the Hamming
distribution has always spread.) So the gap is a valid **per-query miss-risk
certificate**: the engine can flag the ~4% of queries likely to miss and either widen C
for them (which is exactly what the adaptive rule does) or return a confidence with the
result for SLA control.

## Conclusions

1. **Adaptive-margin C cuts mean rerank ~34% at matched ~0.994 recall** — a real win in
   the high-recall operating band, marginal in RAM but a direct latency cut where rerank
   is the cost (disk-resident, per-045).
2. **The simple absolute-margin rule overshoots at the extreme-recall tail** (mean C
   1753 for 0.9997); it wants a cap / relative margin to be globally Pareto.
3. **The Hamming-gap certificate is predictive** (0.78 vs 0.94 recall across gap
   buckets) — a compute-without-truth per-query miss-risk signal, the mechanism behind
   the adaptive rule and a hook for per-query SLAs.

## Caveats

- One cell (N=100k, Cohere). Plain rotated binary (no residual); 046's residual
  composes and would lift the absolute recall but not the fixed-vs-adaptive shape.
- mean-C is the work metric; the in-RAM QPS impact is small (scan-dominated). The disk
  translation uses the 045 ~0.4 ms/read constant, not a re-measured disk run.
- Certificate buckets are coarse (two populated); the monotone trend is the claim, not
  the exact per-bucket numbers.
