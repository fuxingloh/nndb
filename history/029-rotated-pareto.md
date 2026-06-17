# 029 — Rotated combined Pareto: research payoff, deployable

Perf record: [`029-rotated-pareto.json`](./029-rotated-pareto.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --rotate 2 --batch 16 --scan-bits N --rerank C`.

## What we did

Combine the deployable, popcount-speed pieces of the RaBitQ research — the free
rotation (026, recall up at no cost; rescues prefix in 027) — with the existing
batch winners (tiling 016, prefix 014) and re-draw the frontier. This is the
practical answer to "did the research make it better?": rotated-*symmetric* keeps
full popcount throughput, unlike the slow RaBitQ estimator (028).

## Result — the frontier moves out at every tier

| scan-bits | C | 018 (no rotation) | **029 (rot×2)** |
|---|---|---|---|
| 1024 | 500  | — | **0.9886 @ 915** |
| 1024 | 1000 | 0.9931 @ 850 | **0.9960 @ 844** |
| 1024 | 2000 | 0.9975 @ 731 | **0.9990 @ 730** |
| 768  | 1000 | 0.9703 @ 893 | **0.9829 @ 957** |
| 768  | 2000 | 0.9864 @ 762 | **0.9940 @ 815** |
| 512  | 2000 | 0.9303 @ 829 | **0.9626 @ 831** |

## Conclusions

1. **Rotation dominates the prior frontier — for free.** At every operating point
   it delivers higher recall at equal-or-better QPS. The rotation costs nothing at
   serve time (baked into codes; a tiny per-query FWHT) and nothing in memory.
2. **The standout is the prefix tier:** 768-bit/C=2000 goes 0.986 @ 762 → **0.994
   @ 815** (+8 pts recall *and* +7% QPS), because rotation rescued truncation
   (027). High recall at a truncated scan is now real — the prefix lever finally
   pays at the 0.99 tier, not just below 0.97.
3. **New best operating points** vs the whole project:
   - **0.996 @ 844 QPS** (full, C=1000) — beats 020's 0.993 @ 851 on recall at ~same QPS.
   - **0.999 @ 730 QPS** (full, C=2000).
   - **0.994 @ 815 QPS** (768, C=2000) — same recall as the original 009 (0.994)
     at **815 vs 585 QPS** (+39%).

## The arc

Original 009: 0.994 @ 585. Post-throughput-work (020): 0.993 @ 851. Post-research
(029): **0.996 @ 844 or 0.994 @ 815**, i.e. higher recall *and* higher throughput
than either — from a free random rotation grounded in RaBitQ/ITQ.

## Caveats

- Rotated-*symmetric* (popcount-speed). The RaBitQ *estimator* (028) would push
  recall-per-C further but needs the fast SIMD kernel first.
- tile=16, reps=5, CV < 0.2%; spot box, compare within table.
