# 046 — Cell size × residual encoding: centering is a free recall gain

Perf record: [`046-cell-size-residual.json`](./046-cell-size-residual.json).
Granite box (Xeon 6975P-C, 8 vCPU). `src/bin/cell.rs`. Cohere v3 1M × 1024, cosine.
Direction 7 of the breakout loop (the 042 within-cell seam). Energy axis dropped
(J/query ≈ package_power / QPS on a fixed-power CPU — QPS in disguise, not an
independent signal); the salvaged metric is **bytes/query**, the bandwidth currency.

## The hypothesis (042 seam #2)

Inside one IVF cell the vectors cluster around the cell centroid, so the **raw**
sign-bits are dominated by the shared DC/centroid direction and carry little
*within-cell* information. Subtracting the centroid first (**residual encoding**)
should make the sign-bits encode the actual within-cell variation → higher stage-1
recall **at the same bit budget**. Residual affects stage-1 selection only; rerank
and ground truth stay exact L2 on the raw vectors, isolating the effect.

## Result — confirmed, and the gain grows with cell size AND bit pressure

recall@10, raw → residual:

**Full 1024 bits** (128 B/code):

| N | C=50 | C=100 | C=200 |
|---|---|---|---|
| 1,000 | 0.929 → 0.944 (+1.5) | 0.983 → 0.987 | 0.998 → 0.999 |
| 20,000 | 0.889 → 0.912 (+2.3) | 0.955 → 0.967 | 0.986 → 0.991 |
| 100,000 | 0.858 → 0.894 (**+3.6**) | 0.934 → 0.957 (+2.2) | 0.975 → 0.986 (+1.1) |

**Tight 256 bits** (32 B/code, ¼ the scan traffic):

| N | C=50 | C=100 | C=200 | C=500 |
|---|---|---|---|---|
| 1,000 | 0.582 → 0.632 (+5.0) | 0.741 → 0.781 | 0.876 → 0.901 | 0.983 → 0.988 |
| 20,000 | 0.421 → 0.502 (+8.1) | 0.537 → 0.624 (**+8.7**) | 0.665 → 0.734 | 0.808 → 0.862 |
| 100,000 | 0.360 → 0.433 (+7.2) | 0.465 → 0.548 (+8.3) | 0.576 → 0.664 (**+8.8**) | 0.717 → 0.797 (+8.0) |

- **Strictly positive** everywhere (only −0.0001–0.0002 noise at saturated C=500 full
  bits, where both already hit ~1.0).
- **Grows with N:** bigger cells → more shared centroid mass → more to gain by
  removing it.
- **Grows with bit pressure:** at 256 bits the gain is **+7 to +9 pts** vs +1 to +3.6
  at full bits — when sign-bits are scarce, not spending them on the DC direction is
  exactly where the budget should go.

## Why it matters (bytes/query)

The residual win is largest in the **low-bit, bandwidth-saving** regime — the one we
care about for cost/throughput. At N=100k, the 256-bit residual code is **32 B/vec
(4× less scan traffic than full 128 B)** yet recovers ~8 pts of the recall lost to
truncation, so it dominates raw on recall-per-byte at the low-recall end
(0.797 @ 5.25 MB/q residual vs 0.717 @ 5.25 MB/q raw, C=500). Full bits still wins
the very-high-recall corner; residual shifts the entire low-bit curve up for free.

## Conclusion

Residual encoding (centroid subtraction before rotation+binary) is a **free,
strictly-positive recall gain for the within-cell scan**, costing one centroid vector
per cell and nothing at query time beyond a subtract. It's a clean Pareto push,
biggest where the engine is most bandwidth-constrained (large cells, tight bit
budgets). It composes with everything in the funnel (rotation, tiling, the
disk-resident split) since it only changes how the stage-1 codes are formed.

## Caveats

- "Cell = first N base vectors" (contiguous slice, not a k-means cluster). A real
  k-means cell is *tighter* around its centroid, so residual should help **at least
  as much** — this is a conservative estimate of the gain.
- Single dataset (Cohere 1024-D). Recall measured vs within-cell exact GT; the IVF
  router's own recall loss is above this layer.
- Centroid is full-precision f32 (4 KB/cell at 1024-D) — negligible vs the code store.
