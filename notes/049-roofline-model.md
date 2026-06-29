# 049 — Capstone: a predictive roofline model for the funnel (QPS & recall)

Perf record: [`049-roofline-model.json`](./049-roofline-model.json). Model fit/validated
against the measured history (`scale` 047, `cell` 046, `crossover` 043, `tile-retune`
038) — no new run. Granite box (Xeon 6975P-C, 8 vCPU). The capstone of the breakout
loop (10a→7→8→5): two closed-form laws that predict the engine's performance for any
(N, bits, C, cores), and validate against independent runs.

## The two laws

**Throughput (compute roofline):**
```
QPS(N, cores) ≈ 1.10e9 · (cores/8) / N        [tiled, tile≥8; tile=1 uses 0.63e9]
```
The tiled binary scan is **popcount-compute-bound** at a fixed aggregate rate
G₈ ≈ 1.10 Gcmp/s on 8 cores (047). QPS is just that rate divided by the cell size.
This is *why* there is no memory cliff (047): the roofline is compute, not bandwidth,
so it's flat through L3→DRAM. Rerank adds a term — negligible in RAM (scan-dominated),
but ≈ C × 0.4 ms on EBS SSD (045), where it dominates.

**Recall (power law):**
```
miss = 1 − recall ≈ e^12.45 · N^0.28 · C^−1.08 · bits^−1.97       (raw rotated binary)
```
Fit on 38 points (log-miss R² = 0.90). The exponents are physical:
- **C^−1.08** — miss ≈ 1/C: each doubling of the funnel width halves the miss rate.
- **bits^−1.97** — miss ≈ 1/bits²: **doubling the scan bits quarters the miss** (more
  bits ⇒ much tighter Hamming↔true-distance concordance). The strongest lever.
- **N^0.28** — miss grows only sublinearly with cell size; big cells are survivable.

(Residual encoding (046) lowers the constant e^12.45; rotation (026) is baked into the
fit. So the law describes the *current best* stage-1.)

## Validation

**QPS vs 047 (tile=8), predicted = G₈·1e9/N:**

| N | actual | pred | 
|---|---|---|
| 100k | 11246 | 11013 |
| 1M | 1130 | 1101 |
| 10M | 114 | 110 |
| 100M | 11.2 | 11.0 |

Within ~2% across a 1000× range. (038's 1M tile=8 = 921 QPS vs scan-only 1101 — the
gap is the C=1000 rerank overhead, exactly the rerank term.)

**Recall vs 043 — an *independent* crossover run, full 1024 bits:**

mean abs recall error **0.0134** (1.3 pts) across N=1k–100k × C=50–500. Systematic
~+4 pts over-prediction only at C=50 (the power law is slightly optimistic in the
very-low-C / low-recall corner); ≤1 pt everywhere recall ≥ 0.95 — the operating range.

## What it unifies

The model collapses the whole history into one picture:

- **Scan = compute roofline G/N** (047, 038) → QPS predictable from N and cores alone.
- **Recall = an orthogonal knob** set by C, bits, rotation, residual (009/026/043/046) →
  the power law. You move along it without touching the QPS roofline (reranking C is
  ~free in RAM).
- **HNSW-in-cell loses at high D** (043) because a graph walk can't beat the G/N compute
  roofline at cell scale while paying full-D distances — the model says the funnel's QPS
  is set by popcount throughput, which HNSW can't undercut.
- **Disk economics** (045) = the rerank term turning on: C × read-latency. The adaptive
  funnel (048) shrinks mean C, so it cuts exactly that term.
- **Scaling** (047): since QPS = G/N with no cliff, doubling the corpus halves QPS
  predictably — capacity planning is arithmetic.

## Using it (capacity planning becomes arithmetic)

To hit a target recall R at cell size N: pick bits then solve
`C ≈ (e^12.45 · N^0.28 · bits^−1.97 / (1−R))^(1/1.08)`, and read off
`QPS ≈ 1.10e9·(cores/8)/N`. Example predictions (8 cores, tiled): 1M/1024-bit/C=500 →
recall ~0.98 @ ~1100 QPS; 10M/1024-bit/C=500 → ~0.97 @ ~110 QPS.

## Conclusions

1. **QPS is a compute roofline, G/N** — flat through the memory hierarchy, predictable
   within ~2%.
2. **Recall is a clean power law** in N, C, bits (miss ~ N^0.28·C^−1.08·bits^−1.97),
   predictive to ~1 pt on an independent run in the operating range.
3. **The two axes are separable**, which is the funnel's whole design virtue: tune
   recall (C/bits/rotation/residual) without moving the throughput roofline.

## Caveats

- Fit on Cohere v3, rotate×2; the constant (and likely the exponents slightly) are
  dataset-dependent. The *form* (G/N; miss ~ N^a·C^p·bits^b) is the portable claim.
- Power law is optimistic at very low C (<50) / low recall; trust it where recall ≥ 0.95.
- QPS law is scan-only; add the rerank term explicitly when rerank is costly (disk).
- Extrapolation beyond the fitted ranges (e.g. 256-bit at 100M) is unverified — the
  model correctly flags such configs as low-recall but the magnitude is untested.
