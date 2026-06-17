# 038 — Tile re-tuning: +7.5% QPS, free (a clean Pareto win)

Perf record: [`038-tile-retune.json`](./038-tile-retune.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --rotate 2 --batch T --rerank 1000`.

## Why this is the right lever for "no sacrifice"

Two routes failed the no-sacrifice goal first: combining tiling+ADSampling rerank
(036, −5–8% QPS) and bf16 rerank (037, recall loss, no speed). Both tried to make
the *rerank* faster and ran into the kernel/vectorization wall. The one lever that
*cannot* cost anything is the **tile size**: tiling (016) is an exact-equivalent
reordering, so recall is identical at every tile, and single-query latency is
independent of tile (it only changes how the batch amortizes the scan). So any tile
that scans faster is a free win.

## Result — the default tile was sitting in a dip

Recall is **0.9968 at every tile**. QPS (reps 5–8, CV ≤0.2%):

| tile | QPS |
|---|---|
| **8** | **921.6** |
| 16 | 857.4 |
| 24 | 898.2 |
| 32 | 907.1 |
| 48 | 899.4 |
| 64 | 895.2 |
| 96 | 823.5 |
| 128 | 826.3 |

`tile=16` — the value used since 016 and in every "best" config (020/029/030) — is a
**reproducible local minimum** (confirmed at reps=8, CV 0.1%). `tile=8` is the
optimum at **921.6 QPS**, and even `tile=32` (907) beats 16. So:

- **tile 16 → 8: +7.5% QPS (857 → 922), identical recall, identical latency.**

## Conclusions

1. **A genuine clean Pareto win:** +7.5% throughput at zero cost on recall or
   latency. New project-best high-recall throughput point: **0.9968 @ 922 QPS**
   (beats 030's tile=16 result of 0.996 @ 845). The engine had ~7–9% free QPS the
   whole time — we'd parked on a bad tile.
2. **The dip is microarchitectural.** tile=16 is non-monotonically slow (8 and 32
   are both faster) — a cache-resonance effect (the 16-query working set of binary
   rows + heaps hits an unlucky pattern). Not noise: reproducible at CV 0.1%. The
   lesson is to *sweep* the tile per machine rather than hardcode 16.
3. **It confirms the funnel was otherwise Pareto-optimal.** After two failed
   lossless-rerank attempts, the only remaining no-sacrifice gain was a tuning
   constant — which is exactly what you'd expect at a frontier. The default should
   be tile=8 (or swept) on this hardware.

## Caveats

- Optimal tile is hardware-dependent (cache geometry); 8 here on Granite, sweep
  elsewhere. The recommended default moves 16 → 8.
- Exact-equivalent: recall 0.9968 identical across all tiles (the proof it's free).
