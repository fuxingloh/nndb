# 051 — The stacked engine: residual wired in, the full frontier

Perf record: [`051-stacked-engine.json`](./051-stacked-engine.json). Granite box
(Xeon 6975P-C, 8 vCPU). Cohere v3 1M × 1024, in-RAM, reps=5. Stacks every composable
in-RAM win into the main funnel: **binary + rotation×2 + residual + tiling (tile=8)**,
swept over scan-bits and rerank width C. Residual is now wired into `vsearch`
(`--residual`); it was previously only in the `cell.rs` experiment.

## Result — the stacked frontier (recall@10 / QPS / p50)

| id | bits | C | residual | recall | QPS | p50 |
|---|---|---|---|---|---|---|
| A | 1024 | 1000 | off | 0.9960 | 882 | 10.6 ms |
| **B** | 1024 | 1000 | **on** | **0.9986** | 883 | 10.6 ms |
| **I** | 1024 | 500 | on | **0.9952** | **963** | **8.0 ms** |
| C | 768 | 1000 | on | 0.9936 | 819 | 7.3 ms |
| G | 768 | 500 | on | 0.9816 | 884 | 4.4 ms |
| D | 512 | 1000 | on | 0.9637 | 922 | 4.2 ms |
| H | 512 | 500 | on | 0.9296 | 978 | 3.6 ms |
| E | 512 | 1000 | off | 0.9297 | 908 | 4.3 ms |

## Findings

1. **Residual is a free recall win at full bits.** B vs A: 0.9960 → **0.9986** at
   *identical* QPS and latency — it only changes how the stage-1 codes are formed. This
   pushes the whole frontier past the prior best (038: 0.9968 @ 922). New max-recall
   point: **0.9986 @ 883 QPS**.
2. **At low bits residual matters far more.** D vs E (512-bit): 0.9297 → **0.9637**
   (+3.4 pts) — confirms 046 in the live engine (centering stops sign-bits from being
   wasted on the shared DC direction).
3. **Fewer bits is a LATENCY lever, not (much) a QPS lever — correcting the roofline
   projection.** I expected 512-bit → ~1.8× QPS; instead batch QPS barely moved
   (882 → 922) while **p50 latency more than halved** (10.6 → 4.2 ms). Why: the *tiled*
   batch already amortizes the scan across the tile, and the C-rerank is a real fixed
   cost — so the batch isn't purely scan-bandwidth-bound. But a *single* query (latency)
   IS scan-bound, so halving the code bytes halves it. Separately, dropping C 1000→500
   alone gives +9% QPS (B→I) — so **C is the QPS lever, bits is the latency lever.**

## The best stacked operating points

- **Max recall:** B — **0.9986 @ 883 QPS, 10.6 ms** (residual free over 038).
- **Best balanced:** I — **0.9952 @ 963 QPS, 8.0 ms** — beats 038 (0.9968 @ 922) on
  *both* QPS and latency at ~equal recall.
- **Low latency, high recall:** C — 0.9936 @ 819 QPS, **7.3 ms**; or G — 0.982 @ 884
  QPS, **4.4 ms** (58% lower latency than A).

So the stacked engine moves the frontier out on every axis vs the prior best, and
opens a low-latency regime (sub-5 ms at ~0.98 recall) that didn't exist before.

## What's NOT in these numbers (orthogonal add-ons)

- **Adaptive-C** (048): marginal in-RAM (rerank is a small batch slice), so left out of
  the headline; its payoff is on disk, where each rerank = an SSD read.
- **Carousel** (039–041): the *serving* layer — stacks on top for bounded tail latency
  under burst, no throughput change.
- **Disk hybrid** (045): 32× less RAM (codes in RAM, f32 on SSD) — same warm latency.

## Caveats

- In-RAM, single box, 8 cores, one dataset (Cohere 1024). Recall vs the file's GT.
- "Cell = full 1M base"; a real k-means IVF cell is tighter around its centroid, so
  residual should help *at least* as much.
- 768 bits isn't a power of two but is fine — it's a prefix of the rotated (full-1024)
  code.
