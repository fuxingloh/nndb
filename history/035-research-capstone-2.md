# 035 — Capstone: training-free research arc (031–035)

Perf record: [`035-research-capstone-2.json`](./035-research-capstone-2.json).
Cohere v3, 1M × 1024, cosine, Granite box.

## The principle (stated, then confirmed)

> "I don't think you're going to beat binarization QPS, so anything novel must be
> stacked on this principle — value-added."

Correct, and the data proves it. Binarization (1 bit/dim + popcount + tiling) is
the **throughput floor** of this engine — 851 QPS. Every training-free technique we
researched either (a) **stacks on** the binary funnel and adds value, or (b) lives
in a **different Pareto region** (exact / very-high recall) that is useful but never
competitive on QPS.

## Consolidated frontier (this session's measured points)

| tier | config | recall@10 | QPS |
|---|---|---|---|
| **binary value-add** | rotated + tiled, C=1000 (029) | 0.996 | **845** |
| **binary value-add** | rotated + tiled, C=2000 (029) | 0.999 | **730** |
| binary value-add | binads C=4000, ADSampling rerank (034, untiled) | 0.999 | 463 |
| exact accelerator | PDX + ADSampling (033) | 0.999 | 37 |
| exact accelerator | PDX plain (032) | 1.000 | 18 |
| exact baseline | naive brute force | 1.000 | 9 |

## What each technique did (all training-free)

| entry | technique (paper) | outcome |
|---|---|---|
| 031 | ADSampling (SIGMOD'23) | 2.16× exact scan @ 0.999, but bandwidth-limited on row-major |
| 032 | PDX layout (SIGMOD'25) | 2× exact scan, 3× lower latency — free, autovectorized |
| 033 | PDX + ADSampling | 4× naive exact @ 0.999 — layout *restores* the pruning speedup |
| 034 | ADSampling rerank **on the binary funnel** | +5–10% funnel QPS at high C, near-lossless |
| (026–029) | RaBitQ/ITQ rotation **on binary** | free recall; +64% effective via the Pareto shift |

## The synthesis

Two clean conclusions, both consistent with the whole project:

1. **The value-adds that matter stack on binary.** The random rotation (026–029,
   from RaBitQ) gave free recall and reshaped the frontier; ADSampling rerank (034)
   makes the high-recall end of the funnel cheaper. Neither changes the fact that
   the popcount binary scan + tiling is the throughput engine — they make it reach
   higher recall per QPS. That's the right kind of win.

2. **The exact accelerators (PDX, ADSampling, PDX+ADSampling) are a separate
   tier.** 4× over brute force is a real result and genuinely useful when you need
   recall ≥ 0.999 that binary tops out below — but at ~20–40 QPS they are not, and
   were never going to be, in the binary funnel's league. Knowing *where* a
   technique lives on the Pareto is the point: PDX/ADSampling are the **exact
   tier's** accelerators, binary+rotation+tiling is the **approximate tier's**
   engine, and they serve different SLAs.

The deployable recommendation is unchanged and now sharper: **rotated binary +
tiling** for throughput (0.996 @ 845 / 0.999 @ 730), with **ADSampling rerank** when
over-retrieving hard for recall, and **PDX(+ADSampling)** only when the SLA demands
near-exact results.

## Open (still the highest-leverage, but harder)

The fast SIMD `vpshufb`/LUT kernel (parked since 011/028) is the one unbuilt piece
that would let RaBitQ's superior recall-per-bit (028) run at popcount speed — the
only remaining lever that could move the *binary tier itself* rather than stack
beside it. Everything tested here was training-free and CPU-portable.
