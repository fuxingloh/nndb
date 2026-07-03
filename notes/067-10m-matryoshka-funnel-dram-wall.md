# 067 — 10M-scale Matryoshka-256 funnel: recall holds, the DRAM share is the ceiling

Perf record: [`067-10m-matryoshka-funnel-dram-wall.json`](./067-10m-matryoshka-funnel-dram-wall.json).
c8a.4xlarge (Zen5, 16 vCPU) spot, us-east-2 — the **same box size as 065**, so QPS is
directly comparable for the first time across a corpus-size jump. 065/066 asked whether
a Matryoshka-256 binary funnel works at 1M; this asks what survives at **10M**, where the
256-bit codes are 320 MB — far past any LLC — and stage 1 is a genuine DRAM stream.

## Setup

- **Dataset:** Snowflake `arctic-embed-m-v1.5`, precomputed (MSMARCO v2.1 — Snowflake
  published all 71M passage vectors; we stream the first 10M+10k, no embedding to run).
  arctic-m-v1.5's headline feature *is* MRL-trained 256-d truncation, so like OpenAI
  text-embedding-3 (and unlike Cohere v3) this is a *genuine* Matryoshka embedding.
  Native 768-d sliced to **256**, then L2-normalized (the published vectors are
  unnormalized). 10M base + 10k queries, exact GT (`--gt-k 100`, ~8 min on the box).
- **Engine:** the shipped 1-bit funnel unchanged — sign-bit codes (256 bits = 32 B/vec,
  **320 MB total**), rotation ×2 + residual, tile=8, exact f32 rerank of top-C. The f32
  base for rerank is 10.2 GB; peak RSS 20.8 GB fits the 32 GB box.

## Result — recall barely notices 10×; throughput pays the linear stream cost

| rerank C | recall@10 | QPS | p50 | p99 |
|---|---|---|---|---|
| 500 | 0.9220 | 683 | 17.5 ms | 18.4 ms |
| 1000 | 0.9536 | 676 | 17.8 ms | 19.6 ms |
| 2000 | 0.9737 | 650 | 18.7 ms | 19.7 ms |
| **4000** | **0.9861** | **618** | **20.0 ms** | **22.0 ms** |
| 8000 | 0.9932 | 548 | 23.0 ms | 24.3 ms |

**Chosen operating point: C=4000 — recall 0.9861, 618 QPS, p50 20 ms.**

## What it shows

- **The Matryoshka funnel does not collapse at 10×.** Recall climbs smoothly to 0.993;
  the Hamming shortlist still contains the true neighbours. To hold the *same* recall as
  1M you widen C roughly with N — 065 hit 0.988 at C=2000 on 1M; here 0.986 needs C=4000.
  The shortlist competes against 10× more distractors, so the funnel's knob scales, but
  it *has* a knob — the bit-floor dead end (058) had none.
- **Rerank is nearly free at this scale — stage 1 is everything.** 16× the rerank width
  (C=500→8000) costs only 20% QPS. At 1M the same sweep cost 3× (6041→2026 QPS). The
  1-bit scan now utterly dominates the per-query byte budget (40 MB scanned vs 2–8 MB
  gathered), which is exactly what the funnel design wants: the cheap stage does ~all
  the work, and recall is bought at the fine-grained top.
- **Throughput drops linearly with corpus size — and (068 measured) it's compute, not
  DRAM.** 10× the corpus took QPS from 4227 → 650 at C=2000, with per-distance cost
  unchanged from 1M (~0.15 ns): the scan does 10× the popcounts at the same rate. The
  effective stream is ≈ 650 × 42 MB ≈ 27 GB/s — numerically near the box's 16/192
  pro-rata share of 064's socket bandwidth, which this entry originally read as a
  bandwidth wall. The follow-up working-set sweep (068) falsified that: marginal
  ns/code is flat from 32 MB to 320 MB of codes, and tiling gains nothing — 27 GB/s is
  what the compute *demands*, not a ceiling it hit. At 256 bits the code stream is so
  light that even 10× past the LLC, popcount stays the bottleneck.
- **Single-box practicality:** 10M vectors, recall 0.986, 618 QPS, p50 20 ms on 16 vCPU
  with 21 GB RSS — a spot c8a.4xlarge (~$0.25/hr). The full-precision fallback for the
  same corpus would scan 10.2 GB/query — the funnel is what makes 10M-in-RAM viable at
  interactive latency.

## Caveats

- **Embedding model changed along with corpus size** (OpenAI/dbpedia at 1M → arctic/
  MSMARCO at 10M), so per-C recall differences between 065 and this entry confound model
  and scale — same confound flagged in 066. The *shape* (smooth recall-vs-C, no floor)
  is the robust finding; a same-model 1M-slice control run would isolate scale cleanly.
- 10M is 14% of Snowflake's published 71M. The full set needs ~73 GB f32 for rerank —
  a larger box (or dropping the f32 tier for int8, 015) before the same experiment runs.
- p50 grew 3.1 → 20 ms with corpus size: single-query latency is one query's stream and
  no amount of cores fixes it (064) — sharding (040/041) is the latency lever at 10M+.
