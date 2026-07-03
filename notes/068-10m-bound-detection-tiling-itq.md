# 068 — 10M is still compute-bound: the DRAM-wall read of 067 was wrong

Perf record: [`068-10m-bound-detection-tiling-itq.json`](./068-10m-bound-detection-tiling-itq.json).
c8a.4xlarge (Zen5, 16 vCPU) **on-demand**, ap-northeast-1 — after us-east-2 spot
reclaimed two boxes mid-experiment. Three measurements on the 067 dataset (Snowflake
arctic-256, 10M), independently re-fetched; the incidental rerank-C re-run **replicated
067 exactly on recall (4 decimals) and within ~4% on QPS** — a free cross-region check.

## 1. Bound detection: marginal ns/code is flat past the LLC → compute-bound

067 inferred a DRAM wall from an aggregate coincidence (27 GB/s ≈ the box's pro-rata
share of 064's socket bandwidth). The working-set sweep (005's method, on the 1-bit
funnel) says otherwise. Per-N *average* ns/distance falls with N (fixed rerank cost
amortizing); the *marginal* cost per code between successive sizes is the real signal:

| codes | marginal ns/code |
|---|---|
| 32 MB (cache-adjacent) | 0.168 |
| 64 MB | 0.156 |
| 128 MB | 0.151 |
| **320 MB (10× past LLC)** | **0.151** |

No step up crossing the cache boundary — scanning DRAM-resident codes costs the same
as cache-resident ones. The scan demands ~27 GB/s; the box can supply more; popcount
is the bottleneck. 067's "wall" was numerology: the follow-up section there is amended.

## 2. Tiling: still pays nothing — batch=1 wins at 10M

| batch | QPS |
|---|---|
| **1** | **668** |
| 2 | 537 |
| 4–32 | 601–640 |

066 found tiling useless at 1M (32 MB of codes, compute-bound); the expectation here
was that 320 MB would flip it back. It doesn't — consistent with §1: tiling amortizes
*bandwidth*, and bandwidth isn't the constraint. The regime flip of 066 is not about
corpus size at all at these code widths; **256-bit codes are simply too cheap to stream
for DRAM to matter on a 16-core box.** (064's wall was real but at 8× the code bytes ×
12× the cores — scale one or both and the roofline returns.)

## 3. ITQ at 10M: the 066 gain evaporates

Same itq bin as 066 (50k sample, 30 iters), adapted to rotate in place (a second copy
of the 10 GB projection would OOM the box):

| rerank C | random+residual | ITQ+residual | delta |
|---|---|---|---|
| 500 | 0.9400 | 0.9420 | +0.0020 |
| 1000 | 0.9666 | 0.9650 | −0.0016 |
| 2000 | 0.9815 | 0.9804 | −0.0011 |
| 4000 | 0.9915 | 0.9897 | −0.0018 |
| 8000 | 0.9958 | 0.9957 | −0.0001 |

Deltas are noise. On 1M OpenAI-256 (066) ITQ gave +0.7→+0.1 pts; here nothing. Two
candidate explanations, unresolved: the 50k training sample is 0.5% of this corpus
(vs 5% at 1M), or arctic-embed's geometry (MRL-trained specifically for 256) leaves
less binarization error for ITQ to reclaim. Either way, on this stack at 10M the
learned rotation is **not worth carrying** — random FWHT + residual is the config.

## Takeaway

At 256-bit codes the funnel stays compute-bound through at least 10M vectors on 16
cores: no tiling, no learned rotation, batch=1, rerank width C as the single recall
knob. The simplest config is the best config — everything clever that helped at other
operating points (tiling at f32 widths, ITQ at 1M) washes out here.
