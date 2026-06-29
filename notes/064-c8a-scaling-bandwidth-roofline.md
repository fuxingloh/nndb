# 064 — c8a scaling spectrum: where the funnel hits the DDR5 bandwidth wall

062 found AMD Zen5 (c8a) the perf-per-buck winner at the 8-vCPU tier. This entry pushes
*one* family across its whole size range — **8 → 192 cores** — to find where the funnel
stops scaling, and what the ceiling is.

Same engine, same Cohere v3 (1M × 1024), same flags, every box the advised size; recall a
constant **0.9952**, so the only axis is cores. (`xlarge`/4c was dropped — its 8 GB RAM OOMs
the dataset prep; it's a pure-compute point anyway.)

## The measured spectrum

| size | cores | funnel QPS | QPS/core | p50 |
|---|---|---|---|---|
| 2xlarge | 8 | 2,310 | 289 | 3.1 ms |
| 4xlarge | 16 | 4,462 | 279 | 3.2 ms |
| 8xlarge | 32 | 7,658 | 239 | 3.2 ms |
| 12xlarge | 48 | 9,710 | 202 | 3.1 ms |
| 16xlarge | 64 | 13,670 | 214 | 3.2 ms |
| 24xlarge | 96 | 13,555 | 141 | 3.3 ms |
| 48xlarge | 192 | 19,881 | 104 | 3.6 ms |

Two reads jump out. **p50 is flat at ~3 ms** the whole way — latency is one query's work and
doesn't care how many cores the box has; throughput is what scales. And **QPS/core decays**
from 289 to 104: the box stops converting cores into throughput.

## A two-tier bandwidth wall

- **Compute-bound to ~16–32 cores:** ~289 QPS/core, near-linear (popcount on AVX-512).
- **Rolls off through 48c** as memory contention ramps in.
- **Shared-socket plateau ~13,600 QPS at 64–96 cores:** a sub-full-socket instance shares
  the socket's 12 DDR5 channels with co-tenants and tops out around 245 GB/s (~40% of peak).
  64c and 96c land at the same QPS — more cores, no more throughput.
- **Full-socket breakout at 192c → 19,881 QPS:** the 48xlarge owns *all* the channels and
  reaches ~358 GB/s (58% of the 614 GB/s DDR5-6400 peak).

So the wall is memory bandwidth, and it's tiered: you only escape the shared plateau by
renting the **whole socket**. Even then it's heavily sublinear — 2× the cores (96→192) buys
1.47× the QPS — because by 96 cores the funnel is already streaming codes faster than a
shared socket can feed them.

## Why the ceiling sits where it does

The funnel at scale is pure stream: every query scans the whole 1-bit base, tiled across T
queries.

```
S = 1M × 1024 bits = 128 MB   (binary base)
R = 500 × 4 KB     = 2 MB      (rerank gather, un-amortizable)
T = 8                          (tile depth)
bytes/query = S/T + R = 16 + 2 = 18 MB
```

At the saturated full socket, `BW / 18 MB = 358 GB/s / 18 MB ≈ 19,900 QPS` — which is exactly
the measured 19,881. Once bandwidth-bound, the ceiling is just `achievable_BW ÷ 18 MB`; cores
beyond the knee add nothing.

## Takeaway

The funnel is compute-bound only on small boxes, where it's also the best **perf-per-buck**
(~289 QPS/core, linear) — so **scale *out* with small instances**, and reach for a full
socket (48xlarge, ~20k QPS) only when you need that throughput in one footprint. Past the
knee, the DDR5 bus — not the cores — is the ceiling.
