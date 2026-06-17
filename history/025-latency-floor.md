# 025 — Latency floor: the combined single-query stack (021–025 capstone)

Perf record: [`025-latency-floor.json`](./025-latency-floor.json).
Cohere v3, 1M × 1024, cosine, Granite box (8 vCPU).

## What we did

Stack the latency levers from this block — intra-query parallel scan (021,
`--query-threads 8`) + parallel rerank (022) + prefix truncation (014) — and sweep
the prefix/C to find the single-query latency floor at each recall tier.

## Result — ~2.6 ms p50, ~4.9× below single-thread

| config | recall tier | p50 | p99 |
|---|---|---|---|
| single-thread, full, C=1000 | ~0.99 | 12.67 ms | 13.19 ms |
| **stack**, full, C=1000 | ~0.99 | **3.60 ms** | 4.14 ms |
| **stack**, 768-bit, C=2000 | ~0.97 | **2.61 ms** | 3.59 ms |
| stack, 512-bit, C=2000 | lower | 2.62 ms | 2.87 ms |
| stack, 512-bit, C=4000 | mid | 3.88 ms | 4.51 ms |

(recall is the noisy 100-query slice; canonical recall per 018.)

## Conclusions

1. **Single-query latency floor ~2.6 ms** (768-bit, C=2000, recall ~0.97) — ~4.9×
   below the 12.67 ms single-thread baseline, and at recall ~0.99 it's 3.60 ms.
2. **768/C=2000 beats 512/C=4000** (2.61 vs 3.88 ms): once the scan is cheap and
   parallel, a *bigger rerank C* costs more than a *narrower scan* saves — even
   with parallel rerank. The latency optimum is a moderate prefix + moderate C, not
   the most aggressive truncation.
3. **We've hit the floor for this design.** The scan is bandwidth-bound and at the
   VPOPCNTDQ ceiling (012/023); rerank prefetch did nothing under saturation (024);
   parallelism (021/022) cashed the obvious latency win. Lower latency now needs
   *less work*, not faster work — i.e. an approximate index (IVF, out of scope) or
   a Matryoshka embedding to hold recall at fewer bits.

## The 021–025 block, summarized

| entry | lever | outcome |
|---|---|---|
| 021 | intra-query parallel scan | **4.4× lower latency** (13 → 3 ms), recall flat |
| 022 | parallel rerank | p50 −10%, p99 −16% (tail); stack → 2.87 ms |
| 023 | register-tiled kernel | **negative** −40–60% (defeats VPOPCNTDQ) |
| 024 | rerank prefetch | **neutral** (bandwidth saturated, MLP already there) |
| 025 | combined min-latency stack | **floor ~2.6 ms p50** at recall ~0.97 |

Net across the whole effort (012–025): batch **485 → 851 QPS @ 0.993** (020) and
single-query latency **12.7 → 2.6 ms** (here). The wins came from moving fewer
bytes (tiling, prefix) and spending cores per request (021/022); every compute or
prefetch micro-opt (012, 023, 024) confirmed the kernel is already at the hardware
ceiling.

## Caveats

- Latency-vs-throughput trade: the stack uses all cores per request (021), so it's
  for latency-optimized, not throughput-bound, deployment.
- reps=1, 300 latency queries; spot box. Compare within the table.
