# 024 — Rerank candidate prefetch: neutral

Perf record: [`024-rerank-prefetch.json`](./024-rerank-prefetch.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --batch 16 --rerank-pf`.

## The idea

The rerank rescores C candidates by random-access gather into the 3.9 GB f32 store
— unlike the sequential scan, a legit software-prefetch target. The hardware
prefetcher only streams *within* a row once touched; it can't see the next random
candidate. So prefetch the row PF=8 ahead to hide its initial DRAM latency.

## Result — no change

| scan-bits | C | recall | no prefetch | prefetch |
|---|---|---|---|---|
| 1024 | 1000 | 0.9931 | 848.8 | 848.0 |
| 1024 | 2000 | 0.9975 | 731.3 | 729.7 |
| 512  | 4000 | 0.9601 | 644.6 | 634.4 |

Within noise (CV < 1%), slightly negative at high C.

## Why it does nothing here

Prefetch buys latency-hiding *only when there is spare memory bandwidth*. Under the
8-core batch this workload is bandwidth-saturated (it's why tiling worked), so
there's no idle bandwidth for prefetched lines to ride on — the prefetch just
competes for the same saturated bus. And rayon-over-queries already issues many
independent gathers concurrently, so the memory system has plenty of in-flight
requests (memory-level parallelism) without explicit hints. Net: nothing to gain,
and at high C the extra prefetch instructions are slight overhead.

This is consistent with 015 (rerank isn't the batch bottleneck) and 016/023 (the
scan, and the saturated memory bus, are the real constraints).

## Conclusion

Dropped as a default (`--rerank-pf` kept opt-in). Prefetch would only matter in a
*latency* path with one query and spare bandwidth — but there the scan dominates
(021/022), so it still wouldn't move the needle.

## Caveats

- x86-only intrinsic; no-op on other arches. Batch path; reps=6.
