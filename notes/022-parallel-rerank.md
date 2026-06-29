# 022 — Parallel rerank: trim the latency tail

Perf record: [`022-parallel-rerank.json`](./022-parallel-rerank.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--query-threads 8 --rerank-par`.

## The idea

021 parallelized the *scan* across cores (13 → 3 ms) but left the rerank serial:
after an 8-shard scan, one core still rescores all C candidates from f32. At C=1000
that's 1000 L2s + 1000 random gathers on a single thread. Parallelize them too
(`rerank_par`: rayon over candidates, then a serial top-k) and pair with the
8-shard scan to minimize single-query latency.

## Result — small p50, bigger tail improvement

| C | rerank | p50 | p95 | p99 |
|---|---|---|---|---|
| 1000 | serial | 3.16 ms | 4.51 | 4.75 |
| 1000 | parallel | **2.87 ms** | 3.87 | **3.98** |
| 2000 | serial | 4.38 ms | 5.62 | 5.88 |
| 2000 | parallel | **3.80 ms** | 4.65 | **4.90** |

## Conclusions

1. **Parallel rerank shaves p50 ~10% and p99 ~16%**, more at larger C where rerank
   is a bigger slice of the request. The combined latency stack (8-shard scan +
   parallel rerank, C=1000) lands at **2.87 ms p50 / 3.98 ms p99** — 4.5× below the
   13 ms single-thread baseline.
2. **The win is mostly the tail.** Because the scan still dominates the request,
   parallelizing rerank moves p50 only a little, but it removes the serial-rerank
   variance that fattened p99 — useful when tail latency is the SLA.
3. **Diminishing returns confirm the scan is the floor.** Once both stages run on
   all cores, single-query latency is bounded by the bandwidth-bound scan (021),
   not the rerank. To go lower you must scan fewer bytes — i.e. prefix (next/
   capstone), not more parallelism.

## Caveats

- Latency path only; pairs with `--query-threads`. Uses all cores per request
  (the latency-vs-throughput trade from 021).
- reps=1, 300 latency queries; p99 from 300 samples is indicative, not tight.
