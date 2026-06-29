# 021 — Intra-query parallelism: 4.4× lower single-query latency

Perf record: [`021-intra-query-parallel.json`](./021-intra-query-parallel.json).
Cohere v3, 1M × 1024, cosine, Granite box (8 vCPU). `--query-threads N`.

## The idea

Single-query latency (~13 ms) is one core scanning all 1M docs. But the scan is
embarrassingly parallel — split the doc range across N rayon shards, each keeps a
local top-C, then merge. Recall is unchanged (the global top-C is contained in the
union of the shards' top-Cs). This is the opposite knob from the serving model
(002), which keeps each query single-threaded to maximize *throughput*; here we
spend all cores on *one* request to minimize its latency.

## Result — near-halving per doubling, recall flat

| query-threads | recall@10 | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 | 0.983 | 13.08 ms | 13.32 | 13.56 |
| 2 | 0.983 | 7.87 ms | 8.14 | 8.27 |
| 4 | 0.983 | 5.28 ms | 5.63 | 5.92 |
| 8 | 0.983 | **2.99 ms** | 4.41 | 4.89 |

## Conclusions

1. **4.4× lower single-query latency (13 → 3 ms) at identical recall.** A clean
   latency win for the case where one query at a time matters (interactive,
   low-QPS, or tail-latency-critical).
2. **Sub-linear scaling, for the expected reason.** 8× the cores gives 4.4×, not
   8×: the scan is bandwidth-bound, so 8 cores hammering one query saturate the
   memory bus before the compute scales out — plus the merge and the
   single-threaded rerank don't parallelize. The same memory wall that tiling
   exploits (016) caps this.
3. **It's a trade, not a free win.** One request now occupies all cores, so
   aggregate throughput falls to ~1 query at a time. This is the dual of the
   serving model: use it when latency per request beats requests per second.

## Caveats

- Latency pass only (`--query-threads` is wired into the single-query path); the
  batch QPS pass still parallelizes over queries.
- recall reads 0.983 here (100-query throughput slice for the recall estimate);
  the point is it's identical across thread counts — the merge is exact.
