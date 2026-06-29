# 017 — Serve the winner: binary+rerank through HTTP

Perf record: [`017-serving-the-winner.json`](./017-serving-the-winner.json).
Cohere v3, 1M × 1024, cosine, Granite box (8 vCPU). HTTP/JSON, concurrency sweep.

## What we did

002 measured *serving* latency, but only for the f32 exact search. The winning
algorithm — binary scan + f32 rerank (009) — had only ever been measured
in-process (batch QPS). This entry puts the funnel behind the real server and
measures user-facing latency under load, the model that actually matters: one
single-threaded search per request, `Semaphore(cores)` so excess load queues.

`server.rs` now takes `--quant binary --rerank C --scan-bits N` and runs the
funnel per request (binarize query → top-C Hamming → f32 rerank). `server` and
`loadtest` gained `--prefix` so they serve Cohere instead of only SIFT.

## Result — ~50× throughput and ~50× lower latency, end-to-end

| mode | concurrency | QPS | client p50 | client p99 | server compute p50 |
|---|---|---|---|---|---|
| f32    | 1  | 1.4  | 702 ms  | 734 ms  | 701 ms |
| f32    | 8  | 9.3  | 858 ms  | 862 ms  | 858 ms |
| f32    | 16 | 9.3  | 1714 ms | 1721 ms | 857 ms |
| f32    | 32 | 9.6  | 3428 ms | 3454 ms | 857 ms |
| binary | 1  | 82.0 | **12.2 ms** | 12.7 ms | 11.6 ms |
| binary | 8  | **461.7** | 17.3 ms | 24.2 ms | 16.9 ms |
| binary | 16 | 482.6 | 32.6 ms | 43.1 ms | 16.0 ms |
| binary | 32 | 476.2 | 65.8 ms | 80.1 ms | 16.3 ms |

## Conclusions

1. **The funnel's win holds end-to-end through the network.** At concurrency =
   cores (8), binary+rerank serves **462 QPS at p50 17 ms** vs f32's **9.3 QPS at
   858 ms** — ~50× throughput and ~50× lower latency, measured client-side over
   HTTP. The single-request floor is **12 ms** (binary) vs **700 ms** (f32).
2. **The 002 serving model reproduces exactly.** Throughput ceilings at
   concurrency = cores (binary plateaus ~462–483 beyond c=8; f32 ~9.3). Past that,
   **server compute p50 stays flat** (binary 16 ms, f32 857 ms) and only the
   *interface/queue* component grows — the rise in client p50 at c=16/32 is pure
   queuing, exactly what `Semaphore(cores)` is supposed to produce.
3. **HTTP/JSON is not the bottleneck.** At c ≤ cores the interface overhead is
   <0.7 ms; the cost is compute, as in 002. The funnel turned a 700 ms compute
   into a 12 ms compute, and that is the whole story.

## Caveats

- Heap selection, full 1024-bit scan, C=1000, f32 rerank — the serving path uses
  the in-process winner's config, not yet the prefix/tile speedups (tiling is a
  batch optimization and doesn't apply to a single request; prefix would lower
  per-request compute further at a recall cost).
- 400 requests/level; spot box. p99 at low concurrency is tight (CV small); the
  large client-p50 at high concurrency is queuing, not compute variance.
