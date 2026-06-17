# 019 — Serving latency knob: prefix scan under load

Perf record: [`019-serving-latency-knob.json`](./019-serving-latency-knob.json).
Cohere v3, 1M × 1024, cosine, Granite box (8 vCPU). HTTP, concurrency sweep.
(Recall figures carried from 018; serving doesn't score recall.)

## What we did

017 served the funnel at full 1024-bit (per-request compute ~11 ms). Tiling
(016) can't help a single request, so the lever for *serving* latency is the
prefix scan (014): scan fewer bits per request. Measured the server at three
prefix widths, each with the C that holds its recall tier.

## Result — a real latency/throughput knob, biggest under load

| scan-bits | C | recall | c=1 compute p50 | c=8 QPS | c=8 client p50 |
|---|---|---|---|---|---|
| 1024 | 1000 | 0.993 | 11.4 ms | 465 | 17.1 ms |
| 768  | 1000 | 0.970 | 11.1 ms | **634 (+36%)** | 12.3 ms |
| 512  | 2000 | 0.930 | 7.9 ms  | **732 (+57%)** | 10.7 ms |

## Conclusions

1. **Prefix is the serving-side speed lever, and it pays most under load.** At
   concurrency = cores it lifts QPS +36% (768) / +57% (512) and drops client p50
   from 17 ms to 10.7 ms — trading recall 0.993 → 0.93.
2. **Single-request vs under-load tells the bandwidth story again.** At c=1 (one
   core, no contention) 768-bit barely moves compute (11.4 → 11.1 ms): the C=1000
   rerank dominates and the scan saving is small in absolute terms. But at c=8
   (all cores contending for memory) the same prefix gives +36% — because its real
   effect is cutting *aggregate* bandwidth demand, which only bites when every core
   is scanning at once. 512-bit halves the scan enough to help even at c=1
   (11.4 → 7.9 ms).
3. **Pick the width by SLA.** Need 0.99 recall → full bits, 465 QPS, 17 ms. Can
   live with 0.97 → 768, 634 QPS, 12 ms. Latency-critical at 0.93 → 512, 732 QPS,
   11 ms. The knob spans the SLA range without touching the index format.

## Caveats

- Heap selection, f32 rerank, no tiling (single-request path). Recall is the 018
  batch number for the same scan-bits/C; the served top-k is identical logic.
- 400 requests/level, spot box; c=16 client p50 is queuing past the core ceiling.
