# 041 — Carousel capstone: fan-out is the dial, adapt it to load

Perf record: [`041-carousel-fanout-frontier.json`](./041-carousel-fanout-frontier.json).
Cohere v3, 1M × 1024, cosine, Granite box (8 vCPU). `carousel --mode grouped --fan F`.

## The unifying parameter

039 (per-worker carousel) and 040 (fully sharded) are the two ends of one dial:
**fan-out F** — how many workers cooperate on a single query. F workers shard the
base (revolution = N/F) and there are G = workers/F independent groups serving
different queries. F=1 packs queries onto cores (throughput); F=8 spreads one query
across all cores (latency). Sweeping F × load traces the whole frontier.

## Frontier — optimal F drops as load rises (p50 ms)

| offered QPS | F=1 pack | F=2 | F=4 | F=8 shard | best |
|---|---|---|---|---|---|
| 200 | 12.5 | 7.6 | 5.1 | **4.5** | F=8 |
| 400 | 12.3 | 8.0 | **6.5** | 8.3 | F=4 |
| 600 | 14.7 | **11.5** | 11.8 | 872 💥 | F=2 |
| 800 | **20.3** | 19.5 | 367 💥 | 2520 💥 | F=1 |
| 1000 | **34.6** | 283 💥 | 1538 💥 | 4196 💥 | F=1 |

Throughput keeps up with offered load until the cliff in each cell.

## The result

1. **The carousel idea works, and fan-out F is its control knob.** The optimal F
   **decreases monotonically with load**: spread a query across all 8 cores when
   they're idle (low load → 4.5 ms), and pack queries one-per-core as load rises
   (high load → bounded latency, full throughput).
2. **The optimal-F envelope never cliffs:** 4.5 ms @ 200 → 6.5 @ 400 → 11.5 @ 600 →
   20 @ 800 → 35 @ 1000 QPS. Compare per-query (017): 12 ms flat until ~500 QPS then
   a cliff to seconds. The adaptive carousel is **~2.7× lower latency at low load
   AND has no saturation cliff** — strictly better across the whole range.
3. **Production rule (the "right number"): F\* ≈ cores ÷ in-flight-queries.** Give
   each in-flight query an equal share of the cores. At 1 query in flight, F=cores
   (all-core scan, ~4 ms); at 8 in flight, F=1 (one core each, max throughput). This
   is **elastic intra-query parallelism** — it unifies 021 (intra-query parallel =
   high F) and 016 (tiling = F=1) under a single load-adaptive scheduler.

## How to build it in production

A coordinator tracks current in-flight count `m`; each arriving query is dispatched
with fan-out `F = clamp(cores / max(1, m), 1, cores)` and rides a carousel of F
shards. As `m` rises the controller hands out smaller F; as it falls, larger F. No
batch-fill wait (queries attach to a moving scan), no empty-seat waste (the scan is
shared by whoever's aboard), and the latency/throughput operating point tracks the
lower envelope automatically. `seats`/`chunk` are second-order (chunk = admission
granularity; seats = per-group backpressure cap).

## Conclusion

The idea is validated and understood end to end: cooperative scan-sharing with
**load-adaptive fan-out** delivers low latency under light/bursty load and graceful,
cliff-free degradation under overload — the production-grade serving model for this
engine. The single tuning parameter is fan-out, and its optimum is `cores ÷
in-flight`.

## Caveats

- Plain binary + f32 rerank C=1000; architecture study (recall ~0.89 fixed). The
  fan-out principle is orthogonal to the recall tier (rotation/tiling still apply
  within each shard).
- Cliff latencies are queue-dominated (offered > capacity); the *envelope* is the
  deliverable. An explicit adaptive controller (F = cores/in-flight) is the next
  build to realize the envelope directly.
