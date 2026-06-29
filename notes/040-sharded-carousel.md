# 040 — Sharded carousel: lowest latency floor, lower capacity

Perf record: [`040-sharded-carousel.json`](./040-sharded-carousel.json).
Cohere v3, 1M × 1024, cosine, Granite box (8 vCPU). `carousel --mode sharded`.

## The fix attempted

039's carousel ran 8 independent *full-base* carousels, so a revolution was the
whole base → a high latency floor (77 ms at low load). Fix: **shard the docs**
across workers (each owns N/8); a query rides all 8 shards in parallel and merges
partial heaps. A revolution is now N/8 docs → expect ~8× lower floor.

## Result — floor crushed (4.3 ms), but capacity drops

Rate sweep (seats=8):

| rate | throughput | p50 | p99 |
|---|---|---|---|
| 100 | 97 | **4.30** | 6.89 |
| 200 | 190 | 4.56 | 10.0 |
| 400 | 385 | 9.24 | 47.2 |
| 600 | 573 | 1052 💥 | 1775 |
| 800 | 751 | 2761 💥 | 4515 |

Seats sweep at overload (rate 800) — **more seats does not help**:

| seats | throughput | p50 |
|---|---|---|
| 8 | 751 | 2790 ms |
| 32 | 751 | 3001 ms |
| 128 | 753 | 3363 ms |

## Conclusions

1. **Sharding crushes the latency floor: 4.3 ms p50** at low load — lower than the
   per-worker carousel (77 ms), per-query (12 ms), and even close to the 021
   intra-query-parallel floor. Because each query is spread across all 8 cores
   (revolution = N/8), which is exactly *dynamic intra-query parallelism*.
2. **But capacity falls to ~550 QPS and then cliffs.** Spreading one query over all
   cores means queries can't run concurrently across cores — they time-share the
   same 8 workers. At saturation it's **compute/popcount-bound**, so adding seats
   only lengthens the revolution (latency ↑ 2790 → 3363 ms) without raising
   throughput (flat ~751). The per-worker carousel (039) keeps capacity >800 because
   its 8 workers serve *different* queries in parallel.
3. **The two designs are the two ends of the latency↔throughput dial:**
   - **sharded** = spread a query across cores → minimum latency, lower capacity.
   - **per-worker carousel** = pack queries onto cores → maximum throughput, higher floor.
   This is the 021 (intra-query) vs 016 (tiling) duality, now expressed as two
   carousel topologies.

## Toward production

Neither dominates: sharded wins at low/moderate load and tight latency SLAs;
per-worker carousel wins under heavy/bursty overload. The production answer is
**adaptive** — spread a query across cores when cores are idle (low load → ~4 ms),
and pack queries per core as load rises (high load → bounded latency, full
throughput). Synthesized in 041.

## Caveats

- Plain binary + f32 rerank C=1000; architecture study. Saturation latencies are
  queue-dominated (offered > capacity); the cliff location is the signal.
