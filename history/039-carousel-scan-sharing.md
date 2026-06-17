# 039 — Carousel (cooperative scan-sharing) under bursty load

Perf record: [`039-carousel-scan-sharing.json`](./039-carousel-scan-sharing.json).
Cohere v3, 1M × 1024, cosine, Granite box (8 vCPU). `src/bin/carousel.rs`.

## The idea (user's)

Fixed query-tiling (016/038) assumes a full tile. Real traffic is **bursty**, so a
tiled server must either **wait** to fill a tile (adds latency — "the car waits at
the station") or scan a **partial** tile (wastes the 122 MB base read on empty
seats). The proposal: keep each worker **continuously cycling** the base; an
arriving query **attaches** at the current cursor, rides exactly one revolution
(sees all N docs), then leaves with its top-C. Every doc read is shared by whoever
is aboard — *dynamic* tiling with tile = current in-flight count, and no waiting.

This is **cooperative scan-sharing** (Crescando, IBM Blink). Built as a
self-contained benchmark with Poisson (bursty) arrivals, comparing the carousel to
the per-query serving model (017).

## Result — bounded latency, no cliff; higher floor at low load

| offered QPS | per-query p50 / p99 | carousel p50 / p99 |
|---|---|---|
| 100 | **12.5 / 16** | 77 / 172 |
| 200 | **11.5 / 16** | 45 / 104 |
| 400 | **11.8 / 21** | 32 / 63 |
| 600 | 533 / 910 💥 | **30 / 61** |
| 800 | 2045 / 3367 💥 | **35 / 59** |

(Throughput tracks offered rate for both up to ~750; per-query is *saturating* at
600–800 — the high latency is a growing queue — while the carousel is not.)

## Conclusions

1. **The idea works for its intended case.** Above the per-query capacity (~500
   QPS) per-query latency **collapses** (12 → 533 → 2045 ms) as the queue explodes,
   while the carousel stays **flat at ~30–35 ms p50 / ~60 ms p99** through 800 QPS,
   and sustains more throughput (scan-sharing). For bursty traffic — where bursts
   transiently exceed capacity — that's the whole game: the carousel keeps latency
   bounded during the burst; per-query spikes to seconds.
2. **The tradeoff is a higher latency floor at low load** (77 ms vs 12 ms at 100
   QPS): a query waits for a full revolution shared with whoever's aboard, and at
   light/bursty load there's idle-poll + cold-cache overhead. So below capacity,
   per-query is better on latency.
3. **The crossover is the design knob.** Per-query wins when you're comfortably
   under capacity; the carousel wins at/above it. That points at an **adaptive**
   server (per-query at low load, carousel under burst) and at lowering the
   carousel's floor.

## The obvious fix to tune next

This version runs **8 independent full-base carousels** (one per worker), so a
"revolution" is the *whole* base (N docs) → the latency floor is a full scan.
Sharding the docs across workers (each owns N/8, a query rides all shards in
parallel and merges) makes a revolution N/8 docs → ~8× lower floor, while keeping
the sharing. That's 040.

## Caveats

- Plain binary scan + f32 rerank C=1000 (no rotation); architecture comparison, so
  recall (~0.89) isn't the focus here.
- Warmup samples dropped; Poisson arrivals via a seeded LCG; in-process (HTTP adds
  <0.4 ms per 017).
