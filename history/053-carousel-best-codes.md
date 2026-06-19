# 053 — Carousel serves the real engine (best codes), and why fan-out must adapt

Perf record: [`053-carousel-best-codes.json`](./053-carousel-best-codes.json). Granite
box (8 vCPU). `src/bin/carousel.rs`. Cohere 1M × 1024, grouped fan=4, RAM rerank C=1000.

The serving target was always the point, so the carousel should serve the *real* engine,
not plain binary. Wired **rotation×2 + residual** into the carousel's code formation as
defaults (rerank stays exact on raw f32). Recall is now **~0.998** (identical code path
to 051), not 0.89.

## Result

| offered QPS | throughput | p50 | p99 |
|---|---|---|---|
| 200 | 190 | 5.0 ms | 8.4 ms |
| 400 | 385 | 6.5 ms | 18.5 ms |
| 600 | 573 | 12.6 ms | 39.9 ms |
| 800 | 751 | **672 ms** 💥 | 1325 ms |
| 1000 | 936 | **1762 ms** 💥 | 2936 ms |

## Two findings

1. **Rotation+residual compose into the carousel for free.** The latency envelope is
   identical to plain-binary (041) — now at **recall ~0.998 instead of 0.89**. They're
   zero-scan-cost code-formation knobs, fully orthogonal to the serving layer. So the
   carousel now serves the real engine at no latency cost.
2. **A *fixed* fan cliffs.** fan=4 is great to ~600 QPS (12.6 ms) then collapses at 800
   (672 ms) — matching 041, where the optimal fan *drops* as load rises (spread a query
   across cores when idle; pack one-per-core under load). A single fixed fan cannot
   cover the load range without a cliff.

## Conclusion

The carousel is now the default best-codes serving path (recall 0.998, ~5–13 ms to
~600 QPS). But **the adaptive fan-out controller (`F = cores ÷ in-flight`, set live) is
required, not optional** — it's the piece that turns the per-load *optimal* envelope
(no cliff to capacity) into something a single running server achieves automatically.
That controller is the last unbuilt piece.

## Update — the default (fan=1) is already a no-cliff server

fan=1 (pack), best codes, recall ~0.998:

| offered QPS | throughput | p50 | p99 |
|---|---|---|---|
| 200 | 190 | 12.1 ms | 16.0 ms |
| 600 | 573 | 15.1 ms | 31.1 ms |
| 1000 | 937 | 49.5 ms | 78.4 ms |
| 1200 | 1115 | 542 ms 💥 | past capacity |

No cliff to **~937 QPS** (the 8-core compute capacity); it only breaks at 1200,
which is genuinely past what the box can do (a cores problem, not a scheduler one).

**Revised conclusion:** carousel + best codes IS the default serving engine —
recall ~0.998, no-cliff to capacity, 12–49 ms p50, fan=1 the safe default. fan>1
lowers low-load latency (200 QPS: ~4.5 ms at fan=8 vs 12 ms at fan=1) but cliffs
under heavy load. The adaptive fan-out controller (`F = cores ÷ in-flight`) would get
the best of both — but it's a **low-load latency optimization, not a correctness need.**
