# 020 — Capstone: the performance effort (012–020)

Perf record: [`020-capstone.json`](./020-capstone.json).
Cohere v3, 1M × 1024, cosine, Granite box.

## Head-to-head (same run, reps=8)

| config | recall@10 | batch QPS | p50 |
|---|---|---|---|
| 009 baseline — tile=1, full-1024, C=1000 | 0.9931 | 484.8 | 11.84 ms |
| **best throughput** — tile=16, full, C=1000 | 0.9931 | **851.3 (+76%)** | 12.06 ms |
| **best recall** — tile=16, full, C=2000 | **0.9975** | 730.2 (+51%) | 13.88 ms |

Tiling alone buys **+76% batch QPS at bit-identical recall**, or you can spend
some of it on recall and still land **0.9975 @ 730 QPS** — higher recall *and*
+51% throughput vs the baseline. Single-query p50 is unchanged (tiling is a batch
lever).

## What moved the needle, what didn't

The goal was QPS up / latency down. Across 012–020:

- **016 tiled scan — the win.** +73–76% QPS, zero recall cost. The binary scan was
  bandwidth-bound (each query streamed the full 122 MB base); tiling reuses each
  doc across T queries so the base streams once per tile. Free throughput.
- **014 prefix scan — a recall/QPS dial.** Fewer scanned bits ≈ proportionally
  more scan QPS (512-bit ~2×). Best below ~0.97 recall; at high recall the full
  scan + tiling wins (018).
- **017 serving — the funnel's real payoff.** End-to-end through HTTP: 462 QPS @
  17 ms vs f32 exact's 9.3 QPS @ 858 ms — ~50× on both throughput and latency.
- **019 serving prefix knob — latency down under load.** +36–57% serving QPS,
  p50 17 → 11 ms, trading recall.
- **013 counting selection — a latency-only knob** (−14% single-query, −6% batch);
  kept selectable, heap stays the batch default.
- **015 int8 rerank — memory only.** No QPS gain (rerank was never the
  bottleneck), −1.5 recall pts; a 4× store-shrink lever, nothing more.
- **012 multi-accumulator hamming — negative result.** `count_ones` already
  autovectorizes to hardware vector popcount; hand-splitting the reduction made it
  ~2.2× *slower*. Reverted.

## The throughline

Two ideas explain almost every result: **(1)** this workload is bandwidth-bound on
the box, so the levers that win are the ones that move *fewer bytes* (tiling: fewer
re-reads; prefix: fewer bits) — not the ones that touch compute (012, 013) or the
non-bottleneck (015). **(2)** every time a bandwidth lever lands, the scan
re-prices toward compute-bound (the 003→007 cascade, seen again at tile≈16), so
gains taper — the kernel and the memory system trade being the bottleneck.

Net: the binary+rerank funnel went from **~485 QPS @ 0.993 (009 baseline) to 851
QPS @ 0.993 or 730 @ 0.9975** in batch, and serves **462 QPS @ 17 ms** end-to-end
(50× the f32 exact baseline). The remaining headroom is the SIMD asymmetric LUT
(parked, see 011) and a Matryoshka-trained embedding (would make prefix hold recall
at fewer bits, 014/018).
