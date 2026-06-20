# 057 — Capstone: the best stacked implementation

Synthesis of the project (entries 001–056). Cohere v3 1M × 1024, cosine, single 8-core
box (Granite, Xeon 6975P-C). The question this whole project asked: *how fast can exact-ish
top-k search go inside one IVF cell, and what's the best way to build and serve it?*

## The winner

**A 1-bit binary funnel — rotation + residual + tiling — served by the carousel.**

- **Scan:** sign-bit (1 bit/dim) Hamming codes, `count_ones` → VPOPCNTDQ, query-tiled.
- **Codes:** random rotation (026) + residual / centroid-subtraction (046) — both free,
  both pure recall.
- **Rerank:** exact f32 on the top-C candidates.
- **Serving:** carousel (cooperative scan-sharing, fan-out per load) — bounded tail
  latency, no cliff.

### Headline numbers (in-RAM, 8 cores, 1M × 1024)

| objective | recall@10 | QPS | p50 |
|---|---|---|---|
| max recall | **0.9986** | 883 | 10.6 ms |
| balanced | 0.9952 | **963** | 8.0 ms |
| predictable tail (p99≈p50) | 0.995 | ~350 | 12 ms (p99 16 ms) |
| serving envelope | 0.998 | no-cliff to ~940 | 12→49 ms |

Footprint: **128 MB of codes** in RAM (+ f32 for rerank, RAM or SSD).

## The full frontier — everything we tried, and where it landed

Recall vs QPS vs bytes/vector (Cohere 1M, recall@10 with rerank):

| method | bytes/vec | recall | QPS | verdict |
|---|---|---|---|---|
| **binary funnel (rot+residual)** | 128 | 0.995–0.999 | **880–960** | **winner (QPS)** |
| PQ M=64 (054) | 64 | 0.9998 | 62 | footprint, slow |
| PQ M=32 (054) | 32 | 0.992 | 140 | footprint, slow |
| PQ M=16 (054) | 16 | 0.905 | 273 | footprint, slow |
| OPQ M=16 (056) | 16 | ~0.95 (+2.3 vs PQ) | (=PQ) | better bytes, same slow |
| ITQ rotation (055) | (binary) | +1.5 vs random | (=binary) | tiny recall knob |
| HNSW-in-cell (043) | graph | ≤0.98 | ≪ funnel @ high-D | loses in-cell |
| exact / PDX (032/044, cut) | f32 | 1.0 | 9–55 | exact tier only |

Two Pareto frontiers, and they point opposite ways:
- **recall-per-QPS → the binary funnel wins outright** (10–15× the QPS of PQ at equal recall).
- **recall-per-byte → PQ/OPQ win** (32 B at 0.99 vs 128 B), but only matters when RAM is
  the hard constraint.

## The one physical reason it all comes out this way

**Popcount beats gather.** The binary scan's per-doc op is `count_ones` → VPOPCNTDQ (8
u64/instruction, autovectorized). Every alternative's per-doc op is a *data-dependent
gather* — PQ/OPQ's ADC table lookups, the asymmetric LUT (011), int8 reranking — and
gathers **don't vectorize**. So even when PQ uses 8× fewer bytes and fits in cache, it's
**gather-compute-bound** and loses to popcount. This single fact (first seen in 011,
reconfirmed at scale in 050/054) decides the whole frontier: on a CPU, 1-bit + popcount
is the throughput-optimal representation, and "fewer bytes" only helps the *footprint*,
not the *speed*, unless you write SIMD ADC (parked, 050).

## What each knob actually contributes

| knob | axis | effect |
|---|---|---|
| binary funnel (009) | the engine | 1-bit scan + rerank, popcount-speed |
| tiling, tile=8 (016/038) | throughput | +73% QPS (amortize base read) |
| rotation ×2 (026) | recall | free, spreads info across bits |
| **residual (046)** | recall | **free, +3–9 pts at low bits** (biggest recall lever) |
| ITQ (055) | recall | +1.5 pts over random rotation (small, dense-rotation cost) |
| carousel (039–041) | tail latency | no cliff under burst |
| disk hybrid (045) | RAM cost | 32× less committed RAM (when corpus > RAM) |
| adaptive-C (048) | disk rerank | keeps SSD reads affordable |
| PQ/OPQ (054/056) | footprint | 8–32 B/vec, but lose QPS |

These are **orthogonal axes** (049's roofline: `QPS ≈ 1.10e9·cores/N`, separable from
recall = `1 − e^12.45·N^0.28·C^−1.08·bits^−1.97`). You tune recall (rotation/residual/
bits/C) without moving the throughput roofline.

## Verdict

The best stacked implementation is the **rotated-residual binary funnel served by the
carousel**: ~**0.995 recall @ ~960 QPS @ 8 ms**, no-cliff serving to ~940 QPS, 128 MB
codes, on one 8-core box. The PQ/ITQ/OPQ exploration *confirms* it rather than beating
it — PQ/OPQ are the footprint alternatives (use when RAM-bound), ITQ a marginal recall
nudge. The throughput crown belongs to popcount.

### Loose ends (all optional)
- Adaptive fan-out controller (sub-5 ms at light load) — only unbuilt *engine* piece.
- SIMD ADC (4-bit + `vpshufb`) — the one thing that could make PQ QPS-competitive.
- Above-us layer: build the IVF router to make this a full index.
- `web/` writeup + CLAUDE.md thesis refresh.
