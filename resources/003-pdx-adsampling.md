# 003 — PDX & ADSampling (the dimension-pruning / layout frontier)

- **ADSampling:** Gao & Long, SIGMOD 2023. arXiv:2303.09855. Code: github.com/gaoj0017/ADSampling
- **PDX:** Kuffo, Krippner & Boncz, SIGMOD 2025. arXiv:2503.04422. Blog: lkuffo.com/vertical-vector-similarity-search-pdx/
- **Type:** training-free distance-computation acceleration (algorithm + data layout); active line of work from the same DB-research groups as RaBitQ.

## ADSampling — early-terminated distance comparison

Random-rotate (JL), compute squared L2 incrementally in batches of `delta` (=32);
prune a candidate once partial `res ≥ threshold·ratio(D,i)`, where
`ratio(D,i) = (i/D)(1+ε0/√i)²`, `ε0≈2.1`. Survivors get the exact distance
(rotation preserves L2). Training-free; probabilistic recall guarantee via ε0.
We use it on the exact scan (history 031) and as a binary-funnel rerank tier (034).

## PDX — vertical (dimension-major) data layout

Store vectors in blocks, transposed within a block (all vectors' dim 0, then dim 1,
…). Distance over a block = loop over dims with inner loop over vectors
(multiple-vectors-at-a-time) → autovectorizes, no horizontal reduction. ~40% faster
plain scans, and it **restores** dimension-pruning's benefit (ADSampling/BSA) to
2–7× — pruning can *lose* on row-major because early termination breaks SIMD.
PDX-BOND is their preprocessing-free pruning variant. We use it in history 032/033.

## Why tracked

The current training-free frontier for making *exact / high-recall* distance
computation fast on CPU. Complements the quantization frontier ([[002-rabitq]]):
quantization wins the approximate/throughput tier, PDX+pruning wins the exact tier.
Both evolving (SIGMOD'23 → '25), with maintained code.

## Related

- **BSA** — another dimension-pruning method PDX accelerates.
- **SAQ** (arXiv:2509.12086) — quantization w/ dimension segmentation; bridges both.
- RaBitQ / Extended RaBitQ (see [[002-rabitq]]) — same research community.
