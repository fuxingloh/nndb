# 006 — Binary-quant search in production (our funnel, shipped) + SIMD kernels

Engines that ship the **binary-scan + rerank** design we converged on — useful to
validate our numbers and steal serving/recall-ratio choices from.

- **Weaviate** (Go): weaviate.io. **"flat + BQ"** index = brute-force **binary-quantized**
  scan + full-precision rerank — *literally our binary funnel in production*. Also has
  product quantization and a BQ-over-HNSW mode. Closest external analog; learn their
  rerank ratio and when they pick flat-BQ vs graph.
- **Qdrant** (Rust): github.com/qdrant/qdrant. Binary + scalar + product quantization,
  with `oversampling` + rerank (the funnel pattern), and the best **filtered-search**
  design (filterable HNSW + payload index). Same language as us — most readable
  reference codebase.
- **Milvus / Knowhere** (C++): index library wrapping FAISS/HNSW; segment lifecycle
  (growing→sealed) for updates — the dynamic-index angle.
- **usearch / SimSIMD** (unum): github.com/unum-cloud/usearch. Single-header SIMD
  distance kernels across ISAs incl. **hamming/popcount** — the vectorized-distance
  craft reference (relevant to [[history 050]]).
- **Lance / LanceDB**: columnar on-disk vector format + IVF_PQ; the disk-native storage
  angle (relevant to [[history 045]]).

## Why tracked

Weaviate-BQ and Qdrant-BQ are the production proof that our 1-bit-funnel design is the
right one; they're where to check our recall/QPS against shipped systems and to learn
filtered-search + dynamic-segment patterns if we ever expand scope. usearch is the
kernel-craft reference.

## Related
- Our funnel: [[history 009]]. Serving: carousel [[history 041]]. SIMD: [[history 050]].
