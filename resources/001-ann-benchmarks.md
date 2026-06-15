# 001 — ANN-Benchmarks

- **URL:** https://ann-benchmarks.com/index.html
- **Code:** https://github.com/erikbern/ann-benchmarks
- **Type:** living benchmark / leaderboard (results change as algorithms and hardware evolve)

## What it is

The de-facto standard benchmark for approximate nearest-neighbor search. Runs many ANN implementations through a reproducible (Docker-based) harness and plots the **recall vs. queries-per-second** Pareto frontier per dataset, plus index build time and index size. This is the reference frontier to compare our engine's recall/QPS against.

## Algorithms it compares (~37)

`faiss-ivf`, `scann`, `hnswlib`, `hnsw` (several variants), `glass`, `NGT` (`qg`/`panng`/`onng`), `vamana(diskann)`, `pynndescent`, `annoy`, `n2`, `flann`, `pgvector`, `qdrant`, `milvus(knowhere)`, `weaviate`, `vald(NGT-anng)`, `vearch`, and baselines `bruteforce-blas` / `bf`.

Useful angle: **`bruteforce-blas`** is the exact baseline (BLAS GEMM) — directly relevant to our batched/GEMM-style direction. The graph indexes (hnsw/glass/NGT) and IVF (faiss-ivf) sit at the high-recall/high-QPS frontier we'd eventually be measured against.

## Datasets (name — dim — metric — k)

| Dataset | Dim | Metric | k |
|---|---|---|---|
| sift-128-euclidean | 128 | L2 | 10 |
| gist-960-euclidean | 960 | L2 | 10 |
| fashion-mnist-784-euclidean | 784 | L2 | 10 |
| glove-100-angular | 100 | angular/cosine | 10 |
| glove-25-angular | 25 | angular/cosine | 10 |
| nytimes-256-angular | 256 | angular/cosine | 10 |
| sift-256-hamming | 256 | Hamming | 10 |
| word2bits-800-hamming | 800 | Hamming | 10 |
| kosarak-jaccard | — | Jaccard | 10 |

We use **sift-128-euclidean** (as `.fvecs`). Note the variety of metrics — angular/Hamming/Jaccard would need different distance kernels than our L2.

## Methodology notes

- Metric is **recall@k vs QPS** (Pareto frontier), with secondary plots for build time and index size.
- Reproducible runs via Docker; each algorithm contributes a wrapper.
- Tuning matters: each algorithm is swept over its parameters and only the frontier is shown.

## Related

- **big-ann-benchmarks** (big-ann-benchmarks.com) — billion-scale variant.
- **MTEB** — ranks embedding *models*; **BEIR** — text retrieval quality. Different layer (embedding quality, not index speed).
