# database

In-memory vector search. Everything is loaded into RAM and searched there — not
disk-bound. The current milestone is the **exact brute-force baseline** plus an
**ANN-Benchmarks-style eval harness** (recall@k + QPS) on SIFT1M.

## Layout

```
src/
  fvecs.rs   read .fvecs / .ivecs (SIFT1M format)
  search.rs  exact brute-force KNN (squared-L2, bounded heap, rayon over queries)
  eval.rs    recall@k against ground truth
  main.rs    benchmark CLI
scripts/
  download-sift.sh   fetch + extract SIFT1M into data/sift/
```

## Run it

```bash
# 1. get the dataset (~168 MB download, ~500 MB on disk)
bash scripts/download-sift.sh

# 2. run the baseline (1000 queries by default)
cargo run --release -- --queries 1000 --k 10

# full 10k-query run
cargo run --release -- --queries 0
```

Output reports recall@k (1.0 for exact search — it's the oracle), QPS, and mean
per-query latency. These are the numbers every approximate index (HNSW, IVF, PQ,
…) gets compared against next.

## Dataset

[SIFT1M](http://corpus-texmex.irisa.fr/) — 1,000,000 base vectors, 128-dim,
10,000 queries, ground-truth top-100 neighbors (L2). The classic ANN baseline.
