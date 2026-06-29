# nndb — the Rust engine

In-memory vector search: every vector is loaded into RAM and searched there, not
disk-bound. This crate is the engine; the story, results, and the full trail of
numbered experiments live in the repo root `README.md` and at the writeup
(`/notes`).

## Layout

```
src/
  fvecs.rs   read .fvecs / .ivecs (SIFT1M format)
  search.rs  exact brute-force KNN (squared-L2, bounded heap, rayon over queries)
  quant.rs   binary quantization + Hamming funnel, rotations, rerank
  eval.rs    recall@k against ground truth
  main.rs    benchmark CLI (the `vsearch` bin)
  bin/       serving (server, loadtest, carousel) + per-experiment drivers
scripts/
  download-sift.sh   fetch + extract SIFT1M into data/sift/
  fetch-cohere.py    fetch Cohere v3 embeddings into data/cohere/
```

## Run it

```bash
# 1. get the dataset (~168 MB download, ~500 MB on disk)
bash scripts/download-sift.sh

# 2. exact baseline (1000 queries by default)
cargo run --release -- --queries 1000 --k 10

# full 10k-query run
cargo run --release -- --queries 0
```

Release builds are mandatory for any real numbers — the inner distance loop relies
on autovectorization (`opt-level=3`, `lto`, `codegen-units=1`). Output reports
recall@k, QPS, and mean per-query latency.

## Datasets

- [SIFT1M](http://corpus-texmex.irisa.fr/) — 1M base vectors, 128-dim, 10k queries,
  ground-truth top-100 (L2). The classic ANN baseline (uint8 → lossless).
- Cohere v3 — 1M × 1024-dim real embeddings (cosine), the lossy testbed for
  quantization. Fetched via `scripts/fetch-cohere.py`.
