# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A greenfield experiment in **in-memory** top-K vector search: load all vectors into RAM (not disk-bound) and search there. The arc is exact baseline → approximate indexes (HNSW/IVF/PQ), measured the ANN-Benchmarks way (recall-vs-QPS). `database/` is the Rust engine; `web/` will host the eval writeup (not built yet).

## Commands

All Rust work happens in `database/`:

```bash
cd database
bash scripts/download-sift.sh                    # fetch SIFT1M into data/sift/ (~168MB dl, 488MB RAM)
cargo build --release                            # release build is mandatory for any real numbers
cargo test --release                             # run unit tests
cargo test --release returns_k_nearest_ascending # run a single test by name
cargo run --release -- --queries 1000 --k 10     # benchmark on a subset (fast iteration)
cargo run --release -- --queries 0               # benchmark on all 10k queries
```

The dataset (`database/data/`) is gitignored — `download-sift.sh` must be run before any benchmark.

## Architecture

The benchmark mirrors the ANN-Benchmarks contract: a `base` vector set, a `query` set, and ground-truth nearest neighbors per query so recall is measurable. Data flows `fvecs` (load) → `search` (rank) → `eval` (score), orchestrated by `main`.

- **`src/fvecs.rs`** — readers for `.fvecs` (f32) / `.ivecs` (i32), the SIFT1M binary format: flat records of `[i32 dim][dim × value]`, little-endian, dim constant across the file. Vectors are stored row-major in a flat `Vec`; access rows via `.row(i)`.
- **`src/search.rs`** — `knn_batch` is the interface every index implements. Exact brute-force today: squared-L2 (sqrt is monotonic, skip it), a bounded max-heap of size k (O(n log k), not a full sort), parallelized across queries with rayon. **New approximate indexes should slot behind this same `knn_batch(base, queries, k) -> Vec<Vec<u32>>` shape** so the existing harness measures them unchanged.
- **`src/eval.rs`** — `recall_at_k`: mean over queries of |returned ∩ true top-k| / k.
- **`src/main.rs`** — loads, runs, reports recall@k + QPS + per-query scan cost.

## Things that will trip you up

- **Exact search reports recall@10 ≈ 0.9994, not 1.0 — this is correct, not a bug.** SIFT descriptors are uint8 so L2 distances are exact integers in f32 (no float error); the ~6/10000 "misses" are boundary ties where the k-th and (k+1)-th neighbors are equidistant and ground truth picks a different (equally valid) point. Don't "fix" this.
- **`--queries N` truncates queries but not ground truth.** `recall_at_k` therefore asserts `found.len() <= truth.len()`, not `==`. Keep that asymmetry when changing eval.
- Always benchmark with `--release`; debug builds give meaningless (orders-of-magnitude slower) numbers because the inner distance loop relies on autovectorization (`opt-level=3`, `lto`, `codegen-units=1` in Cargo.toml).
- The "single-thread-equivalent ms/query" line (~100ms scanning 1M vectors) is the whole motivation for approximate indexes — it's the number new indexes must beat while holding recall.
