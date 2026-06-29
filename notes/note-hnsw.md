# ♫ HNSW (graph ANN) & the `hnsw_rs` crate

- **Paper:** Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs," IEEE TPAMI 2018.
  arXiv:1603.09320.
- **Reference impl:** github.com/nmslib/hnswlib (C++, the de-facto baseline).
- **Rust crate we use:** `hnsw_rs` (github.com/jean-pierreBoth/hnswlib-rs),
  v0.3.4 — parallel insert, `DistL2`/`DistCosine`, tunable `M` / `ef_construction`
  / `ef_search`. Pulled in `database/Cargo.toml` for the bake-off only.
- **Type:** graph-based ANN index — sub-linear candidate visits via a layered
  navigable small-world graph.

## What it is

Multi-layer proximity graph; search greedily descends layers, then does a
best-first walk on layer 0 with a candidate set of size `ef`. Recall ↔ latency is
tuned by `ef_search`; graph quality/memory by `M` (neighbors per node) and
`ef_construction`. Sub-linear in N, but each visit is a full-precision distance and
traversal is **random-access** (cache-hostile), and results are **approximate**
(recall plateaus below 1.0).

## Why tracked / how it relates to this engine

HNSW is the standard ANN index and the usual **above-us** layer (the coarse
router over all N), which is out of scope per [[history 042]]. We benchmarked it as
a *within-cell* competitor to the binary funnel in **history 043**: at high
dimension (1024-D embeddings) the funnel wins on recall/latency/QPS for all
realistic cell sizes and costs ~100–160× less to build; HNSW-in-cell only pays at
low-D (≤128) large cells — i.e. when it should just *be* the index. So HNSW lives
above the cell, not inside it. Evolving OSS (active crate; hnswlib is the moving
baseline other ANN libs benchmark against).

## Related

- [[002-rabitq]], [[003-pdx-adsampling]] — the within-cell quantization / pruning
  frontier the funnel is built on.
- ANN-Benchmarks ([[001-ann-benchmarks]]) — where HNSW variants are leaderboard-tracked.
