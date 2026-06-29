# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A greenfield experiment in **in-memory** top-K vector search: load all vectors into RAM (not disk-bound) and search there, measured the ANN-Benchmarks way (recall/QPS/latency). `nndb/` is the Rust engine (crate `nndb`); `website/` is the writeup — a Next.js + MDX learning article (Lab + Prose, Ayu-dark) at `website/app/page.mdx`, which also **serves every `notes/` entry as a readable page at `/notes/<slug>`** (dynamic markdown route reading `notes/` directly via `website/lib/notes.ts`). `notes/` stays the source of truth — the measure scripts write there and the site reads from there; don't move it.

**Scope (important):** this engine models the efficient *exact* search **inside a single IVF cell/shard** — the coarse quantizer that routes a query to a cell lives at a layer above us. So the goal is making the within-cell full scan as fast as possible (SIMD, memory layout, cache). Approximate-index layers (IVF router, HNSW) and quantization were investigated and closed out as numbered entries (042–044, 054–056). Directions deliberately *not* built are parked as **♫ notes** in `notes/` (e.g. `note-accelerator-gemm.md` — search-as-GEMM on a GPU/ASIC), not implemented here.

## Commands

All Rust work happens in `nndb/`:

```bash
cd nndb
bash scripts/download-sift.sh                    # fetch SIFT1M into data/sift/ (~168MB dl, 488MB RAM)
cargo build --release                            # release build is mandatory for any real numbers
cargo test --release                             # run unit tests
cargo test --release returns_k_nearest_ascending # run a single test by name
cargo run --release -- --queries 1000 --k 10     # in-process benchmark (the vsearch bin)
cargo run --release -- --queries 0               # benchmark on all 10k queries

# serving path (two extra bins; see Serving below)
cargo run --release --bin server                 # start the HTTP search server
cargo run --release --bin loadtest -- --concurrency 8 --requests 1000
```

The dataset (`nndb/data/`) is gitignored — `download-sift.sh` must be run before anything.

## Notes / measurement

The `notes/` folder holds two kinds of entry, both served on the site at `/notes/<slug>`:

**Numbered experiments** — each improvement is a numbered pair in `notes/`: `NNN-<descriptive-title>.md` (narrative + conclusions) and `NNN-<descriptive-title>.json` (perf data) — e.g. `001-exact-brute-force-baseline.md`, `002-networked-serving.md`. The title should summarize the index/milestone, not say "what we did". Two generators, both stamp date + git commit:
- `notes/measure.sh <out.json> <label>` — in-process algorithm numbers (recall, QPS, latency, memory).
- `notes/measure-serving.sh <out.json> <label>` — starts the server and runs a concurrency sweep for user-facing latency.

When you make an improvement, add the next numbered entry — don't overwrite old ones; the point is the trend. Entries are **retrospective only**: record what was done and what the numbers show. Do not add "Next"/roadmap sections — forward plans go stale. New `notes/*.md` entries appear on the site automatically (the `/notes` route lists everything in `notes/`); keep the first line a single `# NNN — Title` H1 so the slug/title render correctly.

**♫ notes** — non-measured entries: `note-<slug>.md` with a `# ♫ Title` H1. Two uses: (a) **external references** worth tracking (benchmark leaderboards, library docs, papers, OSS projects, SIMD/compiler trackers) — things *external, evolving, and trackable*, NOT universal/timeless concepts (don't write a note explaining what SIMD or GEMM *is*); (b) **parked directions** deliberately not built (e.g. `note-accelerator-gemm.md`). The ♫ in the H1 is the marker the site uses to render them as notes (not experiments) and the cross-reference signal — keep it.

## Architecture

The benchmark mirrors the ANN-Benchmarks contract: a `base` vector set, a `query` set, and ground-truth nearest neighbors per query so recall is measurable. Data flows `fvecs` (load) → `search` (rank) → `eval` (score), orchestrated by `main`.

- **`src/fvecs.rs`** — readers for `.fvecs` (f32) / `.ivecs` (i32), the SIFT1M binary format: flat records of `[i32 dim][dim × value]`, little-endian, dim constant across the file. Vectors are stored row-major in a flat `Vec`; access rows via `.row(i)`.
- **`src/search.rs`** — `knn_batch` is the interface every index implements. Exact brute-force today: squared-L2 (sqrt is monotonic, skip it), a bounded max-heap of size k (O(n log k), not a full sort), parallelized across queries with rayon. **New approximate indexes should slot behind this same `knn_batch(base, queries, k) -> Vec<Vec<u32>>` shape** so the existing harness measures them unchanged.
- **`src/eval.rs`** — `recall_at_k`: mean over queries of |returned ∩ true top-k| / k.
- **`src/main.rs`** — loads, runs, reports recall@k + QPS + per-query scan cost.

## Serving

`server.rs` holds the vectors in `Arc<AppState>` and serves `POST /search`. The serving model is deliberate: one request = one **single-threaded** search; a `Semaphore(cores)` bounds in-flight searches so excess load queues (modeling a CPU-bound service). Do **not** parallelize a single query with rayon inside the handler — that tanks throughput under concurrency. Throughput ceilings at ~cores/compute-time (~100 QPS here); beyond concurrency = cores, only latency grows (queuing). The interface (HTTP/JSON) adds <0.4ms — it is not the bottleneck; compute is.

## Things that will trip you up

- **Exact search reports recall@10 ≈ 0.9994, not 1.0 — this is correct, not a bug.** SIFT descriptors are uint8 so L2 distances are exact integers in f32 (no float error); the ~6/10000 "misses" are boundary ties where the k-th and (k+1)-th neighbors are equidistant and ground truth picks a different (equally valid) point. Don't "fix" this.
- **`--queries N` truncates queries but not ground truth.** `recall_at_k` therefore asserts `found.len() <= truth.len()`, not `==`. Keep that asymmetry when changing eval.
- Always benchmark with `--release`; debug builds give meaningless (orders-of-magnitude slower) numbers because the inner distance loop relies on autovectorization (`opt-level=3`, `lto`, `codegen-units=1` in Cargo.toml).
- The "single-thread-equivalent ms/query" line (~100ms scanning 1M vectors) is the whole motivation for approximate indexes — it's the number new indexes must beat while holding recall.
