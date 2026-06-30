# NNDB — Nearest-Neighbor DB

An in-memory vector-search engine, built from scratch to learn where the time
actually goes. Every vector lives in RAM and is searched there — not disk-bound —
and every change is a numbered, measured experiment. The whole point is the trend
line: pushing **recall · QPS · p50 · p99** as far as first principles allow.

This is a personal exploration, not a product. I learn by building and measuring,
so the repo *is* the lab notebook: ~60 numbered experiments, each with the numbers
that justified (or killed) it.

## What & why

Modern vector databases route a query to a cluster (IVF) and then scan the vectors
*inside* that cell. NNDB is that **within-cell scan**, made as fast as the hardware
allows — SIMD, memory layout, cache, quantization. The coarse router that picks the
cell lives a layer above and is deliberately out of scope (see *Scope & honesty*).

The engine that came out best is a **binary-quantization funnel**: keep one sign
bit per dimension (32× smaller), scan all N with a Hamming/popcount kernel to get a
few-hundred shortlist, then re-rank only those against the real vectors. It buys
roughly two orders of magnitude in throughput over an f32 brute-force scan while
holding recall near the exact baseline. The live numbers, per machine, are in the
writeup and the notes — not pinned here, because they keep moving.

→ **Writeup:** <https://nndb.fuxing.dev> · **Full trail:** [`/notes`](https://nndb.fuxing.dev/notes)

## Layout

```
nndb/       the Rust engine (crate `nndb`) — fvecs/search/quant/eval + serving bins
website/    Next.js + MDX writeup; also serves every notes/ entry at /notes/<slug>
notes/      the source of truth: numbered experiments (NNN-*.md + .json) and
            ♫ notes (note-*.md — parked directions & external references)
infra/      CDK for ephemeral spot boxes used to benchmark across CPUs
```

`notes/` is read directly by the site and written by the measure scripts — it stays
put. Numbered entries are measured experiments; **♫** entries are documented notes,
not measurements.

## Quickstart

```bash
cd nndb
bash scripts/download-sift.sh                 # fetch SIFT1M into data/ (~168 MB)
cargo build --release                         # release is mandatory for real numbers
cargo run --release -- --queries 1000 --k 10  # in-process benchmark
cargo test --release                          # unit tests
```

See [`nndb/README.md`](nndb/README.md) for the serving path and the Cohere dataset.

## Scope & honesty

- **Within-cell scan only.** This is the per-cell exact/funnel search, not a full
  ANN system — no IVF router, no HNSW graph over all N (both were measured and
  parked as notes). It's the layer LanceDB/turbopuffer also build under their
  cluster index.
- **Single-node, CPU-bound serving model.** One request = one single-threaded
  search, bounded by a semaphore; throughput ceilings at ~cores. Cluster-scale and
  accelerator (GPU/FPGA) directions are ♫ notes, not built.
- **The dead ends are kept on purpose.** Plenty of experiments are negative results
  (scalar PQ, HNSW-in-cell, register-tiling, prefetch). They're in `/notes` because
  the failures are the lesson.

## License

[MIT](LICENSE) © 2026 Fuxing Loh
