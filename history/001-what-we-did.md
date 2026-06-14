# 001 — Exact brute-force baseline

Perf record: [`001-what-we-did.json`](./001-what-we-did.json)

## What we did

Stood up the project and the measurement rig:

- **`database/`** Cargo crate with the search core:
  - `.fvecs`/`.ivecs` reader for SIFT1M (all vectors loaded into RAM).
  - Exact brute-force KNN: squared-L2, bounded size-`k` max-heap (O(n log k)), parallelized across queries with rayon.
  - Eval harness: recall@k against ground truth.
  - Benchmark that reports the three axes we track from here on — **recall**, **throughput (QPS)**, **latency distribution (p50/p95/p99)**, and **memory**.
- **Dataset:** [SIFT1M](http://corpus-texmex.irisa.fr/) — 1,000,000 × 128, L2, with ground-truth top-100. The classic ANN baseline.

## Results

| metric | value |
|---|---|
| recall@10 | 0.9994 |
| QPS (1000 searches, 8 threads) | ~98 |
| latency mean / p50 / p95 / p99 | 48.6 / 48.3 / 50.4 / 51.7 ms |
| memory — index (raw vectors) | 488 MB |
| memory — peak RSS | ~996 MB |

Measured on an 8-thread M-series mac. Throughput is a parallel batch; latency is single-query sequential. See the JSON for the exact run.

## Conclusions

1. **recall@10 = 0.9994 is correct, not an error.** Exact search can't miss a true neighbor. SIFT components are integers in [0,157], so L2² is a sum of squared ints — bit-exact in f32. All 6/10000 missed slots are *exact-distance ties* at the k-th boundary (proven by `cargo run --release --example analyze_misses`), where ground truth recorded a different but equidistant near-duplicate. Effectively 1.0.

2. **Latency is flat and predictable (p99/p50 = 1.07).** Every query does identical work — a full 1M-vector scan — so there's no algorithmic tail. This is a property we *lose* with graph/cluster indexes, whose latency varies with hops/probes; worth remembering when we compare tails later.

3. **Parallel scaling is sublinear: ~4.8× on 8 threads (~60% efficiency).** Single-query is 48.6 ms → ~21 QPS on one core; we get ~98 QPS on eight. The kernel is **memory-bandwidth bound** — each query streams all 488 MB of vectors and 8 cores contend for the same memory bus. Adding cores has diminishing returns; the real win is touching less data (what ANN indexes do).

4. **Memory is dominated by raw vectors (488 MB); peak RSS ~996 MB.** The ~500 MB gap is a loader transient: we hold the raw file bytes (516 MB) and the parsed f32 array (488 MB) simultaneously during load. Fixable later by streaming the parse. Steady-state working set is the 488 MB.

5. **48.6 ms/query is the number to beat.** An ANN index should cut this by 100–1000× while holding recall near 0.999. That speedup-vs-recall tradeoff is the whole point of the next entries.

## Decisions

- This entry measures the **algorithm in-process** — the clean floor, no framework noise. It stays the reference for "is the index itself faster."
- We will **also** measure user-facing latency through a real networked server (the interface layer a cluster actually serves through). That's entry 002 — focused on serving latency under production-like traffic, not startup/recovery.

## Next

Entry 002: in-memory vector-search **server** (HTTP API) + a concurrent load generator measuring user-facing latency/QPS through the network.
