# 043 — Binary funnel vs HNSW *inside one cell*: dimensionality decides

Perf record: [`043-hnsw-in-cell-crossover.json`](./043-hnsw-in-cell-crossover.json).
Granite box (Xeon 6975P-C, 8 vCPU). `src/bin/crossover.rs`. Cohere v3 (1024-D,
cosine) and SIFT1M (128-D, L2). HNSW = the `hnsw_rs` crate (real, parallel,
DistL2, M=16, ef_construction=200) — a fair fight, not a strawman.

## The question (from `042`)

If a node *is* one IVF cell, should the within-cell search be our rotated-binary
funnel (scan **all** N at popcount speed → rerank) or an **HNSW graph built over
the cell** (visit ~O(log N) candidates, but random-access + approximate)? They're
**competitors** for the same job. So: measure recall / latency / QPS for both,
sweep cell size N, and find where the winner flips. Each method runs at *its* best
— funnel QPS uses tiling (its real throughput edge), HNSW QPS uses per-query
parallelism (it can't tile a shared scan).

## Result 1 — at 1024-D (Cohere), the funnel wins everywhere tested

Latency at a matched ~0.975–0.98 recall (single query, p50):

| cell N | funnel | HNSW | funnel speedup |
|---|---|---|---|
| 1,000 | 0.983 @ **45 µs** | 0.988 @ 617 µs | ~14× |
| 5,000 | 0.993 @ **117 µs** | 0.989 @ 1353 µs | ~11× |
| 20,000 | 0.986 @ **221 µs** | 0.984 @ 2022 µs | ~9× |
| 50,000 | 0.979 @ **411 µs** | 0.976 @ 2683 µs | ~6.5× |
| 100,000 | 0.975 @ **657 µs** | 0.975 @ 3112 µs | ~4.7× |

The funnel also wins QPS at high recall (e.g. N=100k: 0.995 @ 6209 QPS vs HNSW
0.975 @ 1946). HNSW's *only* wins at 1024-D show up at N=100k in the **low**-recall
corner (ef=10: 0.728 @ 432 µs / 14.8k QPS beats the funnel's cheapest point) — its
sub-linear scaling finally starts paying, but only where recall is too low to want.
Note the funnel's lead **shrinks with N** (14× → 4.7×): HNSW's asymptotic advantage
is real, it just kicks in beyond typical cell sizes at this dimension.

## Result 2 — at 128-D (SIFT), HNSW wins above N ≈ 5–10k

Cheap low-D distances flip it. Latency at matched recall (p50):

| cell N | funnel | HNSW | winner |
|---|---|---|---|
| 1,000 | 0.980 @ 30 µs | 0.987 @ 43 µs | ~tie (funnel QPS edge) |
| 5,000 | 0.972 @ 100 µs | 0.975 @ 95 µs | ~tie |
| 20,000 | 0.903 @ 181 µs* | 0.967 @ **150 µs** | **HNSW** |
| 50,000 | 0.853 @ 257 µs* | 0.962 @ **217 µs** | **HNSW** |
| 100,000 | 0.797 @ 404 µs* | 0.980 @ **557 µs** / ef10 0.800 @ **122 µs** | **HNSW** |

\* the funnel can't even reach 0.97 within the swept C at large low-D N. At N=100k
SIFT, HNSW is ~3× lower latency and ~5× higher QPS at equal recall.

## Result 3 — build & memory: HNSW-in-cell is expensive to *maintain*

A per-cell index must be built and held. At N=100k:

| | Cohere 1024-D | SIFT 128-D |
|---|---|---|
| funnel build / mem | **0.33 s** / ~13 MB codes | **0.04 s** / ~1.6 MB |
| HNSW build / mem | 53.2 s / 678 MB | 11.4 s / 180 MB |

HNSW build is **100–160× slower** and **80–130× more memory** (it stores a full
f32 copy *plus* the graph; the funnel adds only 1-bit-per-dim codes on top of the
vectors it already reranks against). (Funnel mem is the analytical code size;
process-RSS deltas under-resolve it. HNSW mem is measured RSS delta.)

## Conclusions

1. **Dimensionality is the deciding variable.** High-D (modern embeddings —
   768/1024/1536) → the funnel dominates on all three axes for every realistic cell
   size, because each HNSW visit pays a full high-D distance while the funnel's
   compare is dim/64 words. Low-D (≤128) → HNSW's cheap distances + sub-linear
   visits win above a few thousand vectors.
2. **N moves the crossover, recall target gates it.** The funnel is
   exact-equivalent — recall → 1.0 by widening C — so it owns the high-recall region
   (0.99+) outright at high D; HNSW plateaus ~0.98 (graph-approximate). Within a
   cell you *want* high recall (IVF already spent its recall budget at routing), so
   the plateau matters.
3. **Verdict for this engine: keep the binary funnel as the within-cell engine.**
   Our regime is high-D embeddings, where the funnel wins outright and costs
   ~100–160× less to build with ~100× less memory. HNSW-in-cell only pays for
   **low-D, large** cells — and a cell that large is really just "make HNSW the
   index", i.e. the coarse/router layer *above* us (out of scope per `042`). So
   HNSW belongs above the cell, not inside it. Confirmed by measurement, not
   assertion.

## Caveats

- HNSW tuned to common defaults (M=16, ef_construction=200); higher M would raise
  recall/latency and memory further — it doesn't change the high-D verdict (HNSW is
  already both slower and higher-recall-capped there). DistL2 == cosine rank for
  unit-norm Cohere. Single-box (8 vCPU); QPS is the best of 3 reps.
- "Cell = first N base vectors" (a contiguous slice, not a true IVF cluster). Real
  cells are k-means clusters (tighter, possibly easier for both); this measures the
  engine mechanics, not cluster geometry.
