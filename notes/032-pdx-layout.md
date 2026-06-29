# 032 — PDX vertical layout: 2× faster exact scan, free

Perf record: [`032-pdx-layout.json`](./032-pdx-layout.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant pdx --block B`.

## Research → idea

**PDX** (Kuffo, Krippner & Boncz, SIGMOD 2025, arXiv:2503.04422) changes the *data
layout*, not the algorithm. Instead of storing each vector's dims contiguously
(horizontal/row-major), it groups vectors into blocks and stores each block
**dimension-major** (transposed): all vectors' dim 0, then all vectors' dim 1, …

Computing distances for a block becomes a loop over dimensions whose **inner loop
is over vectors** — "multiple-vectors-at-a-time." That inner loop is branch-free
and autovectorizes cleanly, and crucially it needs **no per-vector horizontal
reduction** (the partial sums accumulate into a `partial[v]` array across the dim
loop). The paper reports ~40% over SIMD horizontal *and* that it restores the
benefit of pruning algorithms (next entry). Training-free, pure scalar code that
the compiler vectorizes — no intrinsics.

## How it pushes the boundary

Our exact f32 scan was the slowest path (9 QPS), and 031 showed pruning on the
*horizontal* layout under-delivered. PDX attacks the layout itself: same bytes
read, but arranged so the compute vectorizes far better and pruning (033) can read
only the dims it needs across many vectors at once.

## Result — ~2× throughput, ~3× lower latency, recall identical

| layout | recall@10 | QPS | p50 |
|---|---|---|---|
| horizontal exact (`knn_batch`) | 1.0000 | 9.3 | 715 ms |
| horizontal tiled (batch=16) | 1.0000 | 14.9 | 707 ms |
| **PDX (block=64)** | 1.0000 | **18.4** | **233 ms** |

Block size is insensitive (32→256 all ≈ 18 QPS).

## Conclusions

1. **PDX ~doubles exact-scan throughput (9.3 → 18.4) and cuts single-query latency
   ~3× (715 → 233 ms)** at bit-identical recall — for free, by transposing the
   block layout. It even beats our own bandwidth-tuned tiled horizontal (14.9).
2. **The latency win (3×) exceeds the throughput win (2×)** for the now-familiar
   reason: single-threaded, PDX's vectorized no-reduction kernel is much faster;
   but the 8-core batch re-saturates memory bandwidth (PDX still reads the whole
   base), so the aggregate gain compresses. Same compute-vs-bandwidth tension as
   012–025, seen from the layout side.
3. **It's the substrate for fast pruning.** The real PDX payoff is that
   dimension-pruning (ADSampling) on this layout reads only the dims it needs
   across a whole block at once — tested in 033.

## Caveats

- Exact (recall 1.0); still far below the binary funnel's QPS (851) but that's the
  approximate tier — PDX is the *exact* accelerator.
- We gained ~2× vs the paper's ~40%; our horizontal `l2_sq` baseline is plain
  autovectorized (not a hand-tuned AVX-512 kernel), so PDX has more to gain here.
- Build transposes the base (one-time, ~3.9 GB copy).
