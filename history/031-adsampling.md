# 031 — ADSampling: faster exact scan via early-terminated distances

Perf record: [`031-adsampling.json`](./031-adsampling.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant adsampling --eps0 E --delta D`.

## Research → idea

**ADSampling** (Gao & Long, SIGMOD 2023, arXiv:2303.09855) — same group as RaBitQ.
The insight: most candidates in a scan are obviously far, yet naive search pays the
full D-dim distance for every one. ADSampling computes the squared distance
**incrementally** over a random-rotated vector and **stops early** as soon as the
partial distance proves the candidate can't reach the current k-th best.

Concretely (their code, faithfully ported): rotate base+query (so a partial sum
over the first `i` dims is an unbiased estimate of the full distance — JL),
accumulate in batches of `delta=32`, and after each batch prune if

  res  ≥  threshold · ratio(D, i),   ratio(D,i) = (i/D)·(1 + eps0/√i)²

with `eps0=2.1`. Survivors run to full D and, because the rotation preserves L2,
get the **exact** distance — so results match exact KNN up to a tiny, tunable
probabilistic slack. Applied to the **exact within-cell f32 scan**, which is this
project's stated core scope.

## Result — 2.16× faster at near-exact recall; eps0 is the dial

| method | eps0 | recall@10 | QPS | p50 | speedup |
|---|---|---|---|---|---|
| exact f32 (naive) | — | 1.0000 | 9.2 | 707 ms | 1.00× |
| ADSampling | 3.0 | 1.0000 | 13.2 | 588 ms | **1.43× (lossless)** |
| ADSampling | 2.1 | 0.9990 | 19.9 | 438 ms | **2.16×** |
| ADSampling | 1.5 | 0.9847 | 26.4 | 314 ms | 2.87× |
| ADSampling (delta=16) | 2.1 | 0.9983 | 20.6 | 413 ms | 2.24× |

## Conclusions

1. **It's a real win for exact search:** 2.16× faster at recall 0.999, and 1.43×
   while staying **bit-exact** (eps0=3.0, recall 1.0). `eps0` cleanly trades recall
   for speed; `delta` barely matters (16 ≈ 32).
2. **Below the theoretical ceiling, for understood reasons.** Pruning after ~32–64
   of 1024 dims suggests ~16–30× fewer flops, but we see ~2×. The exact scan is
   bandwidth-bound: ADSampling does read fewer bytes per pruned row (it stops
   mid-row), but it still touches every row's first cache lines, and the per-batch
   branch + heap overhead and the loss of long sequential streaming eat into it.
   It converts a *compute* saving that the memory system only partly rewards.
3. **Where it sits on the Pareto:** ADSampling exact (20 QPS, recall 0.999) does
   **not** beat the binary+rerank funnel on throughput (851 QPS) — but it reaches
   recall the funnel can't (exact / 0.999+ vs binary's ~0.997 ceiling). It's the
   **high-recall end** of the frontier: when you need ~exact results, it's 2.16×
   cheaper than naive brute force.

## Where it could go next

- **ADSampling as the rerank tier:** apply the DCO to the binary funnel's C f32
  rescores (prune candidates mid-rescore) — could cut rerank cost at high C.
- **Compose with tiling:** the bandwidth-amortizing tile (016) plus early
  termination might recover more of the theoretical speedup.

## Caveats

- Needs a random rotation (power-of-two dim for our FWHT); 3 rounds used.
- 300 queries; exact baseline reads recall 1.0 on this slice (canonical 0.9994).
- Implemented as `search::knn_adsampling`; correctness test
  (`adsampling_conservative_matches_exact`) checks it equals exact KNN when eps0 is
  large enough to disable pruning.
