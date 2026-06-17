# 036 — Tiling + ADSampling rerank: not a Pareto win

Perf record: [`036-tiling-plus-adsampling-rerank.json`](./036-tiling-plus-adsampling-rerank.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binads --batch 16` vs `--quant binary --rotate 2 --batch 16`.

## The hypothesis

Tiling (016) and ADSampling rerank (034) are orthogonal stages, so stacking them
should lift QPS at no recall cost — the clean Pareto improvement the goal asks for.
Built `knn_binary_funnel_tiled_ads` (tiled scan + ADSampling-pruned rerank) and
compared to tiled + plain rerank, both rotated, at high C.

## Result — recall held, but QPS drops 5–8%

| C | rerank | recall@10 | QPS | p50 |
|---|---|---|---|---|
| 2000 | plain | 0.9990 | **735** | 12.49 ms |
| 2000 | ADSampling eps0=3.0 | 0.9990 | 687 | 12.46 ms |
| 4000 | plain | 0.9998 | **579** | 15.69 ms |
| 4000 | ADSampling eps0=3.0 | 0.9998 | 532 | 14.58 ms |

Recall is identical and latency is marginally better, but **batch QPS is lower** —
so it sacrifices QPS. Not a Pareto win.

## Why it flipped (vs 034, where ADS rerank won untiled)

The 023 lesson, one layer up. ADSampling rerank only helped in 034 because that was
*untiled* — the scan still dominated, so a slightly-slower-but-pruning rerank was
hidden. Once **tiling** makes the scan cheap, the rerank becomes the dominant cost,
and now the rerank's *own efficiency* matters: plain `l2_sq` is a tight,
fully-autovectorized FMA loop (006), while ADSampling reranks in 32-dim batches with
an early-exit branch that **defeats that vectorization**. The flops saved by pruning
don't repay the lost SIMD throughput. Same trap as the register-tiled kernel (023).

## The redirect

This rules out *fewer flops* as the rerank lever and pinpoints the real one: in the
tiled funnel the rerank is **random-gather bandwidth-bound** — C cache-cold 4 KB f32
rows per query. To improve QPS losslessly, cut the gather *bytes*, not the flops →
a narrower rerank store (bf16). Tested in 037.

## Caveats

- `eps0=2.5` recovers some (705 @ C=2000) but still below plain; lossless throughout.
- tile=16, reps=4, CV < 0.5%.
