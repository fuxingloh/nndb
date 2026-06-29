# 015 — int8 rerank tier: a memory lever, not a speed lever

Perf record: [`015-int8-rerank-tier.json`](./015-int8-rerank-tier.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --rerank-quant f32|i8`.

## The idea

The funnel reranks the top-C candidates by rescoring them from the full **f32**
store (3.9 GB) via random gathers. An int8 store is 4× smaller (977 MB) and 4×
less traffic on that gather, so the hypothesis was: cheaper rerank → more QPS, less
memory. `--rerank-quant i8` reranks by int8 dot instead of f32 L2.

## Result — no speed win, a recall cost, only memory improves

| C | rerank | recall@10 | QPS | p50 |
|---|---|---|---|---|
| 500  | f32 | 0.9826 | 431 | 15.22 ms |
| 500  | i8  | 0.9686 | 429 | 15.19 ms |
| 1000 | f32 | **0.9943** | 420 | 15.92 ms |
| 1000 | i8  | 0.9783 | 420 | 15.72 ms |
| 2000 | f32 | 0.9986 | 398 | 17.01 ms |
| 2000 | i8  | 0.9819 | 395 | 16.44 ms |

## Conclusions

1. **No QPS gain — because rerank was never the bottleneck.** QPS is within noise
   of f32 at every C. 014 already showed the *scan* dominates and is bandwidth-
   bound; the rerank tier (C random gathers) is a small slice of the per-query
   cost, so making it cheaper barely moves the total. This is a useful negative
   result: it rules out the rerank tier as a throughput lever and points all
   remaining speed work back at the scan.
2. **It costs ~1.5 recall points.** int8 rescoring reorders the final top-k less
   accurately than exact f32 (0.9943 → 0.9783 at C=1000). To claw that back you'd
   raise C, which costs QPS — so on the recall/QPS plane i8 rerank is dominated.
3. **The one real benefit is memory.** The rerank store shrinks 3.9 GB → 977 MB
   (4×). Combined with the 122 MB binary scan store, total resident drops
   substantially. So i8 rerank is worth it *only* under memory pressure, accepting
   the recall hit (or recovering it with a larger C and eating the QPS).

## Caveats

- Default is `--rerank-quant f32` (exact); i8 is opt-in.
- int8 here is a single global-scale symmetric quantizer; per-vector or per-dim
  scales would narrow the recall gap, at more store overhead.
