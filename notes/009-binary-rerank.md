# 009 — Binary quantization + two-stage rerank (the Exa funnel)

Perf record: [`009-binary-rerank.json`](./009-binary-rerank.json).
Cohere v3 Wikipedia, 1,000,000 × 1024-d, cosine (unit-norm → L2/dot), measured on the **Granite Rapids** box (memory-bound).

## What we did

Binary quantization (1 bit/dim = sign), Hamming distance via XOR + popcount, then the two-stage funnel: **binary scan → top-C candidates → exact f32 rerank → top-k**. Swept C to get the recall curve. This is Exa's middle layer (`concepts.md` §L218, §L232).

## The full curve

| scheme | recall@10 | QPS | p50 latency | memory |
|---|---|---|---|---|
| f32 (exact) | 1.000 | 9.5 | 644 ms | 3906 MB |
| int8 (008) | 0.983 | 19.8 | 589 ms | 977 MB |
| **binary, no rerank** | **0.468** | 613 | 10 ms | **122 MB** |
| binary + rerank C=20 | 0.630 | 583 | 8 ms | 122 MB |
| binary + rerank C=50 | 0.798 | 573 | 9 ms | 122 MB |
| binary + rerank C=100 | 0.887 | 615 | 8 ms | 122 MB |
| binary + rerank C=200 | 0.942 | 610 | 9 ms | 122 MB |
| binary + rerank C=500 | 0.983 | 598 | 10 ms | 122 MB |
| **binary + rerank C=1000** | **0.994** | 585 | 10 ms | 122 MB |

## Conclusions

1. **Without rerank, binary is unusable: recall 0.468.** Sign-bit Hamming alone loses *half* the true neighbors — but it's blazing (613 QPS = 64× f32, 122 MB = 32× smaller). So binary alone = fast + tiny + wrong.

2. **Rerank is the whole trick, and it's nearly free.** Recall climbs 0.47 → 0.89 (C=100) → 0.98 (C=500) → 0.99 (C=1000), while **QPS barely moves** (615 → 585 across C=10→1000). Exact-f32 rescoring a few hundred candidates costs almost nothing next to the binary scan over 1M. This is *why* the funnel works: stage-1 just has to get the true neighbors into the top-C pool; stage-2 fixes the order for free.

3. **Binary + rerank Pareto-dominates everything.** At C=1000: **recall 0.994 (≈ f32), 62× the throughput (585 vs 9.5 QPS), 65× lower latency (10 vs 644 ms), 32× less memory (122 MB vs 3.9 GB).** It beats both f32 (quality-tied, vastly cheaper) and int8 (better on every axis). This is the headline result of the whole project: **~exact quality at a fraction of the cost.**

4. **"What's a good C": the knee is ~500–1000.** recall 0.98 at C=500, 0.99 at C=1000; diminishing after. And since **QPS is flat across C** (rerank is cheap), over-retrieving is nearly free — pick C for your recall target and stop worrying about its cost. (Confirms the earlier prediction with a curve instead of a guess.)

## Where we are (001 → 009)

| | recall | QPS | memory | vs f32 |
|---|---|---|---|---|
| f32 exact | 1.000 | 9.5 | 3906 MB | 1× |
| binary + rerank C=1000 | 0.994 | 585 | 122 MB | **62× faster, 32× smaller, ~same recall** |

The arc: exact baseline → diagnosed compute-bound → fixed the kernel (FMA) → flipped to memory-bound → quantization (int8 4×, binary 32×) → rerank recovers quality at no cost. The funnel lands ~exactly where Exa's design says it should.

## Caveats

- **Favorable data:** Cohere v3 is compression-aware, so binary loses less than a naive model would; non-compression-aware embeddings would need larger C.
- **Naive binary kernel** (plain XOR+popcount, no `vpshufb`/`vpermb` LUT trick yet) — already 600+ QPS, but Exa's LUT approach would push further.
- Single memory-bound box; no PMU (virtualized). Recall is honest (vs exact GT).
