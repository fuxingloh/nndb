# 010 — Asymmetric scoring (full-precision query × binary doc)

Perf record: [`010-asymmetric.json`](./010-asymmetric.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant asym`.

## What we did

Symmetric binary (009) binarizes *both* sides → Hamming. **Asymmetric keeps the query in full precision** and scores it against the docs' ±1 sign bits: `score = Σ qᵢ·signᵢ = 2·(Σ qᵢ over set bits) − Σq`, so ranking is just the masked query-sum. Docs stay 1 bit (122 MB); the query keeps its magnitude. Direct kernel (iterate set bits + gather) — **no `vpshufb` LUT yet**.

## Result — recall up at every C, but the kernel is slow

| rerank C | symmetric recall | **asym recall** |
|---|---|---|
| none | 0.466 | **0.607** |
| 20 | 0.630 | 0.799 |
| 50 | 0.798 | **0.935** |
| 100 | 0.887 | **0.976** |
| 200 | 0.942 | **0.993** |
| **scan QPS** | **690** | **6.7** |

## Conclusions

1. **Asymmetric lifts stage-1 recall substantially — confirmed.** Keeping the query's real values (not just its sign) makes the coarse ranking much better at every C. Asym reaches **0.993 at C=200**; symmetric needed **C=1000** for 0.994 → **~5× smaller rerank C** for the same final quality. This is the design rationale behind Exa's asymmetric scoring (`stories.md` §L59).

2. **But the naive asym kernel is ~100× slower than Hamming (6.7 vs 690 QPS).** Symmetric is 16 popcounts/doc; asym is ~512 gathers/doc (iterate set bits, index into the float query). So *as implemented*, asym is **not** a Pareto win — symmetric+rerank (0.994 @ 585 QPS, 009) still beats it on speed.

3. **The `vpshufb`/`vpermb` LUT is exactly what closes this gap.** It computes the *same* asymmetric score via precomputed 4-bit tables at popcount-like speed (`concepts.md` §L218) — that's *why* Exa uses lookup tables for asymmetric rather than a direct loop. With the LUT, asym would keep the recall advantage (smaller C) at symmetric-like throughput → then it dominates.

## Where this leaves the funnel

- **Best viable today:** symmetric binary + rerank C=1000 (009) — 0.994 recall, 585 QPS, 122 MB.
- **Asymmetric's promise:** same recall at ~5× smaller C, *if* the LUT brings the scan speed up. Recall half proven here; speed half is the LUT (next).
- At billion scale the smaller-C win compounds (less rerank = less of the expensive f32 tier touched), so the LUT is worth it there; at our scale it's marginal vs symmetric+rerank.

## Caveats

- Direct asym kernel (no LUT) → speed is a lower bound; the LUT is the real implementation.
- Cohere v3 is compression-aware (favorable to binary in general).
