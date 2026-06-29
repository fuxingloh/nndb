# 033 — ADSampling on PDX: pruning speedup restored

Perf record: [`033-pdx-adsampling.json`](./033-pdx-adsampling.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant pdxads --eps0 E --block B`.

## The idea

031 showed dimension-pruning (ADSampling) under-delivered on the row-major layout
(2.16×, not the theoretical 16–30×). The PDX paper's central claim is that the
*layout* is why — and that on the vertical layout pruning is restored to 2–7×. So
combine them: run ADSampling pruning ON the PDX block layout. Per block, accumulate
partial distances dimension-by-dimension over the *alive* vectors; after each batch
drop the ones that can't beat the k-th best, so they skip their remaining dims.

## Result — 4× over naive exact, ~2× over horizontal ADSampling

| method | recall@10 | QPS | p50 |
|---|---|---|---|
| naive exact (031 baseline) | 1.0000 | 9.3 | 715 ms |
| horizontal ADSampling eps0=2.1 (031) | 0.9990 | 19.9 | 416 ms |
| PDX plain (032) | 1.0000 | 17.9 | 222 ms |
| **PDX+ADSampling eps0=2.1** | 0.9990 | **37.4** | 219 ms |
| PDX+ADSampling eps0=3.0 | 1.0000 | 23.1 | 387 ms |
| PDX+ADSampling eps0=1.5 | 0.9850 | 54.8 | 135 ms |

## Conclusions

1. **Layout + pruning compound.** PDX+ADSampling at recall 0.999 is **37.4 QPS — 4×
   the naive exact scan and 1.9× horizontal ADSampling** (031). Lossless (eps0=3.0)
   is 23.1 QPS (2.5×). The PDX paper's "restored to 2–7×" reproduces: pruning needs
   the vertical layout to keep the alive-vector inner loop vectorized while
   skipping pruned vectors' tail dims.
2. **eps0 dials it:** 0.999 @ 37 QPS, or 0.985 @ 55 QPS — the high-recall accelerator
   with a recall knob.

## The honest ceiling (and the pivot)

Even at 4×, this is **~37 QPS — still ~23× below the binary+rerank funnel (851
QPS)**. It's the *exact* tier: it wins where you need recall ≥0.999 that binary
can't reach, but it does not, and will not, beat binarization on throughput.

That's the right read of 031–033: standalone exact-scan accelerators are a
different Pareto region from the binary funnel. The value-add is to **stack these
research techniques onto the binary funnel**, not run them beside it — apply
ADSampling pruning to the funnel's f32 rerank tier (034), where it composes with
the 851-QPS winner instead of competing with it.

## Caveats

- Needs a random rotation (built from the rotated base); 3 rounds.
- 300 queries; recall on this slice; block 64 vs 128 ≈ equal.
