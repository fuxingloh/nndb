# 034 — ADSampling rerank, stacked on the binary funnel

Perf record: [`034-adsampling-rerank-stack.json`](./034-adsampling-rerank-stack.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binads --rerank C --eps0 E`.

## The pivot

031–033 built ADSampling/PDX as *standalone exact* scans — interesting, but ~37 QPS
max, a different Pareto region that will never beat binarization's 851 QPS. The
right move (per the project's own findings and the steer to add value *on top of*
binary): **stack** the research technique onto the binary funnel rather than run it
beside.

So: keep the fast rotated **binary scan** to get C candidates, then rerank them
with **ADSampling-pruned** exact L2 (on the rotated f32 store) instead of full L2 —
candidates that provably can't enter the top-k stop early. This lets the funnel use
a *larger* C (more recall) without the rerank cost growing linearly.

## Result — cheaper rerank, gain grows with C, near-lossless

Same rotated binary scan; plain exact rerank vs ADSampling rerank:

| C | rerank | recall@10 | QPS | Δ QPS |
|---|---|---|---|---|
| 1000 | plain | 0.9970 | 535 | — |
| 1000 | ADSampling | 0.9962 | 563 | **+5%** |
| 2000 | plain | 0.9992 | 494 | — |
| 2000 | ADSampling | 0.9984 | 524 | **+6%** |
| 4000 | plain | 0.9999 | 421 | — |
| 4000 | ADSampling | 0.9991 | 463 | **+10%** |

eps0=1.5 pushes further (+14% @ C=4000) at recall ~0.99.

## Conclusions

1. **It adds value where rerank matters.** The speedup grows with C (+5% → +10%)
   because rerank's share of the per-query cost grows with C, and ADSampling prunes
   most of it. At eps0=2.1 it's near-lossless (recall −0.001). So the *high-recall*
   funnel (C=2000–4000, recall 0.999+) gets 6–10% cheaper for free.
2. **This is the correct shape of a research win here:** stacked on binary, not
   competing with it. Standalone the same technique gave ~37 QPS (033); attached to
   the funnel it lifts an 850-class pipeline at its expensive end.
3. **Bounded by the same truth as everything since 016:** the *scan* still
   dominates at low C, so ADSampling rerank does little there (+5%); it only pays
   once you over-retrieve. Tiling (016) remains the bigger lever; this composes
   with it.

## Caveats

- Untiled here to compare scan-matched plain vs ADSampling rerank cleanly; combined
  with tiling the absolute QPS rises and the rerank-pruning effect still applies.
- Rerank runs on the rotated f32 store (L2 preserved); eps0 trades recall for prune
  aggression, delta=32.
