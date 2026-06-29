# 055 — ITQ: a learned rotation beats random, but only by ~1.5 pts

Perf record: [`055-itq-learned-rotation.json`](./055-itq-learned-rotation.json). Granite
box (8 vCPU). `src/bin/itq.rs`. Cohere 1M × 1024, recall@10 vs exact GT (binary funnel +
exact rerank). 50k training sample, 30 ITQ iterations.

## What ITQ is

Iterative Quantization (Gong & Lazebnik 2011): instead of a *random* rotation before
sign-binarizing, *learn* an orthogonal rotation R that minimizes the binarization error
‖sign(VR) − VR‖. Fit by alternating `B = sign(VR)` and the orthogonal-Procrustes update
`R = U·Wᵀ` from `SVD(Vᵀ·B)`. We test at b=256/512 bits; the b-dim projection is the
first b dims of our FWHT-rotated vector, and ITQ learns a b×b rotation on top of it
(baseline = R=identity = the current random-rotation codes).

## Result — consistent but small

| bits | C | random | ITQ | Δ |
|---|---|---|---|---|
| 256 | 200 | 0.5003 | 0.5190 | **+1.9** |
| 256 | 1000 | 0.7151 | 0.7344 | **+1.9** |
| 512 | 200 | 0.7846 | 0.7998 | +1.5 |
| 512 | 1000 | 0.9286 | 0.9403 | +1.2 |

ITQ beats random rotation everywhere, by **+1.2 to +1.9 pts**.

## The catch: dense rotation

ITQ's R is **dense (b×b)**, so applying it is O(b²)/vector — vs the FWHT random
rotation's O(b log b). At 256 bits that's ~32× more rotation work (still a small absolute
per-query cost, ~1–2% of the scan), plus an offline SVD fit. So it's not quite "free"
like the FWHT rotation.

## Conclusion

A learned rotation is a **real but modest** recall gain (~1.5 pts) over random. For
context, **residual (046) gives +3 to +9 pts** on the same recall axis at zero apply
cost — a much bigger lever. So ITQ is worth it only when squeezing the last point and you
can afford the dense rotation; otherwise random-rotation + residual already captures most
of what a learned rotation would.

## Caveats

- b-dim projection is the FWHT-rotated prefix; classic ITQ uses PCA — PCA-init might lift
  ITQ a little more, untested.
- 50k sample / 30 iters; more could help marginally. The verdict (small gain) is robust.
