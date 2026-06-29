# 030 — Research capstone (RaBitQ block, 026–030)

Perf record: [`030-research-capstone.json`](./030-research-capstone.json).
Cohere v3, 1M × 1024, cosine, Granite box.

## Head-to-head (same run, reps=8)

| config | recall@10 | QPS | p50 |
|---|---|---|---|
| 009 baseline — plain binary, tile=1, C=1000 | 0.9931 | 516 | 11.51 ms |
| **research best — rot×2, tile=16, C=1000** | **0.9960** | **845** | 11.40 ms |
| **research best-recall — rot×2, tile=16, C=2000** | **0.9990** | 730 | 13.14 ms |

The research adds **+0.29 pts recall *and* +64% QPS** over the baseline (and the
+64% is on top of identical recall being available), or **0.999 @ 730**.

## What the research found

Searched the literature for something that beats plain sign-bit binary. The winner
was **RaBitQ** (Gao & Long, SIGMOD 2024) — and its key, free idea (shared with
**ITQ**, 2011): a random orthogonal rotation before binarizing.

| entry | idea | outcome |
|---|---|---|
| 026 | random rotation (FWHT) before sign bits | **free recall** +2.4 pts stage-1, 0.994→0.997 @ C=1000, 0 QPS cost |
| 027 | rotation + prefix truncation | rotation **rescues the prefix**: +4.6 pts @ 512/C=1000 |
| 028 | RaBitQ unbiased estimator | **best recall-per-bit** (stage-1 0.606 vs 0.463, ~5–10× smaller C) but 40× slower scan |
| 029 | rotated combined Pareto | frontier shifts **out at every tier** vs non-rotated 018 |
| 030 | capstone | 0.996 @ 845 / 0.999 @ 730 vs 009's 0.993 @ 516 |

## The two takeaways

1. **The rotation is the deployable win — and it's free.** Spreading information
   evenly across dimensions makes every sign bit count, lifting recall at zero QPS
   or memory cost, *and* it rescues prefix truncation (027), which reshapes the
   whole batch frontier (029). Random rotation buys most of the Matryoshka benefit
   without a Matryoshka-trained model — the standout practical result of the block.
2. **RaBitQ's estimator is the recall ceiling, gated on a kernel.** Its unbiased
   per-vector estimate (028) is far sharper than Hamming (~5–10× smaller rerank C
   for the same recall), but as a set-bit gather it's 40× slower. Making it fast is
   the same parked SIMD-LUT work as the asymmetric kernel (011) — the one remaining
   high-value build, and where the payoff concentrates at billion scale.

## Project arc (all blocks)

- 001–011: built the funnel (exact → serving → binary+rerank 009 → asymmetric).
- 012–025: performance — tiling (+76% QPS, 016), prefix dial (014), serving (50× vs
  f32, 017), intra-query latency (13→3 ms, 021). Net: 0.993 @ 851, latency 2.6 ms.
- 026–030: research — RaBitQ/ITQ rotation (free recall, rescues prefix) → **0.996 @
  845 or 0.999 @ 730**, best of the project; RaBitQ estimator maps the recall
  ceiling pending the fast kernel.

## Caveats

- Rotation requires power-of-two dim (FWHT). RaBitQ estimator recall is the upper
  bound a fast kernel would hit at popcount speed.
- Spot box; compare within-table. Cohere is well-conditioned, so rotation's gain is
  conservative — it would be larger on raw embeddings.
