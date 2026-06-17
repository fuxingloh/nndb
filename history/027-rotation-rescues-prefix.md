# 027 — Rotation rescues prefix truncation (Matryoshka-like)

Perf record: [`027-rotation-rescues-prefix.json`](./027-rotation-rescues-prefix.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --scan-bits N --rotate 2 --batch 16`.

## The idea

014 found prefix truncation cheap on QPS but costly on recall, *because* Cohere
isn't Matryoshka-trained — the first N raw dims aren't the most informative.
026's rotation spreads each vector's information evenly across all dims. So a
**prefix of the rotated vector** is effectively a random projection to N dims (each
output dim mixes all D inputs), not a drop of (D−N) raw dims. Hypothesis: rotation
should make truncation lose much less recall.

## Result — rotation helps *more* the more you truncate

| scan-bits | C | no-rot recall | rot×2 recall | Δ |
|---|---|---|---|---|
| 512 | stage-1 | 0.2609 | 0.3007 | +4.0 pts |
| 512 | 1000 | 0.8842 | **0.9297** | **+4.6 pts** |
| 512 | 2000 | 0.9303 | **0.9626** | +3.2 pts |
| 768 | 1000 | 0.9703 | **0.9829** | +1.3 pts |

Compare the full-bit gain from 026: only +0.27 pts at 1024/C=1000. The rotation's
benefit *grows* as you truncate harder — exactly the hypothesis. QPS is unchanged
(rotation is free).

## Conclusions

1. **Rotation fixes 014's limitation.** Prefix truncation was a recall-lossy lever
   on Cohere; with rotation it's far gentler. 768-bit rot×2 → **0.983 recall @ 923
   QPS** — 014 needed C=2000 to approach that recall and only hit ~517 QPS. The
   prefix lever is now genuinely useful at high recall, not just below 0.97.
2. **The effect compounds with truncation depth** (+4.6 pts at 512 vs +0.27 at
   1024), because the more dims you drop, the more it matters that the kept ones are
   balanced. This is the practical realization of the Matryoshka idea *without* a
   Matryoshka-trained model — a random rotation buys most of the benefit for free.
3. **This reshapes the Pareto frontier** (mapped next): rotation + prefix + tiling
   should push the recall↔QPS curve out beyond the non-rotated 018 frontier.

## Caveats

- Still cosine/Cohere; the absolute recall at 512-bit (0.93) is below the full-bit
  0.997, so prefix remains a recall trade — just a much better one with rotation.
- tile=16, reps=4, CV < 0.5%.
