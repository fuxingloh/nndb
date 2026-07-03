# 070 — Matryoshka-over-bits cascade: Pareto-dominated by the flat funnel (negative)

Perf record: [`070-matryoshka-bits-cascade-negative.json`](./070-matryoshka-bits-cascade-negative.json).
c8a.4xlarge (Zen5) on-demand, Tokyo — same box/dataset as 067–069. The natural
"one more tier" idea, tried and killed with data.

## The idea

The funnel's Matryoshka move (065) truncates *dimensions*; rotation makes every
*bit* of the 256-bit code an independent hyperplane, so the same nesting applies to
the code itself: the first w words are a valid coarser sketch. Pack word 0 of every
code contiguously (80 MB — packing matters: reading word 0 in-place from 32 B rows
would still touch every 64 B cache line) and run three tiers:

1. **64-bit sketch scan** (80 MB stream, ¼ the popcounts) → top-C1 via two-pass
   histogram selection (013's counting trick, minus its per-doc `dists` buffer);
2. **256-bit Hamming** on the C1 survivors (random 32 B gathers) → top-C2;
3. exact f32 rerank of C2.

After 069 the scan is stream-and-compute balanced, so ¼ the bytes *and* ¼ the
popcounts in tier 1 looked like a straight ~2× on paper. `src/bin/cascade.rs`.

## Result: every point loses at recall parity

| config | recall@10 | QPS |
|---|---|---|
| **flat funnel, C=2000 (069)** | **0.9737** | **847** |
| flat funnel, C=500 | 0.9220 | 902 |
| 64-bit sketch, C1=100k | 0.8488 | 1341 |
| 64-bit sketch, C1=400k | 0.9335 | 801 |
| 128-bit sketch, C1=200k | 0.9722 | 447 |
| 192-bit sketch, C1=50k | 0.9736 | 364 |

The 64-bit tier is fast but blind — even a C1 of 400k (4% of the corpus!) only
reaches 0.9335, below what the flat funnel gives at *higher* QPS (902 @ 0.922 ≈
same recall band). Widening the sketch restores recall but the QPS collapses to
half-or-worse of baseline. There is no crossing point; the cascade's whole
recall-QPS curve sits inside the flat funnel's frontier. (SIFT1M control: same
domination, so it's not a 10M artifact.)

## Why it fails — two compounding effects

- **64 random bits are statistically too weak at N=10M.** Hamming on 64 bits
  concentrates hard: distances pile up at 32 ± 4, so the "true neighbor vs random
  doc" margin is a couple of bits wide and ~10⁵ of 10⁷ docs tie within it. The
  sketch can't reliably put true neighbors in its top 0.25–4%. (The binomial math
  says expected count within the discriminating radius is ~10⁵ — exactly where
  recall stalls.) 256 bits was chosen in 058/065 *because* that's where the
  bit-floor clears; 64 re-enters the floor.
- **Fixing it with width pays back the savings.** Two-pass histogram selection
  reads the sketch stream **twice**, so a w-word sketch costs 2w/4 of the full
  scan's bytes: at w=2 (128-bit) that's byte-identical to the flat scan — plus C1
  random gathers and a second selection stage on top. The architecture only ever
  had headroom at w=1, and w=1 doesn't have the recall.

The two constraints pinch out the design space: statistically you need >64 bits,
economically you need ≤64. A single-pass tier-A selection (loose heap or sampled
threshold) would halve the stream cost and reopen a sliver at w=1 for recall
targets ≤0.93 — noted in the JSON as the only surviving variant, parked as not
worth engine complexity for a regime below our operating recall.

## Takeaway

Matryoshka nesting is a property of the *embedding*, not a free license to nest
anywhere: dims 256→64 collapsed recall pre-rotation (014) and bits 256→64
collapses it post-rotation, for the same reason — the information isn't in a
prefix unless something (MRL training) put it there. Rotation spreads information
*uniformly*, which makes every 64-bit subset equally good and equally inadequate.
The flat 256-bit funnel with the 069 kernel remains the operating point:
**0.974 @ 847 QPS / 0.922 @ 902 QPS at 10M on 16 vCPU.**
