# 012 — Multi-accumulator hamming: a negative result (popcount is already SIMD)

Perf record: [`012-popcount-autovectorizes.json`](./012-popcount-autovectorizes.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary`.

## The hypothesis

006 made the f32 kernel ~30% faster by splitting the L2 reduction across many
independent accumulators — a single accumulator serializes on the FMA latency
chain. The binary scan's hot path is `hamming` (`quant.rs`), which sums
`count_ones()` over the XOR'd words into **one** `u32`. Same shape → same fix?
The reasoning: scalar `popcnt` has ~3-cycle latency but 1/cycle throughput, so a
lone accumulator should be latency-bound and multiple accumulators should let the
popcounts pipeline.

So we tried it: 4 independent `u32` accumulators, tree-reduced at the end. Recall
is bit-identical (integer-add reassociation is exact), and all tests pass.

## Result — it's ~2.2× SLOWER, on both architectures

Same box, same run, simple vs 4-accumulator:

| Granite Xeon 6975P-C | simple loop | 4 accumulators |
|---|---|---|
| scan, no rerank (QPS) | **444.5** | 200.0 |
| scan + rerank C=1000 (QPS) | **421.2** | 195.5 |
| single-core scan (QPS) | **71.5** | 43.6 |
| recall@10 (no rerank / C=1000) | 0.4681 / 0.9943 | 0.4681 / 0.9943 |

Cross-checked on the M3 (NEON), single-core scan: **simple 240 vs multiacc 115
QPS** — same ~2× regression. So this isn't a quirk of one CPU.

## Why the f32 lesson does NOT transfer

The premise was wrong. `count_ones()` over a slice is **not** a scalar
`popcnt` — the compiler autovectorizes the naive loop to a *hardware vector
popcount*: `VPOPCNTDQ` on Granite (AVX-512), `CNT` on M3 (NEON). The simple loop
is already SIMD, counting many lanes per instruction. Manually splitting the sum
into four fixed scalar accumulators presents the compiler a shape it can't fold
back into that vector reduction, so it falls back to **scalar** popcounts — and
loses the very vectorization that made it fast.

The f32 FMA kernel had no such hardware primitive: there is no single instruction
that does "L2 of two vectors", so breaking the dependency chain by hand is the
win. popcount is the opposite case — the chip already has the wide instruction,
so the job is to stay out of the autovectorizer's way.

## Conclusion

1. **Reverted.** The simple `hamming` loop is kept; a comment now warns against
   re-introducing the manual split. 009's binary scan was already optimal.
2. **Lesson: measure before transplanting an optimization.** "Multiple
   accumulators help reductions" is true for arithmetic the hardware can't do in
   one instruction; for popcount (and anything with a dedicated SIMD op) the
   naive loop autovectorizes and hand-tuning regresses it.
3. The binary scan's remaining levers are therefore *not* the popcount itself —
   they're the top-C selection (heap → counting/radix on the bounded integer
   distance) and the rerank tier's random f32 gathers. Those stay open.

## Caveats

- Absolute QPS here (444 simple, no-rerank) is below 009's recorded 612 on the
  same box — spot-instance run-to-run variance; the before/after within this run
  is internally consistent (CV < 1.5%), which is what the comparison rests on.
