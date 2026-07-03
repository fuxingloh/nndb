# 071 — The bounds-check tax: types instead of `unsafe`, +13% QPS, p50 −6 ms

Perf record: [`071-typed-kernel-bounds-check-tax.json`](./071-typed-kernel-bounds-check-tax.json).
c8a.4xlarge (Zen5) on-demand, Tokyo — continuing 069/070 on the 10M arctic-256 funnel.

## The finding: the emitted kernel was ~half safety scaffolding

Disassembling the release `vsearch` (objdump on the box) showed the 069 fast path
spends **~6 of its ~13 uops per call** on things the source never wrote: the
`len == 4` gate, then *bounds guards on `b`* — because the gate only tested
`a.len()`, the compiler had to guard all four `b[i]` indexes against panic. Zen5's
binding resource at this operating point is **dispatch width (8 uops/cycle)**, not
any ALU — every guard uop displaces a popcount uop, so "free" predicted branches
aren't free at all here. (The dump also showed rustc's reduction is smarter than
assumed: `vpmovqb` + `vpsadbw` — the SSE-era byte-sum trick — 3 uops, not 5.)

## The fix, in two steps — no `unsafe` anywhere

1. **`a.len() == 4 && b.len() == 4`** — one added condition lets the compiler prove
   all eight indexes in-range; every bounds guard disappears. Two words of diff.
2. **`--features fixed256` + typed `hamming4(&[u64; 4], &[u64; 4])`** — the width
   moves into the type system; the gate itself compiles out. Off-width codes panic
   loudly at conversion (never a wrong answer); the default build stays
   variable-width and byte-identical in results.

## Result at 10M (batch=8, recall bit-identical everywhere)

| build | C=500 QPS | C=2000 QPS | p50 (C=500) |
|---|---|---|---|
| 069 baseline | 902 | 847 | 16.3 ms |
| both-length gate | 994 | 924 | 10.2 ms |
| **fixed256 (typed)** | **1028** | **955** | 10.6 ms |

**+13–14% QPS over 069; C=500 crosses 1000 QPS at 10M on 16 vCPU.** The p50 drop
(16.3 → ~10 ms) is the same uop diet acting on the single-threaded latency pass,
where dispatch pressure isn't hidden behind memory stalls. Cumulative kernel arc at
C=2000: **650 (067) → 847 (069) → 955 (071)** — +47% from two zero-recall-risk
kernel changes.

## Why this is the right shape (and `get_unchecked` wasn't)

The `unsafe` route removes checks by *withholding* the index from scrutiny; the
typed route removes them by *proving* them unnecessary — and the proof composes:
`&[u64; 4]` also enables full inlining and hoisting where the caller's width is
static. Same lesson as 012/069 from a new angle: the compiler is an ally to be
given evidence, not an obstacle to be bypassed. The uop ledger, now assembly-
verified twice: ~6 uops checks (gone), ~4 arithmetic, ~3 reduction, ~5 caller
overhead — the reduction and caller overhead are what the word-planar layout
(parked as the next kernel experiment) would attack.

## Caveats

- The 069-vs-071 baseline rows were measured in different sessions on the same
  box/dataset (069's run used latency-queries=200 vs 50 here); QPS deltas are
  within-session consistent (CV < 0.5%), and the p50 improvement reproduces across
  C. Recall is bit-identical by construction.
- `fixed256` is a build-time commitment: one engine, one code width. The default
  build keeps the dual-path dispatch for mixed-width work (1024-bit Cohere etc.).
