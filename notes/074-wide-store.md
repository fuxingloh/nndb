# 074 — Wide store: the last answer-format tax, ~20 uops/doc and an irreducible loop

Perf record: [`074-wide-store.json`](./074-wide-store.json).
c8a.4xlarge (Zen5) on-demand, Tokyo — closing the last item on 073's disassembly
ledger (see [/alu/073](/alu/073)): the planar kernel finished each doc-group with
8 complete distances sitting in one zmm, then *disassembled it lane-by-lane* — 8
extract/store pairs (`vmovd`/`vpextrd`/`vextracti128`/`vextracti32x4`), each with
its own cmp/je tail-guard, plus an acc-pointer reload between extractions. ~24
uops/doc of store-out tax on results the vector unit had already computed.

## The change

Two moves. First, pad `acc` to whole groups of 8 — the tail bound moves out of
the hot loop entirely (selection still reads only `acc[..t]`, so padding lanes
are never consumed). Second, with no tail-guards left, the whole disassembly
collapses to two instructions: `_mm512_cvtepi64_epi32` (vpmovqd) narrows the
zmm of u64 distances to a ymm of u32, and one 32 B `_mm256_storeu_si256` writes
all 8 at once. The scalar fallback path got the same simplification.

objdump confirms: vpmovqd present in the funnel region (6 sites, including the
remainder paths); the per-lane extraction chain is gone from the group loop.

## Tiling re-swept — the optimum holds at 32

| batch | 16 | **32** | 48 | 64 |
|---|---|---|---|---|
| QPS (C=2000) | 1401 | **1498** | 1379 | 1367 |
| cv | 0.2% | 0.5% | 1.9% | 5.6% |

## Result (batch=32, 10M, recall bit-identical throughout)

| rerank C | recall@10 | QPS | cv | p50 | p99 |
|---|---|---|---|---|---|
| 500 | 0.9220 | **1685** | 0.0% | 11.26 ms | 11.42 ms |
| **2000** | **0.9737** | **1471** | 0.9% | 12.37 ms | 12.70 ms |
| 8000 | 0.9932 | 925 | 0.1% | 16.80 ms | 17.93 ms |

+13% over 073 at C=2000 (1307 → 1471). **Kernel arc at C=2000: 650 (067) → 847
(069) → 955 (071) → 1093 (072) → 1307 (073) → 1471 (074)** — 2.26× cumulative
from kernel-only changes, recall bit-identical at every step.

## The law, found for the third time

This was the last piece of "answer-format tax." 069's horizontal reduction,
072's per-doc scalar demand, 073's lane extraction — same finding each time:
*the arithmetic was never the cost; converting the answer into the consumer's
format was.* With the store now wide, the loop is ~20 uops/doc: 4 broadcasts +
4 folded-load XORs + 4 popcounts + 3 adds + narrow + store + loop control. The
remaining per-doc work is essentially irreducible at this ISA — the scan-side
levers left are bytes (smaller codes) and locality (CCX-resident shards), not
uops.

## Caveats

- The intrinsics path is x86-64 + AVX512-VPOPCNTDQ only; others take the
  (now simplified) scalar fallback — still unmeasured on ARM.
- The acc padding adds ≤7 dummy u32 per tile — harmless, never read.
- QPS deltas are same-box, same-session against the 073 numbers; final
  operating points are 3 reps at latency-queries=100.
- **A follow-up micro-opt was tried and rejected:** the objdump shows LLVM
  re-broadcasting the doc words per unrolled group (~4 uops/group). Hand-hoisting
  them into named zmms outside the group loop measured *worse* (1622 vs 1685 QPS
  at C=500) — the re-broadcasts ride free on load-pipe slack, and the manual hoist
  perturbs LLVM's schedule. Kept the simpler form; noted in the kernel comment.
