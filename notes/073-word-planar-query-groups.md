# 073 — Word-planar query groups: the invariant waste dies, the knee moves to T=32

Perf record: [`073-word-planar-query-groups.json`](./073-word-planar-query-groups.json).
c8a.4xlarge (Zen5) on-demand, Tokyo — executing the plan read off 072's disassembly
(see the [ALU ledger](/alu/072)): ~73% of the vectorized j-loop's dispatch budget was
re-establishing loop invariants — de-interleaving 8 fat pointers into gather addresses
(~10 uops/doc) and 4 `vpgatherqq` re-fetching the *same* 256 B of query words, every
doc, 10M times.

## The change

Queries transpose once per tile into word-planar stack groups — `qw[w][j]` = query
*j*'s word *w*, values not references — and the doc loop becomes four rounds of
broadcast/XOR/popcount/add via AVX-512 intrinsics, one accumulator store, no
reduction, no pointers, no gathers. Groups of 8 queries = one zmm lane-set; short
tails pad with zero-lanes whose garbage distances are never read.

## The detour that cost an afternoon: the autovectorizer refused the shape

The plan was safe scalar code shaped for the vectorizer. It refused — twice:

| attempt | C=2000 QPS | emitted |
|---|---|---|
| scalar planar, u32 lanes | 822 | 32 *scalar* `popcnt`/doc |
| scalar planar, u64 lanes | 736 | still scalar, worse |
| **`_mm512_popcnt_epi64` intrinsics** | **1185** | the 4×(broadcast·xor·popcnt·add) we wanted |

The irony: LLVM *built this exact instruction sequence itself* in 072 — from the
Vec-of-slices shape it recognized as a gather pattern. Handed the same dataflow as
plain arrays, its popcount-reduction idiom matcher doesn't fire, and it scalarizes.
So the kernel is intrinsics now (`_mm512_set1_epi64/xor/popcnt_epi64/add_epi64`),
cfg-gated on `avx512vpopcntdq` with the scalar loop as the portable fallback.
012's law ("stay out of the autovectorizer's way") earns its final asterisk:
*when the pattern-matcher can't see your shape at all, you write the instructions
yourself.*

## Tiling re-swept — the knee moved, as the model said it must

Per-query tile state collapsed from pointer machinery to 32 B in shared zmm rows,
so k in T\*=√(s/k) shrank and the optimum jumped:

| batch | 8 | 16 | 24 | **32** | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|
| QPS | 1220 | 1318 | 1241 | **1403** | 1265 | 1244 | 866 |

(Past 64, the group array + heaps spill L2 and rayon chunking turns noisy — cv up
to 18%.)

## Result (batch=32, 10M, recall bit-identical throughout)

| rerank C | recall@10 | QPS | p50 | p99 |
|---|---|---|---|---|
| 500 | 0.9220 | **1542** | 11.8 ms | 12.0 ms |
| **2000** | **0.9737** | **1307** | 13.0 ms | 13.4 ms |
| 8000 | 0.9932 | 885 | 17.4 ms | 18.4 ms |

objdump confirms the waste is gone: zero gathers in the funnel, no pointer surgery,
query zmm rows hoisted above the doc loop. **Kernel arc at C=2000: 650 (067) → 847
(069) → 955 (071) → 1093 (072) → 1307 (073)** — 2.0× cumulative from kernel work
alone, recall bit-identical at every step. The scan now demands ~44 GB/s of code
stream at C=500; the per-core streaming ceiling (068/064) is no longer far away —
the next factor lives in bytes or shards, not uops.

## Caveats

- The intrinsics path is x86-64 + AVX512-VPOPCNTDQ only (Zen4+/Ice Lake+); others
  take the scalar fallback (M3/NEON: unmeasured — the fallback shape may need its
  own look if ARM becomes a target).
- batch=32 CV at C=2000 was 3.5% (rayon chunk granularity at 2000 queries / 32);
  the 1307 is the median of 3 reps. C=500's 1542 is tight (0.3%).
