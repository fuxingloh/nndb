# 072 — The last check goes: unsafe width cast, +17%, C=2000 crosses 1000 QPS

Perf record: [`072-unsafe-width-cast.json`](./072-unsafe-width-cast.json).
c8a.4xlarge (Zen5) on-demand, Tokyo — the follow-through on 071. Same box, same
10M arctic-256 dataset, batch=8.

## The change

071 replaced bounds checks with a `try_from` proof per call — better, but the
objdump showed the length check still executing inside the 80M-iteration j-loop
(the compiler hoisted the doc-side copy, not the query-side). Under `fixed256`
the width isn't a runtime question at all: it's the build's contract — the whole
feature *means* "codes are 256-bit, guaranteed by the operator." So the check is
now a `debug_assert` + pointer cast:

```rust
debug_assert!(a.len() == 4 && b.len() == 4);
let a4 = unsafe { &*(a.as_ptr() as *const [u64; 4]) };
```

Debug builds still verify; release builds spend zero uops re-proving an invariant
the data structure already owns. This is the repo's first `unsafe` in a hot path,
taken deliberately after the safe alternatives were measured to their limit.

## Result (10M, batch=8, recall bit-identical)

| build | C=500 QPS | C=2000 QPS |
|---|---|---|
| 071 typed/try_from | 1028 | 955 |
| **072 unsafe cast** | **1198** | **1093** |

**+17% over 071. C=2000 crosses 1000 QPS at 10M on 16 vCPU** (recall 0.9737,
CV ≤ 0.9%). Kernel arc at C=2000: 650 (067) → 847 (069) → 955 (071) → **1093** —
+68% cumulative, all recall-risk-free.

## The negative control that almost hid the win

The first attempt bundled the cast with two "obvious" structural improvements
(stack `acc` array to kill an aliasing-forced doc reload; per-tile query
pre-conversion to hoist the checks) — and measured **927/887, slower than 071**.
Unbundled, the cast alone gives 1198/1093: the extras were a −25% drag. A raw-
pointer `&[u64; 4]` array and a cfg-split loop body evidently broke LLVM's
aliasing/unrolling analysis — the compiler lost more than the checks cost.

Two lessons, one old, one new:
- **012/071 still binds:** hand-restructuring that takes structure *away* from
  the compiler loses, even when it looks like it removes work. The only safe
  `unsafe` was the minimal one *inside* the kernel, leaving every loop intact.
- **Bundle nothing.** Three codegen-touching edits in one measurement nearly
  shipped a regression as a win (or discarded a win as a regression).

## Caveats

- `fixed256` + this cast makes the 256-bit contract load-bearing: feeding the
  fixed256 build non-256-bit codes is UB in release (debug catches it). The
  default build keeps the checked paths and is unaffected.
- try_from-vs-cast at −17% is larger than the removed instructions alone explain;
  the cast likely also unlocked better inlining of `hamming4` into the tile loop.
  Not disassembled this time — the QPS delta is the decision-grade number.
