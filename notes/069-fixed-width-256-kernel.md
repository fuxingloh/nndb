# 069 — Fixed-width 256-bit kernel: 3× the popcount, +30% funnel QPS, tiling pays again

Perf record: [`069-fixed-width-256-kernel.json`](./069-fixed-width-256-kernel.json).
c8a.4xlarge (Zen5) on-demand, Tokyo — same box and dataset as 068. Consequence of
068's verdict: if the 10M scan is compute-bound on popcount, the kernel itself is the
only thing left to attack. 012/050 closed that door — but they measured **1024-bit**
codes, where the generic loop autovectorizes cleanly. At **256 bits** it doesn't.

## The blind spot in 012/050

The engine's `hamming` was one generic loop over `a.len()` words. At 16 words
(1024-bit), the compiler emits wide VPOPCNTDQ over full zmm registers and the loop is
optimal — 012/050's conclusion, correct *at that width*. But the Matryoshka funnel's
operating point since 065 is 256-bit codes = **4 words**: at that trip count the loop
control and length checks dominate four popcounts, and the autovectorizer produces a
much weaker body. Nobody re-audited the kernel when the operating point moved — the
"closed" conclusion silently didn't transfer.

## Kernel microbench (`simdbench --bits 256`, single thread)

| variant | Zen5 in-cache | Zen5 at 320 MB | M3 in-cache |
|---|---|---|---|
| generic loop (shipped) | 0.555 | 0.493 | 0.461 |
| **fixed-width unrolled** | **2.230 (4.0×)** | **1.415 (2.9×)** | **1.214 (2.6×)** |
| fixed + doc-pair (zmm) | 5.978 (10.8×) | 1.313 | 1.378 |
| fixed + 4-doc ILP | 2.741 | 1.373 | 1.415 |

Fully unrolling the 4-word body (straight-line XOR+popcount, no loop) is **~3× at
scale, on both ISAs**. The fancier shapes (doc-pair filling a whole zmm — 10.8× in
cache! — and 4-doc interleave) collapse to the same ~1.4 Gcmp/s once the codes stream
from DRAM: past the unroll, the *stream* is the limit, so the simple fix is the right
one. `hamming()` now takes a fixed-width fast path when `len == 4`; other widths keep
the generic loop (1024-bit behavior unchanged).

## Funnel impact at 10M — and the regime flips *again*, informatively

| config | recall@10 | QPS | vs 068 best |
|---|---|---|---|
| batch=1, C=2000 (068's optimum) | 0.9739 | 645 | −3% (noise) |
| **batch=8, C=2000** | **0.9737** | **847** | **+27%** |
| batch=8, C=500 | 0.9220 | 902 | +32% vs 683 |
| batch=8, C=8000 | 0.9932 | 669 | +22% vs 548 |

Recall is bit-identical everywhere (the kernel computes the same integers). And note
the reversal: 068 (slow kernel) found tiling useless — compute was slow enough to hide
DRAM entirely. Tripling the popcount rate re-prices that balance: at batch=1 the fast
kernel is *wasted* waiting on the stream (645 ≈ 668, no gain!), but with tile=8
amortizing the stream the kernel's speed becomes visible: 847 QPS. **A kernel win and
a bandwidth win are complements at this operating point — neither shows up alone.**
Deeper tiles (16/32) return nothing; counting selection still loses (494).

New operating point: **batch=8, C=2000 — recall 0.974, 847 QPS, p50 17.5 ms** (was
650 in 067). At C=500, 902 QPS — past 900 for the first time at 10M.

## Takeaway

- "Closed" performance conclusions are **width-scoped**: 012/050 said "don't touch the
  popcount loop" and were right at 1024 bits, wrong at 4 words. When the operating
  point moves, re-audit the kernel at the new shape.
- The regime story is now three-layered: slow kernel → compute-bound, tiling useless
  (068); fast kernel → balanced, tiling pays (+31%); the next 3× of kernel would make
  it DRAM-bound outright, where only fewer bytes (050's conclusion) helps.
- Zero recall risk: bit-identical distances, all unit tests pass, 1024-bit path untouched.

## Caveats

- The doc-pair kernel's 10.8× in-cache figure is the popcount *ceiling* for this chip;
  reaching it at scale needs the stream cost gone (smaller-than-LLC shards — the
  carousel/sharding direction, 040/041) — parked, not pursued here.
- Microbench uses random codes (kernel-isolated); funnel numbers are the real measure.
