# 023 — Register-tiled hamming kernel: another negative result

Perf record: [`023-register-tiled-kernel.json`](./023-register-tiled-kernel.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --batch 16 --tile-rt`.

## The idea

The tiled scan (016) compares each doc against T queries via T `hamming()` calls.
Hypothesis: reorder to *doc-word outer* — load each doc word once into a register,
reuse it across all T queries in the tile — to cut loads. Classic register
blocking.

## Result — 40–60% slower

| scan-bits | per-query (VPOPCNTDQ) | register-tiled | Δ |
|---|---|---|---|
| 1024 | 846 QPS | 339 QPS | **−60%** |
| 512  | 1056 QPS | 620 QPS | **−41%** |

Recall identical (same arithmetic).

## Why it loses (the 012 lesson, again)

Per-query `hamming` runs `count_ones` over the whole word slice, which the compiler
autovectorizes to **`VPOPCNTDQ`** — popcounting many 64-bit words per instruction.
The register-tiled form moves the popcount *inside* the per-query inner loop,
indexed one word at a time, which forces **scalar `popcnt`** and throws away the
vectorization. And the load it was trying to save wasn't a real cost: a doc is
128 bytes and stays in **L1** across the whole tile, so re-reading `doc[w]` per
query is an L1 hit, not a memory access.

So the trade was: save a free L1 hit, lose vectorized popcount. It loses — exactly
as 012 found for the non-tiled hamming. Two data points now say the same thing:
**don't hand-restructure popcount; let `count_ones` autovectorize.**

## Conclusion

Reverted to per-query as the default (`--tile-rt` kept opt-in to document the
result). Combined with 012, the binary kernel's compute is at the hardware
vector-popcount ceiling — there's no compute lever left that doesn't fight the
autovectorizer.

## Caveats

- Batch path, tile=16, reps=6, CV < 0.7%; clean.
