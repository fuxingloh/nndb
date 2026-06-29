# 007 — Granite Rapids: the cascade closes (batching finally pays)

Perf records: [`007-granite-bound.json`](./007-granite-bound.json), [`007-granite-batch.json`](./007-granite-batch.json).
Box: `c8i.2xlarge` spot (Intel Xeon 6975P-C, **Granite Rapids**, AVX-512 + VBMI), provisioned via `infra/` CDK in ap-southeast-1.

## What we did

Ran the 006 FMA kernel on the newest Intel gen — the chip closest to Exa's stack (has `vpermb`/VBMI). Bound sweep + batch sweep + the forced-512 test, same protocol as Cascade Lake (004/006).

## Results — two firsts

**1. The bound FLIPPED — first machine to show it.**

| working set | ns/dist |
|---|---|
| 0.5 MB (L1) | 8.75 |
| 3.9 MB (L2) | 8.69 |
| 31 MB (L3) | 8.65 |
| 125 MB | 12.20 |
| 488 MB (DRAM) | 13.50 |

Ratio **1.56× → MIXED / memory-sensitive**. Cache-resident the FMA kernel runs ~8.6 ns/distance (near the M3's 7.2); but the full-base scan jumps to 13.5 ns — **the fast kernel outran this box's memory bandwidth**, so at full size it's (partly) memory-bound. Every prior machine was flat (compute-bound); this is the first to step up.

**2. Batching finally HELPS — first machine where it does.**

| batch | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| QPS | 73.0 | 112.3 | 112.3 | 113.1 | 114.3 | 114.5 |

**+57% (73 → 114)**, then a clean plateau. Because the full-base scan is now memory-bound, amortizing each loaded vector across a query tile pays — exactly what did *nothing* on the compute-bound machines (003–006). recall 0.9994 throughout.

## This is the re-pricing cascade, demonstrated end to end

- 003–005: batching did nothing → **compute-bound** (serial reduction).
- 006: fixed the kernel (FMA + accumulators) → ~25–34% faster.
- 007: the faster kernel makes the full scan **memory-bound** on this box → **batching now works (+57%)**, plateauing at the compute ceiling.

The levers unlock each other in order: **kernel first, then batching.** Neither helped alone; together they break the ceiling. The bound detector earned its keep — it flagged MIXED, and independently batching helped (the two methods agree, again).

## The 512-bit question, settled across two Intel gens

Default build (LLVM `prefer-256-bit`) → 256-bit (`ymm`) FMA → 73 QPS / 13.5 ns. Forced 512-bit (`zmm`) → **52.9 QPS / 18.9 ns — 27% slower** (CV 0.9%). So even on Granite Rapids (minimal downclock, VBMI), forcing 512-bit loses, just less catastrophically than Cascade Lake's −32%. **LLVM's 256-bit choice is right on both Intel generations. Width is not the lever.**

## Cross-machine summary (FMA kernel)

| machine | widest SIMD used | ns/dist cache→DRAM | bound | batching | forced-512 |
|---|---|---|---|---|---|
| M3 | NEON-128 | 5.9 → 7.2 (1.23×) | compute (near wall) | none | n/a |
| Cascade Lake | 256-bit (ymm) | ~15 → 16 (1.09×) | compute | none | −32% |
| **Granite Rapids** | 256-bit (ymm) | 8.6 → 13.5 (**1.56×**) | **MIXED/memory** | **+57%** | −27% |

The progression is the whole story: as cores get fast relative to memory, you move from compute-bound (batching useless) toward memory-bound (batching pays). Granite Rapids got there first because it pairs fast cores with a modest VM memory-bandwidth slice.

## Caveats

- Virtualized (no PMU) → bound via the software working-set detector (005), not hardware top-down. The detector's MIXED verdict is corroborated by the independent batching gain.
- This is an 8-vCPU **slice** of a Granite Rapids socket; a full socket has far more memory bandwidth and would flip to memory-bound *later* (need a faster kernel / more cores to outrun it). The *relative* result (kernel→memory→batching) is what carries.
- Once memory-bound, the next lever is **fewer bytes per distance (quantization)** — still parked, but now the data shows exactly why it's next.
