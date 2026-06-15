# 004 — Cross-arch baseline: AVX-512 doesn't flip it either

Perf record: [`004-cascadelake-avx512.json`](./004-cascadelake-avx512.json)

## What we did

Ran the 003 batch sweep on an AWS **Lightsail compute-optimized** box to test the cross-architecture hypothesis: *does wider SIMD (AVX-512) flip the f32 scan from compute-bound to bandwidth-bound, so that batching finally helps?*

- **Box:** Intel Xeon Platinum **8223CL (Cascade Lake)**, 8 vCPU (4 physical cores + HT), 15 GiB, Amazon Linux 2023. Virtualized (no PMU → no `perf` top-down here; this is a QPS/recall baseline).
- **Build:** `RUSTFLAGS="-C target-cpu=native"` → confirmed the binary emits **AVX-512** (`objdump`: 1350 `zmm` refs), i.e. genuine 512-bit, 4× the M3's NEON-128.
- This is the **batch baseline** for this box: `batch=1` = no batching, `batch=4…64` = batching.

## Results

| batch | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| QPS | 50.9 | 48.9 | 48.8 | 48.9 | 48.8 | 49.1 |
| recall@10 | 0.9994 | … | … | … | … | 0.9994 |

Flat. Single-query latency p50 ≈ 140 ms.

vs the M3 (003): | | M3 (NEON-128) | Cascade Lake (AVX-512) |
|---|---|---|
| QPS (batch=1) | 98.6 | 50.9 |
| single-query p50 | ~49 ms | ~140 ms |
| batching effect | none | none |

## Conclusions

1. **Batching does nothing here either → compute-bound, not bandwidth-bound.** 003's finding generalizes from ARM NEON-128 to x86 AVX-512. Bytes were never the constraint on either machine.

2. **4× wider SIMD did NOT move the bound.** AVX-512 was genuinely used (1350 `zmm`), yet the QPS-vs-batch curve is identical-shaped to the M3's. The reason is the same kernel defect, confirmed in the x86 disassembly: **`vfmadd` count = 0** — the compiler vectorized sub/mul to 512-bit but kept the strict-FP **serial reduction** (no FMA, no parallel accumulators). Widening the lanes can't help when the bottleneck is the dependency chain in the sum. This is strong evidence the binding constraint is the **reduction**, not SIMD width and not memory.

3. **Cascade Lake is ~2× slower than the M3 despite 4× wider SIMD.** 50 vs 98 QPS; 140 vs 49 ms/query. Likely causes: only 4 physical cores (the other 4 vCPU are HT), an older/lower-IPC core, a small virtualized memory-bandwidth slice — and plausibly **AVX-512 downclocking** (Cascade Lake is the worst generation for it). With the reduction capping throughput, the 512-bit mul/sub bought nothing while the frequency penalty still applied — a concrete instance of *"wider SIMD buys nothing at the bottleneck, and can hurt via downclock"* (Exa future-optimizations.md §L43). We can't prove the downclock here (no PMU on a virtualized box), but the direction fits.

4. recall@10 = 0.9994 throughout (exact, unchanged).

## Caveats

- Virtualized Lightsail → no hardware PMU, so this is QPS/recall only; the *direct* compute-vs-memory verdict (top-down) still needs a `.metal` box.
- 4-physical-core slice, not a full Cascade Lake socket — absolute QPS isn't representative of a full server; the *batching-is-flat* conclusion is what carries.

## The throughline (001→004)

The f32 scan is **compute-bound on every architecture tested** — M3 NEON-128 and Cascade Lake AVX-512 — and the binding constraint is the **kernel's serial, FMA-less reduction**, not bandwidth (003) and not SIMD width (004). Until that reduction is fixed (FMA + multiple vector accumulators), neither batching nor wider SIMD helps.
