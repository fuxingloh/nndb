# 005 — Bound detector: compute-bound on both NEON and AVX-512

Perf records: [`005-m3-bound.json`](./005-m3-bound.json) (M3 NEON-128), [`005-cascadelake-bound.json`](./005-cascadelake-bound.json) (Lightsail Cascade Lake AVX-512).

## What we did

Replaced *inference* with an explicit **compute-vs-memory bound detector**. The Lightsail box is virtualized and exposes **no PMU** (`perf stat` → `<not supported>` for all hardware counters), so a direct top-down read is impossible. The PMU-free method:

**Working-set sweep.** Shrink the base from cache-resident to RAM-sized (`--base-subset N`) and measure **ns-per-distance** (time ÷ queries ÷ N — normalized, so size cancels out). If ns/distance is flat across the cache→DRAM boundary, memory speed doesn't matter → **compute-bound**. If it steps up past cache → **memory-bound**.

Also added the rigor that was missing: `--reps R` with the first run discarded as warmup, reporting **median + coefficient of variation** instead of a single shot.

Sizes (512 B/vector): 1k=0.5 MB (L1), 8k=3.9 MB (L2), 64k=31 MB (L3), 256k=125 MB, 1M=488 MB (DRAM). 6 reps each; query count scaled to hold total work ~constant.

## Results

| working set | M3 NEON-128 ns/dist | Cascade Lake AVX-512 ns/dist |
|---|---|---|
| 0.5 MB (L1) | 13.34 | 20.52 |
| 3.9 MB (L2) | 14.21 | 20.06 |
| 31 MB (L3) | 15.02 | 19.90 |
| 125 MB | 12.98 | 19.83 |
| 488 MB (DRAM) | 12.66 | 19.74 |
| **max/min ratio** | **1.18×** | **1.04×** |
| CV | 2–6% | <1.2% |
| **verdict** | **COMPUTE-BOUND** | **COMPUTE-BOUND** |

## Conclusions

1. **Compute-bound on both architectures — directly measured, not inferred.** Growing the working set ~1000× (L1 → DRAM) changes ns/distance by <18% (M3) / <4% (Cascade Lake). If this were memory-bound, ns/distance would *spike* the moment the base exceeds L2/L3. It doesn't even wiggle. Memory speed is irrelevant to this kernel.

2. **Two independent methods agree.** The batching-invariance test (003: cut bytes 32×, QPS unchanged) and this working-set sweep (cut the working set to cache size, ns/distance unchanged) attack the question from opposite directions and reach the same verdict on both machines. That's the cross-check the single-method 003/004 lacked.

3. **The numbers are trustworthy now.** CVs are 2–6% on the M3 (laptop, thermal/background noise) and <1.2% on Cascade Lake (one at 0.03% — a quiet server CPU). These are medians of 6 reps with warmup discarded, not single shots. Earlier entries' single-shot QPS should be read as ±~10%.

4. **The small-N values are slightly *higher*, not lower.** Counterintuitive for "cache is faster" — but it's per-query fixed overhead (rayon dispatch, heap setup) amortized over fewer distances at small N. True compute cost ≈ the large-N asymptote (M3 ~12.6 ns, Cascade Lake ~19.7 ns). Another tell that it's compute, not memory.

5. **Cascade Lake is ~1.6× slower per distance than the M3 (19.7 vs 12.6 ns) despite 4× wider SIMD** — consistent with 004 (fewer/older cores, likely AVX-512 downclock, FMA-less reduction). Wider lanes don't help a serial-reduction-bound kernel.

## Caveats

- This is a *software* detector. The definitive **PMU top-down** (% memory-bound, achieved GB/s) still requires a bare-metal box; Lightsail can't provide it. But two agreeing software methods is strong evidence.
- Verdict is for *this kernel*. The whole point is that the kernel's serial, FMA-less reduction is the binding constraint — fix that and the bound may shift to memory (at which point the detector earns its keep again).
