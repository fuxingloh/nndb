# 008 — int8 quantization: a bandwidth win (so it depends on the machine)

Perf records: [`008-granite-i8.json`](./008-granite-i8.json), [`008-granite-f32.json`](./008-granite-f32.json).
Dataset: **Cohere v3 Wikipedia**, 1,000,000 × 1024-d, real floats, cosine (unit-normalized → L2 == cosine). Measured on the **Granite Rapids** box (memory-bound) and the M3 (compute-bound).

## What we did

First real lossy quantization (SIFT was uint8 → lossless; Cohere floats are genuinely lossy). `--quant i8`: symmetric scalar int8, global scale, dot-product ranking. No rerank yet.

## Result — same code, opposite outcome by machine

| machine | bound | f32 QPS | int8 QPS | int8/f32 | f32 ns/dist | int8 ns/dist | int8 recall | int8 mem |
|---|---|---|---|---|---|---|---|---|
| M3 | compute-bound | 15.4 | 10.5 | **0.68×** (slower) | 64.8 | 95.2 | 0.9815 | 977 MB |
| Granite | **memory-bound** | 9.5 | 19.5 | **2.05×** (faster) | 105.6 | 51.3 | 0.9830 | 977 MB |

(f32 index 3906 MB both; recall 1.0 for f32. CV ≤0.5%.)

## Conclusions

1. **int8's speedup is a *bandwidth* win — it only appears where bandwidth binds.** The exact same binary is **0.68× (slower) on the compute-bound M3** and **2.05× (faster) on the memory-bound Granite box.** Streaming 4× fewer bytes only helps when bytes are the constraint. This is the cleanest demonstration yet of "measure on the machine whose bottleneck matches the optimization" — and why running this on the M3 alone would have given the *wrong* conclusion ("quantization is useless").

2. **Quality cost is tiny: recall@10 ≈ 0.983, no rerank.** Cohere v3's compression-aware training showing — int8 barely moves the rankings. Rerank would push it back toward 1.0.

3. **Memory drops 4× everywhere** (3906 → 977 MB) — that's bandwidth-independent, true on both machines.

4. **Only ~2× faster, not 4× (from 4× fewer bytes) — the re-pricing cascade again.** Quantization *eased* the memory wall, so compute starts to re-bind (int8 ns/dist 51 is ~half f32's, not a quarter). Two reasons it's not 4×: (a) the int8 kernel is still naive — `i8→i32` widen + multiply, **not** the hardware int-dot instructions (NEON `SDOT`/`UDOT`, x86 `VNNI`); (b) at 977 MB you're partway back to compute-bound. A VNNI/DotProd kernel should push the speedup further toward 4×.

5. **"Which is better" — answered, with a caveat:** on a **memory-bound serving box**, int8 is strictly better (2× faster, 4× smaller, ~same recall). On a **compute-bound box** with high bandwidth, f32 is faster. *Know your bound before quantizing.*

## Caveats

- **Naive int8 kernel** (no SDOT/VNNI) — leaves speed on the table; the next lever to reach ~4×.
- **No rerank** — recall is 0.983, not the ~1.0 a two-stage funnel would give. Rerank + the binary (32×) path are the next entries.
- Virtualized box → no PMU; bound inferred from the working-set detector + this cross-machine flip.
