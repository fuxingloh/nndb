# Search on an FPGA (AWS F2 / AMD Virtex UltraScale+ HBM)

**Status: parked.** Recorded so the direction is on record without building it. Sibling of
[`accelerator-gemm.md`](accelerator-gemm.md) — the GPU/ASIC question — but it reaches the
*opposite* conclusion, which is exactly why it's worth keeping separate.

## The test (from the Exa concepts notes)

`prep/exa/concepts.md` §"hardware lottery": *does it reduce to multiply-add?* For an FPGA
the more useful question is *what primitive does it reward?* — and an FPGA rewards a
different one than a tensor engine:

- **Binary funnel → XNOR-popcount.** Our actual win is the 1-bit scan: `Hamming = popcount(a
  XNOR b)`. This is **bit-level**, the FPGA's home turf — LUTs do XNOR, hardened popcount
  trees do the rest, at 1 bit/element. **No 8× inflation** to int8 (the penalty that the
  ASIC note flags as killing the binary advantage). The FPGA is the *one* accelerator that
  rides the binary funnel natively instead of forcing a switch to dense GEMM.
- **Exact L2 → MAC.** Maps to DSP slices, fine — but a GPU does dense MAC better per dollar.
  The FPGA's edge isn't raw GEMM, it's the bit-ops and custom dataflow.

## Why this inverts the GPU/ASIC conclusion

| | GPU / ASIC (`accelerator-gemm.md`) | FPGA (this note) |
|---|---|---|
| Rewards | dense exact GEMM — *discard* the funnel | **the funnel itself** — 1-bit XNOR-popcount |
| Binary code | must inflate ±1 → int8 (8× bytes) | native 1-bit, no penalty |
| Dataflow | fixed; you adapt to the engine | **custom** — build the funnel as a pipeline |
| Latency | single query = GEMV, <5% util, must batch | deterministic streaming, low single-query latency |
| Top-K | awkward off-engine (WarpSelect) | a heap/selection unit *is* synthesizable inline |

So an FPGA is the accelerator that says "your clever do-less funnel was right — let me wire
it into silicon" rather than "throw it away and brute-force everything." That's the appeal.

## AWS F2 specifically

F2 (successor to F1) uses **AMD Virtex UltraScale+ HBM** FPGAs (VU47P-class) with on-package
**HBM2** — high bandwidth feeding the binary codes straight into the popcount pipeline.
*(Verify live — specs below are approximate as of 2026-06:)*

- `f2.6xlarge` — 1 FPGA, ~24 vCPU, ~256 GB RAM; FPGA has ~16 GB HBM2 at ~460 GB/s.
- `f2.12xlarge` / `f2.48xlarge` — 2 / 8 FPGAs for scale-out.
- The whole 128 MB binary working set fits in HBM many times over — bandwidth, not capacity,
  is the question, and HBM gives it.

## Why it's parked — the development cost is the wall

- **Toolchain is a different planet.** AWS FPGA Developer AMI + Vitis/Vivado, HDL
  (Verilog/VHDL) or HLS (C++) → synthesis → place-and-route (**hours per build**) → AFI
  (Amazon FPGA Image) registration. Not Rust, not Python, not an afternoon.
- **Clock is slow** (~300 MHz vs GPU GHz) — the win has to come from *width and dataflow*
  (thousands of popcounts per cycle, no instruction overhead), not frequency. If the design
  can't fill the fabric, it loses.
- **Irregular tail maps awkwardly.** The 1-bit scan is perfect; the **rerank ~500 + top-K
  heap** is control-flow-heavy and eats engineering effort to pipeline cleanly.
- **$/hr is high and dev time dominates the bill** — the honest cost is engineer-weeks of HDL
  before a single QPS number exists, not the instance rate.

## What would decide it (if ever un-parked)

1. A binary XNOR-popcount scan kernel on `f2.6xlarge` over SIFT1M 1-bit codes — measure
   QPS/latency and **$/QPS at fixed recall** vs the CPU funnel *and* vs the GPU dense baseline
   from `accelerator-gemm.md`. Three-way $/QPS is the real comparison.
2. Whether the rerank + top-K stage stays on-FPGA (full pipeline) or hands back to host CPU,
   and which keeps the latency win.
3. Whether HLS gets a credible design standing without dropping to hand-written Verilog —
   that ratio decides whether this is ever economically sane vs the GPU path.

## Related

- [`accelerator-gemm.md`](accelerator-gemm.md) — the GPU/ASIC sibling; opposite conclusion
  (rewards dense GEMM, penalizes the binary code). Read both together.
- `resources/005-simd-adc-scann.md` — the CPU SIMD analogue (pshufb LUT); the FPGA is the
  "build the funnel into hardware" extreme of the same idea.
- Exa notes `prep/exa/concepts.md` §"hardware lottery", §16 (re-pricing cascade).
