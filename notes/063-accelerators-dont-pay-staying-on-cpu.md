# 063 — Accelerators (GPU, Trainium, FPGA): more headroom, worse economics — staying on CPU

The hardware sweep (062) crowned a CPU winner: **c8a.2xlarge, 2310 QPS at 0.995 recall
for $3,776/yr** on-demand. The obvious next question — would specialized silicon push
further? The honest answer for *this* engine (the within-cell 1-bit funnel): each one
*can* add headroom, but none of them pays. This entry records why, with real prices, so
the door is closed deliberately rather than left as a vague "maybe GPUs are faster."

Costs are us-east-1 on-demand, annualized (`$/hr × 8760`); the funnel maps to each as noted.

| class | instance | $/yr | fit to the funnel |
|---|---|---|---|
| **CPU (baseline)** | c8a.2xlarge | **$3,776** | native — this *is* the engine |
| GPU | g6.xlarge (L4) | $7,050 | different engine (dense GEMM) |
| GPU | g5.xlarge (A10G) | $8,813 | different engine |
| GPU | g6e.xlarge (L40S) | $16,302 | different engine |
| GPU | p4d.24xlarge (8×A100) | $192,349 | absurd at this scale |
| GPU | p5.48xlarge (8×H100) | $482,150 | absurd at this scale |
| Inferentia | inf2.xlarge | $6,642 | poor (matmul ASIC) |
| Trainium | trn1.2xlarge | $11,771 | poor (training ASIC) |
| Trainium | trn1.32xlarge | $188,340 | poor |
| FPGA | f1.2xlarge | $14,454 | best fit, worst ROI |
| FPGA | f2.6xlarge | $17,345 | best fit, worst ROI |

## GPU — the only credible win, and it's a different product

Tensor cores don't do binary popcount, so you don't run the funnel — you run **dense
exact (or int8) brute-force as a batched GEMM** (cuVS / FAISS-GPU). Batched, a single
L4/A10G-class card plausibly does ~10–40k QPS exact, which on raw throughput-per-dollar
can edge past the CPU. But the caveats are the whole story: it only wins **batched**
(a single query is a GEMV at <5% utilization), top-K selection is off-engine overhead, and
you're buying **exactness you may not need** in place of a 0.995-recall funnel. So a GPU
adds headroom by *changing the engine*, not by accelerating ours. It's worth it only for
huge-batch, exact workloads — the honest evaluation ($/QPS at fixed recall, batch sweep) is
parked in `questions/accelerator-gemm.md`, not built here.

## Trainium / Inferentia — wrong shape

These are matmul ASICs for large, dense GEMMs. The funnel is the *opposite* of a GEMM (its
whole point is doing *less* work), binary codes would have to inflate to int8 — **8× the
bytes**, which deletes the advantage that won the project — top-K doesn't map, and it's a
Neuron-SDK rewrite. You'd pay **$6.6k–$11.8k/yr** for a strictly worse fit than a $3.8k
CPU. Nothing here recommends it.

## FPGA — perfect for the algorithm, terrible for the budget

This is the one accelerator that fits the *math*: XOR + popcount is native to FPGA fabric
(bit-parallel, thousands of popcounts per cycle), so the funnel maps beautifully. But the
scan is still **bandwidth-bound** feeding codes from DRAM — the same wall as the CPU — so
the fabric advantage is capped, while the costs are not: **$14k–17k/yr** (≈4× the CPU) plus
months of HDL/HLS engineering. It only amortizes at hyperscale with custom silicon, which
is a different company than this one.

## Verdict

The CPU funnel at **$3,776/yr** is the perf-per-buck floor, and nothing here clearly beats
it for our workload at our scale:

- **GPU** beats it only by switching to a *different engine* (dense batched exact) for a
  *different workload* (huge batch, exact) — a real but separate product, parked.
- **Trainium/Inferentia** are the wrong architecture and cost more for less.
- **FPGA** fits the algorithm but loses on bandwidth, dollars, and engineering-months.

Every one of them "could push performance" — and every one fails the only test that
matters here, **$/QPS at the recall we ship**. So this is a deliberate stop, not an
oversight: **the engine stays on CPU**, and the accelerator question is closed (the GPU
path remains parked, by $/QPS, in `questions/accelerator-gemm.md`).
