# Search-as-GEMM on an accelerator (GPU / Trainium / Inferentia)

**Status: parked.** Out of scope for "fastest CPU box." Recorded so the direction is on
record without building it. Triggered by asking whether an ML ASIC (e.g. `trn1.2xlarge`)
can accelerate this engine.

## The test (from the Exa concepts notes)

`prep/exa/concepts.md` §"hardware lottery": *when evaluating a new architecture, ask first
— does it reduce to multiply-add?* For vector search the answer is **yes**:

- **Exact L2/cosine:** `‖q−b‖² = ‖q‖² + ‖b‖² − 2·q·b`. The `q·b` term over all base
  vectors is `Q × Bᵀ` — a GEMM. Tensor-core food.
- **Binary funnel too:** map sign bits `{0,1}→{−1,+1}`, then `Hamming = (D − a·b)/2`.
  Hamming is a dot product → also a matmul (the XNOR-popcount trick from binary NNs).

So it passes the test — an ASIC *can* run it. The catch is the re-pricing cascade
(concepts.md §16): the ASIC rewards a **different engine**, not ours.

## Why it's a different engine, not an acceleration of the funnel

| | Our CPU engine (funnel) | What the ASIC wants |
|---|---|---|
| Strategy | **do less** — 1-bit scan, rerank ~500 | **do all of it** — dense `Q×N` GEMM, fast |
| Bytes | fewer (128 MB — the whole win) | int8/f16 in HBM; doesn't reward 1-bit |
| Latency | single query is fine | single query = GEMV, <5% utilization — must **batch** |
| Top-K | cheap heap | **not a matmul** — argmax/partial-sort is awkward off-engine |

The funnel's "don't compute all the distances" is the opposite of feeding a systolic
array. On an ASIC you'd discard the funnel and do **dense brute-force GEMM**, betting raw
throughput beats cleverness — and it might:

**Back-of-envelope:** 1M × 1024-dim exact ≈ 2 GFLOP/query. A Trainium chip is ~O(100+)
TFLOP. Even at 20–30% MFU + selection overhead, exact brute force could land **10k–50k
QPS** — an order of magnitude past our ~960. The ASIC makes "all the work" so cheap that
*exact brute force* out-competes the clever index. That **inverts the premise of this
project** (which exists because doing all the work was expensive on a CPU).

## Why `trn1` specifically is the wrong member of the family

- **Trainium is for training** (BF16/FP8). Search-as-GEMM wants INT8 *inference*
  throughput → **Inferentia (`inf2`)** or a GPU is the natural pick.
- **Binary doesn't map cleanly:** to do Hamming as matmul you'd inflate ±1 into int8
  (Trainium's smallest well-supported type) → **8× more bytes**, killing the advantage
  that won the project. You'd want native 1-bit/4-bit XNOR matmul.
- **Top-K** still has to happen somewhere — FAISS-GPU pours huge effort into custom
  selection kernels (WarpSelect) for exactly this.
- **Porting cost:** Neuron SDK / NKI graph compiler, not Rust. Whole different codebase.

## What would decide it (if ever un-parked)

1. Dense int8 brute-force GEMM on a GPU (`g`/`p` family) or `inf2` — measure exact-search
   QPS and **cost-per-1k-QPS** vs the CPU funnel. The honest comparison is $/QPS, not QPS.
2. Whether on-device top-K (vector-engine WarpSelect-style) keeps the GEMM win or eats it.
3. Whether any available ASIC has native sub-int8 (1-bit/4-bit) matmul, which would let
   the *binary* code ride the tensor engine without the 8× byte penalty.

## How to set it up (if un-parked)

The honest experiment is an afternoon for the baseline, a day or two for the fair fight.
The setup, in order:

1. **Integration point is the ANN-Benchmarks contract, not the Rust `knn_batch` trait.**
   A GPU run is a *separate harness* (Python is fine) that reads the same SIFT1M `fvecs`
   base/query + ground-truth `ivecs` and reports recall@k / QPS / latency the same way.
   Nothing in `database/` changes — it's a sibling experiment, like any `history/` entry.

2. **FAISS-GPU for the dense exact baseline** — `IndexFlatL2` → `index_cpu_to_gpu`. Turnkey,
   and it solves on-device top-K (WarpSelect). Raw PyTorch (`Q @ B.T` + `torch.topk`) is the
   ~10-line fallback to see the GEMM directly.

3. **Instance — pick on memory bandwidth, not generation.** The scan is bandwidth-bound, so
   the older card can win:

   | | GPU | Mem BW | Note |
   |---|---|---|---|
   | g5.2xlarge | A10G (Ampere) | ~600 GB/s | default first data point |
   | g6.2xlarge | L4 (Ada, 72 W) | ~300 GB/s | better $/QPS & INT8, bandwidth-starved |
   | g6e.2xlarge | L40S (Ada) | ~864 GB/s, 48 GB | where the GPU actually wins throughput |

   (`g7` unconfirmed as of 2026-06 — verify it exists before assuming.) Pull **live spot +
   on-demand prices** before deciding; $/QPS is the metric, and it moves with spot capacity.

4. **Infra change is one file.** Extend `infra/lib/spot-stack.ts` to take a `g`-family
   `instanceType` and swap the AMI from Amazon Linux 2023 to a **Deep Learning AMI** (CUDA +
   drivers preinstalled — hand-installing CUDA is the only annoying part). Spot/SSH/teardown
   plumbing carries over unchanged.

5. **Bake in the honesty caveats** (see the table above — the GPU runs a different engine):
   report **$/QPS at fixed recall**, not raw QPS, and sweep **batch size** (single query =
   GEMV at <5% utilization; the GPU only wins batched). Note both or the comparison misleads.

Done = FAISS-GPU exact on SIFT1M on a g-family spot box, recall/QPS/latency across a batch
sweep, $/QPS vs the CPU funnel at fixed recall, recorded as the next `history/NNN-*` pair.

## Related

- `resources/005-simd-adc-scann.md` — the CPU analogue (pshufb LUT) of "gather can be fast
  if vectorized"; the same logic at ASIC scale is this note.
- Exa notes `prep/exa/concepts.md` §16 (re-pricing cascade), §"hardware lottery", line 594
  (MFU / <5% single-user utilization).
