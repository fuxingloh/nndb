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

## Related

- `resources/005-simd-adc-scann.md` — the CPU analogue (pshufb LUT) of "gather can be fast
  if vectorized"; the same logic at ASIC scale is this note.
- Exa notes `prep/exa/concepts.md` §16 (re-pricing cascade), §"hardware lottery", line 594
  (MFU / <5% single-user utilization).
