# 054 — Product Quantization: wins recall-per-byte, loses recall-per-QPS

Perf record: [`054-product-quantization.json`](./054-product-quantization.json). Granite
box (8 vCPU). `src/bin/pq.rs`. Cohere 1M × 1024, K=256/subspace, trained on a 100k
sample (12 k-means iters), recall@10 vs exact GT. Closes the open branch from `044`.

## What PQ is

Split each vector into M subvectors, k-means each subspace (K=256 → 1 byte/subspace), so
a vector is **M bytes**. Search by **ADC**: per query, precompute an M×256 table of
subvector→centroid distances; each doc's distance is M table lookups summed.

## Result

| M | bytes/vec | recall C=0 | recall C=200 | recall C=1000 | QPS |
|---|---|---|---|---|---|
| 8 | 8 | 0.071 | 0.373 | 0.675 | ~590 |
| 16 | 16 | 0.206 | 0.683 | 0.905 | ~290 |
| 32 | 32 | 0.405 | 0.924 | **0.992** | 140 |
| 64 | 64 | 0.593 | 0.993 | **0.9998** | 62 |
| **binary funnel** (051) | **128** | — | — | **0.998** | **963** |

## The verdict: better bytes, worse speed

- **Recall-per-byte: PQ wins.** M=32 matches the funnel's recall (0.992) at **¼ the
  bytes** (32 vs 128); M=64 edges it (0.9998).
- **Recall-per-QPS: PQ loses, badly.** M=32 runs **140 QPS vs the funnel's 963 — 7×
  slower** at the same recall; M=64 is 62 QPS (15×).

## Why — the 011 lesson, at scale

PQ's ADC is **M data-dependent gathers per doc** (look up `table[s][code[s]]`), and
gathers **don't autovectorize**. The binary funnel's per-doc op is `count_ones` →
**VPOPCNTDQ** (8 u64/instr). And PQ here is **cache-resident** (16–64 MB ≪ L3), so it's
*not* bandwidth-bound — it's **gather-compute-bound**, and scalar gather loses to vector
popcount. This is exactly `011` (the asymmetric LUT lost to popcount) reproduced at 1M
scale: fewer bytes don't buy QPS when the per-doc op is a slow gather.

## Conclusion

**The binary funnel stays the QPS winner.** PQ is the **footprint / recall-per-byte**
option — valuable only when RAM is the binding constraint (8–32 B/vec vs 128 B), and even
then the funnel + disk hybrid (045) is an alternative. To make PQ *QPS*-competitive you'd
need a **SIMD ADC** (4-bit codes + `vpshufb` lookup, the FAISS PQ4 trick) — which is the
same unsafe-SIMD path parked in `050`. Scalar PQ on CPU does not beat popcount.

## Caveats

- Scalar ADC (the slow path); SIMD ADC (4-bit + pshufb) is the known fast variant and
  would change the QPS story — untested (needs the parked SIMD kernel).
- K=256, 100k training sample, 12 iters; more training/iters lifts recall a little, not
  the QPS verdict. OPQ (learned rotation + PQ) improves recall-per-byte further (056) but
  not the gather-vs-popcount QPS gap.
- One dataset (Cohere 1024). Recall vs within-set exact GT.
