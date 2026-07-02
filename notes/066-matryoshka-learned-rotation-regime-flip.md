# 066 — Matryoshka-256: learned rotation, code uniqueness, and the regime flip

Perf record: [`066-matryoshka-learned-rotation-regime-flip.json`](./066-matryoshka-learned-rotation-regime-flip.json).
c8a.4xlarge (Zen5, 16 vCPU), continuing 065 on the OpenAI Matryoshka-256 dataset.
Three results, each of which closes a question.

## 1. Learned rotation (ITQ) stacks with residual — +0.3 pts, free at query time

055 shelved ITQ on Cohere-1024 (+1.5 pts, "dense rotation too costly"). Re-tested at
256 bits with residual, via `itq --residual` (the bin's random column reproduces the
shipped funnel's numbers exactly, validating the comparison):

| rerank C | random+residual | ITQ+residual | delta |
|---|---|---|---|
| 500 | 0.9474 | 0.9541 | +0.0067 |
| 1000 | 0.9731 | 0.9780 | +0.0049 |
| **2000** | **0.9878** | **0.9907** | **+0.0029** |
| 4000 | 0.9944 | 0.9968 | +0.0024 |
| 8000 | 0.9979 | 0.9992 | +0.0013 |

The two levers overlap: ITQ alone was +0.55 at C=2000, but +0.29 on top of residual —
both improve code quality, so there's less left. The 055 cost objection dissolves on a
static index: the 256×256 rotation is applied once at build; the query-time scan is
byte-identical popcount. **Free recall, smaller than hoped, real.**

## 2. All 990k codes are unique — recall loss is ranking error, not identity loss

Counted exact duplicates of the funnel's stage-1 codes (residual → rotate → sign):
**990,000 / 990,000 unique, zero collisions** (with or without residual). Expected —
2^256 code space, birthday bound ~2^128 — but worth pinning: the 1-bit code loses no
identity. The recall gap is purely *ranking* (true neighbors landing a few Hamming
bits outside the top-C), which is exactly what rerank width and better rotations fix,
and why the funnel dials smoothly to 0.999.

It also answers "would sorted/compressed codes help?" — no. Rotation maximizes
entropy per bit by design, so sorted codes share only ~log2(N)≈20 leading bits (~8%
compressible, best case), and any decode adds instructions to a loop that costs ~4.

## 3. The regime flip: at 32 MB of codes, tiling hurts — the scan is compute-bound

The Cohere funnel (122 MB codes, DRAM-streamed) needed tiling: +73% (016). At 256
bits the store is **31.7 MB ≈ L3-resident on Zen5**, and tiling *costs*:

| batch | scan-only QPS | C=2000 QPS |
|---|---|---|
| **1** | **8,861** | **4,966** |
| 8 | 6,792 | 4,148 |
| 32 | 6,251 | — |

Same lesson as 038 (sweep the tile per machine), inverted: when codes fit in cache,
the right tile is **no tile**. The 003→007 re-pricing cascade again — every byte
reduction slides the workload toward compute-bound, and levers must be re-measured.

## Final operating point (Matryoshka-256, this box)

**ITQ + residual + batch=1 + C=2000 → recall@10 0.9907, ~4,966 QPS, p50 3.1 ms,
p99 3.3 ms**, 31.7 MB codes + 1.0 GB f32 store, 990k vectors, 16 vCPU.

## Caveats

- QPS on 16 vCPU (4xlarge) — the 062/064 sweep used 8 vCPU; don't compare across.
- ITQ recall measured in the `itq` bench bin; not yet wired into `vsearch`/server as
  a flag (the scan+rerank workload is identical, but end-to-end serving numbers with
  ITQ codes were not separately measured).
- One embedding (OpenAI text-embedding-3-large, closed model / open vectors). The
  fully-open reproduction path is Nomic v1.5 (MRL, Apache-2.0), unbuilt.
