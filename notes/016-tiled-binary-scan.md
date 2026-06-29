# 016 — Tiled binary scan: +73% QPS at identical recall

Perf record: [`016-tiled-binary-scan.json`](./016-tiled-binary-scan.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --batch T`.

## The idea

014/015 nailed down that the binary **scan** is the bottleneck and it's bandwidth-
bound: each query independently streams the whole 122 MB binary corpus. So do what
007 did for f32 — **tile the queries**: load each doc's 128-bit code once and
compare it against a tile of T queries before moving to the next doc. The base
streams once per *tile* instead of once per *query*, cutting base bandwidth ~T×.
The result is exact-equivalent (same candidates, just reordered work).

## Result — large throughput win, recall untouched

| tile | scan-bits | recall@10 | QPS |
|---|---|---|---|
| 1  | 1024 | 0.9931 | 485 |
| 4  | 1024 | 0.9931 | 797 |
| 8  | 1024 | 0.9931 | **838 (+73%)** |
| 16 | 1024 | 0.9931 | 850 |
| 32 | 1024 | 0.9931 | 858 (+77%) |
| 8  | 512  | 0.8842 | 958 |
| 16 | 512  | 0.8842 | 984 |

## Conclusions

1. **Tiling is a pure win here: +73% QPS at tile=8, recall bit-identical.** This is
   the single biggest throughput gain of the funnel work, and unlike prefix
   truncation (014) it costs *nothing* in recall — it only reorders computation to
   reuse each doc across T queries.
2. **The gain saturates ~tile=16–32.** Once base bandwidth is amortized, the scan
   re-prices back toward **compute-bound** (the popcount work, which tiling doesn't
   reduce — it's still T×N hammings). The knee at tile=8 captures most of it; past
   tile=16 returns are small. This is the 003→007 re-pricing cascade again, now on
   the binary kernel: kill the bandwidth wall and the compute wall reappears.
3. **It composes with prefix truncation (014).** tile=16 + 512-bit → 984 QPS (at
   the prefix's lower recall). The two levers stack: tiling cuts *re-reads* of the
   base, prefix cuts the *size* of the base.

## Notes

- Applies to the batch/throughput path only — a single-query request (the serving
  model) can't tile, so this raises batch QPS, not single-query latency. The
  serving entry measures the single-query side.
- `--batch` default is 1 (per-query); tiling is opt-in. Heap selection + f32
  rerank; the tiled path already respects `--scan-bits` (bbase is prefix-packed).
- recall reads 0.9931 here (2000-query slice) vs 0.9943 elsewhere (1000-query) —
  different query subset, same ballpark.
