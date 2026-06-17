# 013 — Counting selection for the binary top-C (a latency/throughput tradeoff)

Perf record: [`013-counting-selection.json`](./013-counting-selection.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --select heap|count`.

## The idea

009's binary scan keeps the top-C candidates with a bounded max-heap — O(n log C)
with a branchy compare-and-maybe-sift on every doc. But Hamming distance is a
small bounded integer (∈ [0, dim]), so selection doesn't need comparisons at all:
histogram the distances (O(n), branch-free), find the threshold bucket holding the
C-th smallest, and collect every id at or below it. New `--select count` picks
this; `--select heap` is the original.

Candidate *order* isn't preserved, which is fine — the rerank stage rescores and
re-sorts anyway.

## Result — lower latency, lower batch throughput

| C | select | recall@10 | batch QPS | p50 (single-query) |
|---|---|---|---|---|
| 0    | heap  | 0.4681 | **587** | 13.87 ms |
| 0    | count | 0.4681 | 561 | **12.61 ms** |
| 200  | heap  | 0.9419 | **584** | 14.33 ms |
| 200  | count | 0.9419 | 549 | **12.85 ms** |
| 1000 | heap  | 0.9943 | **550** | 15.32 ms |
| 1000 | count | 0.9943 | 511 | **13.16 ms** (−14%) |

Recall is identical (same candidate set up to ties). Counting cuts single-query
latency ~9–14%, but loses ~5–7% batch QPS.

## Why the split — and why counting still matters for serving

The two passes measure different things:

- **Batch QPS** runs all queries across all 8 cores at once. The box is
  memory-bound (007), so aggregate bandwidth is the ceiling. Counting writes and
  re-reads a 2 MB `dists` buffer per query *on top of* the 122 MB scan; with 8
  cores doing that simultaneously, the extra traffic costs throughput. (Reusing
  the buffer via per-thread scratch did **not** help — bandwidth, not allocation,
  is the constraint; it measured slightly worse, so it was dropped.)
- **Single-query latency** runs one search on one core. Bandwidth is plentiful, so
  the branch-free histogram beats the branchy heap — the heap mispredicts on every
  candidate that displaces the current C-th best.

This matters because the **serving model** (002) is one single-threaded search per
request, where server throughput = cores ÷ per-query *latency*. By that measure
counting is the win: 15.3 → 13.2 ms/query is ~16% more server throughput per core,
even though the all-cores batch number drops. The batch benchmark and the serving
benchmark reward opposite choices here.

## Conclusions

1. **Kept both; default stays `heap`.** Heap maximizes the batch-QPS headline.
   `--select count` is the choice for latency-bound / serving workloads (revisited
   in the serving entry).
2. **Selection was never the scan's main cost** — the popcount scan dominates, so
   swapping the selector moves things only single-digit percent. The bigger levers
   are the scan's bandwidth and the rerank tier (next entries).

## Caveats

- Counting's `dists` is `u16` (Hamming ≤ dim = 1024 fits); fine up to dim 65535.
- Numbers are a single before/after run (CV < 1.5%); spot-instance level drifts
  run-to-run, so compare within the table, not against other entries' absolutes.
