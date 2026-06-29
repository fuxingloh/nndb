# 014 — Truncated (prefix) binary scan: trade recall for QPS

Perf record: [`014-prefix-scan.json`](./014-prefix-scan.json).
Cohere v3, 1M × 1024, cosine, Granite box. `--quant binary --scan-bits N`.

## The idea

013 showed the *selector* isn't the cost — the popcount scan is, and on this box
it's bandwidth-bound (reads the full 122 MB binary corpus every query). The
biggest scan lever is therefore reading **fewer bytes**: pack only the first `N`
dimensions of each binary code (a Matryoshka-style prefix) and scan those. Stage-1
recall drops because we threw away dims, but the full-f32 rerank recovers it — so
the question is purely: how much QPS do we buy per point of recall?

`QuantBinary::from_f32_prefix(v, bits)` packs the prefix; `--scan-bits` selects it
(0 = full). The rerank tier is unchanged (full f32).

## Result — scanning fewer bits ≈ proportional QPS, recall recovers with C

| scan-bits | C | recall@10 | QPS | p50 |
|---|---|---|---|---|
| **1024 (full)** | 1000 | **0.9943** | 419 | 15.65 ms |
| 768  | 1000 | 0.9706 | 584 | 11.19 ms |
| 768  | 2000 | 0.9868 | 517 | 12.49 ms |
| 512  | 1000 | 0.8906 | 810 | 5.54 ms |
| 512  | 2000 | 0.9359 | 699 | 7.66 ms |
| 512  | 4000 | 0.9649 | 551 | 11.36 ms |
| 256  | 2000 | 0.6930 | 976 | 5.49 ms |
| 256  | 4000 | 0.7688 | 720 | 7.96 ms |

## Conclusions

1. **The scan is bandwidth-bound — bits-read is the throughput knob.** 512-bit
   scan hits 810 QPS vs 419 at full 1024 (≈2× for half the bytes), and p50 latency
   drops 15.6 → 5.5 ms. The relationship is roughly proportional to bytes scanned,
   which is exactly the signature of a bandwidth-bound kernel.
2. **Prefix truncation extends the recall↔QPS Pareto frontier.** You get a clean
   dial: recall 0.987 @ 517 QPS (768/C=2000, +23% over full at the same recall
   tier), or 0.97 @ 584, or 0.89 @ 810. Over-retrieving more (larger C) buys back
   recall but costs the f32-rerank tier, so each prefix has a knee.
3. **The near-iso-recall sweet spot is 768/C=2000:** 0.987 recall at +23% QPS and
   −20% latency vs the full-scan 0.994 — a small recall give-up for a real
   throughput win.

## Why it doesn't hold 0.994 at 768 bits

Cohere v3 is **not** Matryoshka-trained, so the first 768 dims are not the most
informative 768 — truncation discards information roughly uniformly. A
Matryoshka-trained embedding concentrates importance in the prefix, so the same
trick would hold recall much closer at fewer bits (this is why Exa pairs binary
with Matryoshka). That's a dataset/model property, not a kernel limit.

## Caveats

- Spot-instance absolute QPS drifts run-to-run (full-scan baseline read 419 here
  vs 550 in 013) — all rows above are one back-to-back run; compare within the
  table, not across entries.
- `--scan-bits` default is 0 (full) — no behavior change unless set.
