# 052 — Carousel × disk: the scan shares, the rerank doesn't

Perf record: [`052-carousel-disk-stack.json`](./052-carousel-disk-stack.json). Granite
box (8 vCPU, 15 GB RAM, EBS gp3). `src/bin/carousel.rs --disk`. Cohere 1M × 1024,
grouped fan=4, Poisson rate=400, 8 workers. Stacks the **serving** layer (carousel,
039–041) with the **storage** layer (disk-resident f32, 045): 1-bit codes stay in RAM
for the shared scan; the f32 rerank store is mmap'd from SSD.

## Result

| store | C | throughput | p50 | note |
|---|---|---|---|---|
| RAM | 1000 | 386 | **6.05 ms** | baseline |
| disk (cold) | 1000 | 386 | **178,673 ms** | death-spiral backlog |
| disk (warm) | 1000 | 386 | **6,052 ms** | still overloads |
| **disk (warm)** | **128** | 385 | **4.68 ms** | **viable** |

## The finding: the carousel shares the scan, but rerank is per-query

The carousel's whole trick is **scan-sharing** — all riders ride one base scan. But the
**rerank is not shared**: each query independently reads its C candidate vectors. On
disk that's **C random SSD reads per query**, and at C=1000 under 400 QPS / 8 workers it
can't keep up → the queue explodes (cold: 178 s; even warm: 6 s — 1000 reads × any
cache-miss penalty, amplified 8× over C=128, tips into overload).

Drop to **C=128 and disk is fine — 4.68 ms p50, even beating RAM C=1000** (fewer rerank
ops). So disk-resident serving is viable *only at small C*.

## What this means for the stack

1. **Disk serving requires small C** → this is the concrete motivation for **adaptive-C
   (048)**: keep the per-query read count low (and spend it only where the Hamming
   margin says it's needed).
2. **Small C costs recall** → which is exactly what **rotation + residual (051)** buy
   back. So you run small C on disk *and* hold recall via better codes.
3. **The 32× committed-RAM saving only realizes when corpus > RAM.** When the dataset
   fits (4 GB < 15 GB here), the page cache holds the f32 anyway (RSS ~full) — the
   *committed* (non-evictable) footprint is the 128 MB of codes vs RAM-mode's 4.2 GB,
   but the OS still uses the spare RAM as cache. The saving bites when you genuinely
   can't fit the f32 — and there, cache misses make small-C/adaptive-C mandatory.

## Conclusion

The naive stack (carousel + disk + C=1000) **fails** — the unshared per-query rerank is
the disk bottleneck. The **coherent production stack** is **carousel + disk + small/
adaptive-C + rotation + residual**: the carousel shares the scan and bounds tail
latency, disk cuts committed RAM, small/adaptive-C keeps the unshared rerank affordable
on SSD, and rotation+residual hold recall at that small C. Every piece we built has a
role; they only compose in the *right* configuration.

## Caveats

- "Warm" = `cat` the f32 file first; under memory pressure the 4 GB isn't fully pinned,
  so C=1000 still sees enough misses to overload — the qualitative result (small C
  mandatory on disk) holds regardless.
- EBS gp3; local NVMe (lower read latency + higher IOPS) would raise the viable-C
  ceiling but not change the "rerank is the unshared disk cost" structure.
- Recall not measured here (serving/latency study, fixed plain-binary scan); the recall
  side is 051.
