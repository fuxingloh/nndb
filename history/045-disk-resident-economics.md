# 045 — Disk-resident vectors: how slow without RAM, and the economics

Perf record: [`045-disk-resident-economics.json`](./045-disk-resident-economics.json).
Granite box (Xeon 6975P-C, 8 vCPU, **15 GB RAM**, EBS gp3 root nvme). `src/bin/disk.rs`,
`scripts/disk-bench.sh`. Cohere v3 1M × 1024 f32 = **4.096 GB**. Single-thread,
serial per-query latency (so cross-config *ratios* are the result, not absolute QPS —
the real engine is parallel/tiled, history 038).

Measured cold disk bandwidth: **135 MB/s** (`cat` the 4 GB store cold in 30.24 s).

## The question

For exact "matrix" search you eventually need everything in RAM — is that right,
and how slow is it if **nothing** is in RAM? And can we trade RAM (expensive) for
SSD (cheap) to scale down cost?

## Result 1 — exact search confirms: it MUST be in RAM

| config | RAM resident | p50 latency |
|---|---|---|
| exact / RAM | 4096 MB | **709 ms** |
| exact / disk **cold** | 0 | **30,200 ms** (30 s!) |
| exact / disk warm (cached) | (cache) | 703 ms |

Exact search reads **every** vector per query. Cold from disk that's the whole
4 GB / 135 MB/s = **30 s per query** — exactly bandwidth-bound, ~42× slower than
in-RAM. So yes: for full-scan exact search, nothing-in-RAM is a non-starter. (Even
*in* RAM, single-thread exact is 709 ms because it streams 4 GB/query — that cost is
the entire reason the funnel exists.)

## Result 2 — the binary funnel breaks the RAM requirement, cheaply

The funnel scans only the **128 MB of 1-bit codes** and reads only the **C=200**
rerank vectors per query. So the 4 GB of f32 can live on SSD:

| config | RAM resident | p50 (cold) | p50 (warm) |
|---|---|---|---|
| funnel / RAM | 4224 MB | — | **11.7 ms** |
| funnel / **hybrid** (codes RAM, f32 disk) | **128 MB** | **92 ms** | **11.5 ms** |
| funnel / all-disk (codes+f32 mmap) | ~0 | — | 10.2 ms |

- **32× less RAM** (128 MB vs 4.2 GB) and still **7.7× faster than even exact-in-RAM**,
  330× faster than exact-cold-disk.
- **Cold cost is just the C random reads:** ~80 ms for 200 reads ≈ 0.4 ms/read serial
  (EBS random 4 KB). Parallelizing the rerank reads (io_uring / threads — NVMe loves
  queue depth) would cut this sharply; it wasn't done here.
- **Warm = full-RAM speed at 128 MB committed.** Because 4 GB < 15 GB RAM, the OS
  page-caches the f32 after first touch — so hybrid converges to 11.5 ms = identical
  to all-in-RAM, while you only *commit* 128 MB. Spare RAM becomes free cache.

## The scale-out unlock: O(C) vs O(N) disk cost

This is the crux for cost-driven scaling:

- **Exact on disk = O(N):** reads the whole dataset per query. 10× the data → 10× the
  cold latency (30 s → 300 s). Hopeless.
- **Funnel on disk = O(C):** reads C=200 vectors per query **regardless of N**. 10× the
  data → cold latency basically unchanged (~90 ms); only the in-RAM code scan grows
  (and codes are 1/32 the bytes).

So to cut cost by pushing vectors to SSD, the funnel is the *only* one of the two
that survives: its disk traffic is independent of dataset size.

## Economics

RAM is ~40–50× more expensive per GB than EBS gp3. Per-vector storage cost for
Cohere drops accordingly: full-RAM commits 4.2 GB of RAM; hybrid commits 128 MB of
RAM + 4 GB of (cheap) disk — roughly a **~20× lower storage bill** for the same
result, free if the working set fits page cache, and a bounded ~90 ms cold penalty
(further reducible with parallel reads) when it doesn't. The funnel's two-tier
shape (tiny hot codes / big cold vectors) is what makes vectors-on-SSD viable.

## Conclusions

1. **Exact/matrix search must be RAM-resident** — cold disk is 30 s/query (O(N)
   bandwidth). The user's intuition is correct *for exact search*.
2. **The binary funnel removes that requirement**: only the 128 MB of codes must be
   hot; the 4 GB of f32 can sit on SSD and be read C times/query. 32× less committed
   RAM, full-RAM speed when cached, bounded cold penalty when not.
3. **Funnel disk traffic is O(C), independent of N** — the property that makes
   cost-driven scale-down (RAM→SSD) actually work as the corpus grows.

## Caveats

- Single-thread serial latency; the production engine is parallel/tiled (~900 QPS,
  038). Ratios transfer; absolute QPS doesn't.
- Dataset (4 GB) < RAM (15 GB), so the page cache flatters warm numbers — that *is*
  the realistic small-corpus economics, but the genuinely disk-bound regime is
  dataset > RAM, untested here (needs a bigger corpus or O_DIRECT).
- EBS gp3 @ 135 MB/s; local NVMe instance storage is ~10–30× faster and would shrink
  both the 30 s exact-cold and the 0.4 ms/read rerank cost.
- `funnel/all-disk` writes the codes file during the run, so its codes weren't truly
  cold; its number reflects cached-codes mmap. The exact-cold (30 s) is the clean
  "nothing in RAM" measurement.
