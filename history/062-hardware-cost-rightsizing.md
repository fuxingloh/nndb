# 062 — Hardware cost right-sizing: throughput is physical cores × popcount width

The engine had only ever run on one box (Intel Granite Rapids, the old ~960-QPS
headline). This entry runs the *same* engine across seven instances — three vendors ×
two families plus an Apple-Silicon ringer — to answer one question: **at a fixed spec,
which silicon gives the most QPS per dollar?**

## Method — everything held constant except the silicon

Identical committed code, identical Cohere v3 (1M × 1024) data, identical flags on every
box: the shipped 1-bit funnel (`--rotate 2 --residual --batch 8 --rerank 500`),
built with `-C target-cpu=native` so each CPU autovectorizes to its
*widest* popcount (AVX-512 `VPOPCNTDQ` on Intel/AMD, NEON `CNT` on Graviton/Apple —
verified in the disassembly). Every box is the **advised `.2xlarge` (8 vCPU)** instance —
exactly the unit AWS sells; the physical-core counts below are just what's *under* that
8-vCPU label. Because nothing varies but the chip, **recall is a constant 0.9952 across
all seven** — the comparison is purely QPS, p50, and QPS-per-dollar. (`$/hr` = us-east-2
on-demand, one region for all, to strip the regional pricing of the AMD boxes — which only
run in Tokyo — from the comparison.)

## Result

All are 8-vCPU instances; "phys" is what's under that vCPU label (Intel = 4 cores + SMT).
Cost is us-east-2 on-demand, annualized (24×365h).

| instance (8 vCPU) | vendor | vCPU | phys | popcount | funnel QPS | p50 | $/yr | **QPS per $1k/yr** | QPS/vCPU |
|---|---|---|---|---|---|---|---|---|---|
| **c8a.2xlarge** | AMD Zen5 | 8 | 8 | AVX-512 | **2310** | 3.1 ms | $3,776 | **612** | 289 |
| m8a.2xlarge | AMD Zen5 | 8 | 8 | AVX-512 | 2313 | 3.1 ms | $4,265 | 542 | 289 |
| c8g.2xlarge | Graviton4 | 8 | 8 | NEON | 986 | 6.6 ms | $2,785 | 354 | 123 |
| m8g.2xlarge | Graviton4 | 8 | 8 | NEON | 981 | 6.8 ms | $3,145 | 312 | 123 |
| c8i.2xlarge | Intel Granite | 8 | 4 | AVX-512 | 934 | 10.6 ms | $3,283 | 285 | 117 |
| m8i.2xlarge | Intel Granite | 8 | 4 | AVX-512 | 924 | 11.1 ms | $3,709 | 249 | 116 |
| mac2.metal | Apple M1 | 8 | 8 | NEON | 712 | 5.7 ms | $5,681 | 125 | 89 |

The kicker is **QPS/vCPU**: you pay per vCPU, and an AMD vCPU does 289 QPS while an Intel
vCPU does 117 — *2.5× less for the same line item* — because half of Intel's vCPUs are SMT
threads sharing four real cores.

**AMD Zen5 wins outright** — best throughput, best latency, best $/QPS — and the old
Intel Granite headline (934, matching c8i exactly) is *2.5× off the frontier*. Compute vs
general (c vs m) is a wash within every vendor: same silicon, and the ~4 GB working set
fits both, so the extra RAM buys nothing.

## Why — two multiplicative levers, no magic

The funnel is compute-bound on popcount, so:

```
funnel QPS  ≈  C_isa · P / N
```

with `P` = **physical cores**, `N` = vectors scanned per query, and `C_isa` the per-core
rate measured here (1024-dim, with rerank): **AVX-512 core ≈ 2.9e8** (AMD) / 2.3e8
(Intel, it downclocks a touch under heavy AVX-512); **NEON core ≈ 1.2e8** (Graviton4) /
0.9e8 (M1). Two findings make the constant concrete:

**1. `P` is physical cores, not vCPUs — SMT is a mirage.** The "8 vCPU" tier is 8 real
cores on AMD/Graviton but **4 cores + hyperthreading** on Intel. A thread-scaling sweep
on c8i settled it:

```
1 thread → 211   2 → 413 (1.96×)   4 → 850 (4.03×)   8 → 808 (−5%)
```

Dead-linear to 4 physical cores, then the SMT second thread *regresses* — it contends for
the same execution units the kernel already saturates. So Intel's 8-vCPU box does the work
of 4 cores; AMD's does 8. That alone is the 2× gap.

**2. Popcount width is the other factor.** Graviton4 has the *same* 8 cores as AMD yet
~half the funnel QPS (986 vs 2310) — because NEON `CNT` is 128-bit vs AVX-512
`VPOPCNTDQ`'s 512-bit. Per-core: AVX-512 ≈ 2× NEON. This is "lever two" (the instruction,
not the math) showing up as a purchasing decision.

So `QPS ≈ cores × popcount-width`. AMD wins by maxing both (8 × 512-bit). Intel:
4 × 512. Graviton4: 8 × 128. Apple M1 (the for-the-LOLs floor): 8 × 128 on a narrower
consumer core, and on a dedicated host its $/QPS is 5× worse than c8a.

## The right-sizing rule

`QPS/$ ≈ C_isa · P / (N · price)` → **maximize `cores × popcount-width` per dollar**:

- **Buy AVX-512 physical cores, not vCPUs.** The densest-real-cores-per-dollar instance
  with `VPOPCNTDQ` wins. Here that's the AMD c-family.
- **Don't buy RAM you won't use.** The hot scan footprint is 128 MB (codes); even with the
  f32 rerank store it's ~4 GB, far under any 2xlarge. Compute-optimized (least RAM/core,
  cheapest) beats general/memory — the spare DRAM on m/r is wasted spend.
- **The RAM slack is the IVF lever.** A box is CPU-bound with ~12–28 GB idle, so size the
  two axes independently: **RAM holds the whole shard's cells** (all resident, cheap),
  **cores scan only the `nprobe` cells a query touches**. Capacity is RAM-bound (huge);
  throughput is core-bound (`P = QPS_target · N_scanned / C_isa`). e.g. 10k QPS at N=1M on
  AVX-512 → ~35 cores ≈ 5× c8a, whose 80 GB could hold ~19× more corpus than it can scan
  hot. Scale cores for QPS; let spare RAM absorb more cells.

This updates the project's headline: on the best-value box the shipped engine does
**~2310 QPS at 3.1 ms / 0.995 recall** — the old ~960 was an Intel-core-count artifact,
not the engine's ceiling.
