# 065 — Matryoshka-256 binary funnel + rerank (OpenAI text-embedding-3)

Perf record: [`065-matryoshka-256-openai-funnel.json`](./065-matryoshka-256-openai-funnel.json).
c8a.4xlarge (Zen5, 16 vCPU) spot, us-east-2. First run on a *genuine* Matryoshka
embedding — closes the question left open by the prefix experiments (014/027): Cohere
v3 isn't Matryoshka, so truncating it to 256 collapsed recall to ~0.72. Here we use an
embedding *trained* to be truncatable.

## Setup

- **Dataset:** OpenAI `text-embedding-3-large`, precomputed (Qdrant's dbpedia-entities
  1M set — no embedding to run). text-embedding-3-large is MRL-trained (256 is a
  documented operating point). We slice the native 1536-d vectors to **256** and
  L2-renormalize (OpenAI's Matryoshka recipe). 990k base + 10k queries, exact GT.
- **Engine:** the shipped 1-bit binary funnel — sign-bit codes (256 bits = 32 B/vec),
  rotation ×2 + residual (both free recall), tile=8 — then exact f32 rerank of the
  top-C shortlist. Same code path as the Cohere runs; only the data differs.

## Result — recall dials from 0.95 to 0.998 on the rerank width

| rerank C | recall@10 | QPS | p50 | p99 |
|---|---|---|---|---|
| 500 | 0.9474 | 6041 | 2.05 ms | 2.10 ms |
| 1000 | 0.9731 | 5275 | 2.41 ms | 2.49 ms |
| **2000** | **0.9878** | **4227** | **3.09 ms** | **3.15 ms** |
| 4000 | 0.9944 | 3092 | 4.40 ms | 4.58 ms |
| 8000 | 0.9979 | 2026 | 6.54 ms | 6.88 ms |

**Chosen operating point: C=2000 — recall 0.9878, 4227 QPS, p50 3.1 ms, p99 3.15 ms.**
(+4 recall points over C=500 for ~30% fewer QPS and <1 ms more latency.)

## What it shows

- **Matryoshka holds — no collapse.** A non-Matryoshka 256-prefix capped low (0.72,
  the "bit-floor" dead end). Here recall climbs smoothly to **0.998**: the 256-bit
  code + rotation/residual produces a high-quality Hamming shortlist, so the true
  neighbours *are* in the top-C and rerank recovers them. 256-bit + rerank is a real,
  tunable operating point on a Matryoshka embedding.
- **256 bits is compact and fast** — 31.7 MB of codes for 990k vectors; even at
  0.998 recall it holds ~2000 QPS at ~6.5 ms p50.

## Caveats

- **Box size differs from the hardware sweep.** This ran on a **16-vCPU** c8a.4xlarge;
  the Cohere sweep (062/064) used **8-vCPU** .2xlarge. So the QPS here is not directly
  comparable to the ~2310 QPS Zen5 Cohere-1024 number. Recall is CPU-independent, so
  the recall figures are final; the throughput needs an equal-box re-run to compare.
- One dataset/embedding (OpenAI text-embedding-3-large). The *model* is closed; only
  the vectors are open. A fully-open reproduction would embed with Nomic v1.5 (MRL).
- Only the 256 dim measured here; the full 512/768/1024/1536 recall-vs-bits curve on
  the same box was not run.
