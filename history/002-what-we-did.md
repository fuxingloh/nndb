# 002 — Networked serving + user-facing latency

Perf record: [`002-what-we-did.json`](./002-what-we-did.json)

## What we did

Added the interface layer and measured what a **client actually sees**, not just in-process compute.

- **`database/src/bin/server.rs`** — in-memory vector-search HTTP server. Loads the 1M vectors into RAM once; `POST /search` takes `{vector, k}` and returns the neighbor ids plus the server-side compute time. Each request is one single-threaded search; a semaphore caps concurrent searches at the core count, so excess load **queues** (that queue is real user-facing latency). The CPU-bound search runs on `spawn_blocking` so it never stalls the reactor.
- **`database/src/bin/loadtest.rs`** — closed-loop load generator at fixed concurrency, firing real SIFT query vectors and recording client-side end-to-end latency. It also reads the server's reported compute time, so each request splits into **compute** vs **interface+queue overhead**.
- **`history/measure-serving.sh`** — starts the server, runs a concurrency sweep, writes the record.

Focus per the brief: **user-facing latency under production-like traffic** — not startup/recovery time.

## Results (concurrency sweep, 8-core mac)

| concurrency | QPS | client p50 | client p99 | server compute p50 | interface+queue p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 49 ms | 52 ms | 49 ms | 0.2 ms |
| 8 (= cores) | **100** | 77 ms | 145 ms | 76 ms | 0.3 ms |
| 16 | 100 | 157 ms | 216 ms | 76 ms | 79 ms |
| 32 | 103 | 310 ms | 355 ms | 75 ms | 234 ms |

## Conclusions

1. **The interface layer is not the cost — compute and queuing are.** Up to core count, HTTP + JSON serialization adds **<0.4 ms** on top of a ~50–77 ms search. The network/API is noise here. This confirms the earlier reasoning: at tens of ms per query, the lever is the *algorithm*, not the transport.

2. **Throughput ceilings at ~100 QPS around concurrency = cores.** Past that, more load buys *zero* extra QPS — the 8 cores are already saturated. Adding clients only lengthens the queue.

3. **Beyond the ceiling, latency grows linearly with load, and it's pure queuing.** p50 goes 77 → 157 → 310 ms as concurrency goes 8 → 16 → 32, while *server compute stays ~76 ms*. The growth is entirely the "interface+queue" term (0.3 → 79 → 234 ms): requests waiting for a core. This is the classic latency-vs-load knee — the system is stable left of the knee (concurrency ≤ cores) and degrades right of it.

4. **Concurrent compute is slower than isolated compute (49 → 76 ms).** Even at exactly core-count concurrency, with no queuing, p50 compute rises from 49 ms (1 client) to 76 ms (8 clients). That's the **memory-bandwidth contention** from entry 001 surfacing as latency: 8 cores all streaming 488 MB compete for the same memory bus. So the effective throughput ceiling (~100 QPS) is well below 8 × single-core (~160 QPS).

5. **Implication for capacity planning:** to hold user-facing p50 near the floor (~50–77 ms), keep offered concurrency ≤ cores. To raise the QPS ceiling *and* drop latency, the answer isn't more cores (sublinear) or a faster transport (already negligible) — it's **touching less memory per query**, i.e. an approximate index.

## Next

Entry 003: first approximate index (IVF or HNSW). Re-run both harnesses — in-process (001) for the algorithm floor and this serving sweep (002) for user-facing latency — to show the recall/QPS/latency tradeoff end to end.
