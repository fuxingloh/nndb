#!/usr/bin/env bash
# Detect compute-bound vs memory-bound WITHOUT hardware counters, by sweeping the
# working-set size (base subset) from cache-resident to RAM and watching
# ns-per-distance. Flat across the cache->DRAM boundary => compute-bound; a step
# up => memory-bound past that size.
#
# Usage:  notes/measure-bound.sh notes/005-bound.json <label>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: measure-bound.sh <output.json relative to repo root> <label>}"
LABEL="${2:-brute-force}"

cd "$ROOT/nndb"
: "${RUSTFLAGS:=-C target-cpu=native}"; export RUSTFLAGS
cargo build --release -q

# Working-set sizes in vectors. 512 B/vector (128-dim f32):
#   1k=0.5MB(L1/L2) 8k=4MB(L2) 64k=32MB(L3) 256k=128MB 1M=488MB(DRAM)
SIZES="1000 8000 64000 256000 1000000"
TMP="$(mktemp)"
for N in $SIZES; do
  # Hold total work ~constant: fewer queries for big N, more for small N.
  Q=$(( 200000000 / N )); [ "$Q" -lt 50 ] && Q=50; [ "$Q" -gt 1000 ] && Q=1000
  ./target/release/vsearch \
    --base-subset "$N" --queries "$Q" --latency-queries 1 --k 10 \
    --reps 6 --label "$LABEL" --json >> "$TMP"
done

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 0)"

python3 - "$TMP" > "$ROOT/$OUT" <<PY
import json, sys
sweep = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ns = [s["ns_per_distance"] for s in sweep]
lo, hi = min(ns), max(ns)
ratio = hi / lo if lo > 0 else float("inf")
# Heuristic verdict: ns/distance roughly constant across cache->DRAM => compute.
if ratio < 1.3:
    verdict = "COMPUTE-BOUND (ns/distance flat across cache->DRAM; memory speed irrelevant)"
elif ratio < 2.0:
    verdict = "MIXED (ns/distance rises {:.2f}x; partly memory-sensitive)".format(ratio)
else:
    verdict = "MEMORY-BOUND (ns/distance rises {:.2f}x past cache)".format(ratio)
print(json.dumps({
    "date": "$DATE", "commit": "$COMMIT", "label": "$LABEL", "cores": $CORES,
    "experiment": "working-set sweep (cache->DRAM bound detector)",
    "ns_per_distance_ratio_max_over_min": round(ratio, 3),
    "verdict": verdict,
    "sweep": sweep,
}, indent=2))
PY
rm -f "$TMP"
echo "wrote $OUT"
python3 -c "import json;o=json.load(open('$ROOT/$OUT'));print('VERDICT:',o['verdict']);[print(f\"  N={s['n_base']:>8}  ws={s['memory_bytes']['index']/1048576:>7.1f}MB  ns/dist={s['ns_per_distance']:.4f}  CV={s['qps_cv']*100:.1f}%\") for s in o['sweep']]"
