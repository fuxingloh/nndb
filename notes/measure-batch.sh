#!/usr/bin/env bash
# Sweep query batch (tile) size and record QPS/recall at each.
# Usage:  notes/measure-batch.sh notes/003-query-batching.json brute-force
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: measure-batch.sh <output.json relative to repo root> <label>}"
LABEL="${2:-brute-force}"

cd "$ROOT/nndb"
cargo build --release -q

TMP="$(mktemp)"
for B in 1 4 8 16 32 64; do
  ./target/release/vsearch \
    --queries 1000 --latency-queries 30 --k 10 --batch "$B" \
    --label "$LABEL" --json >> "$TMP"
done

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 0)"

python3 - "$TMP" > "$ROOT/$OUT" <<PY
import json, sys
sweep = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(json.dumps({
    "date": "$DATE", "commit": "$COMMIT", "label": "$LABEL",
    "cores": $CORES, "experiment": "query-batching (tiled scan)", "sweep": sweep,
}, indent=2))
PY
rm -f "$TMP"
echo "wrote $OUT"
cat "$ROOT/$OUT"
