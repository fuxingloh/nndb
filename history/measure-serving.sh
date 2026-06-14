#!/usr/bin/env bash
# Start the in-memory server, run a concurrency sweep through the network, and
# write a serving perf record (user-facing latency under load).
# Usage:  history/measure-serving.sh history/002-what-we-did.json brute-force
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: measure-serving.sh <output.json relative to repo root> <label>}"
LABEL="${2:-brute-force}"
ADDR="127.0.0.1:8080"

cd "$ROOT/database"
cargo build --release -q

./target/release/server --data data/sift --addr "$ADDR" > /tmp/vsearch-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://$ADDR/health" >/dev/null 2>&1 && break
  sleep 1
done

TMP="$(mktemp)"
for c in 1 8 16 32; do
  reqs=$(( c < 4 ? 200 : 1000 ))
  ./target/release/loadtest --url "http://$ADDR/search" --data data/sift \
    --concurrency "$c" --requests "$reqs" --warmup 50 --k 10 --label "$LABEL" --json >> "$TMP"
done

kill $SRV 2>/dev/null || true

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 0)"

python3 - "$TMP" > "$ROOT/$OUT" <<PY
import json, sys
sweep = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(json.dumps({
    "date": "$DATE", "commit": "$COMMIT", "label": "$LABEL",
    "transport": "http", "cores": $CORES, "sweep": sweep,
}, indent=2))
PY
rm -f "$TMP"
echo "wrote $OUT"
cat "$ROOT/$OUT"
