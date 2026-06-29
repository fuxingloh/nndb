#!/usr/bin/env bash
# Measure the current build and write a perf record for one notes entry.
# Usage:  notes/measure.sh notes/001-exact-brute-force-baseline.json brute-force
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: measure.sh <output.json relative to repo root> <label>}"
LABEL="${2:-brute-force}"

cd "$ROOT/nndb"
cargo build --release -q

JSON="$(./target/release/vsearch \
  --queries 1000 --latency-queries 200 --k 10 \
  --label "$LABEL" --json)"

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '%s' "$JSON" | python3 -c \
  "import json,sys; o=json.load(sys.stdin); print(json.dumps({'date':'$DATE','commit':'$COMMIT',**o}, indent=2))" \
  > "$ROOT/$OUT"

echo "wrote $OUT"
cat "$ROOT/$OUT"
