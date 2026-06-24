#!/usr/bin/env bash
# Drive the hardware sweep FROM the M3 (we have the SSH key; the boxes don't talk
# to each other). For each "label=ip": rsync the committed source up (NOT the
# 8 GB data/ or target/), run the per-box bench, stream its log, and save the two
# JSON results under history/hw/.
#
# Usage:
#   bash history/sweep-hardware.sh \
#     c8i.2xlarge=1.2.3.4  c8a.2xlarge=1.2.3.5  c8g.2xlarge=1.2.3.6 \
#     m8i.2xlarge=1.2.3.7  m8a.2xlarge=1.2.3.8  m8g.2xlarge=1.2.3.9
#
# Env: KEY=~/.ssh/vps-bench.pem (default), USER=ec2-user (default).
set -uo pipefail

KEY="${KEY:-$HOME/.ssh/vps-bench.pem}"
SSH_USER="${USER_OVERRIDE:-ec2-user}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/history/hw"; mkdir -p "$OUT"
SSHOPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30)

one() {
  local label="${1%%=*}" ip="${1#*=}"
  echo "==================== $label @ $ip ===================="
  # Push committed source only; data/target/node_modules excluded (boxes self-fetch data).
  rsync -az --delete \
    --exclude 'database/data' --exclude 'database/target' --exclude 'web/node_modules' \
    --exclude 'web/.next' --exclude 'infra/node_modules' --exclude 'infra/cdk.out' \
    --exclude '.git' \
    -e "ssh ${SSHOPTS[*]}" "$ROOT/" "${SSH_USER}@${ip}:~/vps/" || { echo "RSYNC FAILED $label"; return 1; }
  # Run the bench; tee full log, keep the two JSON lines.
  ssh "${SSHOPTS[@]}" "${SSH_USER}@${ip}" \
    "cd ~/vps && bash history/run-cohere-bench.sh '$label'" 2>&1 \
    | tee "$OUT/${label}.log" \
    | grep -E '^(EXACT_JSON|FUNNEL_JSON)=' \
    | sed -E 's/^(EXACT|FUNNEL)_JSON=//' > "$OUT/${label}.jsonl"
  echo "saved -> history/hw/${label}.jsonl ($(wc -l < "$OUT/${label}.jsonl") lines)"
}

# Run all boxes in parallel (independent; each ~20-30 min including data fetch).
pids=()
for spec in "$@"; do one "$spec" & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "==================== SUMMARY ===================="
for spec in "$@"; do
  label="${spec%%=*}"
  echo "--- $label ---"; cat "$OUT/${label}.jsonl" 2>/dev/null || echo "(no results)"
done
exit $fail
