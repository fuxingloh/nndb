#!/usr/bin/env bash
# Per-box hardware benchmark for the cost-right-sizing sweep (history 062).
#
# Runs ON a fresh Amazon Linux 2023 box (x86_64 OR arm64). Every box runs the
# IDENTICAL workload — same committed code, same dataset, same flags — so recall
# is held constant by construction and the only variables are the silicon and its
# spot price. We compare QPS, p50, and (downstream) QPS-per-dollar.
#
# Usage (on the box, from the repo root):  bash history/run-cohere-bench.sh <label>
#   <label> e.g. c8i.2xlarge  (free-form; just tags the JSON)
#
# Emits two JSON lines to stdout, prefixed EXACT_JSON= and FUNNEL_JSON=, plus a
# CPU identity block. The M3 driver (sweep-hardware.sh) parses these.
set -euo pipefail

LABEL="${1:-$(uname -m)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OS="$(uname -s)"
echo "################ CPU IDENTITY [$LABEL] ($OS) ################"
if [ "$OS" = "Darwin" ]; then
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  echo "cores: $(sysctl -n hw.physicalcpu) phys / $(sysctl -n hw.logicalcpu) logical"
  echo "mem:   $(( $(sysctl -n hw.memsize) / 1024 / 1024 )) MB"
  echo "vector ISA: NEON (Apple Silicon)"
else
  command -v lscpu >/dev/null && lscpu | grep -E "Architecture|Model name|^CPU\(s\)|Thread|Core|BogoMIPS|Flags" | head -20 || true
  echo -n "vector ISA: "
  grep -o -m1 -E 'avx512[a-z_]*|asimd|neon' /proc/cpuinfo 2>/dev/null | tr '\n' ' ' || echo "(unknown)"
  echo; echo "mem: $(grep MemTotal /proc/meminfo 2>/dev/null || true)"
fi

echo "################ TOOLCHAIN ################"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
. "$HOME/.cargo/env"
# C toolchain: macOS ships clang via Xcode CLT; Linux installs gcc.
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  if [ "$OS" = "Darwin" ]; then xcode-select --install 2>/dev/null || true
  else sudo dnf -y install gcc || (sudo apt-get update && sudo apt-get install -y build-essential); fi
fi
# uv for the deterministic dataset fetch (avoids system-python wheel gaps).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "################ DATASET (deterministic: same rows + GT on every box) ################"
DATA="$ROOT/database/data/cohere"
mkdir -p "$DATA"
cd "$ROOT/database"
export RUSTFLAGS="-C target-cpu=native"   # autovectorize to the widest ISA this CPU has

if [ ! -f "$DATA/cohere_base.fvecs" ]; then
  echo "fetching Cohere v3 1M+10k from HuggingFace (streamed, in order = reproducible) ..."
  uv run --python 3.12 --with datasets --with numpy \
    "$ROOT/database/scripts/fetch-cohere.py" --target 1000000 --queries 10000 --out "$DATA"
fi

echo "################ BUILD (target-cpu=native) ################"
# Only the benchmark bin — other bins (pq4simd/funnel3) need nightly portable_simd.
cargo build --release --bin vsearch -q

if [ ! -f "$DATA/cohere_groundtruth.ivecs" ]; then
  echo "writing exact ground truth (deterministic) ..."
  ./target/release/vsearch --data data/cohere --prefix cohere \
    --write-ground-truth data/cohere/cohere_groundtruth.ivecs --gt-k 100
fi

echo "################ WORKLOAD (identical flags on every box) ################"
# reps=3 → median of 3 (1st is warmup, discarded); 2000-query throughput pass.
run() {
  ./target/release/vsearch --data data/cohere --prefix cohere \
    --queries 2000 --latency-queries 200 --k 10 --reps 3 --json "$@"
}

# The shipped engine — 1-bit funnel, rotation+residual, tile=8, exact rerank.
# C=500 is the balanced operating point (~0.995 recall in 061). The exact f32 path
# is not measured here — only the funnel matters for the hardware comparison.
FUNNEL="$(run --quant binary --rotate 2 --residual --batch 8 --rerank 500 --label "${LABEL} funnel")"

echo "FUNNEL_JSON=${FUNNEL}"
echo "################ DONE [$LABEL] ################"
