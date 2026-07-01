#!/usr/bin/env bash
# OpenAI text-embedding-3-large Matryoshka recall-vs-bits experiment — run ON a box.
#
# Downloads PRECOMPUTED OpenAI text-embedding-3-large vectors (Qdrant's 1536-d
# dbpedia set — no embedding to run), slices each Matryoshka dim, then runs the
# 1-bit binary funnel + f32 rerank per dim. Answers: how does 256-bit (and
# 512/768/…) binary-quant + full-precision rerank behave on a REAL Matryoshka
# embedding — QPS, p50, p99, recall — on this box's CPU.
#
# Usage on the box (repo root):  bash notes/run-openai-bench.sh [label]
# Emits one OAI_JSON= line per dim to stdout.
#
# Env knobs: DIMS="1536 1024 768 512 256 128 64"  TARGET=1000000  QUERIES=10000  RERANK=500
set -euo pipefail
LABEL="${1:-$(uname -m)}"
DIMS="${DIMS:-1536 1024 768 512 256 128 64}"
TARGET="${TARGET:-1000000}"; QUERIES="${QUERIES:-10000}"; RERANK="${RERANK:-500}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "################ TOOLCHAIN ################"
if ! command -v cargo >/dev/null 2>&1; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; fi
. "$HOME/.cargo/env"
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  sudo dnf -y install gcc || (sudo apt-get update && sudo apt-get install -y build-essential); fi
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export RUSTFLAGS="-C target-cpu=native"

DATA="$ROOT/nndb/data/oai"; mkdir -p "$DATA"
cd "$ROOT/nndb"

echo "################ DATASET (precomputed — download + slice, NO embedding) ################"
if [ ! -f "$DATA/oai256_base.fvecs" ]; then
  uv run --python 3.12 --with datasets --with numpy \
    "$ROOT/nndb/scripts/fetch-openai-dbpedia.py" --target "$TARGET" --queries "$QUERIES" \
    --dims "$(echo "$DIMS" | tr ' ' ',')" --out "$DATA"
fi

echo "################ BUILD (target-cpu=native) ################"
cargo build --release --bin vsearch -q

is_pow2() { local n="$1"; [ "$n" -gt 0 ] && [ $(( n & (n - 1) )) -eq 0 ]; }

echo "################ FUNNEL per Matryoshka dim (QPS / p50 / p99 / recall) ################"
for D in $DIMS; do
  P="oai${D}"
  [ -f "$DATA/${P}_base.fvecs" ] || { echo "skip $P (no data)"; continue; }
  if [ ! -f "$DATA/${P}_groundtruth.ivecs" ]; then
    ./target/release/vsearch --data data/oai --prefix "$P" \
      --write-ground-truth "$DATA/${P}_groundtruth.ivecs" --gt-k 100
  fi
  # binary funnel + f32 rerank. Rotation (FWHT) needs a power-of-2 dim; others run plain.
  is_pow2 "$D" && ROT="--rotate 2 --residual --batch 8" || ROT="--batch 8"
  J="$(./target/release/vsearch --data data/oai --prefix "$P" \
        --queries 2000 --latency-queries 200 --k 10 --reps 3 --json \
        --quant binary $ROT --rerank "$RERANK" --label "${LABEL} ${P}")"
  echo "OAI_JSON=${J}"
done
echo "################ DONE [$LABEL] ################"
