#!/usr/bin/env bash
# Snowflake arctic-embed-m-v1.5 Matryoshka-256 funnel at 10M scale — run ON a box.
#
# Downloads PRECOMPUTED arctic-embed-m-v1.5 vectors (Snowflake's MSMARCO v2.1 71M
# set — no embedding to run), slices 10M+10k to 256-d, then runs the 1-bit binary
# funnel + f32 rerank across a rerank-C sweep. Answers: at 10M vectors the 256-bit
# codes are 320 MB (>> LLC) — where does the funnel sit when stage 1 is genuinely
# streaming from DRAM, vs the 1M runs (065/066) where codes fit in cache?
#
# Usage on the box (repo root):  bash notes/run-snowflake-bench.sh [label]
# Emits one ARCTIC_JSON= line per rerank width to stdout.
#
# Env knobs: DIM=256  TARGET=10000000  QUERIES=10000  RERANKS="500 1000 2000 4000 8000"
set -euo pipefail
LABEL="${1:-$(uname -m)}"
DIM="${DIM:-256}"
TARGET="${TARGET:-10000000}"; QUERIES="${QUERIES:-10000}"
RERANKS="${RERANKS:-500 1000 2000 4000 8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "################ TOOLCHAIN ################"
if ! command -v cargo >/dev/null 2>&1; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; fi
. "$HOME/.cargo/env"
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  sudo dnf -y install gcc || (sudo apt-get update && sudo apt-get install -y build-essential); fi
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export RUSTFLAGS="-C target-cpu=native"

DATA="$ROOT/nndb/data/snowflake"; mkdir -p "$DATA"
cd "$ROOT/nndb"
P="arctic${DIM}"

echo "################ DATASET (precomputed — stream 10M of 71M, slice, NO embedding) ################"
if [ ! -f "$DATA/${P}_base.fvecs" ]; then
  uv run --python 3.12 --with "huggingface_hub[hf_transfer]" --with pyarrow --with numpy \
    "$ROOT/nndb/scripts/fetch-snowflake.py" --target "$TARGET" --queries "$QUERIES" \
    --dims "$DIM" --out "$DATA"
fi

echo "################ BUILD (target-cpu=native) ################"
cargo build --release --bin vsearch -q

echo "################ GROUND TRUTH (exact KNN over ${TARGET} — this is the slow part) ################"
if [ ! -f "$DATA/${P}_groundtruth.ivecs" ]; then
  ./target/release/vsearch --data data/snowflake --prefix "$P" \
    --write-ground-truth "$DATA/${P}_groundtruth.ivecs" --gt-k 100
fi

echo "################ FUNNEL rerank-C sweep (QPS / p50 / p99 / recall) ################"
for C in $RERANKS; do
  J="$(./target/release/vsearch --data data/snowflake --prefix "$P" \
        --queries 2000 --latency-queries 200 --k 10 --reps 3 --json \
        --quant binary --rotate 2 --residual --batch 8 --rerank "$C" \
        --label "${LABEL} ${P} C=${C}")"
  echo "ARCTIC_JSON=${J}"
done
echo "################ DONE [$LABEL] ################"
