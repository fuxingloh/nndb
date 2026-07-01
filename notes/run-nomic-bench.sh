#!/usr/bin/env bash
# Nomic v1.5 Matryoshka recall-vs-bits experiment — run ON a box, not a laptop.
#
# Embeds a text corpus (dbpedia-entity) with Nomic Embed v1.5 (open MRL model) once
# at several Matryoshka dims, then runs the 1-bit binary funnel + f32 rerank at each
# dim. Answers: how does 256-bit (and 512/128/64) binary-quant + full-precision
# rerank behave on a REAL Matryoshka embedding — the thing Cohere v3 couldn't give us.
#
# Usage on the box (repo root):  bash notes/run-nomic-bench.sh [label]
# Emits one NOMIC_JSON= line per dim to stdout (sweep-hardware.sh can parse these).
#
# Env knobs: DIMS="768 512 256 128 64"  TARGET=1000000  QUERIES=10000  RERANK=500
set -euo pipefail
LABEL="${1:-$(uname -m)}"
DIMS="${DIMS:-768 512 256 128 64}"
TARGET="${TARGET:-1000000}"; QUERIES="${QUERIES:-10000}"; RERANK="${RERANK:-500}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "################ TOOLCHAIN ################"
if ! command -v cargo >/dev/null 2>&1; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; fi
. "$HOME/.cargo/env"
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
  sudo dnf -y install gcc || (sudo apt-get update && sudo apt-get install -y build-essential); fi
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export RUSTFLAGS="-C target-cpu=native"   # autovectorize to the widest ISA this CPU has

DATA="$ROOT/nndb/data/nomic"; mkdir -p "$DATA"
cd "$ROOT/nndb"

echo "################ EMBED (Nomic v1.5 — all dims in one pass) ################"
# The heavy step. On a GPU box torch auto-uses CUDA; on a CPU box it uses all cores
# (slower — ~1-2h at 1M on a c8a.4xlarge). max_seq_length is capped inside the script.
if [ ! -f "$DATA/nomic256_base.fvecs" ]; then
  uv run --python 3.12 --with sentence-transformers --with datasets --with einops --with numpy \
    "$ROOT/nndb/scripts/fetch-nomic.py" --target "$TARGET" --queries "$QUERIES" \
    --dims "$(echo "$DIMS" | tr ' ' ',')" --out "$DATA"
fi

echo "################ BUILD (target-cpu=native) ################"
cargo build --release --bin vsearch -q

is_pow2() { local n="$1"; [ "$n" -gt 0 ] && [ $(( n & (n - 1) )) -eq 0 ]; }

echo "################ FUNNEL per Matryoshka dim (recall vs bits) ################"
for D in $DIMS; do
  P="nomic${D}"
  [ -f "$DATA/${P}_base.fvecs" ] || { echo "skip $P (no data)"; continue; }
  if [ ! -f "$DATA/${P}_groundtruth.ivecs" ]; then
    ./target/release/vsearch --data data/nomic --prefix "$P" \
      --write-ground-truth "$DATA/${P}_groundtruth.ivecs" --gt-k 100
  fi
  # binary funnel + f32 rerank. Rotation (FWHT) needs a power-of-2 dim; 768 runs plain.
  is_pow2 "$D" && ROT="--rotate 2 --residual --batch 8" || ROT="--batch 8"
  J="$(./target/release/vsearch --data data/nomic --prefix "$P" \
        --queries 2000 --latency-queries 200 --k 10 --reps 3 --json \
        --quant binary $ROT --rerank "$RERANK" --label "${LABEL} ${P}")"
  echo "NOMIC_JSON=${J}"
done
echo "################ DONE [$LABEL] ################"
