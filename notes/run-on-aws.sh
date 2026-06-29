#!/usr/bin/env bash
# Bootstrap + baseline benchmark on a fresh Linux box (Amazon Linux 2023 / Ubuntu).
# Run from the repo root ON THE BOX:  bash notes/run-on-aws.sh [label]
#
# Builds with -C target-cpu=native so the kernel autovectorizes to the WIDEST
# SIMD this CPU supports (AVX-512 on Sapphire Rapids/Zen, NEON on Graviton).
set -euo pipefail

LABEL="${1:-$(uname -m)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================ CPU ================"
if command -v lscpu >/dev/null; then lscpu | grep -E "Model name|Architecture|^CPU\(s\)|^Thread|^Core"; fi
echo -n "SIMD ISA present: "
grep -o -m1 -E 'avx512f|avx2|asimd|neon' /proc/cpuinfo 2>/dev/null || echo "(unknown)"
echo "(avx512 flags:)"; grep -o -m1 -E 'avx512[a-z]+' /proc/cpuinfo 2>/dev/null | tr '\n' ' ' || true; echo

echo "================ toolchain ================"
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
. "$HOME/.cargo/env"
if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
  if command -v dnf >/dev/null; then
    sudo dnf -y install gcc
  else
    sudo apt-get update && sudo apt-get install -y build-essential
  fi
fi

echo "================ dataset ================"
if [ ! -f "$ROOT/nndb/data/sift/sift_base.fvecs" ]; then
  bash "$ROOT/nndb/scripts/download-sift.sh"
fi

echo "================ build (target-cpu=native) ================"
export RUSTFLAGS="-C target-cpu=native"   # inherited by measure-batch.sh's build too
cd "$ROOT/nndb"
cargo build --release

echo "================ batch sweep ================"
bash "$ROOT/notes/measure-batch.sh" "notes/004-${LABEL}-baseline.json" "$LABEL"

echo "================ done -> notes/004-${LABEL}-baseline.json ================"
