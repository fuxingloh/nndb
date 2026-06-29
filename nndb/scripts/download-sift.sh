#!/usr/bin/env bash
# Download SIFT1M (1M base vectors x 128 dim) from the corpus-texmex source.
# Extracts to nndb/data/sift/{sift_base.fvecs,sift_query.fvecs,sift_groundtruth.ivecs,sift_learn.fvecs}
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

if [ ! -f sift.tar.gz ]; then
  echo "Downloading SIFT1M (~168 MB) from $URL"
  # If ftp is blocked on your network, grab sift.tar.gz manually from
  # http://corpus-texmex.irisa.fr/ and drop it in nndb/data/
  curl -L --fail -o sift.tar.gz "$URL"
fi

echo "Extracting..."
tar xzf sift.tar.gz

echo "Done. Files in $DATA_DIR/sift:"
ls -la "$DATA_DIR/sift"
