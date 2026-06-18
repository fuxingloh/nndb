#!/usr/bin/env bash
# Disk-economics sweep: run each mode with a CONTROLLED page-cache state.
# "cold" = drop the OS page cache first (needs sudo) so mmap faults from disk;
# "warm" = run again immediately so the file is cached. Run ON THE BOX.
#
#   bash scripts/disk-bench.sh data/cohere cohere
set -euo pipefail
DATA="${1:-data/cohere}"
PREFIX="${2:-cohere}"
BIN=./target/release/disk
. "$HOME/.cargo/env" 2>/dev/null || true

drop() { sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }

echo "===== raw disk read bandwidth (cold) ====="
drop
# time reading the flat f32 file straight through (created on first disk.rs run)
if [ -f "$DATA/${PREFIX}.f32raw" ]; then
  SZ=$(stat -c %s "$DATA/${PREFIX}.f32raw")
  T=$( { /usr/bin/time -f "%e" sh -c "cat '$DATA/${PREFIX}.f32raw' > /dev/null"; } 2>&1 )
  echo "f32raw bytes=$SZ  cat_seconds=$T  ->  $(awk "BEGIN{printf \"%.0f MB/s\", $SZ/1e6/$T}")"
fi

run() { # mode queries [cold|warm]
  local mode="$1" q="$2" state="${3:-warm}"
  [ "$state" = cold ] && drop
  echo "----- $mode ($state, q=$q) -----"
  $BIN --data "$DATA" --prefix "$PREFIX" --mode "$mode" --queries "$q" --c 200
}

echo; echo "===== EXACT (touches all N per query) ====="
run exact-ram    200 warm
run exact-disk     3 cold      # first_ms = cold full-dataset scan; rest warm from cache
run exact-disk   200 warm      # now cached -> RAM speed

echo; echo "===== FUNNEL (scan codes, rerank C) ====="
run funnel-ram    1000 warm
run funnel-hybrid 1000 cold    # codes in RAM, f32 faulted from disk
run funnel-hybrid 1000 warm    # f32 now cached
run funnel-disk    500 cold    # codes AND f32 from disk: nothing in RAM
run funnel-disk    500 warm

echo "DONE"
