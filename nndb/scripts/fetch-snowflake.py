#!/usr/bin/env python3
"""Prepare precomputed Snowflake arctic-embed-m-v1.5 (Matryoshka) vectors as .fvecs.

NO embedding inference — Snowflake already embedded MSMARCO v2.1 (71M passages)
and published the 768-d vectors. arctic-embed-m-v1.5's headline feature IS the
MRL-trained 256-d truncation (their card benchmarks 256 ≈ 768 NDCG), so this is
the 10M-scale companion to fetch-openai-dbpedia.py: slice to each Matryoshka dim,
L2-normalize (the published vectors are UNNORMALIZED — normalizing is mandatory),
and study the 1-bit funnel + f32 rerank where the corpus no longer fits the LLC.

Scale changes the shape of this script vs the 1M fetchers: 10M x 768 f32 is ~31GB,
too big to materialize. So we download the ~5.5GB parquet shards ONE AT A TIME,
stream row-batches through slice+normalize, append straight to the .fvecs outputs,
and delete each shard before fetching the next (peak disk ~= 1 shard + outputs).

Ground truth is generated per dim by the Rust tool's --write-ground-truth (exact KNN).

Run via uv (no torch — just download + slice):
  uv run --python 3.12 --with "huggingface_hub[hf_transfer]" --with pyarrow --with numpy \
    nndb/scripts/fetch-snowflake.py --target 10000000 --queries 10000 --out nndb/data/snowflake
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Snowflake/msmarco-v2.1-snowflake-arctic-embed-m-v1.5")
ap.add_argument("--col", default="embedding")
ap.add_argument("--target", type=int, default=10_000_000, help="base vector count")
ap.add_argument("--queries", type=int, default=10_000)
ap.add_argument("--dims", default="256",
                help="Matryoshka dims to emit (comma-sep); slice the native 768-d vectors")
ap.add_argument("--out", default="nndb/data/snowflake")
ap.add_argument("--name", default="arctic")
ap.add_argument("--keep-shards", action="store_true", help="don't delete parquet shards after use")
a = ap.parse_args()

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

dims = [int(d) for d in a.dims.split(",") if d.strip()]
total = a.target + a.queries
base_n = a.target
os.makedirs(a.out, exist_ok=True)

shards = sorted(f for f in list_repo_files(a.repo, repo_type="dataset")
                if f.startswith("corpus/") and f.endswith(".parquet"))
print(f"{len(shards)} corpus shards in {a.repo}; need {total} rows")

# Incremental .fvecs writer: same flat [i32 dim][dim x f32] records as write_fvecs
# in the 1M fetchers, but append-per-chunk so we never hold the corpus in memory.
class FvecsWriter:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.path = path
        self.n = 0
    def append(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        n, dim = x.shape
        out = np.empty((n, dim + 1), dtype=np.int32)
        out[:, 0] = dim
        out[:, 1:] = x.view(np.int32)
        out.tofile(self.f)
        self.n += n
    def close(self):
        self.f.close()
        print(f"  wrote {self.path}: {self.n} vectors")

writers = {D: {"base": FvecsWriter(f"{a.out}/{a.name}{D}_base.fvecs"),
               "query": FvecsWriter(f"{a.out}/{a.name}{D}_query.fvecs")} for D in dims}

def emit(x, done):
    """Route a chunk of full-dim rows into per-dim base/query files.
    Rows [0, base_n) are base, [base_n, total) are queries — same
    first-N/last-nq split as the 1M fetchers, just streamed."""
    for D in dims:
        # arctic MRL recipe: slice to first D dims, then L2-normalize
        # (published vectors are unnormalized, so normalize even at native dim).
        s = x[:, :D].copy()
        s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
        nb = max(0, min(len(s), base_n - done))
        if nb:
            writers[D]["base"].append(s[:nb])
        if nb < len(s):
            writers[D]["query"].append(s[nb:])

done = 0
native = None
for shard in shards:
    if done >= total:
        break
    print(f"shard {shard} ({done}/{total}) ...")
    path = hf_hub_download(a.repo, shard, repo_type="dataset")
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=65_536, columns=[a.col]):
        col = batch.column(0)
        # flatten() respects list offsets (values does not, and batches are slices)
        x = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)
        if native is None:
            native = x.shape[1]
            print(f"native dim={native}")
        if done + len(x) > total:
            x = x[: total - done]
        emit(x, done)
        done += len(x)
        if done >= total:
            break
    if not a.keep_shards:
        # hf cache stores the blob behind a symlink; drop both so peak disk stays ~1 shard
        blob = os.path.realpath(path)
        if os.path.islink(path):
            os.remove(path)
        os.remove(blob)
    print(f"  {done}/{total}")

if done < total:
    print(f"WARNING: only {done} rows available (wanted {total}); base/query split still honored")

for D in dims:
    writers[D]["base"].close()
    writers[D]["query"].close()

print(f"\ndone. next, per dim (e.g. 256):")
print(f"  cargo run --release --bin vsearch -- --data {a.out} --prefix {a.name}256 \\")
print(f"      --write-ground-truth {a.out}/{a.name}256_groundtruth.ivecs --gt-k 100")
print(f"  cargo run --release --bin vsearch -- --data {a.out} --prefix {a.name}256 --quant binary --rerank 2000")
