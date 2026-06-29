#!/usr/bin/env python3
"""Prepare a Cohere v3 Wikipedia embedding subset as .fvecs for the benchmark.

NO training, NO embedding inference — Cohere already embedded Wikipedia and
published the vectors. We stream a subset, unit-normalize (so L2 ranking ==
cosine), split into base/queries, and write our .fvecs format. Ground truth is
generated separately by the Rust tool's --write-ground-truth (exact KNN).

Run via uv (avoids Python 3.14 wheel gaps):
  uv run --python 3.12 --with datasets --with numpy \
    nndb/scripts/fetch-cohere.py --target 1000000 --out nndb/data/cohere
"""
import argparse, os
import numpy as np
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Cohere/wikipedia-2023-11-embed-multilingual-v3")
ap.add_argument("--lang", default="en")
ap.add_argument("--target", type=int, default=1_000_000, help="base vector count")
ap.add_argument("--queries", type=int, default=10_000)
ap.add_argument("--col", default="emb")
ap.add_argument("--out", default="nndb/data/cohere")
a = ap.parse_args()

total = a.target + a.queries
print(f"streaming {total} rows from {a.repo} [{a.lang}] ...")
ds = load_dataset(a.repo, a.lang, split="train", streaming=True)

arr = None
n = 0
for row in ds:
    v = row[a.col]
    if arr is None:
        dim = len(v)
        arr = np.empty((total, dim), dtype=np.float32)
        print(f"dim={dim}")
    arr[n] = v
    n += 1
    if n % 100_000 == 0:
        print(f"  {n}/{total}")
    if n >= total:
        break
if n < total:
    print(f"only {n} rows available; using those")
    arr = arr[:n]
    total = n

# unit-normalize -> L2 ranking == cosine ranking
norms = np.linalg.norm(arr, axis=1, keepdims=True)
norms[norms == 0] = 1.0
arr /= norms

nq = min(a.queries, total // 10)
base = np.ascontiguousarray(arr[: total - nq])
queries = np.ascontiguousarray(arr[total - nq :])

def write_fvecs(path, x):
    n, dim = x.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = x.view(np.int32)
    out.tofile(path)

os.makedirs(a.out, exist_ok=True)
write_fvecs(f"{a.out}/cohere_base.fvecs", base)
write_fvecs(f"{a.out}/cohere_query.fvecs", queries)

frac = float(np.mean(base[:1000] != np.round(base[:1000])))
print(f"base={base.shape} queries={queries.shape} dim={base.shape[1]}")
print(f"non-integer fraction (want ~1.0 => quantization is lossy): {frac:.3f}")
print(f"wrote {a.out}/cohere_base.fvecs , cohere_query.fvecs")
