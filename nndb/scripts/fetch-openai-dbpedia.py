#!/usr/bin/env python3
"""Prepare precomputed OpenAI text-embedding-3-large (Matryoshka) vectors as .fvecs.

NO embedding inference — OpenAI already embedded dbpedia and Qdrant published the
1536-d vectors. text-embedding-3-large is MRL-trained (256 is a documented, trained
operating point), so we slice the 1536-d vectors to each Matryoshka dim, renormalize
(OpenAI's recipe: truncate then L2-normalize so L2 ranking == cosine), and study the
1-bit binary funnel + f32 rerank recall-vs-bits on a *genuine* Matryoshka embedding
(unlike Cohere v3, which isn't Matryoshka). Same dbpedia corpus the SOTA quant
benchmarks (RaBitQ/Qdrant) use, so numbers are comparable.

Ground truth is generated per dim by the Rust tool's --write-ground-truth (exact KNN).

Run via uv (no torch — just download + slice):
  uv run --python 3.12 --with datasets --with numpy \
    nndb/scripts/fetch-openai-dbpedia.py --target 1000000 --queries 10000 --out nndb/data/oai
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M")
ap.add_argument("--split", default="train")
ap.add_argument("--col", default="text-embedding-3-large-1536-embedding")
ap.add_argument("--target", type=int, default=1_000_000, help="base vector count (dataset has 1M total)")
ap.add_argument("--queries", type=int, default=10_000)
ap.add_argument("--dims", default="1536,1024,768,512,256,128,64",
                help="Matryoshka dims to emit (comma-sep); slice the native 1536-d vectors")
ap.add_argument("--out", default="nndb/data/oai")
ap.add_argument("--name", default="oai")
a = ap.parse_args()

dims = [int(d) for d in a.dims.split(",") if d.strip()]
want = a.target + a.queries

from datasets import load_dataset
print(f"loading up to {want} rows from {a.repo}[{a.split}] (sliced, non-streaming) ...")
# .with_format('numpy') → Arrow buffers come out as numpy directly; extract in chunks so
# we never materialize the column as a giant Python list (that OOMs a 32GB box at 1M×1536).
ds = load_dataset(a.repo, split=f"{a.split}[:{want}]").with_format("numpy")
N = len(ds)
native = int(np.asarray(ds[0][a.col]).shape[0])
emb = np.empty((N, native), dtype=np.float32)
CH = 50_000
for i in range(0, N, CH):
    emb[i : i + CH] = np.asarray(ds[i : i + CH][a.col], dtype=np.float32)
    print(f"  extracted {min(i + CH, N)}/{N}")
del ds
print(f"vectors: {emb.shape} (native dim {native})")

nq = min(a.queries, N // 10)
base_n = N - nq
print(f"base={base_n} queries={nq}")

def write_fvecs(path, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, dim = x.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = x.view(np.int32)
    out.tofile(path)

os.makedirs(a.out, exist_ok=True)
for D in dims:
    if D > native:
        print(f"  skip D={D} (> native {native})"); continue
    # OpenAI Matryoshka: slice to first D dims, then L2-normalize.
    x = emb[:, :D].copy()
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    write_fvecs(f"{a.out}/{a.name}{D}_base.fvecs", x[:base_n])
    write_fvecs(f"{a.out}/{a.name}{D}_query.fvecs", x[base_n:])
    print(f"  wrote {a.name}{D}: base=({base_n},{D}) query=({nq},{D})")

print(f"\ndone. next, per dim (e.g. 256):")
print(f"  cargo run --release --bin vsearch -- --data {a.out} --prefix {a.name}256 \\")
print(f"      --write-ground-truth {a.out}/{a.name}256_groundtruth.ivecs --gt-k 100")
print(f"  cargo run --release --bin vsearch -- --data {a.out} --prefix {a.name}256 --quant binary --rerank 500")
