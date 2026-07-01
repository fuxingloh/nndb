#!/usr/bin/env python3
"""Embed a text corpus with Nomic Embed v1.5 (Matryoshka) → .fvecs for the benchmark.

Unlike fetch-cohere.py (which downloads *precomputed* vectors), Nomic v1.5 is an
open MRL model we run ourselves — so we can emit the SAME corpus at several
Matryoshka dimensions and study recall-vs-bits (esp. 256-d binary funnel + f32
rerank). Nomic v1.5 is Apache-2.0 (weights+data+code), so this is fully reproducible.

Pipeline: stream a text corpus (default BeIR/dbpedia-entity, the same corpus
Qdrant embedded with OpenAI text-embedding-3) → embed at native 768-d with the
`search_document:` task prefix → for each requested dim D, apply Nomic's official
Matryoshka recipe (layer-norm the full 768-d, slice to D, L2-normalize) → split
base/queries → write <out>/nomic<D>_{base,query}.fvecs.

Ground truth is generated separately, per dim, by the Rust tool:
  cargo run --release -- --data nndb/data/nomic --prefix nomic256 \
      --write-ground-truth nndb/data/nomic/nomic256_groundtruth.ivecs

Run via uv (avoids Python 3.14 wheel gaps; downloads torch + the model on first run):
  uv run --python 3.12 --with sentence-transformers --with datasets \
      --with einops --with numpy \
      nndb/scripts/fetch-nomic.py --target 1000000 --queries 10000

Queries are held-out corpus documents with the same task prefix as the base, i.e.
in-distribution doc-to-doc nearest-neighbor search (matches how sift/cohere are set
up, and what the recall model assumes). Use --query-prefix to make it asymmetric.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="BeIR/dbpedia-entity")
ap.add_argument("--config", default="corpus")
ap.add_argument("--split", default="corpus")
ap.add_argument("--target", type=int, default=1_000_000, help="base vector count")
ap.add_argument("--queries", type=int, default=10_000)
ap.add_argument("--dims", default="768,512,256,128,64",
                help="Matryoshka dims to emit (comma-sep); embed once at 768, slice each")
ap.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
ap.add_argument("--doc-prefix", default="search_document: ", help="Nomic task prefix for base+query")
ap.add_argument("--query-prefix", default=None, help="override prefix for queries (asymmetric); default = doc-prefix")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max-seq", type=int, default=512,
                help="cap tokens/doc — uncapped, one long doc blows up MPS O(seq^2) attention")
ap.add_argument("--out", default="nndb/data/nomic")
ap.add_argument("--name", default="nomic", help="file prefix stem -> <name><D>_base.fvecs")
a = ap.parse_args()

dims = [int(d) for d in a.dims.split(",") if d.strip()]
total = a.target + a.queries
qpref = a.query_prefix if a.query_prefix is not None else a.doc_prefix

# --- 1. load the corpus text (non-streaming sliced split; streaming over the HF
#        filesystem hits a client-lifecycle bug — "client has been closed") ---
from datasets import load_dataset
n_take = total + 2000  # small buffer for empty docs we skip
print(f"loading {n_take} docs from {a.repo}/{a.config}[{a.split}] (sliced, non-streaming) ...")
ds = load_dataset(a.repo, a.config, split=f"{a.split}[:{n_take}]")
texts, n = [], 0
for row in ds:
    t = ((row.get("title") or "") + ". " + (row.get("text") or "")).strip(" .")
    if not t:
        continue
    texts.append(t)
    n += 1
    if n % 100_000 == 0:
        print(f"  {n}/{total}")
    if n >= total:
        break
total = len(texts)
nq = min(a.queries, total // 10)
print(f"collected {total} docs; base={total - nq} queries={nq}")

# base and query get task prefixes (see module docstring re: symmetric vs asymmetric)
prefixed = [a.doc_prefix + t for t in texts[: total - nq]] + [qpref + t for t in texts[total - nq :]]

# --- 2. embed at native 768 with Nomic v1.5 ---
import torch
from sentence_transformers import SentenceTransformer
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"loading {a.model} on {device} ...")
model = SentenceTransformer(a.model, trust_remote_code=True, device=device)
model.max_seq_length = a.max_seq  # bound O(seq^2) attention so a long doc can't OOM MPS
emb = model.encode(prefixed, batch_size=a.batch, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
print(f"embedded: {emb.shape}")

# --- 3. Nomic Matryoshka recipe: layer-norm full dim, then per-D slice + L2-normalize ---
mean = emb.mean(axis=1, keepdims=True)
var = emb.var(axis=1, keepdims=True)
emb_ln = (emb - mean) / np.sqrt(var + 1e-5)   # F.layer_norm(normalized_shape=(768,)), no affine

def write_fvecs(path, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, dim = x.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = x.view(np.int32)
    out.tofile(path)

os.makedirs(a.out, exist_ok=True)
for D in dims:
    if D > emb_ln.shape[1]:
        print(f"  skip D={D} (> native {emb_ln.shape[1]})"); continue
    x = emb_ln[:, :D].copy()
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)  # renormalize the truncated prefix
    base, queries = x[: total - nq], x[total - nq :]
    write_fvecs(f"{a.out}/{a.name}{D}_base.fvecs", base)
    write_fvecs(f"{a.out}/{a.name}{D}_query.fvecs", queries)
    print(f"  wrote {a.name}{D}: base={base.shape} query={queries.shape}")

print(f"\ndone. next, per dim (e.g. 256):")
print(f"  cargo run --release -- --data {a.out} --prefix {a.name}256 \\")
print(f"      --write-ground-truth {a.out}/{a.name}256_groundtruth.ivecs")
print(f"  cargo run --release -- --data {a.out} --prefix {a.name}256 --quant binary --rerank 200")
