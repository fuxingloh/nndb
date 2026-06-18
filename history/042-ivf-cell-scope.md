# 042 — IVF is the premise, not a component (scope entry)

Metadata: [`042-ivf-cell-scope.json`](./042-ivf-cell-scope.json). No perf numbers —
this is an architecture/scoping entry, written to close `questions/IVF.md` (nothing
was *built* for IVF, so there's no existing entry to point to).

## The question

Is IVF needed in this engine? If this node is already one IVF cell, what's the
point of IVF here — isn't it not required?

## The answer

**IVF-the-router is not required here, and we deliberately don't build it** — but
IVF is the *premise* that justifies this entire engine. Two things to keep separate:

1. **IVF-the-router is out of scope (and not built).** The coarse quantizer that
   decides *which* cells a query visits lives a layer above us. If you assume "this
   node is already an IVF cell," routing is somebody else's job; nothing in our scan
   needs it to exist at runtime.

2. **IVF is *why* the within-cell scan is worth optimizing at all.** Without IVF, a
   query must scan all N vectors → the ~100 ms brute-force number from `001`. IVF
   partitions N into ~√N cells and routes each query to only `nprobe` of them, which
   is what makes a "cell" small enough that an *exact* within-cell scan is fast
   enough to matter. Our fast scan is precisely the kernel that runs `nprobe` times
   per query *because* IVF narrowed the candidate set. So IVF is the reason our work
   exists — a premise, not a component we own.

## In-scope seam questions (logged, not resolved)

These are NOT "building IVF" — they're "how the cell scan should behave given it
lives inside an IVF." Future entries if we revisit:

1. **Cell sizing.** Cell size sets our scan length (N per cell): too big → scan
   dominates; too small → recall loss + more cells for the router. The sweet spot is
   a within-cell concern that our scan throughput directly informs.
2. **Residual encoding.** IVF cells often store residuals (vector − centroid), which
   quantize better. Does the rotation+binary funnel (`026`/`029`) gain recall on
   residuals vs raw vectors? A real quantization-on-the-cell question.
3. **Per-cell top-C / cross-cell merge.** A query hits `nprobe` cells; each returns
   top-C and the router merges. What C per cell, and does multi-cell merging change
   the optimal funnel width?

## Conclusion

`questions/IVF.md` is removed: the "is IVF required?" question is answered (no, the
router isn't built here; IVF is the premise). The remaining IVF interaction
questions are genuinely in-scope but unbuilt, and are parked above rather than in a
separate question file.
