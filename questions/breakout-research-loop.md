# Breakout Research — loop agenda

Full-auto loop. Four directions, in this order. One history entry each (expand only
if a result opens a real sub-question); build → measure → record → commit → next;
**stop after the fourth.** 10 was split: instrument energy first so 7/8/5 carry a
byte/Joule number, build the model last so it can fit them.

1. **10a — bytes-per-answer + Joules.** Make bytes/query (deterministic from
   quant·scan_bits·batch·C·store) and J/query (`powermetrics`, M3) first-class in the
   harness/JSON. Headline becomes recall-per-Joule / recall-per-byte. *M3.*

2. **7 — cell-size × residual (042 seam).** Sweep within-cell N for the scan/recall
   optimum (subset + fresh GT). Add `--residual` (subtract cell centroid) and test
   whether rotation+binary gains recall on residuals vs raw. *M3.*

3. **8 — scale & the hierarchy cliff.** Push N to 10M–100M (replicate/synthetic) and
   find where the index leaves L3→DRAM — does "tiling wins / carousel never cliffs"
   survive? Binary-scan-only (f32 store infeasible at 100M). *AWS box.*

4. **5 — query-adaptive funnel + certificate.** Set C (and/or scan-bits) per query
   from the stage-1 Hamming gap; emit a per-query miss-risk bound. Beat fixed-C on
   mean QPS at matched mean recall. *M3.*

Capstone after 4: **10b — predictive roofline model** that predicts QPS+recall for
any (N, bits, C, cores) and validates against all `history/*.json`.
