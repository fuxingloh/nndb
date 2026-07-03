"use client";

import { useState } from "react";

// The 10M kernel arc: measured QPS at C=2000, recall 0.9737 (bit-identical at
// every step), c8a.4xlarge (Zen5, 16 vCPU), Snowflake arctic-256 x 10M.
type Step = {
  v: string;
  label: string;
  qps: number;
  slug: string;
  note: string;
};

const ARC: Step[] = [
  {
    v: "067",
    label: "baseline at 10M",
    qps: 650,
    slug: "067-10m-matryoshka-funnel-dram-wall",
    note: "The shipped 1M funnel config, scaled 10×. Generic hamming loop; batch=8. Recall dials cleanly — the engine survives the jump; throughput pays linearly.",
  },
  {
    v: "069",
    label: "fixed-width unroll",
    qps: 847,
    slug: "069-fixed-width-256-kernel",
    note: "012/050 said 'the naive loop is optimal' — a width-scoped verdict. At 4 words, loop control drowned 4 popcounts. Unrolled: ~3× kernel, +30% funnel, and tiling started paying again.",
  },
  {
    v: "071",
    label: "bounds checks proven away",
    qps: 955,
    slug: "071-typed-kernel-bounds-check-tax",
    note: "objdump showed ~6 of 13 uops/call were length gate + slice guards. Fix: test both lengths, add a typed &[u64;4] kernel. Types, not unsafe — dispatch slots are the binding resource.",
  },
  {
    v: "072",
    label: "unsafe cast → LLVM re-vectorizes",
    qps: 1093,
    slug: "072-unsafe-width-cast",
    note: "The width is the build's contract; stop re-proving it per call. With a provable fixed trip count, LLVM re-vectorized the whole tile loop: 8 queries per doc in SIMD lanes.",
  },
  {
    v: "073",
    label: "word-planar query transpose",
    qps: 1307,
    slug: "073-word-planar-query-groups",
    note: "072's ASM spent ~73% of dispatch re-establishing loop invariants (pointer surgery + 4 gathers/doc for never-changing query words). Transpose once per tile; gathers die. Tile knee moves 8→32.",
  },
  {
    v: "074",
    label: "wide store — last extraction dies",
    qps: 1471,
    slug: "074-wide-store",
    note: "073 built 8 distances in one zmm, then disassembled it lane by lane (~24 uops). vpmovqd + one 32 B store. The loop is now ~20 uops/doc-group: movement, math, one store — the floor.",
  },
];

const MAX = 1600;

export function KernelArcLab() {
  const [sel, setSel] = useState<string>("074");
  const active = ARC.find((s) => s.v === sel)!;

  return (
    <div>
      <div className="space-y-2.5">
        {ARC.map((s, i) => {
          const isSel = sel === s.v;
          const isLast = i === ARC.length - 1;
          const gain = i > 0 ? ((s.qps / ARC[i - 1].qps - 1) * 100).toFixed(0) : null;
          return (
            <button
              key={s.v}
              onClick={() => setSel(s.v)}
              onMouseEnter={() => setSel(s.v)}
              className="block w-full text-left"
              aria-label={`${s.v} ${s.label}: ${s.qps} QPS`}
            >
              <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
                <span className={isSel ? "text-text" : "text-dim"}>
                  <span className="text-amber">{s.v}</span> — {s.label}
                </span>
                <span className="tabular-nums">
                  {gain ? <span className="mr-2 text-dim">+{gain}%</span> : null}
                  <span style={{ color: isLast ? "var(--color-green)" : "var(--color-accent)" }}>
                    {s.qps.toLocaleString()} QPS
                  </span>
                </span>
              </div>
              <div className="h-5 overflow-hidden rounded bg-grid">
                <div
                  className="h-full rounded transition-opacity"
                  style={{
                    width: `${(s.qps / MAX) * 100}%`,
                    background: isLast ? "var(--color-green)" : "var(--color-accent)",
                    opacity: isSel ? 0.95 : 0.45 + 0.4 * ((i + 1) / ARC.length) * 0.5,
                  }}
                />
              </div>
            </button>
          );
        })}
      </div>

      <div className="mt-4 min-h-[84px] rounded-lg border border-border bg-grid/40 px-4 py-3 text-[13px] leading-6">
        <span className="font-mono text-[12px] text-text">
          {active.v} — {active.label}
        </span>
        <p className="mt-1 text-dim">{active.note}</p>
        <a href={`/notes/${active.slug}`} className="font-mono text-[11px] text-accent hover:underline">
          notes/{active.v} →
        </a>
      </div>

      <p className="pt-3 text-[13px] leading-6 text-dim">
        C=2000, recall 0.9737 — <span className="text-text">bit-identical results at every
        step</span>; the only thing that changed is how many dispatch slots each comparison
        costs. <span className="text-green">2.26×</span> from kernel work alone, on a kernel
        the project had twice concluded was done. Not shown: two variants that measured{" "}
        <em>slower</em> and were reverted (a bundled refactor in 072, a hand-hoisted broadcast
        in 074) — the arc is monotonic only because every step was gated by measurement.
      </p>
    </div>
  );
}
