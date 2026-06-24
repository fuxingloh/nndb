"use client";

// The QPS climb across the key milestones — ~100× faster at essentially the same
// recall. Each step is a measured history entry.
const STEPS = [
  { name: "exact brute force", qps: 9, recall: "1.000", note: "scan all 4 GB / query", entry: "001" },
  { name: "binary funnel", qps: 516, recall: "0.993", note: "1-bit scan + rerank", entry: "009" },
  { name: "+ query tiling", qps: 922, recall: "0.997", note: "amortize the base read", entry: "038" },
  { name: "+ rotation + residual", qps: 963, recall: "0.995", note: "make the bits count", entry: "051" },
];

const maxQ = Math.max(...STEPS.map((s) => s.qps));

export function JourneyLab() {
  return (
    <div className="space-y-3">
      {STEPS.map((s, i) => (
        <div key={s.name}>
          <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
            <span className="text-text">
              <span className="text-dim">{s.entry} ·</span> {s.name}
            </span>
            <span className="text-cyan tabular-nums">{s.qps} QPS</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-5 flex-1 overflow-hidden rounded bg-grid">
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.max(2, (Math.log10(s.qps) / Math.log10(maxQ)) * 100)}%`,
                  background: i === STEPS.length - 1 ? "#5ccfe6" : "#3d4a63",
                }}
              />
            </div>
            <span className="w-28 shrink-0 text-right font-mono text-[11px] text-dim">
              recall {s.recall}
            </span>
          </div>
          <p className="mt-0.5 text-[12px] text-dim">{s.note}</p>
        </div>
      ))}
      <p className="pt-1 text-[13px] leading-6 text-dim">
        ~<span className="text-cyan">100×</span> the throughput of exact brute force, at the same
        ~0.99 recall — every step a measured entry, not a guess. Bars are log-scaled QPS.
      </p>
    </div>
  );
}
