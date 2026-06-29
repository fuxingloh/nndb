"use client";

// QPS climbs ~100× across the key milestones while recall stays ~0.99. Linear scale,
// with the ×-vs-baseline called out so the jump is legible (a log axis hid it).
const STEPS = [
  { name: "exact brute force", qps: 9, recall: "1.000", note: "scan all 4 GB / query", entry: "001" },
  { name: "binary funnel", qps: 516, recall: "0.993", note: "1-bit scan + rerank", entry: "009" },
  { name: "+ query tiling", qps: 922, recall: "0.997", note: "amortize the base read", entry: "038" },
  { name: "+ rotation + residual", qps: 963, recall: "0.995", note: "make the bits count", entry: "051" },
];

const maxQ = Math.max(...STEPS.map((s) => s.qps));
const base = STEPS[0].qps;

export function JourneyLab() {
  return (
    <div className="space-y-4">
      {STEPS.map((s, i) => {
        const last = i === STEPS.length - 1;
        return (
          <div key={s.name}>
            <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
              <span className="text-text">
                <span className="text-dim">{s.entry} ·</span> {s.name}
              </span>
              <span className="tabular-nums">
                <span className={last ? "text-accent" : "text-text"}>{s.qps} QPS</span>
                <span className="ml-2 text-dim">{Math.round(s.qps / base)}×</span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="relative h-6 flex-1 overflow-hidden rounded bg-grid">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${Math.max(1.5, (s.qps / maxQ) * 100)}%`,
                    background: last ? "#818cf8" : "#3b3b46",
                  }}
                />
              </div>
              <span className="w-24 shrink-0 text-right font-mono text-[11px] text-dim">
                recall {s.recall}
              </span>
            </div>
            <p className="mt-0.5 text-[12px] text-dim">{s.note}</p>
          </div>
        );
      })}
      <p className="border-t border-border pt-3 text-[13px] leading-6 text-dim">
        Bar length is QPS (linear). The throughput goes <span className="text-accent">~107×</span> from
        9 to 963 — yet the <span className="text-text">recall column barely moves</span> (≈ 0.99
        throughout). That&apos;s the whole point: massive speed, ~no recall lost.
      </p>
    </div>
  );
}
