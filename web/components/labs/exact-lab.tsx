"use client";

// The other regime: when you need ~exact recall (0.999+), the funnel's lossy codes
// are off the table — you must scan full f32. Two tricks make that scan 4× faster.
// Vertical (column) layout autovectorizes for free; ADSampling early-terminates a
// vector's distance once it can't beat the running threshold. Recall stays ~1.0.
const BARS = [
  { name: "naive f32 scan", qps: 9.2, recall: "1.000", color: "#3b3b46", note: "row-major, full distance every vector" },
  { name: "+ PDX vertical layout", qps: 18.0, recall: "1.000", color: "#f4b860", note: "column-major — autovectorizes, free 2×" },
  { name: "+ ADSampling pruning", qps: 37.4, recall: "0.999", color: "#34d399", note: "early-stop a vector once it can't win" },
];
const maxQ = Math.max(...BARS.map((b) => b.qps));
const base = BARS[0].qps;

export function ExactLab() {
  return (
    <div className="space-y-4">
      {BARS.map((b, i) => {
        const last = i === BARS.length - 1;
        return (
          <div key={b.name}>
            <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
              <span className="text-text">{b.name}</span>
              <span className="tabular-nums">
                <span style={{ color: b.color }}>{b.qps.toFixed(1)} QPS</span>
                <span className="ml-2 text-dim">{(b.qps / base).toFixed(1)}×</span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-6 flex-1 overflow-hidden rounded bg-grid">
                <div
                  className="h-full rounded"
                  style={{ width: `${Math.max(2, (b.qps / maxQ) * 100)}%`, background: b.color }}
                />
              </div>
              <span className="w-24 shrink-0 text-right font-mono text-[11px] text-dim">
                recall {b.recall}
              </span>
            </div>
            <p className="mt-0.5 text-[12px] text-dim">{b.note}</p>
            {last && (
              <p className="mt-0.5 font-mono text-[11px] text-green">
                4× the naive exact scan, recall still ~1.0
              </p>
            )}
          </div>
        );
      })}
      <p className="border-t border-border pt-3 text-[13px] leading-6 text-dim">
        This is the <span className="text-text">exact</span> regime — full f32, recall ~1.0 —
        not the funnel. Even here the two levers hold: the layout change is free SIMD,
        and pruning <span className="text-accent">moves fewer bytes</span> per vector. The funnel
        is still far faster when you can spend recall; this is the corner where you can&apos;t.
      </p>
    </div>
  );
}
