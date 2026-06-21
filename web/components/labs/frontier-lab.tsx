"use client";

import { useState } from "react";

// Measured recall@10 vs QPS (Cohere 1M × 1024, one 8-core box). The binary funnel
// dominates the upper-right; PQ wins bytes-per-recall but loses QPS (gather vs popcount).
type Pt = { qps: number; recall: number; label: string };
const FUNNEL: Pt[] = [
  { qps: 1028, recall: 0.978, label: "C=200" },
  { qps: 963, recall: 0.9952, label: "C=500" },
  { qps: 883, recall: 0.9986, label: "C=1000" },
];
const PQ: Pt[] = [
  { qps: 273, recall: 0.905, label: "M=16" },
  { qps: 140, recall: 0.992, label: "M=32" },
  { qps: 62, recall: 0.9998, label: "M=64" },
];

const W = 420;
const H = 300;
const PAD = 44;
// x = QPS on a log scale (40 → 1100), y = recall (0.88 → 1.0)
const x0 = Math.log10(45);
const x1 = Math.log10(1100);
const sx = (q: number) => PAD + ((Math.log10(q) - x0) / (x1 - x0)) * (W - 2 * PAD);
const sy = (r: number) => H - PAD - ((r - 0.88) / (1.0 - 0.88)) * (H - 2 * PAD);

function Series({ pts, color, show }: { pts: Pt[]; color: string; show: boolean }) {
  if (!show) return null;
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.qps)},${sy(p.recall)}`).join(" ");
  return (
    <g>
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} opacity={0.5} />
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={sx(p.qps)} cy={sy(p.recall)} r={5} fill={color} />
          <text x={sx(p.qps) + 8} y={sy(p.recall) + 3} fontSize={9} className="font-mono" fill={color}>
            {p.label}
          </text>
        </g>
      ))}
    </g>
  );
}

export function FrontierLab() {
  const [funnel, setFunnel] = useState(true);
  const [pq, setPq] = useState(true);
  const xticks = [50, 100, 250, 500, 1000];
  const yticks = [0.9, 0.95, 1.0];

  return (
    <div>
      <div className="mb-3 flex gap-2">
        <Toggle on={funnel} set={setFunnel} color="#5ccfe6" label="binary funnel (128 B)" />
        <Toggle on={pq} set={setPq} color="#ffb454" label="PQ (16–64 B)" />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {yticks.map((r) => (
          <g key={r}>
            <line x1={PAD} y1={sy(r)} x2={W - PAD} y2={sy(r)} stroke="#1c2330" />
            <text x={PAD - 8} y={sy(r) + 3} textAnchor="end" fontSize={9} className="fill-dim font-mono">
              {r.toFixed(2)}
            </text>
          </g>
        ))}
        {xticks.map((q) => (
          <text key={q} x={sx(q)} y={H - PAD + 14} textAnchor="middle" fontSize={9} className="fill-dim font-mono">
            {q}
          </text>
        ))}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} className="fill-dim">
          QPS (log) →
        </text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={10} className="fill-dim" transform={`rotate(-90 12 ${H / 2})`}>
          recall@10 →
        </text>
        <Series pts={PQ} color="#ffb454" show={pq} />
        <Series pts={FUNNEL} color="#5ccfe6" show={funnel} />
      </svg>
      <p className="mt-2 text-[13px] leading-6 text-dim">
        Up-and-right is better. The funnel holds ~0.99 recall at <span className="text-cyan">~900 QPS</span>;
        PQ reaches the same recall only near <span className="text-amber">~140 QPS</span> — 7× slower, because
        its ADC is a gather, not a popcount.
      </p>
    </div>
  );
}

function Toggle({
  on,
  set,
  color,
  label,
}: {
  on: boolean;
  set: (f: (b: boolean) => boolean) => void;
  color: string;
  label: string;
}) {
  return (
    <button
      onClick={() => set((b) => !b)}
      className="flex items-center gap-1.5 rounded-md border border-border bg-grid px-2.5 py-1 font-mono text-[11px] text-text"
      style={{ opacity: on ? 1 : 0.4 }}
    >
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
      {label}
    </button>
  );
}
