"use client";

// Latency vs offered load. A naive per-query server is fine until ~capacity, then its
// queue explodes and p50 cliffs to seconds. The carousel (shared scan) stays bounded.
type Pt = { qps: number; ms: number };
const PERQUERY: Pt[] = [
  { qps: 100, ms: 12.5 }, { qps: 200, ms: 11.5 }, { qps: 400, ms: 11.8 },
  { qps: 600, ms: 533 }, { qps: 800, ms: 2045 },
];
const CAROUSEL: Pt[] = [
  { qps: 200, ms: 4.5 }, { qps: 400, ms: 6.5 }, { qps: 600, ms: 11.5 },
  { qps: 800, ms: 20 }, { qps: 1000, ms: 35 },
];

const W = 420, H = 300, PAD = 48;
const xmax = 1000;
const ylo = Math.log10(4), yhi = Math.log10(3000); // ms, log
const sx = (q: number) => PAD + (q / xmax) * (W - 2 * PAD);
const sy = (ms: number) => H - PAD - ((Math.log10(ms) - ylo) / (yhi - ylo)) * (H - 2 * PAD);

function Line({ pts, color, dash }: { pts: Pt[]; color: string; dash?: boolean }) {
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.qps)},${sy(p.ms)}`).join(" ");
  return (
    <g>
      <path d={d} fill="none" stroke={color} strokeWidth={2} strokeDasharray={dash ? "4 3" : undefined} />
      {pts.map((p, i) => <circle key={i} cx={sx(p.qps)} cy={sy(p.ms)} r={4} fill={color} />)}
    </g>
  );
}

export function ServingLab() {
  const yticks = [10, 100, 1000];
  const xticks = [200, 600, 1000];
  return (
    <div>
      <div className="mb-3 flex gap-3 font-mono text-[11px]">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-full bg-rose" />per-query (naive)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-full bg-accent" />carousel</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {yticks.map((ms) => (
          <g key={ms}>
            <line x1={PAD} y1={sy(ms)} x2={W - PAD} y2={sy(ms)} stroke="#1b1b21" />
            <text x={PAD - 8} y={sy(ms) + 3} textAnchor="end" fontSize={9} className="fill-dim font-mono">{ms}ms</text>
          </g>
        ))}
        {xticks.map((q) => (
          <text key={q} x={sx(q)} y={H - PAD + 14} textAnchor="middle" fontSize={9} className="fill-dim font-mono">{q}</text>
        ))}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} className="fill-dim">offered QPS →</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={10} className="fill-dim" transform={`rotate(-90 12 ${H / 2})`}>p50 latency (log) →</text>
        <Line pts={PERQUERY} color="#fb7185" dash />
        <Line pts={CAROUSEL} color="#818cf8" />
      </svg>
      <p className="mt-2 text-[13px] leading-6 text-dim">
        Same throughput ceiling, opposite tails. Past ~500 QPS the naive server
        <span className="text-rose"> cliffs to seconds</span>; the carousel stays
        <span className="text-accent"> bounded</span> (~35 ms at 1000 QPS). Note the log y-axis.
      </p>
    </div>
  );
}
