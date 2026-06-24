"use client";

// c8a (AMD Zen5) funnel QPS measured across the whole size range, 8→192 cores. Same
// engine/data, recall held 0.9952. It climbs ~linearly while compute-bound, then DDR5-6400
// saturates: sub-full-socket boxes share the 12 channels and plateau ~13.6k (64–96c); the
// full-socket 48xlarge gets them all and tops out at ~19.9k. Bandwidth is the wall.
const PTS = [
  { cores: 8, qps: 2310, size: "2xl" },
  { cores: 16, qps: 4462, size: "4xl" },
  { cores: 32, qps: 7658, size: "8xl" },
  { cores: 48, qps: 9710, size: "12xl" },
  { cores: 64, qps: 13670, size: "16xl" },
  { cores: 96, qps: 13555, size: "24xl" },
  { cores: 192, qps: 19881, size: "48xl" },
];

const W = 560, H = 360, PAD = 52, YMAX = 22000;
const LX0 = Math.log2(8), LX1 = Math.log2(192);
const sx = (c: number) => PAD + ((Math.log2(c) - LX0) / (LX1 - LX0)) * (W - 2 * PAD);
const sy = (q: number) => H - PAD - (Math.min(q, YMAX) / YMAX) * (H - 2 * PAD);

export function ScalingLab() {
  const actual = PTS.map((p) => `${sx(p.cores)},${sy(p.qps)}`).join(" ");
  const yticks = [5000, 10000, 15000, 20000];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {yticks.map((q) => (
          <g key={q}>
            <line x1={PAD} y1={sy(q)} x2={W - PAD} y2={sy(q)} stroke="#1b1b21" />
            <text x={PAD - 8} y={sy(q) + 3} textAnchor="end" fontSize={9} className="fill-dim font-mono">
              {q / 1000}k
            </text>
          </g>
        ))}
        {PTS.map((p) => (
          <text key={p.cores} x={sx(p.cores)} y={H - PAD + 14} textAnchor="middle" fontSize={8.5} className="fill-dim font-mono">
            {p.cores}
          </text>
        ))}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} className="fill-dim">physical cores (log) →</text>
        <text x={13} y={H / 2} textAnchor="middle" fontSize={10} className="fill-dim" transform={`rotate(-90 13 ${H / 2})`}>funnel QPS →</text>

        {/* measured curve */}
        <polyline points={actual} fill="none" stroke="#818cf8" strokeWidth={2.5} />
        {PTS.map((p) => (
          <circle key={p.cores} cx={sx(p.cores)} cy={sy(p.qps)} r={4}
            fill={p.cores === 192 ? "#34d399" : "#818cf8"} />
        ))}

        {/* observed-region labels (not model lines) */}
        <text x={sx(13)} y={sy(5200)} fontSize={9} className="fill-dim font-mono">compute-bound</text>
        <text x={sx(70)} y={sy(15600)} fontSize={9} className="fill-amber font-mono">shared-socket plateau ~13.6k</text>
        <text x={sx(192)} y={sy(19881) - 8} textAnchor="end" fontSize={9} className="fill-green font-mono">full socket 19.9k</text>
      </svg>
      <p className="mt-2 text-[13px] leading-6 text-dim">
        QPS climbs ~linearly with cores while compute-bound, then <span className="text-text">DDR5-6400 saturates</span>.
        Sub-full-socket boxes share the socket&rsquo;s 12 channels and pin at <span className="text-amber">~13.6k</span>
        {" "}(64–96c); only the full-socket 48xlarge gets them all and tops out at <span className="text-green">~19.9k</span>
        {" "}(≈358 GB/s, 58% of the 614 GB/s peak). Beyond that the memory bus, not the cores, is the ceiling.
      </p>
    </div>
  );
}
