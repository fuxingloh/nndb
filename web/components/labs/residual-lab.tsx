"use client";

import { useState } from "react";

// A cluster of points near a common center. Sign-binarizing the RAW vectors gives
// every point the SAME 2-bit code (they share the centroid's quadrant) — useless.
// Subtract the centroid first (residual) and the codes spread out → distinguishable.
const PTS = [
  [0.62, 0.55],
  [0.78, 0.48],
  [0.55, 0.71],
  [0.84, 0.66],
  [0.70, 0.40],
  [0.66, 0.78],
];

const W = 360;
const H = 300;
const PAD = 36;
const codeColor = (c: string) =>
  ({ "++": "#5ccfe6", "+-": "#ffb454", "-+": "#7fd962", "--": "#f07178" })[c] ?? "#8a93a5";

export function ResidualLab() {
  const [residual, setResidual] = useState(false);

  const cx = PTS.reduce((s, p) => s + p[0], 0) / PTS.length;
  const cy = PTS.reduce((s, p) => s + p[1], 0) / PTS.length;
  // origin of the sign-axes: (0,0) for raw, the centroid for residual
  const ox = residual ? cx : 0.5;
  const oy = residual ? cy : 0.5;
  // for raw we binarize around 0.5 (the data lives in [0,1]); for residual around centroid
  const codeOf = (p: number[]) =>
    `${p[0] - ox >= 0 ? "+" : "-"}${p[1] - oy >= 0 ? "+" : "-"}`;
  const codes = PTS.map(codeOf);
  const distinct = new Set(codes).size;

  const sx = (x: number) => PAD + x * (W - 2 * PAD);
  const sy = (y: number) => H - PAD - y * (H - 2 * PAD);

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <button
          onClick={() => setResidual((r) => !r)}
          className="rounded-md border border-border-hover bg-grid px-3 py-1.5 font-mono text-xs text-text transition-colors hover:border-cyan"
        >
          {residual ? "● residual (vector − centroid)" : "○ raw vectors"}
        </button>
        <span className="font-mono text-xs text-dim">
          distinct codes:{" "}
          <span className={distinct > 1 ? "text-green" : "text-rose"}>
            {distinct}/{PTS.length}
          </span>
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* sign-axis crosshair (the binarization boundary) */}
        <line x1={sx(ox)} y1={PAD} x2={sx(ox)} y2={H - PAD} stroke="#222a39" strokeWidth={1} />
        <line x1={PAD} y1={sy(oy)} x2={W - PAD} y2={sy(oy)} stroke="#222a39" strokeWidth={1} />
        {/* centroid marker */}
        <circle cx={sx(cx)} cy={sy(cy)} r={4} fill="none" stroke="#8a93a5" strokeDasharray="2 2" />
        <text x={sx(cx) + 7} y={sy(cy) - 6} className="fill-dim font-mono" fontSize={9}>
          centroid
        </text>
        {/* points, colored by their 2-bit sign code */}
        {PTS.map((p, i) => (
          <g key={i}>
            <circle cx={sx(p[0])} cy={sy(p[1])} r={6} fill={codeColor(codes[i])} opacity={0.9} />
            <text
              x={sx(p[0]) + 9}
              y={sy(p[1]) + 3}
              className="font-mono"
              fontSize={10}
              fill={codeColor(codes[i])}
            >
              {codes[i]}
            </text>
          </g>
        ))}
      </svg>

      <p className="mt-2 text-[13px] leading-6 text-dim">
        {residual
          ? "Subtracting the centroid re-centers the sign-axes on the cluster — now each point's code encodes how it deviates. The codes separate."
          : "Raw: every point sits in the same quadrant, so every 2-bit code is identical. The bits carry the shared center, not what makes points different."}
      </p>
    </div>
  );
}
