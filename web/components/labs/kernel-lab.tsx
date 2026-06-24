"use client";

// The other lever: fewer/cheaper instructions. Single-thread scan throughput (Gcmp/s).
// A table gather is slow; popcount is one SIMD instruction; a SIMD lookup (pshufb) on
// tiny 4-bit codes is faster still — the frontier that surprised me.
const BARS = [
  { name: "scalar PQ — table gather", g: 0.14, color: "#fb7185" },
  { name: "binary — popcount (VPOPCNTDQ)", g: 0.20, color: "#818cf8" },
  { name: "SIMD PQ4 — pshufb (8 B/vec)", g: 1.83, color: "#34d399" },
];
const max = Math.max(...BARS.map((b) => b.g));

export function KernelLab() {
  return (
    <div className="space-y-3">
      {BARS.map((b) => (
        <div key={b.name}>
          <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
            <span className="text-text">{b.name}</span>
            <span className="tabular-nums" style={{ color: b.color }}>
              {b.g.toFixed(2)} Gcmp/s
            </span>
          </div>
          <div className="h-5 overflow-hidden rounded bg-grid">
            <div className="h-full rounded" style={{ width: `${(b.g / max) * 100}%`, background: b.color }} />
          </div>
        </div>
      ))}
      <p className="pt-1 text-[13px] leading-6 text-dim">
        Same job, different instruction. The gather can&apos;t vectorize; popcount is one
        instruction over 8 words; the SIMD lookup does 16 lookups per instruction on codes
        16× smaller — <span className="text-green">9×</span> the popcount scan. Single-thread,
        random codes.
      </p>
    </div>
  );
}
