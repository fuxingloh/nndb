"use client";

// The bandwidth wall: a query's cost is bytes moved from RAM. Float scans 4 GB;
// 1-bit codes scan 128 MB — 32× fewer bytes, the lever that broke the wall.
const ROWS = [
  { name: "float32 (4 KB/vec)", mb: 4096, color: "#f07178", note: "memory-bound — 4 GB/query" },
  { name: "int8 (1 KB/vec)", mb: 1024, color: "#ffb454", note: "4× smaller" },
  { name: "1-bit codes (128 B/vec)", mb: 128, color: "#5ccfe6", note: "32× smaller — fits the bus" },
];
const max = Math.log10(4096);

export function BytesLab() {
  return (
    <div className="space-y-3">
      {ROWS.map((r) => (
        <div key={r.name}>
          <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
            <span className="text-text">{r.name}</span>
            <span className="tabular-nums" style={{ color: r.color }}>
              {r.mb >= 1024 ? `${(r.mb / 1024).toFixed(1)} GB` : `${r.mb} MB`}/query
            </span>
          </div>
          <div className="h-5 overflow-hidden rounded bg-grid">
            <div className="h-full rounded" style={{ width: `${(Math.log10(r.mb) / max) * 100}%`, background: r.color }} />
          </div>
          <p className="mt-0.5 text-[12px] text-dim">{r.note}</p>
        </div>
      ))}
      <p className="pt-1 text-[13px] leading-6 text-dim">
        Scanning 1M vectors moves the whole base from RAM every query. No kernel trick moves
        4 GB faster than the bus — so the lever is <span className="text-cyan">fewer bytes</span>.
        Bars are log-scaled bytes/query.
      </p>
    </div>
  );
}
