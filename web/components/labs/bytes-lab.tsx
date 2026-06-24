"use client";

// The bandwidth wall: a query's cost is bytes moved from RAM. Linear scale so the 32×
// cut is obvious — the 1-bit bar is a sliver next to float, which is the whole point.
const ROWS = [
  { name: "float32 — 4 KB/vec", mb: 4096, color: "#fb7185", note: "memory-bound: 4 GB / query" },
  { name: "int8 — 1 KB/vec", mb: 1024, color: "#f4b860", note: "4× smaller" },
  { name: "1-bit code — 128 B/vec", mb: 128, color: "#818cf8", note: "32× smaller — fits the bus" },
];
const max = ROWS[0].mb;

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
          <div className="h-6 overflow-hidden rounded bg-grid">
            <div className="h-full rounded" style={{ width: `${Math.max(1, (r.mb / max) * 100)}%`, background: r.color }} />
          </div>
          <p className="mt-0.5 text-[12px] text-dim">{r.note}</p>
        </div>
      ))}
      <p className="border-t border-border pt-3 text-[13px] leading-6 text-dim">
        Bar length is bytes read per query (linear). Scanning 1M vectors moves the whole base
        from RAM every query — no kernel trick moves 4 GB faster than the bus, so the lever is
        <span className="text-accent"> fewer bytes</span>. 1-bit is a <span className="text-accent">32×</span> cut.
      </p>
    </div>
  );
}
