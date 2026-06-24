// The funnel: a cheap 1-bit scan over the whole cell narrows to a few hundred
// candidates, then an exact rerank ranks only those. Coarse-but-fast filters;
// exact-but-slow ranks the survivors.
const STAGES = [
  { w: 100, count: "1,000,000 vectors", op: "1-bit Hamming scan · 128 B/vec · popcount", color: "#3b3b46" },
  { w: 34, count: "~500 candidates", op: "exact f32 rerank · 4 KB/vec · only these", color: "#818cf8" },
  { w: 9, count: "top 10", op: "the answer", color: "#34d399" },
];

export function FunnelDiagram() {
  return (
    <div className="space-y-2">
      {STAGES.map((s, i) => (
        <div key={i}>
          <div className="mx-auto flex h-12 items-center justify-center rounded" style={{ width: `${s.w}%`, background: s.color }}>
            <span className={`font-mono text-[13px] ${i === 0 ? "text-text" : "text-ink"} font-medium`}>{s.count}</span>
          </div>
          <p className="mt-1 text-center text-[12px] text-dim">{s.op}</p>
          {i < STAGES.length - 1 && <div className="mx-auto my-1 text-center text-dim">↓</div>}
        </div>
      ))}
      <p className="border-t border-border pt-3 text-[13px] leading-6 text-dim">
        Every vector gets the <span className="text-accent">cheap</span> 1-bit comparison; only the
        few hundred survivors pay for the <span className="text-accent">exact</span> one. You read
        4 KB/vector for 500 vectors, not a million.
      </p>
    </div>
  );
}
