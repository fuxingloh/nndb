"use client";

// Same engine, same 1M×1024 Cohere data, same flags on every box — so recall is a
// constant 0.9952 and the only variables are the silicon and its price. The funnel is
// compute-bound on popcount, so throughput ≈ (physical cores) × (popcount width). The
// "8 vCPU" tier hides the real story: AMD/Graviton give 8 physical cores, Intel gives
// 4 + SMT (and SMT adds ~nothing here). $/hr is us-east-1 on-demand, normalized.
type Row = {
  name: string; vendor: string; cores: number; isa: "AVX-512" | "NEON";
  qps: number; p50: number; usd: number; color: string;
};
// NOTE: m8g filled from the live run before commit.
const ROWS: Row[] = [
  { name: "c8a.2xlarge", vendor: "AMD Zen5",      cores: 8, isa: "AVX-512", qps: 2310, p50: 3.1,  usd: 0.4311, color: "#a78bfa" },
  { name: "m8a.2xlarge", vendor: "AMD Zen5",      cores: 8, isa: "AVX-512", qps: 2313, p50: 3.1,  usd: 0.4869, color: "#a78bfa" },
  { name: "c8g.2xlarge", vendor: "Graviton4",     cores: 8, isa: "NEON",    qps: 986,  p50: 6.6,  usd: 0.3190, color: "#34d399" },
  { name: "m8g.2xlarge", vendor: "Graviton4",     cores: 8, isa: "NEON",    qps: 981,  p50: 6.8,  usd: 0.3590, color: "#34d399" },
  { name: "c8i.2xlarge", vendor: "Intel Granite", cores: 4, isa: "AVX-512", qps: 934,  p50: 10.6, usd: 0.3748, color: "#818cf8" },
  { name: "m8i.2xlarge", vendor: "Intel Granite", cores: 4, isa: "AVX-512", qps: 924,  p50: 11.1, usd: 0.4234, color: "#818cf8" },
  { name: "mac2.metal",  vendor: "Apple M1",      cores: 8, isa: "NEON",    qps: 712,  p50: 5.7,  usd: 0.6485, color: "#fb7185" },
];

type Metric = "perbuck" | "qps";

import { useState } from "react";

export function HardwareLab() {
  const [metric, setMetric] = useState<Metric>("perbuck");
  // perbuck = QPS per $1,000/yr of on-demand spend ($/hr × 8760 / 1000).
  const val = (r: Row) => (metric === "perbuck" ? r.qps / (r.usd * 8.76) : r.qps);
  const unit = metric === "perbuck" ? "QPS per $1k/yr" : "funnel QPS";
  const rows = [...ROWS].sort((a, b) => val(b) - val(a));
  const max = Math.max(...rows.map(val));

  return (
    <div>
      <div className="mb-3 flex gap-2 font-mono text-[11px]">
        {(["perbuck", "qps"] as Metric[]).map((m) => (
          <button
            key={m}
            onClick={() => setMetric(m)}
            className={`rounded px-2 py-1 ${metric === m ? "bg-accent text-ink" : "bg-grid text-dim hover:text-text"}`}
          >
            {m === "perbuck" ? "perf per $" : "raw QPS"}
          </button>
        ))}
        <span className="ml-auto self-center text-dim">recall 0.9952 — held constant</span>
      </div>
      <div className="space-y-2.5">
        {rows.map((r) => {
          const v = val(r);
          return (
            <div key={r.name}>
              <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
                <span className="text-text">
                  {r.name} <span className="text-dim">· 8 vCPU ({r.cores} phys) · {r.isa}</span>
                </span>
                <span className="tabular-nums" style={{ color: r.color }}>
                  {metric === "perbuck" ? Math.round(v).toLocaleString() : `${Math.round(v)} QPS`}
                  <span className="ml-2 text-dim">${Math.round(r.usd * 8760).toLocaleString()}/yr</span>
                </span>
              </div>
              <div className="h-5 overflow-hidden rounded bg-grid">
                <div className="h-full rounded" style={{ width: `${Math.max(2, (v / max) * 100)}%`, background: r.color }} />
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-3 border-t border-border pt-3 text-[13px] leading-6 text-dim">
        {unit}, same engine on every box.{" "}
        <span className="text-violet">AMD Zen5</span> wins both raw and per-dollar — not magic silicon,
        but <span className="text-text">8 real AVX-512 cores</span> at the &ldquo;8 vCPU&rdquo; tier vs Intel&rsquo;s
        <span className="text-text"> 4 + hyperthreading</span>. <span className="text-green">Graviton4</span> has 8 cores too
        but half-width NEON popcount; <span className="text-rose">Apple M1</span> is the for-the-LOLs floor.
        Throughput ≈ <span className="text-text">physical cores × popcount width</span>.
      </p>
    </div>
  );
}
