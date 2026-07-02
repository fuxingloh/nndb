"use client";

import { useState } from "react";

// Measured: 256-bit binary funnel + exact f32 rerank on OpenAI text-embedding-3-large
// (MRL) truncated to 256 dims — 990k base, Zen5 16 vCPU (notes 065/066). The slider is
// the rerank width C: recall climbs 0.947 → 0.998 with no plateau, because the true
// neighbors ARE in the Hamming shortlist (all 990k codes unique — ranking error only).
// The ghost line is the counterfactual: the same 256-bit budget on a NON-Matryoshka
// embedding (Cohere bit-floor, note 058) capped at ~0.72 — the embedding's training,
// not the engine, decides whether 256 bits works.
type Pt = { c: number; recall: number; itq: number; qps: number; p50: number; p99: number };
const SWEEP: Pt[] = [
  { c: 500, recall: 0.9474, itq: 0.9541, qps: 6041, p50: 2.05, p99: 2.1 },
  { c: 1000, recall: 0.9731, itq: 0.978, qps: 5275, p50: 2.41, p99: 2.49 },
  { c: 2000, recall: 0.9878, itq: 0.9907, qps: 4227, p50: 3.09, p99: 3.15 },
  { c: 4000, recall: 0.9944, itq: 0.9968, qps: 3092, p50: 4.4, p99: 4.58 },
  { c: 8000, recall: 0.9979, itq: 0.9992, qps: 2026, p50: 6.54, p99: 6.88 },
];
const NON_MRL = 0.72; // the bit-floor collapse: non-Matryoshka embedding at 256 bits

const W = 420;
const H = 300;
const PAD = 44;
const x0 = Math.log10(400);
const x1 = Math.log10(10000);
const sx = (c: number) => PAD + ((Math.log10(c) - x0) / (x1 - x0)) * (W - 2 * PAD);
const sy = (r: number) => H - PAD - ((r - 0.68) / (1.0 - 0.68)) * (H - 2 * PAD);

export function MatryoshkaLab() {
  const [i, setI] = useState(2); // default C=2000, the chosen operating point
  const [itq, setItq] = useState(true);
  const p = SWEEP[i];
  const recall = itq ? p.itq : p.recall;

  const path = (key: "recall" | "itq") =>
    SWEEP.map((s, j) => `${j ? "L" : "M"}${sx(s.c)},${sy(s[key])}`).join(" ");

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <label className="flex items-center gap-2 font-mono text-[11px] text-dim">
          rerank C
          <input
            type="range"
            min={0}
            max={SWEEP.length - 1}
            step={1}
            value={i}
            onChange={(e) => setI(Number(e.target.value))}
            className="w-36 accent-[#818cf8]"
          />
          <span className="text-accent tabular-nums">{p.c}</span>
        </label>
        <button
          onClick={() => setItq((b) => !b)}
          className="flex items-center gap-1.5 rounded-md border border-border bg-grid px-2.5 py-1 font-mono text-[11px] text-text"
          style={{ opacity: itq ? 1 : 0.4 }}
        >
          <span className="inline-block h-2 w-2 rounded-full bg-[#34d399]" />
          learned rotation (ITQ)
        </button>
      </div>

      <div className="mb-3 grid grid-cols-4 gap-2">
        <Metric label="recall@10" value={recall.toFixed(4)} accent />
        <Metric label="QPS" value={p.qps.toLocaleString()} />
        <Metric label="p50" value={`${p.p50} ms`} />
        <Metric label="p99" value={`${p.p99} ms`} />
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {[0.7, 0.8, 0.9, 1.0].map((r) => (
          <g key={r}>
            <line x1={PAD} y1={sy(r)} x2={W - PAD} y2={sy(r)} stroke="#1b1b21" />
            <text x={PAD - 8} y={sy(r) + 3} textAnchor="end" fontSize={9} className="fill-dim font-mono">
              {r.toFixed(2)}
            </text>
          </g>
        ))}
        {SWEEP.map((s) => (
          <text key={s.c} x={sx(s.c)} y={H - PAD + 14} textAnchor="middle" fontSize={9} className="fill-dim font-mono">
            {s.c}
          </text>
        ))}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} className="fill-dim">
          rerank shortlist C (log) →
        </text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={10} className="fill-dim" transform={`rotate(-90 12 ${H / 2})`}>
          recall@10 →
        </text>

        {/* the counterfactual: non-Matryoshka 256-bit ceiling */}
        <line x1={PAD} y1={sy(NON_MRL)} x2={W - PAD} y2={sy(NON_MRL)} stroke="#fb7185" strokeWidth={1} strokeDasharray="4 4" opacity={0.7} />
        <text x={W - PAD} y={sy(NON_MRL) - 6} textAnchor="end" fontSize={9} className="font-mono" fill="#fb7185">
          non-Matryoshka @ 256 bits ≈ 0.72 (bit-floor)
        </text>

        {/* measured curves */}
        <path d={path("recall")} fill="none" stroke="#818cf8" strokeWidth={1.5} opacity={itq ? 0.35 : 0.9} />
        {itq && <path d={path("itq")} fill="none" stroke="#34d399" strokeWidth={1.5} opacity={0.9} />}
        {SWEEP.map((s, j) => (
          <circle
            key={s.c}
            cx={sx(s.c)}
            cy={sy(itq ? s.itq : s.recall)}
            r={j === i ? 6 : 3.5}
            fill={itq ? "#34d399" : "#818cf8"}
            opacity={j === i ? 1 : 0.55}
          />
        ))}
        {/* active point callout */}
        <text x={sx(p.c)} y={sy(recall) - 10} textAnchor="middle" fontSize={10} className="font-mono" fill={itq ? "#34d399" : "#818cf8"}>
          {recall.toFixed(4)}
        </text>
      </svg>

      <p className="mt-2 text-[13px] leading-6 text-dim">
        Matryoshka-256 codes (32 B/vec, a 31.7 MB index for 990k vectors): recall dials
        smoothly to <span className="text-green">0.999</span> because every code is unique —
        the shortlist just needs to be wide enough. The dashed line is the same 256-bit
        budget on an embedding <em>not</em> trained for truncation: it caps at ~0.72 no
        matter the C. The model, not the engine, decides whether 256 bits is enough.
        (QPS/latency shown are the 065 sweep, tile=8; the final config drops tiling for
        ~+18% QPS — 0.9907 @ ~4,966 QPS at C=2000, note 066.)
      </p>
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-grid/40 px-3 py-2">
      <div className={`font-mono text-sm tabular-nums ${accent ? "text-green" : "text-text"}`}>{value}</div>
      <div className="mt-0.5 text-[10px] text-dim">{label}</div>
    </div>
  );
}
