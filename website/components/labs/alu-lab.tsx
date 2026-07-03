"use client";

import { useState } from "react";

// ── 072 kernel: where the machine spends itself, per doc (8 comparisons) ──────
// uop estimates from the objdump walk-through (notes 072) + Zen5 port model:
// gathers are 1 instruction but ~12 uops of machinery each on Zen5.
type Seg = { key: string; name: string; uops: number; kind: "waste" | "work" | "book"; note: string };

const K072: Seg[] = [
  {
    key: "ptr",
    name: "Block 1 — rebuild 8 query pointers",
    uops: 10,
    kind: "waste",
    note: "vmovq/vpermq/vpunpcklqdq/vinserti64x4: de-interleave 8 fat pointers (ptr,len) into one zmm of addresses. Loop-invariant — the addresses never change across 10M docs — but the heap acc store blocks hoisting.",
  },
  {
    key: "gather",
    name: "4× vpgatherqq — refetch query words",
    uops: 48,
    kind: "waste",
    note: "Each gather fetches 8 qwords from 8 computed addresses (~12 uops of machinery). Re-reads the SAME 256 B of query data every doc. The doc side proves the counterfactual: its 4 words sit hoisted in zmm1–4 at zero per-doc cost.",
  },
  {
    key: "math",
    name: "the actual math — XOR·POPCNT·ADD",
    uops: 18,
    kind: "work",
    note: "4× vpxorq + 4× vpopcntq + 4× vpmovqd + 3× vpaddd. Query-planar: lane j accumulates query j's full 256-bit distance. The old horizontal-reduction tax (vpmovqb/vpsadbw/vmovd per doc) is GONE — distances are born lane-separated.",
  },
  {
    key: "store",
    name: "store acc + loop",
    uops: 3,
    kind: "book",
    note: "One 32 B vmovdqu writes all 8 distances (the scalar kernel did 8 stores). add/cmp/jne close the doc loop. This heap store is also what forces Blocks 1–2 to re-run — LLVM can't prove it doesn't alias the query pointers.",
  },
];

const K073: Seg[] = [
  {
    key: "bcast",
    name: "4× vpbroadcastq — doc words to lanes",
    uops: 4,
    kind: "work",
    note: "Doc's four words, each broadcast across 8 lanes. The only per-doc data movement left.",
  },
  {
    key: "math",
    name: "the actual math — XOR·POPCNT·ADD",
    uops: 15,
    kind: "work",
    note: "Same query-planar math. Queries live in 4 hoisted zmm registers (loaded once per tile from a stack array — provably non-aliasing, so LLVM pins them). No pointers, no gathers.",
  },
  {
    key: "store",
    name: "store acc + loop",
    uops: 3,
    kind: "book",
    note: "Unchanged: one store, three loop uops.",
  },
];

// ── Zen5 port pressure: what each resource has vs what the kernel asks of it ──
// capacity = uops/cycle the resource accepts; demand = kernel uops per doc-iteration
// mapped onto it (072). The binding resource reads 100%.
type Port = { name: string; cap: string; used072: number; used073: number; note: string };
const PORTS: Port[] = [
  {
    name: "dispatch (8 uops/cyc)",
    cap: "8/cyc",
    used072: 100,
    used073: 34,
    note: "THE binding resource. 79 uops/doc ÷ 8-wide ≈ 10 cycles/doc. Every bookkeeping uop displaces math here.",
  },
  {
    name: "load/AGU (2×512b + AGUs)",
    cap: "2×512b/cyc",
    used072: 85,
    used073: 12,
    note: "Gathers hammer the AGUs with 32 scattered qword fetches per doc. Planar: one 32 B doc line per doc.",
  },
  {
    name: "vector ALU (4×512b)",
    cap: "4 pipes",
    used072: 38,
    used073: 55,
    note: "XOR/add — capacity to spare in both. The shuffle-class subset (2 pipes) eats Block 1's permutes in 072.",
  },
  {
    name: "popcount (2 of the 4)",
    cap: "2 pipes",
    used072: 20,
    used073: 62,
    note: "4 zmm popcounts per doc. The theoretical floor: 2/cyc = 4 docs/cyc. 073 moves it toward binding — which is the goal.",
  },
  {
    name: "scalar ALU (6)",
    cap: "6/cyc",
    used072: 10,
    used073: 18,
    note: "Loop control, heap compares. Six wide — never the limiter; it steers while vector pipes chew.",
  },
];

const COLOR: Record<Seg["kind"], string> = {
  waste: "var(--color-rose)",
  work: "var(--color-accent)",
  book: "var(--color-dim)",
};

export function AluLedger() {
  const [mode, setMode] = useState<"072" | "073">("072");
  const [hover, setHover] = useState<string | null>(null);
  const segs = mode === "072" ? K072 : K073;
  const total = segs.reduce((s, x) => s + x.uops, 0);
  const cycles = (total / 8).toFixed(1);
  const perCmp = (total / 8 / 8).toFixed(2);
  const active = segs.find((s) => s.key === hover);

  return (
    <div>
      <div className="mb-4 flex items-center gap-2 font-mono text-[12px]">
        {(["072", "073"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded border px-3 py-1 transition-colors ${
              mode === m
                ? "border-border-hover bg-grid text-text"
                : "border-border text-dim hover:text-body"
            }`}
          >
            {m === "072" ? "072 — as shipped" : "073 — planar (predicted)"}
          </button>
        ))}
        <span className="ml-auto text-dim">
          <span className="text-text tabular-nums">{total}</span> uops/doc ·{" "}
          <span className="text-text tabular-nums">≈{cycles}</span> cyc/doc ·{" "}
          <span className="text-text tabular-nums">{perCmp}</span> cyc/comparison
        </span>
      </div>

      {/* the ledger bar: one doc's dispatch budget, 8 comparisons */}
      <div className="flex h-9 w-full gap-[2px] overflow-hidden rounded">
        {segs.map((s) => (
          <button
            key={s.key}
            onMouseEnter={() => setHover(s.key)}
            onMouseLeave={() => setHover(null)}
            onFocus={() => setHover(s.key)}
            onBlur={() => setHover(null)}
            className="relative h-full min-w-0 rounded-[3px] transition-opacity"
            style={{
              width: `${(s.uops / (mode === "072" ? 79 : 79)) * 100}%`,
              background: COLOR[s.kind],
              opacity: hover && hover !== s.key ? 0.35 : s.kind === "book" ? 0.55 : 0.9,
            }}
            aria-label={`${s.name}: ${s.uops} uops`}
          >
            {s.uops >= 8 ? (
              <span className="pointer-events-none absolute inset-0 flex items-center justify-center font-mono text-[11px] text-[#0a0a0c]">
                {s.uops}
              </span>
            ) : null}
          </button>
        ))}
        {mode === "073" ? (
          <div
            className="h-full rounded-[3px] border border-dashed border-border"
            style={{ width: `${((79 - total) / 79) * 100}%` }}
            aria-label="reclaimed dispatch budget"
          />
        ) : null}
      </div>

      {/* legend + direct labels (identity never by color alone) */}
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-dim">
        <span>
          <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: COLOR.waste }} />
          re-established every doc (invariant — waste)
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: COLOR.work }} />
          hamming math
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: COLOR.book, opacity: 0.55 }} />
          store + loop
        </span>
        {mode === "073" ? (
          <span>
            <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm border border-dashed border-border align-[-1px]" />
            reclaimed (58 uops)
          </span>
        ) : null}
      </div>

      {/* hover detail */}
      <div className="mt-3 min-h-[72px] rounded-lg border border-border bg-grid/40 px-4 py-3 text-[13px] leading-6 text-body">
        {active ? (
          <>
            <span className="font-mono text-[12px] text-text">{active.name}</span>
            <span className="font-mono text-[12px] text-dim"> · {active.uops} uops</span>
            <p className="mt-1 text-dim">{active.note}</p>
          </>
        ) : (
          <p className="text-dim">
            Hover a segment. Width = share of the doc-loop&apos;s dispatch budget (8 uops/cycle,
            the binding resource). One iteration = one doc × 8 register-resident queries.
          </p>
        )}
      </div>
    </div>
  );
}

export function AluPorts() {
  const [mode, setMode] = useState<"072" | "073">("072");
  return (
    <div>
      <div className="mb-4 flex items-center gap-2 font-mono text-[12px]">
        {(["072", "073"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded border px-3 py-1 transition-colors ${
              mode === m
                ? "border-border-hover bg-grid text-text"
                : "border-border text-dim hover:text-body"
            }`}
          >
            {m === "072" ? "072 — as shipped" : "073 — planar (predicted)"}
          </button>
        ))}
      </div>
      <div className="space-y-3">
        {PORTS.map((p) => {
          const v = mode === "072" ? p.used072 : p.used073;
          const binding = v >= 95;
          return (
            <div key={p.name}>
              <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
                <span className="text-text">
                  {p.name}
                  {binding ? <span className="ml-2 text-rose">← binds</span> : null}
                </span>
                <span className="tabular-nums text-dim">{v}%</span>
              </div>
              <div className="h-4 overflow-hidden rounded bg-grid">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${v}%`,
                    background: binding ? "var(--color-rose)" : "var(--color-accent)",
                    opacity: binding ? 0.9 : 0.7,
                  }}
                />
              </div>
              <p className="mt-1 text-[12px] leading-5 text-dim">{p.note}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── the annotated hot loop, block by block ─────────────────────────────────────
type AsmBlock = {
  key: string;
  label: string;
  kind: Seg["kind"];
  lines: [string, string][]; // [asm, gloss]
};

const ASM: AsmBlock[] = [
  {
    key: "b1",
    label: "Block 1 · a128f — rebuild 8 query pointers (every doc; they never change)",
    kind: "waste",
    lines: [
      ["vmovq  0x70(%rax),%xmm7", "load fat-ptr pieces from qrows (Vec<&[u64]>: ptr,len,ptr,len…)"],
      ["vpermq $0xe8,(%rax),%ymm6", "pluck the even qwords — addresses; discard the lens"],
      ["vpunpcklqdq %xmm7,%xmm8,%xmm7", "interleave two more pointers"],
      ["vinserti64x4 $1,%ymm7,%zmm5,%zmm5", "stitch → zmm5 = 8 query ADDRESSES"],
    ],
  },
  {
    key: "b2a",
    label: "Block 2a · a12e6 — gather query words (same 256 B, refetched 10M×)",
    kind: "waste",
    lines: [
      ["kxnorb %k0,%k0,%k1", "all-ones mask (each gather consumes k1)"],
      ["vpgatherqq 0x0(,%zmm5),%zmm6{%k1}", "word 0 of ALL 8 queries — ~12 uops of machinery"],
      ["vpgatherqq 0x8(,%zmm5),%zmm7{%k1}", "word 1 of all 8 (words 2,3 gathered below)"],
    ],
  },
  {
    key: "b2b",
    label: "Block 2b — the math (query-planar: lane j = query j)",
    kind: "work",
    lines: [
      ["vpxorq   %zmm2,%zmm6,%zmm6", "doc.w0 (broadcast, hoisted ✓) ^ 8 queries' w0"],
      ["vpopcntq %zmm6,%zmm6", "8 lane-counts, ONE instruction (2-of-4 pipes)"],
      ["vpmovqd  %zmm6,%ymm6", "narrow u64 → u32 lanes"],
      ["vpaddd   %ymm6,%ymm7,%ymm6", "accumulate per lane — NO horizontal reduction, ever"],
    ],
  },
  {
    key: "b3",
    label: "Block 3 · a1376 — store + loop (the store that blocks all hoisting)",
    kind: "book",
    lines: [
      ["vmovdqu %ymm5,(%rdx,%rcx,4)", "acc[0..8] = 8 distances, one 32 B store — HEAP Vec"],
      ["add $0x8,%rcx / cmp / jne", "tile cursor += 8; next doc"],
    ],
  },
  {
    key: "b5",
    label: "Block 5 · a13ec — selection (per query; hot path = one cmp, rare insert)",
    kind: "book",
    lines: [
      ["mov (%rax,%r15,4),%eax", "h = acc[j]"],
      ["cmp (%rcx),%r13 / jae skip", "vs heap root — L1-pinned, 0.06% mispredict, 99.83% end here"],
      ["call grow_one", "Vec-grow check on the (rare) insert path — pre-sized, never grows"],
    ],
  },
];

export function AluAsm() {
  const [open, setOpen] = useState<string | null>("b1");
  return (
    <div className="space-y-2">
      {ASM.map((b) => {
        const isOpen = open === b.key;
        return (
          <div key={b.key} className="overflow-hidden rounded-lg border border-border">
            <button
              onClick={() => setOpen(isOpen ? null : b.key)}
              className="flex w-full items-center gap-3 bg-grid/40 px-4 py-2.5 text-left font-mono text-[12px] text-body hover:text-text"
            >
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-sm"
                style={{ background: COLOR[b.kind], opacity: b.kind === "book" ? 0.55 : 0.9 }}
              />
              <span className="min-w-0 flex-1">{b.label}</span>
              <span className="shrink-0 text-dim">{isOpen ? "−" : "+"}</span>
            </button>
            {isOpen ? (
              <div className="divide-y divide-border/60 bg-panel">
                {b.lines.map(([asm, gloss]) => (
                  <div key={asm} className="grid gap-1 px-4 py-2 sm:grid-cols-[minmax(0,22rem)_1fr] sm:gap-4">
                    <code className="font-mono text-[12px] text-text">{asm}</code>
                    <span className="text-[12px] leading-5 text-dim">{gloss}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
