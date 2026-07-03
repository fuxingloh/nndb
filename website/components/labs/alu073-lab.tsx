"use client";

import { useState } from "react";

// ── 073 kernel as SHIPPED (objdump a0eac–a0fe5): one doc × 8 queries ──────────
type Seg = { key: string; name: string; uops: number; kind: "waste" | "work" | "book"; note: string };

const K073: Seg[] = [
  {
    key: "bcast",
    name: "4× vpbroadcastq — doc words to lanes",
    uops: 4,
    kind: "work",
    note: "Doc's four words broadcast from the streamed code array, load+broadcast fused to one uop each. The only per-doc data movement left on the input side.",
  },
  {
    key: "math",
    name: "XOR·POPCNT·ADD — 8 distances in one zmm",
    uops: 11,
    kind: "work",
    note: "4× vpxorq with the query load FOLDED (qw rows are compile-time offsets off the stack array — 0x00/0x40/0x80/0xc0 ARE the transpose), 4× vpopcntq, 3× vpaddq in a pairwise tree (LLVM's gift: shorter dependency chain than serial accumulation). No gathers, no pointer surgery, no horizontal reduction.",
  },
  {
    key: "storeout",
    name: "the store-out — 8 lane extractions",
    uops: 24,
    kind: "waste",
    note: "vmovd/vpextrd/vextracti128/vextracti32x4 per lane, each with its own cmp/je tail-guard, plus a reload of the acc pointer between extractions (the 072 aliasing ghost, back in miniature). The kernel builds 8 distances in one register — then disassembles it lane by lane. ~24 uops un-vectorizing what the math just vectorized.",
  },
  {
    key: "loop",
    name: "loop control",
    uops: 3,
    kind: "book",
    note: "add/cmp + group bookkeeping. Fine.",
  },
];

// 074 (predicted): one vpmovqd narrows the 8 qword counts to 8 dwords; one vmovdqu
// stores all of acc[0..8]. Tail handled outside the loop.
const K074: Seg[] = [
  { key: "bcast", name: "4× vpbroadcastq — doc words to lanes", uops: 4, kind: "work", note: "Unchanged." },
  {
    key: "math",
    name: "XOR·POPCNT·ADD — 8 distances in one zmm",
    uops: 11,
    kind: "work",
    note: "Unchanged — the math was already right.",
  },
  {
    key: "store",
    name: "vpmovqd + vmovdqu — one wide store",
    uops: 2,
    kind: "work",
    note: "_mm512_cvtepi64_epi32 narrows the 8 qword counts to 8 dwords; one 32 B store writes the whole group's acc. The 8 extract/store pairs and 8 tail-guards collapse to 2 uops; tail bounds move outside the loop.",
  },
  { key: "loop", name: "loop control", uops: 3, kind: "book", note: "Unchanged." },
];

const COLOR: Record<Seg["kind"], string> = {
  waste: "var(--color-rose)",
  work: "var(--color-accent)",
  book: "var(--color-dim)",
};

const TOTAL_072 = 79; // shared x-scale with /alu/072 so widths are comparable

export function Alu073Ledger() {
  const [mode, setMode] = useState<"073" | "074">("073");
  const [hover, setHover] = useState<string | null>(null);
  const segs = mode === "073" ? K073 : K074;
  const total = segs.reduce((s, x) => s + x.uops, 0);
  const active = segs.find((s) => s.key === hover);

  return (
    <div>
      <div className="mb-4 flex items-center gap-2 font-mono text-[12px]">
        {(["073", "074"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded border px-3 py-1 transition-colors ${
              mode === m
                ? "border-border-hover bg-grid text-text"
                : "border-border text-dim hover:text-body"
            }`}
          >
            {m === "073" ? "073 — as shipped (measured)" : "074 — wide store (predicted)"}
          </button>
        ))}
        <span className="ml-auto text-dim">
          <span className="text-text tabular-nums">{total}</span> uops/doc ·{" "}
          <span className="text-text tabular-nums">≈{(total / 8).toFixed(1)}</span> cyc/doc ·{" "}
          <span className="text-text tabular-nums">{(total / 8 / 8).toFixed(2)}</span> cyc/comparison
        </span>
      </div>

      {/* same x-scale as the 072 ledger: the empty space IS the 072→073 win */}
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
              width: `${(s.uops / TOTAL_072) * 100}%`,
              background: COLOR[s.kind],
              opacity: hover && hover !== s.key ? 0.35 : s.kind === "book" ? 0.55 : 0.9,
            }}
            aria-label={`${s.name}: ${s.uops} uops`}
          >
            {s.uops >= 6 ? (
              <span className="pointer-events-none absolute inset-0 flex items-center justify-center font-mono text-[11px] text-[#0a0a0c]">
                {s.uops}
              </span>
            ) : null}
          </button>
        ))}
        <div
          className="h-full rounded-[3px] border border-dashed border-border"
          style={{ width: `${((TOTAL_072 - total) / TOTAL_072) * 100}%` }}
          aria-label="budget reclaimed vs 072"
        />
      </div>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-dim">
        <span>
          <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: COLOR.work }} />
          hamming math + data movement
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: COLOR.waste }} />
          answer-format tax (store-out)
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: COLOR.book, opacity: 0.55 }} />
          loop
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm border border-dashed border-border align-[-1px]" />
          reclaimed vs 072&apos;s 79-uop budget
        </span>
      </div>

      <div className="mt-3 min-h-[72px] rounded-lg border border-border bg-grid/40 px-4 py-3 text-[13px] leading-6 text-body">
        {active ? (
          <>
            <span className="font-mono text-[12px] text-text">{active.name}</span>
            <span className="font-mono text-[12px] text-dim"> · {active.uops} uops</span>
            <p className="mt-1 text-dim">{active.note}</p>
          </>
        ) : (
          <p className="text-dim">
            Hover a segment. Same x-scale as the 072 ledger — the dashed region is the budget
            073 reclaimed. The red that remains is the last answer-format tax in the pipeline.
          </p>
        )}
      </div>
    </div>
  );
}

// ── the kernel arc: measured QPS at C=2000, recall 0.9737, 10M, 16 vCPU ───────
const ARC = [
  { v: "067", label: "generic loop", qps: 650 },
  { v: "069", label: "fixed-width unroll", qps: 847 },
  { v: "071", label: "typed + both-length gate", qps: 955 },
  { v: "072", label: "unsafe cast → LLVM re-vectorizes", qps: 1093 },
  { v: "073", label: "word-planar groups (T=32)", qps: 1307 },
];
const ARC_MAX = 1500;

export function KernelArc() {
  return (
    <div className="space-y-3">
      {ARC.map((a, i) => (
        <div key={a.v}>
          <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
            <span className="text-text">
              {a.v} <span className="text-dim">— {a.label}</span>
            </span>
            <span className="tabular-nums" style={{ color: i === ARC.length - 1 ? "var(--color-green)" : "var(--color-accent)" }}>
              {a.qps} QPS
            </span>
          </div>
          <div className="h-5 overflow-hidden rounded bg-grid">
            <div
              className="h-full rounded"
              style={{
                width: `${(a.qps / ARC_MAX) * 100}%`,
                background: i === ARC.length - 1 ? "var(--color-green)" : "var(--color-accent)",
                opacity: 0.5 + 0.5 * ((i + 1) / ARC.length),
              }}
            />
          </div>
        </div>
      ))}
      <p className="pt-1 text-[13px] leading-6 text-dim">
        C=2000, recall 0.9737 — bit-identical at every step. 2.0× from kernel work alone;
        recall never moved. Each step was one change, one measurement, one objdump.
      </p>
    </div>
  );
}

// ── the emitted hot loop, block by block ───────────────────────────────────────
type AsmBlock = {
  key: string;
  label: string;
  kind: Seg["kind"];
  lines: [string, string][];
};

const ASM: AsmBlock[] = [
  {
    key: "bcast",
    label: "a0eac — doc words: broadcast from the stream",
    kind: "work",
    lines: [
      ["vpbroadcastq 0x18(%rax,%rdi,8),%zmm0", "doc.w3 → all 8 lanes — load+broadcast fused, one uop"],
      ["vpbroadcastq 0x10(%rax,%rdi,8),%zmm1", "doc.w2"],
      ["vpbroadcastq 0x8(%rax,%rdi,8),%zmm2", "doc.w1"],
      ["vpbroadcastq (%rax,%rdi,8),%zmm2", "doc.w0 (register reused after w1 consumed)"],
    ],
  },
  {
    key: "math",
    label: "a0ec9 — the math: folded loads, pairwise add tree",
    kind: "work",
    lines: [
      ["vpxorq 0xc0(%r8,%rdx,1),%zmm0,%zmm0", "qw[3][0..8] ^ doc.w3 — the offset IS the transpose; no gather"],
      ["vpxorq 0x80(%r8,%rdx,1),%zmm1,%zmm1", "qw[2] ^ doc.w2 (0x00/0x40/0x80/0xc0 = compile-time rows)"],
      ["vpopcntq %zmm0,%zmm0", "8 lane-counts for w3"],
      ["vpopcntq %zmm1,%zmm1", "8 lane-counts for w2"],
      ["vpaddq %zmm0,%zmm1,%zmm0", "w3+w2 — pairwise tree, shorter dep chain than serial"],
      ["vpxorq 0x40(...)/vpxorq (...) …", "w1, w0 same shape; final vpaddq → zmm0 = 8 complete distances"],
    ],
  },
  {
    key: "storeout",
    label: "a0f1d — the store-out: 8 lane extractions (the 074 target)",
    kind: "waste",
    lines: [
      ["vmovd  %xmm0,-0x20(%r9,%rcx,4)", "lane 0 → acc[0]"],
      ["vpextrd $0x2,%xmm0,-0x1c(%r9,%rcx,4)", "lane 1 — each extraction pairs with a cmp/je tail-guard"],
      ["vextracti128 $0x1,%ymm0,%xmm1", "lanes 2–3 need the upper half pulled down first"],
      ["vextracti32x4 $0x3,%zmm0,%xmm0", "…and lanes 6–7 the top quarter. 8 extract/store pairs total"],
      ["mov 0x18(%rsp),%r9  (between each!)", "acc pointer reloaded per extraction — the aliasing ghost again"],
    ],
  },
];

export function Alu073Asm() {
  const [open, setOpen] = useState<string | null>("math");
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
                  <div key={asm} className="grid gap-1 px-4 py-2 sm:grid-cols-[minmax(0,24rem)_1fr] sm:gap-4">
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
