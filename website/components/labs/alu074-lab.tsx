"use client";

import { useState } from "react";

/* ── fig 1: the 512-bit register, stage by stage ─────────────────────────────
   One zmm = 8 × 64-bit lanes. The kernel's whole trick is WHAT each lane holds:
   lane j always belongs to query j, from broadcast through the final store. */

type Stage = { key: string; title: string; insn: string; lanes: string[]; note: string };

const STAGES: Stage[] = [
  {
    key: "bcast",
    title: "1 · broadcast — one doc word fills all 8 lanes",
    insn: "vpbroadcastq (%rax,%rdi,8),%zmm0",
    lanes: ["d.w0", "d.w0", "d.w0", "d.w0", "d.w0", "d.w0", "d.w0", "d.w0"],
    note: "The doc's word 0 (8 bytes from the streamed code array) is copied into every lane. Load + broadcast fuse into ONE uop on Zen5 — the register is 'primed' with the same question for 8 different queries. The doc's 256 bits are never together in one register; they arrive one word at a time.",
  },
  {
    key: "xor",
    title: "2 · XOR — against 8 different queries at once",
    insn: "vpxorq (%rdx),%zmm0,%zmm0",
    lanes: ["d.w0^q0.w0", "^q1.w0", "^q2.w0", "^q3.w0", "^q4.w0", "^q5.w0", "^q6.w0", "^q7.w0"],
    note: "The memory operand is a word-planar row: 8 queries' word-0, contiguous (the once-per-tile transpose put them there). The load folds into the XOR — one uop. After this, lane j holds the differing-bits pattern between the doc and QUERY j. The lanes have diverged: same doc, 8 different comparisons.",
  },
  {
    key: "popcnt",
    title: "3 · popcount — 8 partial distances, one instruction",
    insn: "vpopcntq %zmm0,%zmm0",
    lanes: ["14", "31", "9", "22", "28", "17", "25", "30"],
    note: "VPOPCNTDQ counts bits within each lane independently. Lane j now holds 'how many of word 0's 64 bits differ between the doc and query j' — 8 partial Hamming distances from one instruction. Only 2 of Zen5's 4 vector pipes accept popcount; this is the kernel's structural ceiling.",
  },
  {
    key: "add",
    title: "4 · accumulate — distances grow vertically, per lane",
    insn: "vpaddq %zmm0,%zmm1,%zmm0",
    lanes: ["Σ q0", "Σ q1", "Σ q2", "Σ q3", "Σ q4", "Σ q5", "Σ q6", "Σ q7"],
    note: "Steps 1–3 repeat for words 1, 2, 3, and the adds fold the four partials as a pairwise tree: (w0+w1)+(w2+w3). Each query's total builds up in ITS OWN lane — no lane ever needs a value from another lane, so there is no shuffle, no horizontal reduction, no vector→scalar crossing anywhere in the loop. That absence is the whole 069→074 arc.",
  },
  {
    key: "store",
    title: "5 · narrow + store — 8 answers leave in one instruction",
    insn: "vpmovqd %zmm0,(%rsi)",
    lanes: ["u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32"],
    note: "The 074 fix: vpmovqd narrows the 8 × u64 counts to 8 × u32 AND stores them in a single instruction (32 B written to the padded acc array). Its 073 predecessor disassembled this register lane by lane — 8 extract/store pairs, 24 uops. The answer format finally matches the register format.",
  },
];

export function ZmmAnatomy() {
  const [stage, setStage] = useState(0);
  const s = STAGES[stage];
  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2 font-mono text-[12px]">
        {STAGES.map((x, i) => (
          <button
            key={x.key}
            onClick={() => setStage(i)}
            className={`rounded border px-3 py-1 transition-colors ${
              stage === i ? "border-border-hover bg-grid text-text" : "border-border text-dim hover:text-body"
            }`}
          >
            {i + 1}
          </button>
        ))}
        <span className="ml-auto self-center text-dim">zmm0 — 512 bits = 8 × 64-bit lanes</span>
      </div>

      <p className="mb-2 font-mono text-[13px] text-text">{s.title}</p>
      <code className="mb-3 block font-mono text-[12px] text-amber">{s.insn}</code>

      <div className="grid grid-cols-8 gap-[2px]">
        {s.lanes.map((l, j) => (
          <div key={j} className="rounded-[3px] border border-border bg-grid px-1 py-3 text-center">
            <div className="font-mono text-[10px] text-dim">lane {j}</div>
            <div className="mt-1 break-words font-mono text-[11px] text-accent">{l}</div>
            <div className="mt-1 font-mono text-[9px] text-dim">q{j}</div>
          </div>
        ))}
      </div>

      <p className="mt-3 min-h-[96px] rounded-lg border border-border bg-grid/40 px-4 py-3 text-[13px] leading-6 text-dim">
        {s.note}
      </p>
    </div>
  );
}

/* ── fig 2: the emitted loop with per-instruction port routing ───────────────
   The real objdump (a0bd0–a0cb8): LLVM 2×-unrolled the group loop. One
   iteration = one doc × 2 groups of 8 queries = 16 comparisons, 37 insns. */

type Insn = { asm: string; port: "load" | "vec-any" | "popcnt" | "store" | "scalar"; gloss: string };

const PORT_LABEL: Record<Insn["port"], { name: string; color: string }> = {
  load: { name: "load pipes (2×512b/cyc)", color: "var(--color-amber)" },
  "vec-any": { name: "vector ALU (4 pipes)", color: "var(--color-accent)" },
  popcnt: { name: "popcount (2 of the 4)", color: "var(--color-violet)" },
  store: { name: "store pipe (1×512b/cyc)", color: "var(--color-green)" },
  scalar: { name: "scalar ALU (6 — free)", color: "var(--color-dim)" },
};

const LOOP: Insn[] = [
  { asm: "vpbroadcastq (%rax,%rdi,8),%zmm0", port: "load", gloss: "doc.w0 → 8 lanes (load+broadcast, 1 uop)" },
  { asm: "vpbroadcastq 0x8(...),%zmm1", port: "load", gloss: "doc.w1 → 8 lanes" },
  { asm: "vpbroadcastq 0x18(...),%zmm2", port: "load", gloss: "doc.w3 → 8 lanes (scheduler reordered w3 before w2)" },
  { asm: "vpbroadcastq 0x10(...),%zmm3", port: "load", gloss: "doc.w2 → 8 lanes" },
  { asm: "add $0x2,%rcx", port: "scalar", gloss: "group counter — hides under the vector work" },
  { asm: "vpxorq (%rdx),%zmm0,%zmm0", port: "vec-any", gloss: "^ 8 queries' w0 (row at +0x00; load folds in)" },
  { asm: "vpxorq 0x40(%rdx),%zmm1,%zmm1", port: "vec-any", gloss: "^ 8 queries' w1 (row +0x40)" },
  { asm: "vpxorq 0xc0(%rdx),%zmm2,%zmm2", port: "vec-any", gloss: "^ 8 queries' w3 (row +0xc0)" },
  { asm: "vpopcntq %zmm0,%zmm0", port: "popcnt", gloss: "8 partial distances for w0" },
  { asm: "vpopcntq %zmm1,%zmm1", port: "popcnt", gloss: "8 partials for w1 — pairs with w0's on the 2 pipes" },
  { asm: "vpopcntq %zmm2,%zmm2", port: "popcnt", gloss: "8 partials for w3" },
  { asm: "vpaddq %zmm0,%zmm1,%zmm0", port: "vec-any", gloss: "w0+w1 — first branch of the pairwise tree" },
  { asm: "vpxorq 0x80(%rdx),%zmm3,%zmm1", port: "vec-any", gloss: "^ 8 queries' w2 — interleaved INTO the adds: the scheduler fills popcount latency with the next word's XOR" },
  { asm: "vpopcntq %zmm1,%zmm1", port: "popcnt", gloss: "8 partials for w2" },
  { asm: "vpaddq %zmm2,%zmm1,%zmm1", port: "vec-any", gloss: "w3+w2 — second branch" },
  { asm: "vpaddq %zmm1,%zmm0,%zmm0", port: "vec-any", gloss: "(w0+w1)+(w2+w3) → 8 complete distances" },
  { asm: "vpmovqd %zmm0,-0x20(%rsi)", port: "store", gloss: "narrow u64→u32 + store 32 B — group 1 done, ONE instruction" },
  { asm: "— group 2: same 16 insns against rows +0x100…+0x1c0 —", port: "vec-any", gloss: "LLVM unrolled 2× so two groups' dependency chains interleave in the ROB; the second group's broadcasts reload the same doc words (measured: hand-hoisting them was SLOWER — they ride free load slots)" },
  { asm: "add $0x200,%rdx / add $0x40,%rsi", port: "scalar", gloss: "advance query-row + acc pointers" },
  { asm: "cmp %rcx,%r8 / jne a0bd0", port: "scalar", gloss: "loop — predicted taken, ~free" },
];

export function PortRouting() {
  const [sel, setSel] = useState<number | null>(null);
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-dim">
        {Object.entries(PORT_LABEL).map(([k, v]) => (
          <span key={k}>
            <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: v.color }} />
            {v.name}
          </span>
        ))}
      </div>
      <div className="divide-y divide-border/60 overflow-hidden rounded-lg border border-border bg-panel">
        {LOOP.map((x, i) => (
          <button
            key={i}
            onClick={() => setSel(sel === i ? null : i)}
            className="grid w-full grid-cols-[10px_minmax(0,20rem)_1fr] items-start gap-3 px-3 py-1.5 text-left hover:bg-grid/40"
          >
            <span className="mt-1.5 h-2.5 w-2.5 rounded-sm" style={{ background: PORT_LABEL[x.port].color }} />
            <code className="font-mono text-[11.5px] leading-5 text-text">{x.asm}</code>
            <span className={`text-[11.5px] leading-5 ${sel === i ? "text-body" : "text-dim"}`}>{x.gloss}</span>
          </button>
        ))}
      </div>
      <p className="pt-3 text-[13px] leading-6 text-dim">
        37 instructions per iteration = one doc × <span className="text-text">16 comparisons</span> ≈
        2.3 uops/comparison. Note two scheduler moves worth savoring: the w2 XOR is threaded{" "}
        <em>between</em> the adds (filling popcount latency), and w3 is processed before w2 —
        LLVM ordering by readiness, not by our source order.
      </p>
    </div>
  );
}

/* ── fig 3: port pressure per iteration — what binds now ─────────────────── */

const PRESSURE = [
  { name: "dispatch (8 uops/cyc)", used: 84, note: "~37 uops / 8-wide ≈ 4.6 cyc floor per 16 comparisons. No longer pinned at 100% — the diet worked." },
  { name: "load (2×512b/cyc)", used: 100, note: "16 loads/iter (8 broadcasts + 8 folded XOR rows) ÷ 2 per cycle = 8 cycles of load issue — THE binding pipe in-cache. The next uop to remove is a load, not an ALU op." },
  { name: "popcount (2 of 4 pipes)", used: 87, note: "8 vpopcntq ÷ 2 pipes = 4 cycles. Near-saturated — exactly where a popcount kernel should sit." },
  { name: "vector ALU (4 pipes)", used: 55, note: "8 XORs + 6 adds across 4 pipes. Headroom — XOR capacity was never the problem." },
  { name: "store (1×512b/cyc)", used: 25, note: "2 vpmovqd per iteration. The 073 store-out burned ~6× this." },
  { name: "scalar ALU (6)", used: 12, note: "4 pointer/counter ops + compare/branch. Six-wide means it never queues." },
];

export function PortPressure074() {
  return (
    <div className="space-y-3">
      {PRESSURE.map((p) => {
        const binding = p.used >= 95;
        return (
          <div key={p.name}>
            <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
              <span className="text-text">
                {p.name}
                {binding ? <span className="ml-2 text-rose">← binds</span> : null}
              </span>
              <span className="tabular-nums text-dim">{p.used}%</span>
            </div>
            <div className="h-4 overflow-hidden rounded bg-grid">
              <div
                className="h-full rounded"
                style={{ width: `${p.used}%`, background: binding ? "var(--color-rose)" : "var(--color-accent)", opacity: binding ? 0.9 : 0.7 }}
              />
            </div>
            <p className="mt-1 text-[12px] leading-5 text-dim">{p.note}</p>
          </div>
        );
      })}
    </div>
  );
}
