import type { Metadata } from "next";
import Link from "next/link";
import { Card, Eyebrow, Stat } from "@/components/shell";
import { PortPressure074, PortRouting, ZmmAnatomy } from "@/components/labs/alu074-lab";

export const metadata: Metadata = {
  title: "074 · Inside the 512-bit register — the loop at its floor",
  description:
    "The final kernel, instruction by instruction: what each zmm lane holds at every stage, which Zen5 port every instruction routes to, and why the next factor of two isn't in the uops.",
};

export default function Alu074() {
  return (
    <article>
      <Link href="/notes/074-wide-store" className="font-mono text-xs text-dim hover:text-text">
        <span className="text-accent">←</span> notes 074
      </Link>

      <header className="mt-6 mb-10 border-b border-border pb-8">
        <Eyebrow>Instrument · kernel autopsy Ⅲ</Eyebrow>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight leading-tight text-text">
          074: inside the 512-bit register
        </h1>
        <p className="mt-5 text-[17px] leading-8 text-body">
          The loop is finished — ~20 uops per doc-group, nothing left that isn&apos;t data
          movement, math, or one store. This page is the full breakdown of <em>how</em> it uses
          the machine: what sits in each lane of the zmm register at every stage, and which of
          Zen5&apos;s execution ports every instruction routes to. Read alongside{" "}
          <Link href="/alu/072" className="text-accent hover:underline">/alu/072</Link> and{" "}
          <Link href="/alu/073" className="text-accent hover:underline">/alu/073</Link> — this is
          where their predictions landed.
        </p>
      </header>

      <div className="mb-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat value="1,471" label="QPS @ 10M, C=2000, recall 0.974" tone="green" />
        <Stat value="~2.3" label="uops per comparison (was ~18 in 067)" />
        <Stat value="8×" label="comparisons per instruction — lane parallelism" />
        <Stat value="2×" label="loop unroll LLVM chose (16 comparisons/iter)" />
      </div>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          The design question a 512-bit register forces:{" "}
          <span className="text-text">what varies across the 8 lanes?</span> Put one doc&apos;s 4
          words there and the register is half-empty and the answer needs a horizontal sum
          (069&apos;s tax). Put 2 docs there and you need <em>two</em> horizontal sums. The
          shipped kernel puts <span className="text-text">8 queries&apos; same-word</span> there
          — the register is full, each query&apos;s distance accumulates vertically in its own
          lane, and no lane ever needs another lane&apos;s value. The tile of 8 isn&apos;t a
          bandwidth trick anymore; it&apos;s literally the shape of the register.
        </p>
      </section>

      <Card n="fig. 1" title="one zmm, five stages — step through the register">
        <ZmmAnatomy />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          Now the emitted loop itself (objdump <code className="font-mono text-[13px]">a0bd0–a0cb8</code>,
          the committed 074 binary). LLVM unrolled it 2× — one iteration handles two groups of 8
          queries against one doc — and made two scheduling moves worth noticing: word 3 is
          processed <em>before</em> word 2, and the last XOR is threaded between the adds. The
          machine orders by operand readiness, not by anyone&apos;s source code.
        </p>
      </section>

      <Card n="fig. 2" title="the loop, instruction by instruction — port routing">
        <PortRouting />
      </Card>

      <Card n="fig. 3" title="port pressure — what binds at the floor">
        <PortPressure074 />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          The pressure profile is the story of the whole arc inverted. In 067–072 the binding
          resource was <span className="text-text">dispatch</span> — the front-end couldn&apos;t
          issue bookkeeping fast enough to feed the ALUs. Now dispatch has slack and the{" "}
          <span className="text-text">load pipes</span> bind: 16 loads per iteration against 2
          per cycle. Even so, half those loads are the <em>same doc words twice</em> (the 2×
          unroll re-broadcasts them) — and hand-hoisting them measured <em>slower</em> (1622 vs
          1685 QPS at C=500): the re-broadcasts ride otherwise-idle load slots, and the manual
          hoist perturbed LLVM&apos;s schedule. The negative control is in{" "}
          <Link href="/notes/074-wide-store" className="text-accent hover:underline">notes 074</Link>.
        </p>
        <p>
          Which is the closing statement of the uop war:{" "}
          <span className="text-text">
            when removing work makes the loop slower, the loop is done.
          </span>{" "}
          At 10M the scan interleaves this in-cache picture with the DRAM stream (~44 GB/s at
          C=500, nearing the per-core concurrency ceiling of 068) — so the next factor of two
          lives in <span className="text-text">bytes</span> (smaller codes move the stream) or{" "}
          <span className="text-text">locality</span> (CCX-resident shards move the source), not
          in any instruction this page could show.
        </p>
        <p className="text-dim">
          Port mappings: Zen5 has 4 × 512-bit vector pipes (VPOPCNTQ accepted by 2), 2 × 512-bit
          loads + 1 store per cycle to L1, 6 scalar ALUs, 8-wide dispatch. Utilization figures
          are steady-state estimates from instruction counts against those widths; QPS is
          measured (5 reps, CV ≤ 1%).
        </p>
      </section>

      <nav className="mt-12 flex justify-between gap-4 border-t border-border pt-6 font-mono text-xs">
        <Link href="/alu/073" className="text-dim hover:text-text">
          <span className="text-accent">←</span> /alu/073 — kernel autopsy Ⅱ
        </Link>
        <Link href="/notes/074-wide-store" className="text-dim hover:text-text">
          notes 074 <span className="text-accent">→</span>
        </Link>
      </nav>
    </article>
  );
}
