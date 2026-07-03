import type { Metadata } from "next";
import Link from "next/link";
import { Card, Eyebrow, Stat } from "@/components/shell";
import { AluAsm, AluLedger, AluPorts } from "@/components/labs/alu-lab";

export const metadata: Metadata = {
  title: "072 · The ALU ledger — where the kernel's cycles actually go",
  description:
    "The 072 hot loop, disassembled and mapped onto Zen5's execution resources: what binds, what's waste, and what 073 reclaims.",
};

export default function Alu072() {
  return (
    <article>
      <Link href="/notes/072-unsafe-width-cast" className="font-mono text-xs text-dim hover:text-text">
        <span className="text-accent">←</span> notes 072
      </Link>

      <header className="mt-6 mb-10 border-b border-border pb-8">
        <Eyebrow>Instrument · kernel autopsy</Eyebrow>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight leading-tight text-text">
          The ALU ledger: one doc through the 072 kernel
        </h1>
        <p className="mt-5 text-[17px] leading-8 text-body">
          After the unsafe width cast, LLVM re-vectorized the tile loop: one iteration now
          compares <span className="text-text">one doc against 8 queries in SIMD lanes</span>.
          This page maps the emitted assembly (objdump, Zen5 EPYC 9R45) onto the machine&apos;s
          execution resources — so the next optimization can be read straight off the ledger.
        </p>
      </header>

      <div className="mb-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat value="1106" label="QPS @ 10M, C=2000, recall 0.974 (072, batch=8)" />
        <Stat value="~79" label="uops per doc-iteration (8 comparisons)" />
        <Stat value="~73%" label="of that budget re-establishes loop-invariant facts" />
        <Stat value="8/cyc" label="dispatch width — the resource that binds" />
      </div>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          The rule this page keeps applying:{" "}
          <span className="text-text">
            throughput = uops per item ÷ dispatch width, and the uops that count are the ones
            your answer format forces.
          </span>{" "}
          The math below is ~18 uops per doc. Everything else is the machine re-proving, ten
          million times, facts that were fixed before the loop began.
        </p>
      </section>

      <Card n="fig. 1" title="the dispatch ledger — one doc's uop budget">
        <AluLedger />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          The red segments are the finding. <span className="text-text">Block 1</span> rebuilds
          the 8 query addresses from Rust fat pointers — lane surgery that exists because{" "}
          <code className="font-mono text-[13px]">qrows</code> is a{" "}
          <code className="font-mono text-[13px]">Vec&lt;&amp;[u64]&gt;</code>. The{" "}
          <span className="text-text">gathers</span> then refetch the same 256 bytes of query
          data, every doc. Neither computes anything about Hamming distances. LLVM cannot hoist
          either one: the accumulator store writes through a heap <code className="font-mono text-[13px]">Vec</code>,
          and the compiler can&apos;t prove that store doesn&apos;t overwrite the query pointers.
          The doc side shows the counterfactual — its four words sit pre-broadcast in{" "}
          <code className="font-mono text-[13px]">zmm1–4</code>, hoisted, free.
        </p>
        <p>
          Toggling to <span className="text-text">073 — planar</span> shows the same iteration
          with queries transposed once per tile into a stack array{" "}
          <code className="font-mono text-[13px]">[[u64; 8]; 4]</code>: addresses become
          compile-time offsets, gathers become four hoistable loads, the aliasing question
          evaporates (stack can&apos;t alias heap), and 58 of 79 uops come off the ledger.
        </p>
      </section>

      <Card n="fig. 2" title="port pressure — what binds, before and after">
        <AluPorts />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          Dispatch is pinned at 100% in 072 — the front-end cannot issue the bookkeeping fast
          enough to keep the vector pipes busy. The popcount pipes (the theoretical ceiling:
          2×512-bit per cycle = 4 docs/cycle) idle at ~20%. 073 flips the profile: dispatch
          relaxes, popcount rises toward binding —{" "}
          <span className="text-text">
            which is the goal. A kernel is done when its dedicated instruction is the bottleneck.
          </span>
        </p>
      </section>

      <Card n="fig. 3" title="the emitted hot loop, block by block (objdump)">
        <AluAsm />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          Worth savoring in Block 2b: there is no horizontal reduction anywhere. Every previous
          kernel paid 3–5 shuffle-class uops per doc converting lane counts into the one scalar
          the heap demands; the vectorized loop keeps lane <em>j</em> as query <em>j</em>&apos;s
          running distance and stores all eight at once. The compiler independently arrived at
          the answer-format fix we designed by hand — once the unsafe cast gave it a provable
          fixed trip count.
        </p>
        <p className="text-dim">
          Numbers: uop counts and port mappings are estimates from the objdump walk-through and
          the Zen5 port model (notes 071/072); QPS is measured (CV ≤ 0.9%). The 073 column is a
          prediction — the experiment exists to falsify it.
        </p>
      </section>

      <nav className="mt-12 flex justify-between gap-4 border-t border-border pt-6 font-mono text-xs">
        <Link href="/notes/072-unsafe-width-cast" className="text-dim hover:text-text">
          <span className="text-accent">←</span> 072 unsafe width cast
        </Link>
        <Link href="/notes" className="text-dim hover:text-text">
          all notes <span className="text-accent">→</span>
        </Link>
      </nav>
    </article>
  );
}
