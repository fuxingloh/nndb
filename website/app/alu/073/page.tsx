import type { Metadata } from "next";
import Link from "next/link";
import { Card, Eyebrow, Stat } from "@/components/shell";
import { Alu073Asm, Alu073Ledger, KernelArc } from "@/components/labs/alu073-lab";

export const metadata: Metadata = {
  title: "073 · The ALU ledger — the waste is dead; one tax remains",
  description:
    "The 073 word-planar kernel, disassembled: gathers and pointer surgery gone, the math near-minimal, and the store-out as the last answer-format tax.",
};

export default function Alu073() {
  return (
    <article>
      <Link href="/notes/073-word-planar-query-groups" className="font-mono text-xs text-dim hover:text-text">
        <span className="text-accent">←</span> notes 073
      </Link>

      <header className="mt-6 mb-10 border-b border-border pb-8">
        <Eyebrow>Instrument · kernel autopsy Ⅱ</Eyebrow>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight leading-tight text-text">
          073: the invariant waste is dead — one tax remains
        </h1>
        <p className="mt-5 text-[17px] leading-8 text-body">
          <Link href="/alu/072" className="text-accent hover:underline">/alu/072</Link> predicted
          what killing the pointer rebuild and gathers would reclaim. This is the follow-up with
          the measurement in hand: queries transposed once per tile into word-planar stack rows,
          the doc loop rewritten as AVX-512 intrinsics — and the emitted assembly re-read for
          what to do next.
        </p>
      </header>

      <div className="mb-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat value="1307" label="QPS @ 10M, C=2000, recall 0.974 (batch=32)" />
        <Stat value="1542" label="QPS @ C=500, recall 0.922" />
        <Stat value="0" label="vpgatherqq in the funnel (was 4/doc)" />
        <Stat value="T=32" label="new tiling optimum (was 8) — √(s/k) as predicted" />
      </div>

      <Card n="fig. 1" title="the dispatch ledger — same x-scale as 072's 79-uop budget">
        <Alu073Ledger />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          The math came out exactly as designed: doc words broadcast from the stream, query rows
          as <span className="text-text">folded loads at compile-time offsets</span> — the stack
          transpose means <code className="font-mono text-[13px]">0x00/0x40/0x80/0xc0</code>{" "}
          <em>are</em> the addressing, no pointers to distill, nothing to gather. LLVM even
          arranged the adds as a pairwise tree.
        </p>
        <p>
          But one red block survived, and it&apos;s new: the{" "}
          <span className="text-text">store-out</span>. The kernel finishes with 8 complete
          distances in one zmm — and then disassembles that register lane by lane: 8
          extract/store pairs, each with its own tail-guard branch, ~24 uops un-vectorizing what
          the math just vectorized. The same answer-format law from 072, one stage later:{" "}
          <span className="text-text">
            the cost was never the arithmetic — it&apos;s the format the next consumer demands.
          </span>{" "}
          074 is the two-instruction fix: <code className="font-mono text-[13px]">vpmovqd</code>{" "}
          narrows the lanes, one <code className="font-mono text-[13px]">vmovdqu</code> writes
          the group. Toggle the ledger to see the predicted shape.
        </p>
      </section>

      <Card n="fig. 2" title="the kernel arc — measured, recall bit-identical throughout">
        <KernelArc />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p>
          Two things this arc teaches. First, the detour that didn&apos;t ship: written as safe
          scalar code, the planar loop <em>scalarized</em> — LLVM&apos;s popcount-idiom matcher
          doesn&apos;t recognize the shape it happily built itself in 072 from a gather pattern
          (822 and 736 QPS on the way to 1185). The kernel is intrinsics now; 012&apos;s
          &ldquo;stay out of the autovectorizer&apos;s way&rdquo; ends with an asterisk —{" "}
          <span className="text-text">
            when the pattern-matcher can&apos;t see your shape, you write the instructions
            yourself.
          </span>
        </p>
        <p>
          Second, the tiling knee moved 8 → 32 exactly as √(s/k) predicts when per-query tile
          state collapses from pointer machinery to 32 shared bytes in zmm rows. The scan now
          pulls ~44 GB/s of code stream at C=500 — within sight of the per-core streaming
          ceiling (068). The next factor of two lives in{" "}
          <span className="text-text">bytes or CCX-resident shards, not uops.</span>
        </p>
      </section>

      <Card n="fig. 3" title="the emitted hot loop (objdump a0eac–a0fe5)">
        <Alu073Asm />
      </Card>

      <section className="space-y-4 text-[15px] leading-7 text-body">
        <p className="text-dim">
          QPS measured on c8a.4xlarge (Zen5, 16 vCPU), 10M Snowflake arctic-256, exact ground
          truth, 3 reps (CV ≤ 3.5%). Uop counts read from the disassembly; the 074 column is a
          prediction — the experiment exists to falsify it.
        </p>
      </section>

      <nav className="mt-12 flex justify-between gap-4 border-t border-border pt-6 font-mono text-xs">
        <Link href="/alu/072" className="text-dim hover:text-text">
          <span className="text-accent">←</span> /alu/072 — the 072 autopsy
        </Link>
        <Link href="/notes/073-word-planar-query-groups" className="text-dim hover:text-text">
          notes 073 <span className="text-accent">→</span>
        </Link>
      </nav>
    </article>
  );
}
