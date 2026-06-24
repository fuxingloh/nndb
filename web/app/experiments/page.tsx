import Link from "next/link";
import { Eyebrow } from "@/components/shell";
import { listExperiments } from "@/lib/experiments";

export const metadata = {
  title: "Experiments · A 1-bit vector search engine",
  description: "Every measured experiment, in order — the full experiment log.",
};

export default function ExperimentsIndex() {
  const items = listExperiments();
  return (
    <div>
      <Link href="/" className="font-mono text-xs text-dim hover:text-text">
        <span className="text-accent">←</span> the writeup
      </Link>
      <header className="mt-6 mb-10 border-b border-border pb-8">
        <Eyebrow>Experiment log · {items.length} entries</Eyebrow>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-text">Experiments</h1>
        <p className="mt-4 max-w-2xl text-[16px] leading-7 text-body">
          Every idea was a numbered, measured entry — wins and dead ends alike. The writeup
          curates what worked; this is the raw trail, in order.
        </p>
      </header>
      <ol className="space-y-0">
        {items.map((e) => (
          <li key={e.slug}>
            <Link
              href={`/experiments/${e.slug}`}
              className="group flex items-baseline gap-4 border-b border-border py-3 hover:bg-panel"
            >
              <span className="font-mono text-sm text-amber tabular-nums">{e.num}</span>
              <span className="text-[16px] leading-6 text-body group-hover:text-text">{e.title}</span>
            </Link>
          </li>
        ))}
      </ol>
    </div>
  );
}
