import Link from "next/link";
import { notFound } from "next/navigation";
import { Eyebrow } from "@/components/shell";
import { Markdown } from "@/components/markdown";
import { getExperiment, listExperiments } from "@/lib/experiments";

export function generateStaticParams() {
  return listExperiments().map((e) => ({ slug: e.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = getExperiment(slug);
  return { title: e ? `${e.num} · ${e.title}` : "Experiment" };
}

export default async function ExperimentPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = getExperiment(slug);
  if (!e) notFound();

  const all = listExperiments();
  const i = all.findIndex((x) => x.slug === slug);
  const prev = i > 0 ? all[i - 1] : null;
  const next = i < all.length - 1 ? all[i + 1] : null;

  return (
    <article>
      <Link href="/experiments" className="font-mono text-xs text-dim hover:text-text">
        <span className="text-cyan">←</span> experiments
      </Link>
      <header className="mt-6 mb-8 border-b border-border pb-6">
        <Eyebrow>Experiment {e.num}</Eyebrow>
        <h1 className="mt-3 font-serif text-3xl font-medium italic leading-tight text-text">
          {e.title}
        </h1>
      </header>
      <Markdown>{e.src}</Markdown>
      <nav className="mt-12 flex justify-between gap-4 border-t border-border pt-6 font-mono text-xs">
        {prev ? (
          <Link href={`/experiments/${prev.slug}`} className="text-dim hover:text-text">
            <span className="text-cyan">←</span> {prev.num} {prev.title}
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/experiments/${next.slug}`} className="text-right text-dim hover:text-text">
            {next.num} {next.title} <span className="text-cyan">→</span>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </article>
  );
}
