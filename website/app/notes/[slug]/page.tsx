import Link from "next/link";
import { notFound } from "next/navigation";
import { Eyebrow } from "@/components/shell";
import { Markdown } from "@/components/markdown";
import { getNote, listNotes } from "@/lib/notes";

export function generateStaticParams() {
  return listNotes().map((e) => ({ slug: e.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = getNote(slug);
  if (!e) return { title: "Note" };
  return { title: e.isNote ? `♫ ${e.title}` : `${e.num} · ${e.title}` };
}

export default async function NotePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = getNote(slug);
  if (!e) notFound();

  const all = listNotes();
  const i = all.findIndex((x) => x.slug === slug);
  const prev = i > 0 ? all[i - 1] : null;
  const next = i < all.length - 1 ? all[i + 1] : null;

  return (
    <article>
      <Link href="/notes" className="font-mono text-xs text-dim hover:text-text">
        <span className="text-accent">←</span> notes
      </Link>
      <header className="mt-6 mb-8 border-b border-border pb-6">
        <Eyebrow>{e.isNote ? "♫ Note" : `Experiment ${e.num}`}</Eyebrow>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight leading-tight text-text">
          {e.title}
        </h1>
      </header>
      <Markdown>{e.src}</Markdown>
      <nav className="mt-12 flex justify-between gap-4 border-t border-border pt-6 font-mono text-xs">
        {prev ? (
          <Link href={`/notes/${prev.slug}`} className="text-dim hover:text-text">
            <span className="text-accent">←</span> {prev.num} {prev.title}
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/notes/${next.slug}`} className="text-right text-dim hover:text-text">
            {next.num} {next.title} <span className="text-accent">→</span>
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </article>
  );
}
