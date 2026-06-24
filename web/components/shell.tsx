import type { ReactNode } from "react";

// Monospace uppercase eyebrow above the title.
export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[11px] uppercase tracking-[0.35em] text-amber">{children}</p>
  );
}

// Page header: clean sans title with eyebrow + intro.
export function PageHeader({
  eyebrow,
  title,
  children,
}: {
  eyebrow: ReactNode;
  title: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="mb-10 border-b border-border pb-10">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h1 className="mt-3 text-5xl font-semibold tracking-tight leading-[1.05] text-text sm:text-6xl">
        {title}
      </h1>
      {children ? <div className="mt-6 text-[18px] leading-8 text-body">{children}</div> : null}
    </header>
  );
}

// Numbered experiment figure — the interactive instrument, framed editorially.
export function Card({ n, title, children }: { n: string; title: string; children: ReactNode }) {
  return (
    <figure className="my-8 overflow-hidden rounded-2xl border border-border bg-panel shadow-[0_1px_2px_rgba(0,0,0,0.3),0_8px_24px_-12px_rgba(0,0,0,0.5)]">
      <figcaption className="flex items-baseline gap-3 border-b border-border bg-grid/40 px-5 py-3">
        <span className="font-mono text-[11px] text-amber">{n}</span>
        <span className="font-mono text-[12px] uppercase tracking-wider text-dim">{title}</span>
      </figcaption>
      <div className="p-5">{children}</div>
    </figure>
  );
}

// A compact stat, e.g. headline numbers.
export function Stat({ value, label }: { value: ReactNode; label: ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-panel px-4 py-3">
      <div className="font-mono text-2xl text-accent tabular-nums">{value}</div>
      <div className="mt-1 text-xs text-dim">{label}</div>
    </div>
  );
}
