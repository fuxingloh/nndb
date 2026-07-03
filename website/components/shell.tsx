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

// A compact stat, e.g. headline numbers. `tone` lifts one stat out of the
// indigo default — used for the era-defining corpus sizes (1M amber, 10M green)
// so the scale jump reads at a glance: tinted value, border, and wash.
export function Stat({
  value,
  label,
  tone,
}: {
  value: ReactNode;
  label: ReactNode;
  tone?: "amber" | "green";
}) {
  const styles =
    tone === "amber"
      ? { box: "border-amber/50 bg-amber/10", val: "text-amber" }
      : tone === "green"
        ? { box: "border-green/50 bg-green/10", val: "text-green" }
        : { box: "border-border bg-panel", val: "text-accent" };
  return (
    <div className={`rounded-lg border px-4 py-3 ${styles.box}`}>
      <div className={`font-mono text-2xl tabular-nums ${styles.val}`}>{value}</div>
      <div className="mt-1 text-xs text-dim">{label}</div>
    </div>
  );
}
