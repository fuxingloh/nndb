import type { MDXComponents } from "mdx/types";
import Link from "next/link";
import { Card, Stat } from "@/components/shell";

// Editorial prose styling for every .mdx page, plus globally-available experiment figures.
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    p: (props) => <p className="my-4 text-[17px] leading-8 text-body" {...props} />,
    h2: (props) => (
      <h2 className="mt-14 mb-3 font-serif text-3xl font-medium italic text-text" {...props} />
    ),
    h3: (props) => (
      <h3 className="mt-8 mb-2 font-mono text-sm uppercase tracking-wider text-amber" {...props} />
    ),
    ul: (props) => <ul className="my-4 space-y-1" {...props} />,
    ol: (props) => <ol className="my-4 space-y-1" {...props} />,
    li: (props) => <li className="ml-5 list-disc text-[17px] leading-8 text-body" {...props} />,
    strong: (props) => <strong className="font-medium text-text" {...props} />,
    em: (props) => <em className="italic" {...props} />,
    code: (props) => (
      <code className="rounded bg-grid px-1.5 py-0.5 font-mono text-[0.85em] text-text" {...props} />
    ),
    pre: (props) => (
      <pre
        className="my-5 overflow-x-auto rounded-lg border border-border bg-panel p-4 font-mono text-[13px] leading-6 text-text [&>code]:bg-transparent [&>code]:p-0"
        {...props}
      />
    ),
    a: ({ href, ...rest }) => (
      <Link href={href ?? "#"} className="text-cyan hover:underline" {...rest} />
    ),
    blockquote: (props) => (
      <blockquote
        className="my-6 rounded-r-md border-l-[3px] border-amber bg-grid/60 px-5 py-4 text-[15px] leading-7 text-body [&>p]:my-0 [&_strong]:text-text"
        {...props}
      />
    ),
    hr: () => <hr className="my-12 border-border" />,
    table: (props) => (
      <div className="my-6 overflow-x-auto">
        <table className="w-full border-collapse text-left text-sm" {...props} />
      </div>
    ),
    thead: (props) => <thead className="border-b border-border-hover" {...props} />,
    th: (props) => (
      <th
        className="px-3 py-2 font-mono text-[11px] font-medium uppercase tracking-wider text-dim"
        {...props}
      />
    ),
    tr: (props) => <tr className="border-b border-border last:border-0" {...props} />,
    td: (props) => (
      <td className="px-3 py-2 text-[15px] leading-7 text-body tabular-nums" {...props} />
    ),
    Card,
    Stat,
    ...components,
  };
}
