"use client";

import { Streamdown } from "streamdown";
import "streamdown/styles.css";

// Render a raw note (.md) with the same editorial styling as the article.
// Streamdown (react-markdown-compatible components API, GFM built in) so
// arbitrary note prose — `<`, math, JSON — never breaks parsing; its table
// controls add copy/download to the notes' many result tables.
// mode="static": notes are files, not model streams — no caret, no animation.
export function Markdown({ children }: { children: string }) {
  return (
    <Streamdown
      mode="static"
      parseIncompleteMarkdown={false}
      lineNumbers={false}
      controls={{ table: { copy: true, download: true, fullscreen: false }, code: false, mermaid: false }}
      components={{
        h1: () => null, // the page header renders the title
        h2: (props) => <h2 className="mt-10 mb-2 text-2xl font-semibold tracking-tight text-text" {...props} />,
        h3: (props) => <h3 className="mt-6 mb-2 font-mono text-xs uppercase tracking-wider text-amber" {...props} />,
        p: (props) => <p className="my-3 text-[15px] leading-7 text-body" {...props} />,
        ul: (props) => <ul className="my-3 space-y-1" {...props} />,
        ol: (props) => <ol className="my-3 space-y-1" {...props} />,
        li: (props) => <li className="ml-5 list-disc text-[15px] leading-7 text-body" {...props} />,
        strong: (props) => <strong className="font-medium text-text" {...props} />,
        a: (props) => <a className="text-accent hover:underline" {...props} />,
        code: (props) => <code className="rounded bg-grid px-1 py-0.5 font-mono text-[0.85em] text-text" {...props} />,
        pre: (props) => (
          <pre className="my-4 overflow-x-auto rounded-lg border border-border bg-panel p-3 font-mono text-[12px] leading-6 text-text [&_code]:bg-transparent [&_code]:p-0" {...props} />
        ),
        blockquote: (props) => (
          <blockquote className="my-4 rounded-r-md border-l-[3px] border-amber bg-grid/60 px-4 py-3 text-[14px] leading-6 text-body [&>p]:my-0 [&_strong]:text-text" {...props} />
        ),
        // Tables: NO overrides — Streamdown owns them completely (its native
        // look: styled wrapper, header row, borders, plus copy/download
        // controls). Its theme reads the shadcn-style tokens set in globals.css.
        hr: () => <hr className="my-8 border-border" />,
      }}
    >
      {children}
    </Streamdown>
  );
}
