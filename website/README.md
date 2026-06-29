# web — the writeup

A learning article (Lab + Prose) on the in-memory vector-search engine: what was built,
why, and what turned out best. Next.js (App Router) + MDX, Ayu-dark editorial style.

- `app/page.mdx` — the article (prose + interactive labs in `<Card>`s)
- `components/labs/` — interactive labs: `residual-lab` (the centroid-subtraction
  intuition), `frontier-lab` (measured recall-vs-QPS, funnel vs PQ)
- `components/shell.tsx` — PageHeader / Card / Stat ; `mdx-components.tsx` — prose styling

```bash
npm install
npm run dev    # http://localhost:3000
npm run build
```

Numbers come from the measured `history/` entries; this is the curated "things that
worked" view, not the full history.
