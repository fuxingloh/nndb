import fs from "node:fs";
import path from "node:path";

// History lives in the repo root (the source of truth that the measure scripts write
// to). The site reads it directly and serves each entry at /experiments/<slug>.
const DIR = path.join(process.cwd(), "..", "history");

export type Experiment = {
  slug: string; // e.g. "046-cell-size-residual"
  num: string; // "046"
  title: string; // first H1, sans leading "NNN — "
  src: string; // raw markdown
};

function titleOf(src: string, slug: string): string {
  const h1 = src.match(/^#\s+(.+)$/m)?.[1] ?? slug;
  return h1.replace(/^\d+\s*[—–-]\s*/, "");
}

export function listExperiments(): Experiment[] {
  return fs
    .readdirSync(DIR)
    .filter((f) => f.endsWith(".md"))
    .map((f) => {
      const slug = f.replace(/\.md$/, "");
      const src = fs.readFileSync(path.join(DIR, f), "utf8");
      return { slug, num: slug.slice(0, 3), title: titleOf(src, slug), src };
    })
    .sort((a, b) => a.slug.localeCompare(b.slug));
}

export function getExperiment(slug: string): Experiment | null {
  const file = path.join(DIR, `${slug}.md`);
  if (!fs.existsSync(file)) return null;
  const src = fs.readFileSync(file, "utf8");
  return { slug, num: slug.slice(0, 3), title: titleOf(src, slug), src };
}
