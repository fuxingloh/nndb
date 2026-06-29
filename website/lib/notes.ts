import fs from "node:fs";
import path from "node:path";

// Notes live in the repo root /notes — the source of truth the measure scripts
// write to. The site reads it directly and serves each entry at /notes/<slug>.
// Two kinds share the folder: numbered, measured *experiments* (001–NNN, each with
// a .json sibling) and ♫-marked *notes* (filename prefix `note-`, no measurement) —
// parked directions and external references. The ♫ is the cross-reference marker.
const DIR = path.join(process.cwd(), "..", "notes");

export type Note = {
  slug: string; // e.g. "046-cell-size-residual" or "note-hnsw"
  num: string; // "046" for experiments, "♫" for notes
  title: string; // first H1, sans leading "NNN — " or "♫ "
  src: string; // raw markdown
  isNote: boolean; // true = a ♫ note, not a measured experiment
};

function titleOf(src: string, slug: string): string {
  const h1 = src.match(/^#\s+(.+)$/m)?.[1] ?? slug;
  return h1.replace(/^\d+\s*[—–-]\s*/, "").replace(/^♫\s*/, "");
}

function toNote(file: string): Note {
  const slug = file.replace(/\.md$/, "");
  const src = fs.readFileSync(path.join(DIR, file), "utf8");
  const isNote = slug.startsWith("note-");
  return { slug, num: isNote ? "♫" : slug.slice(0, 3), title: titleOf(src, slug), src, isNote };
}

export function listNotes(): Note[] {
  return fs
    .readdirSync(DIR)
    .filter((f) => f.endsWith(".md"))
    .map(toNote)
    .sort((a, b) => a.slug.localeCompare(b.slug)); // experiments (0–9) first, ♫ notes after
}

export function getNote(slug: string): Note | null {
  const file = path.join(DIR, `${slug}.md`);
  if (!fs.existsSync(file)) return null;
  return toNote(`${slug}.md`);
}
