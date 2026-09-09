import catalog from "./data/style-atlas.json";

export type StyleEntry = {
  id: string;
  category: string;
  lead: string;
  rest: string;
  text: string;
  caption?: string;
  clips: string[];
};

export type StyleCatalog = {
  upstream: string;
  revision: string;
  dataset: string;
  categories: string[];
  styles: StyleEntry[];
};

export const STYLE_ATLAS = catalog as StyleCatalog;

const SAFE_LEAD = /^(A|An|The|In|Inside|On|At|Across|Under|Over|Through|Along|Beside|Behind|Beneath|Outside|From|Two|Three|Four|Five|Six)\b/;

let vocabulary: string[] = [];

export function setStyleVocabulary(phrases: string[]) {
  vocabulary = [...new Set(phrases.map((p) => p.trim()).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .map((p) => p.toLowerCase());
}

function leadLength(text: string): number {
  const lower = text.toLowerCase();
  for (const phrase of vocabulary) {
    if (!lower.startsWith(phrase)) continue;
    const after = text[phrase.length];
    if (after === undefined || /[\s,.;:—-]/.test(after)) return phrase.length;
  }
  return 0;
}

/** Continuity `leadWithStyle`: style leads, existing prompt follows after a comma. Re-applying swaps the previous style. */
export function leadWithStyle(text: string, phrase: string): string {
  const descriptor = String(phrase ?? "").trim().replace(/[,.;:\s]+$/, "");
  const body = String(text ?? "").trim();
  const rest = body.slice(leadLength(body)).replace(/^[\s,.;:—-]+/, "");
  if (!descriptor) return rest;
  if (!rest) return descriptor;
  const tail = SAFE_LEAD.test(rest) ? rest[0].toLowerCase() + rest.slice(1) : rest;
  return `${descriptor}, ${tail}`;
}

export function ensureStyleVocabulary() {
  if (vocabulary.length) return;
  setStyleVocabulary(STYLE_ATLAS.styles.flatMap((s) => [s.text, s.caption ?? ""]));
}

export function styleThumbUrl(id: string): string {
  return `/style-atlas/${id}.webp`;
}

export function filterStyles(query: string, category: string): StyleEntry[] {
  const q = query.trim().toLowerCase();
  return STYLE_ATLAS.styles.filter((s) => {
    if (category && s.category !== category) return false;
    if (!q) return true;
    return `${s.lead} ${s.text} ${s.caption ?? ""} ${s.category}`.toLowerCase().includes(q);
  });
}

ensureStyleVocabulary();
