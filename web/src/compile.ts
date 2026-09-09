import type { CastMember, ReferenceItem, RoutingMode } from "./types";

const VIDEO_KINDS = new Set(["silent_video", "video", "video_audio"]);

export interface CompileInput {
  prompt: string;
  refs: ReferenceItem[];
  castMembers: CastMember[];
  selectedCastIds: string[];
  routing: RoutingMode;
  imagePath?: string | null;
  endImagePath?: string | null;
}

export interface CompileToken {
  handle: string;
  token: string;
  name: string;
  kind: ReferenceItem["kind"];
}

export interface CompileResult {
  compiledPrompt: string;
  refs: ReferenceItem[];
  modeHint: string;
  tokens: CompileToken[];
  warnings: string[];
  hadAtTokens: boolean;
}

export function isRefEnabled(ref: ReferenceItem): boolean {
  return ref.enabled !== false;
}

export function activeRefs(refs: ReferenceItem[]): ReferenceItem[] {
  return refs.filter(isRefEnabled);
}

export function isVideoKind(kind: ReferenceItem["kind"]): boolean {
  return VIDEO_KINDS.has(kind);
}

export function kindHandlePrefix(kind: ReferenceItem["kind"]): "img" | "vid" | "aud" {
  if (kind === "image") return "img";
  if (kind === "audio") return "aud";
  return "vid";
}

export function modelTokenFor(kind: ReferenceItem["kind"], index: number): string {
  if (kind === "image") return `Picture ${index}`;
  if (kind === "audio") return `Audio ${index}`;
  return `Video ${index}`;
}

/** 1-based index of this ref among unmuted refs of the same handle family. */
export function handleIndex(refs: ReferenceItem[], index: number): number {
  const ref = refs[index];
  const prefix = kindHandlePrefix(ref.kind);
  let n = 0;
  for (let i = 0; i <= index; i++) {
    if (kindHandlePrefix(refs[i].kind) === prefix) n++;
  }
  return n;
}

export function handleForRef(refs: ReferenceItem[], index: number): string {
  const ref = refs[index];
  return `@${kindHandlePrefix(ref.kind)}-${handleIndex(refs, index)}`;
}

export function tokensForRefs(refs: ReferenceItem[]): CompileToken[] {
  return refs.map((ref, i) => ({
    handle: handleForRef(refs, i),
    token: modelTokenFor(ref.kind, handleIndex(refs, i)),
    name: ref.name,
    kind: ref.kind,
  }));
}

function slugName(name: string): string {
  return name.trim().replace(/\s+/g, "");
}

function findCastByMention(members: CastMember[], mention: string): CastMember | undefined {
  const q = mention.toLowerCase();
  const exact = members.filter((m) => slugName(m.name).toLowerCase() === q);
  if (exact.length === 1) return exact[0];
  if (exact.length > 1) return undefined;
  const prefix = members.filter((m) => slugName(m.name).toLowerCase().startsWith(q));
  if (prefix.length === 1) return prefix[0];
  return undefined;
}

function isMediaHandleName(name: string): boolean {
  return /^(img|vid|aud)(-\d+)?$/i.test(name);
}

function mediaToRef(member: CastMember, media: CastMember["media"][number]): ReferenceItem {
  const kind: ReferenceItem["kind"] =
    media.type === "video" ? "video" : media.type === "audio" ? "audio" : "image";
  return {
    id: `cast-${member.id}-${media.id}`,
    kind,
    path: media.path,
    name: media.label || `${member.name} ${media.type}`,
    enabled: true,
    previewUrl: media.thumbnailUrl,
    source: "cast",
    castId: member.id,
    durationS: media.durationS,
  };
}

function mergeCastRefs(
  refs: ReferenceItem[],
  members: CastMember[],
  involvedIds: Set<string>,
): { refs: ReferenceItem[]; descriptions: string[] } {
  const paths = new Set(refs.map((r) => r.path));
  const next = [...refs];
  const descriptions: string[] = [];
  for (const member of members) {
    if (!involvedIds.has(member.id)) continue;
    const media = member.media ?? [];
    if (media.length === 0) {
      if (member.description?.trim()) {
        descriptions.push(`${member.name}: ${member.description.trim()}`);
      }
      continue;
    }
    for (const item of media) {
      if (!item.path || paths.has(item.path)) continue;
      paths.add(item.path);
      next.push(mediaToRef(member, item));
    }
  }
  return { refs: next, descriptions };
}

const AT_MEDIA = /@(img|vid|aud)-(\d+)/gi;
const AT_NAME = /@([A-Za-z][A-Za-z0-9_-]*)/g;

export function resolveMode(input: {
  routing: RoutingMode;
  refs: ReferenceItem[];
  imagePath?: string | null;
  endImagePath?: string | null;
}): string {
  const { routing, imagePath, endImagePath } = input;
  const refs = activeRefs(input.refs);
  if (routing === "ref2va") return "ref2va";
  if (routing === "fl2va") {
    if (imagePath && endImagePath) return "fl2va";
    if (endImagePath && !imagePath) return "last_frame";
    if (imagePath) return "first_frame";
    return "t2va";
  }
  if (refs.length > 0) return "ref2va";
  if (imagePath && endImagePath) return "fl2va";
  if (endImagePath && !imagePath) return "last_frame";
  if (imagePath) return "first_frame";
  return "t2va";
}

export function compilePrompt(input: CompileInput): CompileResult {
  const warnings: string[] = [];
  const base = activeRefs(input.refs);
  const mentioned = new Set<string>();
  const prompt = input.prompt ?? "";

  for (const match of prompt.matchAll(AT_NAME)) {
    const name = match[1];
    if (isMediaHandleName(name)) continue;
    const member = findCastByMention(input.castMembers, name);
    if (member) mentioned.add(member.id);
    else warnings.push(`Unknown @${name}`);
  }

  const involved = new Set<string>([...input.selectedCastIds, ...mentioned]);
  const merged = mergeCastRefs(base, input.castMembers, involved);
  const refs = merged.refs;
  const tokens = tokensForRefs(refs);
  const byHandle = new Map(tokens.map((t) => [t.handle.toLowerCase(), t]));

  let hadAtTokens = false;
  let compiled = prompt.replace(AT_MEDIA, (full, kind: string, num: string) => {
    hadAtTokens = true;
    const handle = `@${kind.toLowerCase()}-${num}`;
    const hit = byHandle.get(handle);
    if (!hit) {
      warnings.push(`No reference for ${handle}`);
      return full;
    }
    return hit.token;
  });

  compiled = compiled.replace(AT_NAME, (full, name: string) => {
    if (isMediaHandleName(name)) return full;
    hadAtTokens = true;
    const member = findCastByMention(input.castMembers, name);
    if (!member) return full;
    const cited = tokens.filter((_, i) => refs[i]?.castId === member.id).map((t) => t.token);
    if (cited.length) return cited.join(", ");
    return member.name;
  });

  if (!hadAtTokens) {
    compiled = prompt;
  }
  if (merged.descriptions.length && (involved.size > 0)) {
    const extra = merged.descriptions.join("\n");
    compiled = compiled.trim() ? `${compiled.trim()}\n${extra}` : extra;
  }

  const modeHint = resolveMode({
    routing: input.routing,
    refs,
    imagePath: input.imagePath,
    endImagePath: input.endImagePath,
  });

  return {
    compiledPrompt: compiled,
    refs: modeHint === "ref2va" ? refs : [],
    modeHint,
    tokens: modeHint === "ref2va" ? tokens : [],
    warnings,
    hadAtTokens,
  };
}

export function nearestDurationId(
  seconds: number,
  presets: { id: string; seconds?: number; num_frames?: number }[],
  fps = 24,
): string | null {
  if (!presets.length) return null;
  const withSec = presets.map((p) => ({
    id: p.id,
    seconds: p.seconds ?? (p.num_frames ? p.num_frames / fps : 0),
  }));
  const snapUp = withSec
    .filter((p) => p.seconds + 0.05 >= seconds)
    .sort((a, b) => a.seconds - b.seconds)[0];
  if (snapUp) return snapUp.id;
  return withSec.reduce((a, b) => (a.seconds > b.seconds ? a : b)).id;
}

export function mentionCandidates(
  refs: ReferenceItem[],
  castMembers: CastMember[],
): { handle: string; label: string }[] {
  const active = activeRefs(refs);
  const tokens = tokensForRefs(active);
  const out = tokens.map((t) => ({ handle: t.handle, label: `${t.handle} → ${t.token} (${t.name})` }));
  for (const m of castMembers) {
    out.push({ handle: `@${slugName(m.name)}`, label: `@${slugName(m.name)} (${m.name})` });
  }
  return out;
}
