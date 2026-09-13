import type { Clip, ClipRecipe, ClipRecipeRef, Config, LoraPreset, ReferenceItem, RoutingMode } from "./types";
import { generateId } from "./utils";

export function clipDisplayPrompt(prompt: string): string {
  return prompt.replace(/\s*\(×\d+ merged\)\s*$/i, "").trim();
}

export function resolutionIdForClip(clip: Clip, config: Config | null): string {
  const fromRecipe = clip.recipe?.resolution_id;
  if (fromRecipe && config?.resolution_presets.some((r) => r.id === fromRecipe)) {
    return fromRecipe;
  }
  if (!clip.width || !clip.height) return "512x512";
  const presets = config?.resolution_presets ?? [];
  const native = presets.find(
    (r) => r.width === clip.width && r.height === clip.height && !r.render_width,
  );
  if (native) return native.id;
  const match = presets.find((r) => r.width === clip.width && r.height === clip.height);
  return match?.id ?? `${clip.width}x${clip.height}`;
}

export function durationIdForClip(clip: Clip, config: Config | null): string {
  const fromRecipe = clip.recipe?.duration_id;
  if (fromRecipe && config?.duration_presets.some((d) => d.id === fromRecipe)) {
    return fromRecipe;
  }
  if (clip.num_frames && config?.duration_presets) {
    const match = config.duration_presets.find((d) => d.num_frames === clip.num_frames);
    if (match) return match.id;
  }
  if (clip.duration_seconds != null) {
    const match = config?.duration_presets.find((d) => d.seconds === clip.duration_seconds);
    if (match) return match.id;
  }
  return config?.duration_presets[0]?.id ?? "1s";
}

export interface ClipEditorSnapshot {
  prompt: string;
  mode: string;
  routing: RoutingMode;
  resolutionId: string;
  durationId: string;
  clipMultiplier: number;
  numSteps: number;
  layers: number;
  reuse: number;
  seed: string;
  quality: string;
  refs: ReferenceItem[];
  imagePath: string | null;
  imageName: string | null;
  endImagePath: string | null;
  endImageName: string | null;
  selectedCastIds: string[];
  tokenReduction: boolean | null;
  ssdStreaming: boolean | null;
  turboEnabled: boolean;
  turboTier: string | null;
  loraPresetIds: string[];
  loraScales: Record<string, number>;
  missingMedia: string[];
}

function basename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

function recipeRefToItem(ref: ClipRecipeRef, missing: string[]): ReferenceItem | null {
  const path = ref.path;
  const available = ref.available !== false && Boolean(path);
  if (!available) {
    missing.push(ref.name || basename(path) || "reference");
    return null;
  }
  const audioPath = ref.audioPath || ref.audio_path;
  if (audioPath && ref.audio_available === false) {
    missing.push(ref.audioName || ref.audio_name || basename(audioPath) || "audio");
  }
  return {
    id: generateId(),
    kind: ref.kind,
    path,
    name: ref.name || basename(path),
    audioPath: audioPath || undefined,
    audioName: ref.audioName || ref.audio_name,
    enabled: ref.enabled !== false,
    durationS: ref.durationS ?? ref.duration_s,
    refSize: ref.refSize ?? ref.ref_size ?? "max",
    previewUrl: ref.preview_url,
    source: ref.source,
    castId: ref.castId || ref.cast_id,
  };
}

function routingFromClip(clip: Clip): RoutingMode {
  const fromRecipe = clip.recipe?.routing;
  if (fromRecipe === "auto" || fromRecipe === "fl2va" || fromRecipe === "ref2va") {
    return fromRecipe;
  }
  if (clip.mode === "ref2va") return "ref2va";
  if (clip.mode === "t2va") return "auto";
  return "fl2va";
}

export function loraScalesFromClip(clip: Clip, catalog: LoraPreset[]): Record<string, number> {
  const scales: Record<string, number> = {};
  const specs = [...(clip.recipe?.loras ?? []), ...(clip.loras ?? [])];
  for (const entry of specs) {
    const match = catalog.find(
      (p) => (entry.id && p.id === entry.id) || (entry.spec && p.spec === entry.spec),
    );
    if (!match || entry.scale == null) continue;
    const scale = Number(entry.scale);
    if (Number.isFinite(scale)) scales[match.id] = Math.max(0, Math.min(2, scale));
  }
  return scales;
}

export function loraIdsFromClip(clip: Clip, catalog: LoraPreset[]): string[] {
  const ids = new Set<string>();
  const recipe = clip.recipe;
  for (const id of recipe?.lora_preset_ids ?? []) {
    if (catalog.some((p) => p.id === id)) ids.add(id);
  }
  const specs = [...(recipe?.loras ?? []), ...(clip.loras ?? [])];
  for (const entry of specs) {
    const match = catalog.find(
      (p) => (entry.id && p.id === entry.id) || (entry.spec && p.spec === entry.spec),
    );
    if (match) ids.add(match.id);
  }
  return [...ids];
}

export function snapshotFromClip(
  clip: Clip,
  config: Config | null,
  defaults: { numSteps: number; layers: number; reuse: number; quality: string },
  catalog: LoraPreset[] = [],
): ClipEditorSnapshot {
  const recipe: ClipRecipe | undefined = clip.recipe;
  const missingMedia: string[] = [];
  const refs: ReferenceItem[] = [];
  for (const ref of recipe?.refs ?? []) {
    const item = recipeRefToItem(ref, missingMedia);
    if (item) refs.push(item);
  }

  let imagePath = recipe?.image_path ?? null;
  let endImagePath = recipe?.end_image_path ?? null;
  if (imagePath && recipe?.image_path_available === false) {
    missingMedia.push(recipe.image_name || basename(imagePath));
    imagePath = null;
  }
  if (endImagePath && recipe?.end_image_path_available === false) {
    missingMedia.push(recipe.end_image_name || basename(endImagePath));
    endImagePath = null;
  }

  return {
    prompt: clipDisplayPrompt(recipe?.composer_prompt || clip.prompt),
    mode: recipe?.mode || clip.mode || "t2va",
    routing: routingFromClip(clip),
    resolutionId: resolutionIdForClip(clip, config),
    durationId: durationIdForClip(clip, config),
    clipMultiplier: clip.clip_count ?? 1,
    numSteps: clip.num_steps ?? defaults.numSteps,
    layers: clip.layers ?? defaults.layers,
    reuse: clip.reuse ?? defaults.reuse,
    seed: clip.seed != null ? String(clip.seed) : "",
    quality: recipe?.quality ?? clip.quality ?? defaults.quality,
    refs,
    imagePath,
    imageName: imagePath ? recipe?.image_name || basename(imagePath) : null,
    endImagePath,
    endImageName: endImagePath ? recipe?.end_image_name || basename(endImagePath) : null,
    selectedCastIds: [...(recipe?.selected_cast_ids ?? [])],
    tokenReduction: recipe?.token_reduction ?? null,
    ssdStreaming: recipe?.ssd_streaming ?? null,
    turboEnabled: Boolean(recipe?.turbo_enabled),
    turboTier: recipe?.turbo_tier ?? null,
    loraPresetIds: loraIdsFromClip(clip, catalog),
    loraScales: loraScalesFromClip(clip, catalog),
    missingMedia,
  };
}
