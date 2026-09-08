export interface LoraPreset {
  id: string;
  label: string;
  spec: string;
  scale: number;
  custom?: boolean;
  cached?: boolean;
  steps?: number;
  layers?: number;
  reuse?: number;
  guidance?: string | null;
}

export interface QualityPreset {
  id: string;
  label: string;
  steps?: number;
  layers?: number;
  reuse?: number;
  token_reduction?: boolean;
  guidance?: string | null;
}

export interface PresetOption {
  id: string;
  label: string;
  width?: number;
  height?: number;
  render_width?: number;
  render_height?: number;
  seconds?: number;
  num_frames?: number;
  guidance?: string;
  aspect?: string;
  group?: string;
}

export interface Config {
  server_connected: boolean;
  server_url: string;
  engine_ok?: boolean;
  engine_error?: string | null;
  h3_bin?: string;
  model_dir?: string;
  ram_gb?: number | null;
  recommend_ssd_streaming?: boolean;
  metal4?: boolean;
  quality_presets: QualityPreset[];
  lora_presets?: LoraPreset[];
  resolution_presets: PresetOption[];
  duration_presets: PresetOption[];
  generation_modes: { id: string; label: string }[];
  ref_kinds?: { id: string; label: string; flag: string }[];
  defaults: {
    num_frames: number;
    width: number;
    height: number;
    num_steps: number;
    layers?: number;
    reuse?: number;
    fps: number;
    quality?: string;
  };
  model_note: string;
  clip_multiplier_max?: number;
  embedded?: boolean;
  web_url?: string;
  pyav_available?: boolean;
}

export interface Clip {
  id: string;
  prompt: string;
  label: string;
  video_url: string;
  filename: string;
  path?: string;
  chain_id: string;
  clip_index: number;
  mode: string;
  status: string;
  created_at: string;
  elapsed_s?: number;
  bytes?: number;
  error?: string;
  num_frames?: number;
  width?: number;
  height?: number;
  seed?: number;
  num_steps?: number;
  layers?: number;
  reuse?: number;
  duration_seconds?: number;
  clip_count?: number;
  autocontinue?: boolean;
  autoconcat?: boolean;
  quality?: string;
}

export interface LibraryFrame {
  id: string;
  label: string;
  path: string;
  image_url: string;
  filename: string;
  width?: number;
  height?: number;
  source_clip_id?: string;
  time_s?: number;
  created_at: string;
}

export interface ModelProgress {
  stage?: string;
  step?: number;
  total?: number;
  pct?: number;
  eta_s?: number;
  avg_step_s?: number;
  elapsed_s?: number;
  label?: string;
}

export type RefKind = "image" | "silent_video" | "video" | "video_audio" | "audio";

export interface ReferenceItem {
  id: string;
  kind: RefKind;
  path: string;
  name: string;
  audioPath?: string;
  audioName?: string;
}

export interface ProgressState {
  phase: string;
  message: string;
  elapsed_s?: number;
  kb?: number;
  stage?: string;
  step?: number;
  total?: number;
  pct?: number;
  eta_s?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cast System Types (Phase 5)
// ─────────────────────────────────────────────────────────────────────────────

export interface CastMember {
  id: string;
  name: string;
  description?: string;
  media: CastMedia[];
  createdAt: string;
  updatedAt: string;
}

export interface CastMedia {
  id: string;
  type: "image" | "video";
  path: string;
  thumbnailUrl?: string;
  label?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset System Types (Phase 5)
// ─────────────────────────────────────────────────────────────────────────────

export interface GenerationPreset {
  id: string;
  name: string;
  description?: string;
  mode: string;
  quality: string;
  resolutionId: string;
  durationId: string;
  numSteps: number;
  layers: number;
  reuse: number;
  loraIds: string[];
  turboEnabled: boolean;
  tokenReduction: boolean;
  ssdStreaming: boolean;
  createdAt: string;
  updatedAt: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline Types (Phase 3)
// ─────────────────────────────────────────────────────────────────────────────

export interface TimelineClip {
  id: string;
  clipId: string;
  chainId: string;
  index: number;
  thumbnailUrl?: string;
  durationMs: number;
  prompt: string;
  status: "pending" | "generating" | "done" | "failed";
}

export interface TimelineSeam {
  id: string;
  beforeClipId: string;
  afterClipId: string;
  transitionType: "cut" | "blend";
  blendDurationMs?: number;
}

export interface Timeline {
  id: string;
  chainId: string;
  clips: TimelineClip[];
  seams: TimelineSeam[];
  totalDurationMs: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference Scope Types (Phase 5)
// ─────────────────────────────────────────────────────────────────────────────

export type ScopeType = "face" | "object" | "scene" | "style";

export interface ReferenceScope {
  face: number;    // 0-1 contribution
  object: number;
  scene: number;
  style: number;
}

export interface ScopedReference extends ReferenceItem {
  scope?: ReferenceScope;
}

// ─────────────────────────────────────────────────────────────────────────────
// Turbo Mode Types (Phase 1)
// ─────────────────────────────────────────────────────────────────────────────

export interface TurboState {
  enabled: boolean;
  loraSpec: string;
  steps: number;
  layers: number;
  reuse: number;
  scale: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pill Control Types (Phase 1)
// ─────────────────────────────────────────────────────────────────────────────

export interface PillOption<T = string> {
  id: T;
  label: string;
  shortLabel?: string;
  description?: string;
  disabled?: boolean;
}

export interface PillGroup<T = string> {
  id: string;
  label: string;
  options: PillOption<T>[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene Queue Types (Phase 4)
// ─────────────────────────────────────────────────────────────────────────────

export type SceneStatus = "pending" | "generating" | "done" | "failed" | "cancelled";

export interface SceneQueueItem {
  id: string;
  prompt: string;
  mode: string;
  quality: string;
  resolutionId: string;
  durationId: string;
  numSteps: number;
  layers: number;
  reuse: number;
  seed: string;
  loraPresetIds: string[];
  turboEnabled: boolean;
  turboTier?: string;
  refs: ReferenceItem[];
  imagePath?: string | null;
  endImagePath?: string | null;
  clipMultiplier: number;
  autocontinue: boolean;
  autoconcat: boolean;
  tokenReduction: boolean;
  ssdStreaming: boolean;
  status: SceneStatus;
  runId?: string;
  error?: string;
  createdAt: string;
}
