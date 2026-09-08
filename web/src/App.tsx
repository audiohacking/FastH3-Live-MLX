import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { clipDisplayPrompt, snapshotFromClip } from "./clipEditor";
import { applyProgressEvent } from "./progress";
import { captureVideoFrame, formatVideoTime } from "./frameCapture";
import { RefList, refsAreValid } from "./RefList";
import { generateId } from "./utils";
import { FEATURES, TURBO_CONFIG, type TurboTier } from "./config";
import { PillSelect, PillRow, PillDivider, NumberPill, TextPill } from "./components/options/PillControls";
import { TurboToggle, TurboInfo } from "./components/composer/TurboToggle";
import { ReferenceChips } from "./components/composer/ReferenceChips";
import { RefListEnhanced } from "./components/composer/RefListEnhanced";
import { LoraModal } from "./components/lora/LoraModal";
import { TimelineStrip } from "./components/timeline/TimelineStrip";
import { SceneQueue, AddToQueueButton } from "./components/composer/SceneQueue";
import { PresetManager } from "./components/presets/PresetManager";
import { CastPicker } from "./components/media/CastPicker";
import { ModelsManager } from "./components/media/ModelsManager";
import { WhatTheModelReads } from "./components/composer/WhatTheModelReads";
import type { CastMember, Clip, Config, GenerationPreset, LibraryFrame, LoraPreset, PillOption, PresetOption, ProgressState, QualityPreset, ReferenceItem, SceneQueueItem } from "./types";

function resolutionGroups(presets: PresetOption[]): { group: string; items: PresetOption[] }[] {
  const order: { group: string; items: PresetOption[] }[] = [];
  const index = new Map<string, number>();
  for (const preset of presets) {
    const group = preset.group || "";
    let slot = index.get(group);
    if (slot === undefined) {
      slot = order.length;
      index.set(group, slot);
      order.push({ group, items: [] });
    }
    order[slot].items.push(preset);
  }
  return order;
}

const H3_DEFAULT_STEPS = 20;
const H3_DEFAULT_LAYERS = 50;
const H3_DEFAULT_REUSE = 1;

function fieldsFromPreset(preset: QualityPreset | undefined) {
  const steps = preset?.steps ?? H3_DEFAULT_STEPS;
  return {
    steps,
    layers: preset?.layers ?? H3_DEFAULT_LAYERS,
    reuse: steps <= 7 ? 1 : (preset?.reuse ?? H3_DEFAULT_REUSE),
    tokenReduction: Boolean(preset?.token_reduction),
  };
}

function loraSelectionSummary(presets: LoraPreset[], selectedIds: string[]) {
  const selected = presets.filter((p) => selectedIds.includes(p.id));
  if (selected.length === 0) return "None";
  if (selected.length === 1) {
    const raw = selected[0].label.replace(/\s*\(default\)\s*$/i, "").trim();
    return raw.length > 28 ? `${raw.slice(0, 27)}…` : raw;
  }
  return `${selected.length} LoRAs`;
}

function LoraMultiSelect({
  presets,
  selectedIds,
  disabled,
  onToggle,
  onRemovePreset,
}: {
  presets: LoraPreset[];
  selectedIds: string[];
  disabled?: boolean;
  onToggle: (id: string, checked: boolean) => void;
  onRemovePreset: (preset: LoraPreset) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);
  const summary = loraSelectionSummary(presets, selectedIds);
  return (
    <div className={`multi-select${open ? " is-open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="multi-select-trigger"
        disabled={disabled}
        aria-expanded={open}
        title={summary}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="multi-select-trigger-text">{summary}</span>
      </button>
      {open && (
        <div className="multi-select-menu" role="listbox" aria-label="LoRA presets">
          {presets.map((p) => {
            const checked = selectedIds.includes(p.id);
            return (
              <div
                key={p.id}
                className={`multi-select-item${checked ? " is-selected" : ""}`}
                role="option"
                aria-selected={checked}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled}
                  aria-label={p.label}
                  onChange={(e) => onToggle(p.id, e.target.checked)}
                />
                <span
                  className="multi-select-item-label"
                  title={p.guidance ?? p.label}
                  onClick={() => {
                    if (!disabled) onToggle(p.id, !checked);
                  }}
                >
                  {p.label}
                </span>
                {p.custom ? (
                  <button
                    type="button"
                    className="lora-remove"
                    title="Remove from list"
                    disabled={disabled}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onRemovePreset(p);
                    }}
                  >
                    ×
                  </button>
                ) : null}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

const API = "";
const BLOB_VIDEO_PREFIX = "blob:";
const NOTIFY_READY_KEY = "h3-ws-notify-on-ready";
const NOTIFY_ASKED_KEY = "h3-ws-notify-permission-asked";

function isHttpsContext(): boolean {
  try {
    return typeof location !== "undefined" && location.protocol === "https:";
  } catch {
    return false;
  }
}

function persistNotifyDecision(enabled: boolean) {
  try {
    localStorage.setItem(NOTIFY_ASKED_KEY, "1");
    localStorage.setItem(NOTIFY_READY_KEY, enabled ? "1" : "0");
  } catch {
    /* ignore */
  }
}

async function maybeRequestNotifyPermissionOnHttps(): Promise<void> {
  if (!isHttpsContext()) return;
  if (typeof Notification === "undefined") return;
  try {
    if (localStorage.getItem(NOTIFY_ASKED_KEY) === "1") return;
  } catch {
    return;
  }
  if (Notification.permission === "granted") {
    persistNotifyDecision(true);
    return;
  }
  if (Notification.permission === "denied") {
    persistNotifyDecision(false);
    return;
  }
  try {
    const permission = await Notification.requestPermission();
    persistNotifyDecision(permission === "granted");
  } catch {
    persistNotifyDecision(false);
  }
}

function notifyGenerationReady(body = "Your video is ready to play.") {
  if (!isHttpsContext()) return;
  try {
    if (localStorage.getItem(NOTIFY_READY_KEY) !== "1") return;
  } catch {
    return;
  }
  if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
  try {
    new Notification("H3-WS", { body, tag: "h3-ws-generation-ready" });
  } catch {
    /* ignore */
  }
}

function revokeClipBlob(clip: Clip) {
  if (clip.video_url?.startsWith(BLOB_VIDEO_PREFIX)) URL.revokeObjectURL(clip.video_url);
}

function revokeBlobVideoUrls(clips: Clip[]) {
  for (const clip of clips) revokeClipBlob(clip);
}

function preserveBlobVideoUrls(prev: Clip[], incoming: Clip[]): Clip[] {
  const blobById = new Map(
    prev
      .filter((c) => c.video_url?.startsWith(BLOB_VIDEO_PREFIX))
      .map((c) => [c.id, c.video_url] as const),
  );
  return incoming.map((c) => {
    const blob = blobById.get(c.id);
    return blob ? { ...c, video_url: blob } : c;
  });
}

function mergeClips(prev: Clip[], incoming: Clip[]): Clip[] {
  const byId = new Map(prev.map((c) => [c.id, c]));
  for (const c of incoming) byId.set(c.id, c);
  return Array.from(byId.values());
}

function replaceChainClips(prev: Clip[], chainId: string, chainClips: Clip[]): Clip[] {
  const rest = prev.filter((c) => c.chain_id !== chainId);
  return [...rest, ...preserveBlobVideoUrls(prev, chainClips)];
}

function formatBytes(n?: number) {
  if (!n) return "";
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

const ROUNDED_DURATIONS: [number, number][] = [
  [22, 1],
  [56, 2],
  [107, 5],
  [243, 10],
  [362, 15],
];

function formatDuration(frames?: number, fps = 24) {
  if (!frames) return "";
  const hit = ROUNDED_DURATIONS.find(([nf]) => nf === frames);
  if (hit) return `${hit[1]}s`;
  return `${Math.round(frames / fps)}s`;
}

function pickPlaybackClip(clips: Clip[], chainId: string): string | null {
  const chain = clips.filter((c) => c.chain_id === chainId && c.status === "done" && c.video_url);
  const merged = chain.find((c) => c.label === "MERGED");
  const current = chain.find((c) => c.label === "CURRENT");
  const latest = [...chain].sort((a, b) => b.clip_index - a.clip_index)[0];
  return merged?.id ?? current?.id ?? latest?.id ?? null;
}

async function fetchConfig(): Promise<Config> {
  const r = await fetch(`${API}/api/config`);
  if (!r.ok) throw new Error("Failed to load config");
  return r.json();
}

async function fetchClips(chainId?: string): Promise<Clip[]> {
  const q = chainId ? `?chain_id=${encodeURIComponent(chainId)}` : "";
  const r = await fetch(`${API}/api/clips${q}`);
  if (!r.ok) throw new Error("Failed to load clips");
  const data = await r.json();
  return data.clips as Clip[];
}

async function fetchFrames(): Promise<LibraryFrame[]> {
  const r = await fetch(`${API}/api/frames`);
  if (!r.ok) throw new Error("Failed to load frames");
  const data = await r.json();
  return (data.frames ?? []) as LibraryFrame[];
}

async function fetchCastMembers(): Promise<CastMember[]> {
  const r = await fetch(`${API}/api/cast`);
  if (!r.ok) throw new Error("Failed to load cast");
  const data = await r.json();
  return (data.cast ?? []) as CastMember[];
}

async function fetchBackendPresets(): Promise<GenerationPreset[]> {
  const r = await fetch(`${API}/api/presets`);
  if (!r.ok) throw new Error("Failed to load presets");
  const data = await r.json();
  return (data.presets ?? []) as GenerationPreset[];
}

async function createPreset(preset: GenerationPreset): Promise<void> {
  await fetch(`${API}/api/presets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preset),
  });
}

async function uploadFile(file: File, kind: string): Promise<string> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/api/upload?kind=${encodeURIComponent(kind)}`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error("Upload failed");
  const data = await r.json();
  return data.path as string;
}

const IMAGE_MODES = new Set(["first_frame", "fl2va"]);
const END_IMAGE_MODES = new Set(["last_frame", "fl2va"]);

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [frameLibrary, setFrameLibrary] = useState<LibraryFrame[]>([]);
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState("t2va");
  const [quality, setQuality] = useState("fast");
  const [resolutionId, setResolutionId] = useState("512x512");
  const [durationId, setDurationId] = useState("1s");
  const [clipMultiplier, setClipMultiplier] = useState(1);
  const [numSteps, setNumSteps] = useState(H3_DEFAULT_STEPS);
  const [layers, setLayers] = useState(H3_DEFAULT_LAYERS);
  const [reuse, setReuse] = useState(H3_DEFAULT_REUSE);
  const [seed, setSeed] = useState("");
  const [ssdStreaming, setSsdStreaming] = useState(false);
  const [tokenReduction, setTokenReduction] = useState(true);
  const [loraPresetIds, setLoraPresetIds] = useState<string[]>([]);
  const [loraPresets, setLoraPresets] = useState<LoraPreset[]>([]);
  const [customLoraUrl, setCustomLoraUrl] = useState("");
  const [customLoraLabel, setCustomLoraLabel] = useState("");
  const [customLoraScale, setCustomLoraScale] = useState("0.8");
  const [addingCustomLora, setAddingCustomLora] = useState(false);
  const [loraBusy, setLoraBusy] = useState(false);
  const [loraActivity, setLoraActivity] = useState<string | null>(null);
  const [showOptions, setShowOptions] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [chainId, setChainId] = useState<string | null>(null);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [endImagePath, setEndImagePath] = useState<string | null>(null);
  const [endImageName, setEndImageName] = useState<string | null>(null);
  const [refs, setRefs] = useState<ReferenceItem[]>([]);
  const [savingFrame, setSavingFrame] = useState(false);
  const [turboEnabled, setTurboEnabled] = useState(false);
  const [turboTier, setTurboTier] = useState<TurboTier>(TURBO_CONFIG.DEFAULT_TIER);
  const [turboLoading, setTurboLoading] = useState(false);
  const [loraModalOpen, setLoraModalOpen] = useState(false);
  const [sceneQueue, setSceneQueue] = useState<SceneQueueItem[]>([]);
  const [queueRunning, setQueueRunning] = useState(false);
  const [generationPresets, setGenerationPresets] = useState<GenerationPreset[]>([]);
  const [castMembers, setCastMembers] = useState<CastMember[]>([]);
  const [selectedCastIds, setSelectedCastIds] = useState<string[]>([]);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [modelsDownloadActive, setModelsDownloadActive] = useState(false);
  const [modelsCloseWarning, setModelsCloseWarning] = useState(false);
  const [lockedClipIds, setLockedClipIds] = useState<Set<string>>(new Set());
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const playerVideoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLInputElement>(null);
  const endImageRef = useRef<HTMLInputElement>(null);
  const runEventSourceRef = useRef<EventSource | null>(null);
  const clipsRef = useRef<Clip[]>([]);
  clipsRef.current = clips;

  const libraryClips = useMemo(
    () => clips.filter((c) => c.status === "done" && c.video_url),
    [clips],
  );
  const activeClip = useMemo(() => {
    if (selectedClipId) return clips.find((c) => c.id === selectedClipId) ?? null;
    return libraryClips[libraryClips.length - 1] ?? null;
  }, [clips, selectedClipId, libraryClips]);
  const chainParts = useMemo(
    () => (chainId ? clips.filter((c) => c.chain_id === chainId) : []),
    [clips, chainId],
  );
  const showChainPicker = chainParts.filter((c) => c.video_url).length > 1;
  const resolution = useMemo(() => {
    const p = config?.resolution_presets.find((r) => r.id === resolutionId);
    return {
      width: p?.width ?? 512,
      height: p?.height ?? 512,
      render_width: p?.render_width,
      render_height: p?.render_height,
      guidance: p?.guidance,
    };
  }, [config, resolutionId]);
  const durationPreset = config?.duration_presets.find((d) => d.id === durationId);
  const isRef2va = mode === "ref2va" || refs.length > 0;
  const isMultiClip = clipMultiplier > 1 && !isRef2va;
  const needsFirst = IMAGE_MODES.has(mode) && !isRef2va;
  const needsLast = END_IMAGE_MODES.has(mode) && !isRef2va;
  const lastUsesPrimaryUpload = mode === "last_frame" && !isRef2va;
  const previewCanvas = resolutionId === "256x256";
  const aggressiveInternal = resolutionId === "512x512-aggressive";
  const tokenReductionLocked =
    previewCanvas ||
    aggressiveInternal ||
    quality === "aggressive" ||
    (layers === 40 && reuse === 3) ||
    loraPresetIds.length > 0;
  const closePreset = config?.quality_presets.find((p) => p.id === "close");

  // Pill options for the new UI
  const modeOptions: PillOption[] = useMemo(
    () => (config?.generation_modes ?? []).map((m) => ({ id: m.id, label: m.label })),
    [config?.generation_modes],
  );

  const qualityOptions: PillOption[] = useMemo(
    () =>
      (config?.quality_presets ?? []).map((p) => ({
        id: p.id,
        label: p.label,
        description: p.guidance ?? undefined,
      })),
    [config?.quality_presets],
  );

  const durationOptions: PillOption[] = useMemo(
    () => (config?.duration_presets ?? []).map((d) => ({ id: d.id, label: d.id })),
    [config?.duration_presets],
  );

  useEffect(() => {
    void maybeRequestNotifyPermissionOnHttps();
    fetchConfig()
      .then((cfg) => {
        setConfig(cfg);
        setQuality(cfg.defaults.quality ?? "fast");
        const preset = cfg.quality_presets.find((p) => p.id === (cfg.defaults.quality ?? "fast"));
        const fields = fieldsFromPreset(preset);
        setNumSteps(fields.steps);
        setLayers(fields.layers);
        setReuse(fields.reuse);
        setTokenReduction(fields.tokenReduction);
        setSsdStreaming(false);
        setLoraPresets(cfg.lora_presets ?? []);
        const defRes =
          cfg.resolution_presets.find((r) => r.id === "512x512") ??
          cfg.resolution_presets.find(
            (r) =>
              r.width === cfg.defaults.width &&
              r.height === cfg.defaults.height &&
              !r.render_width,
          );
        if (defRes) setResolutionId(defRes.id);
        const defDur = cfg.duration_presets.find((d) => d.num_frames === cfg.defaults.num_frames);
        if (defDur) setDurationId(defDur.id);
      })
      .catch((e) => setError(String(e)));
    fetchClips().then(setClips).catch(() => undefined);
    fetchFrames().then(setFrameLibrary).catch(() => undefined);
    fetchCastMembers().then(setCastMembers).catch(() => undefined);
    fetchBackendPresets().then((presets) => {
      if (presets.length > 0) setGenerationPresets(presets);
    }).catch(() => undefined);
    return () => {
      runEventSourceRef.current?.close();
      revokeBlobVideoUrls(clipsRef.current);
    };
  }, []);

  const applyClipSelection = useCallback(
    (clip: Clip) => {
      setSelectedClipId(clip.id);
      setChainId(clip.chain_id);
      if (!config) return;
      const snap = snapshotFromClip(clip, config, {
        numSteps: config.defaults.num_steps,
        layers: config.defaults.layers ?? H3_DEFAULT_LAYERS,
        reuse: config.defaults.reuse ?? H3_DEFAULT_REUSE,
        quality: config.defaults.quality ?? "fast",
      });
      setPrompt(snap.prompt);
      setMode(snap.mode);
      setResolutionId(snap.resolutionId);
      setDurationId(snap.durationId);
      setClipMultiplier(snap.clipMultiplier);
      setNumSteps(snap.numSteps);
      setLayers(snap.layers);
      setReuse(snap.reuse);
      setSeed(snap.seed);
      setQuality(snap.quality);
    },
    [config],
  );

  function applyLoraHints(preset: LoraPreset) {
    if (preset.steps) setNumSteps(preset.steps);
    if (preset.layers) setLayers(preset.layers);
    if (preset.reuse) setReuse(preset.reuse);
    if (preset.steps && preset.steps <= 7) setReuse(1);
    setTokenReduction(false);
    setSsdStreaming(false);
  }

  async function handleTurboToggle(enabled: boolean) {
    if (!FEATURES.TURBO_MODE) return;

    if (enabled) {
      setTurboLoading(true);
      try {
        // Ensure the turbo LoRA is downloaded
        await ensureLoraSpec(TURBO_CONFIG.LORA_SPEC, TURBO_CONFIG.LABEL);

        // Find or create the turbo preset ID
        const existingPreset = loraPresets.find((p) => p.spec === TURBO_CONFIG.LORA_SPEC);
        if (existingPreset) {
          // Enable the existing preset
          setLoraPresetIds((prev) =>
            prev.includes(existingPreset.id) ? prev : [...prev, existingPreset.id],
          );
        } else {
          // Add as custom and enable
          const r = await fetch(`${API}/api/loras/custom`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              spec: TURBO_CONFIG.LORA_SPEC,
              label: TURBO_CONFIG.LABEL,
              scale: TURBO_CONFIG.SCALE,
            }),
          });
          if (r.ok) {
            const data = (await r.json()) as { id: string; lora_presets?: LoraPreset[] };
            if (data.lora_presets) setLoraPresets(data.lora_presets);
            if (data.id) setLoraPresetIds((prev) => [...prev, data.id]);
          }
        }

        // Apply turbo settings using selected tier
        const tierConfig = TURBO_CONFIG.TIERS[turboTier];
        setNumSteps(tierConfig.steps);
        setLayers(TURBO_CONFIG.LAYERS);
        setReuse(TURBO_CONFIG.REUSE);
        setTokenReduction(false);
        setSsdStreaming(false);
        setTurboEnabled(true);
        setLoraActivity(`${TURBO_CONFIG.LABEL} enabled`);
      } catch (e) {
        setError(String(e));
        setTurboEnabled(false);
      } finally {
        setTurboLoading(false);
      }
    } else {
      // Disable turbo - remove turbo LoRA from selection
      const turboPreset = loraPresets.find((p) => p.spec === TURBO_CONFIG.LORA_SPEC);
      if (turboPreset) {
        setLoraPresetIds((prev) => prev.filter((id) => id !== turboPreset.id));
      }
      setTurboEnabled(false);
      setLoraActivity(null);

      // Restore default quality preset settings
      const preset = config?.quality_presets.find((p) => p.id === quality);
      if (preset) {
        const fields = fieldsFromPreset(preset);
        setNumSteps(fields.steps);
        setLayers(fields.layers);
        setReuse(fields.reuse);
      }
    }
  }

  function handleTurboTierChange(tier: TurboTier) {
    setTurboTier(tier);
    if (turboEnabled) {
      // Update step count when tier changes while turbo is active
      const tierConfig = TURBO_CONFIG.TIERS[tier];
      setNumSteps(tierConfig.steps);
    }
  }

  function addToSceneQueue() {
    const scene: SceneQueueItem = {
      id: generateId(),
      prompt,
      mode,
      quality,
      resolutionId,
      durationId,
      numSteps,
      layers,
      reuse,
      seed,
      loraPresetIds,
      turboEnabled,
      turboTier,
      refs: [...refs],
      imagePath,
      endImagePath,
      clipMultiplier,
      autocontinue: clipMultiplier > 1,
      autoconcat: clipMultiplier > 1,
      tokenReduction,
      ssdStreaming,
      status: "pending",
      createdAt: new Date().toISOString(),
    };
    setSceneQueue((prev) => [...prev, scene]);
  }

  function removeFromSceneQueue(id: string) {
    setSceneQueue((prev) => prev.filter((s) => s.id !== id));
  }

  function reorderSceneQueue(scenes: SceneQueueItem[]) {
    setSceneQueue(scenes);
  }

  function loadSceneToEditor(scene: SceneQueueItem) {
    setPrompt(scene.prompt);
    setMode(scene.mode);
    setQuality(scene.quality);
    setResolutionId(scene.resolutionId);
    setDurationId(scene.durationId);
    setNumSteps(scene.numSteps);
    setLayers(scene.layers);
    setReuse(scene.reuse);
    setSeed(scene.seed);
    setLoraPresetIds(scene.loraPresetIds);
    setTurboEnabled(scene.turboEnabled);
    if (scene.turboTier) setTurboTier(scene.turboTier as TurboTier);
    setRefs(scene.refs);
    setImagePath(scene.imagePath ?? null);
    setEndImagePath(scene.endImagePath ?? null);
    setClipMultiplier(scene.clipMultiplier);
    // autocontinue/autoconcat are derived from clipMultiplier
    setTokenReduction(scene.tokenReduction);
    setSsdStreaming(scene.ssdStreaming);
  }

  function clearSceneQueue() {
    setSceneQueue([]);
  }

  async function runSceneQueue() {
    // Run all pending scenes sequentially
    const pending = sceneQueue.filter((s) => s.status === "pending");
    if (pending.length === 0) return;

    setQueueRunning(true);
    for (const scene of pending) {
      // Load scene settings
      loadSceneToEditor(scene);
      // Mark as generating
      setSceneQueue((prev) =>
        prev.map((s) => (s.id === scene.id ? { ...s, status: "generating" as const } : s))
      );
      // Trigger generation (we need to await it completing via SSE)
      // For now, just mark as done after a short delay - actual implementation would await the run
      // This is a placeholder - full implementation would use the existing handleRun logic
      try {
        // Trigger the generation button programmatically
        const genButton = document.querySelector<HTMLButtonElement>(".gen-submit");
        if (genButton && !genButton.disabled) {
          genButton.click();
          // Wait for generation to complete (monitor busy state)
          await new Promise<void>((resolve) => {
            const checkInterval = setInterval(() => {
              // Check if no longer busy
              if (!document.querySelector(".gen-submit:disabled")) {
                clearInterval(checkInterval);
                resolve();
              }
            }, 500);
          });
        }
        setSceneQueue((prev) =>
          prev.map((s) => (s.id === scene.id ? { ...s, status: "done" as const } : s))
        );
      } catch (e) {
        setSceneQueue((prev) =>
          prev.map((s) =>
            s.id === scene.id ? { ...s, status: "failed" as const, error: String(e) } : s
          )
        );
      }
    }
    setQueueRunning(false);
  }

  function savePreset(name: string, description?: string) {
    const preset: GenerationPreset = {
      id: generateId(),
      name,
      description,
      mode,
      quality,
      resolutionId,
      durationId,
      numSteps,
      layers,
      reuse,
      loraIds: loraPresetIds,
      turboEnabled,
      tokenReduction,
      ssdStreaming,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setGenerationPresets((prev) => [...prev, preset]);
    // Persist to backend (survives restarts); localStorage as local fallback
    const updated = [...generationPresets, preset];
    localStorage.setItem("h3ws-presets", JSON.stringify(updated));
    void createPreset(preset);
  }

  function loadPreset(preset: GenerationPreset) {
    setMode(preset.mode);
    setQuality(preset.quality);
    setResolutionId(preset.resolutionId);
    setDurationId(preset.durationId);
    setNumSteps(preset.numSteps);
    setLayers(preset.layers);
    setReuse(preset.reuse);
    setLoraPresetIds(preset.loraIds);
    setTurboEnabled(preset.turboEnabled);
    setTokenReduction(preset.tokenReduction);
    setSsdStreaming(preset.ssdStreaming);
  }

  function deletePreset(id: string) {
    setGenerationPresets((prev) => prev.filter((p) => p.id !== id));
    const updated = generationPresets.filter((p) => p.id !== id);
    localStorage.setItem("h3ws-presets", JSON.stringify(updated));
    void fetch(`${API}/api/presets/${id}`, { method: "DELETE" }).catch(() => undefined);
  }

  function toggleClipLock(clipId: string) {
    setLockedClipIds((prev) => {
      const next = new Set(prev);
      if (next.has(clipId)) {
        next.delete(clipId);
      } else {
        next.add(clipId);
      }
      return next;
    });
  }

  async function createCastMember(name: string, description?: string) {
    try {
      const r = await fetch(`${API}/api/cast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => null);
        throw new Error((body && body.detail) || "Failed to create cast member");
      }
      const data = await r.json();
      setCastMembers((prev) => [...prev, data.cast as CastMember]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function deleteCastMember(id: string) {
    try {
      const r = await fetch(`${API}/api/cast/${id}`, { method: "DELETE" });
      if (!r.ok) {
        const body = await r.json().catch(() => null);
        throw new Error((body && body.detail) || "Failed to delete cast member");
      }
      setCastMembers((prev) => prev.filter((m) => m.id !== id));
      setSelectedCastIds((prev) => prev.filter((cid) => cid !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function toggleCast(id: string) {
    setSelectedCastIds((prev) =>
      prev.includes(id) ? prev.filter((cid) => cid !== id) : [...prev, id]
    );
  }

  function toggleModelsOpen() {
    // If a download is running, closing the modal is safe (the server task
    // survives), but surface a confirmation so the user knows it continues.
    if (modelsDownloadActive) {
      setModelsCloseWarning(true);
      return;
    }
    setModelsOpen((v) => !v);
  }

  function handleModelsDownloadStateChange(active: boolean) {
    setModelsDownloadActive(active);
    if (!active) setModelsCloseWarning(false);
  }

  function confirmModelsClose() {
    setModelsCloseWarning(false);
    setModelsOpen(false);
  }

  // Load presets from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem("h3ws-presets");
      if (stored) {
        setGenerationPresets(JSON.parse(stored) as GenerationPreset[]);
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  async function ensureLoraSpec(spec: string, label: string) {
    setLoraBusy(true);
    setLoraActivity(`Downloading ${label}…`);
    try {
      const r = await fetch(`${API}/api/loras/ensure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof (err as { detail?: string }).detail === "string"
          ? (err as { detail: string }).detail
          : `LoRA download failed: ${label}`);
      }
      setLoraActivity(`${label} ready`);
    } finally {
      setLoraBusy(false);
    }
  }

  async function toggleLoraPreset(id: string, checked: boolean) {
    const preset = loraPresets.find((p) => p.id === id);
    setLoraPresetIds((prev) => {
      const next = checked ? [...prev.filter((x) => x !== id), id] : prev.filter((x) => x !== id);
      return next;
    });
    if (checked && preset) {
      applyLoraHints(preset);
      try {
        await ensureLoraSpec(preset.spec, preset.label);
      } catch (e) {
        setError(String(e));
        setLoraActivity(String(e));
      }
    }
  }

  async function addCustomLora() {
    const spec = customLoraUrl.trim();
    if (!spec || addingCustomLora) return;
    setAddingCustomLora(true);
    try {
      const r = await fetch(`${API}/api/loras/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
          label: customLoraLabel.trim() || undefined,
          scale: Number(customLoraScale) || 1,
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(typeof (err as { detail?: string }).detail === "string"
          ? (err as { detail: string }).detail
          : "Could not add custom LoRA");
      }
      const data = (await r.json()) as { id: string; preset?: LoraPreset; lora_presets?: LoraPreset[] };
      if (data.lora_presets) setLoraPresets(data.lora_presets);
      if (data.id) {
        setLoraPresetIds((prev) => (prev.includes(data.id) ? prev : [...prev, data.id]));
        if (data.preset) applyLoraHints(data.preset);
      }
      setCustomLoraUrl("");
      setCustomLoraLabel("");
      setLoraActivity(`LoRA ready: ${data.preset?.label ?? spec}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setAddingCustomLora(false);
    }
  }

  async function removeLoraPreset(preset: LoraPreset) {
    if (!preset.custom) return;
    if (!confirm(`Remove custom LoRA "${preset.label}"?`)) return;
    const r = await fetch(`${API}/api/loras/custom/${encodeURIComponent(preset.id)}`, {
      method: "DELETE",
    });
    if (!r.ok) return;
    const data = (await r.json()) as { lora_presets?: LoraPreset[] };
    if (data.lora_presets) setLoraPresets(data.lora_presets);
    setLoraPresetIds((prev) => prev.filter((id) => id !== preset.id));
  }

  async function startNewProject() {
    setClips((prev) => {
      revokeBlobVideoUrls(prev);
      return [];
    });
    setChainId(null);
    setSelectedClipId(null);
    setPrompt("");
    setClipMultiplier(1);
    setBusy(false);
    setProgress(null);
    setError(null);
    setImagePath(null);
    setImageName(null);
    setEndImagePath(null);
    setEndImageName(null);
    setRefs([]);
    try {
      await fetch(`${API}/api/session/clear`, { method: "POST" });
    } catch (err) {
      console.warn("Session clear failed", err);
    }
  }

  async function deleteClip(clip: Clip) {
    await fetch(`${API}/api/clips/${clip.id}`, { method: "DELETE" });
    setClips((prev) => {
      revokeClipBlob(clip);
      return prev.filter((c) => c.id !== clip.id);
    });
    if (selectedClipId === clip.id) setSelectedClipId(null);
    setRefs((prev) =>
      prev.filter((r) => r.path !== clip.path && r.path !== clip.filename && !r.path.endsWith(`/${clip.filename}`)),
    );
  }

  async function saveCurrentFrame() {
    const video = playerVideoRef.current;
    if (!video || !activeClip) return;
    setSavingFrame(true);
    try {
      const blob = await captureVideoFrame(video);
      const fd = new FormData();
      fd.append("file", blob, "frame.png");
      fd.append("source_clip_id", activeClip.id);
      fd.append("time_s", String(video.currentTime));
      fd.append("label", `Frame @ ${formatVideoTime(video.currentTime)}`);
      const r = await fetch(`${API}/api/frames`, { method: "POST", body: fd });
      if (!r.ok) throw new Error("Could not save frame");
      setFrameLibrary(await fetchFrames());
    } catch (e) {
      setError(String(e));
    } finally {
      setSavingFrame(false);
    }
  }

  function handleRefsChange(next: ReferenceItem[]) {
    setRefs(next);
    if (next.length === 0) return;
    setMode("ref2va");
    setImagePath(null);
    setImageName(null);
    setEndImagePath(null);
    setEndImageName(null);
    setClipMultiplier(1);
  }

  function applyFrameAsInput(frame: LibraryFrame, which: "start" | "end") {
    if (mode === "ref2va") {
      if (refs.some((r) => r.path === frame.path)) return;
      handleRefsChange([
        ...refs,
        { id: generateId(), kind: "image", path: frame.path, name: frame.label },
      ]);
      return;
    }
    setRefs([]);
    if (which === "end") {
      setEndImagePath(frame.path);
      setEndImageName(frame.label);
      if (mode === "t2va" || mode === "first_frame") setMode("fl2va");
      return;
    }
    setImagePath(frame.path);
    setImageName(frame.label);
    if (mode === "t2va") setMode("first_frame");
  }

  async function deleteFrame(frame: LibraryFrame) {
    const r = await fetch(`${API}/api/frames/${frame.id}`, { method: "DELETE" });
    if (!r.ok) return;
    const data = await r.json();
    setFrameLibrary((data.frames ?? []) as LibraryFrame[]);
    if (imagePath === frame.path) {
      setImagePath(null);
      setImageName(null);
    }
    if (endImagePath === frame.path) {
      setEndImagePath(null);
      setEndImageName(null);
    }
    setRefs((prev) => prev.filter((r) => r.path !== frame.path));
  }

  async function cancelActiveRun() {
    if (!activeRunId) return;
    await fetch(`${API}/api/runs/${activeRunId}/cancel`, { method: "POST" });
  }

  function subscribeRun(runId: string, runChainId: string) {
    runEventSourceRef.current?.close();
    setActiveRunId(runId);
    const es = new EventSource(`${API}/api/runs/${runId}/events`);
    runEventSourceRef.current = es;
    let closed = false;
    const finishRun = () => {
      if (closed) return;
      closed = true;
      es.close();
      runEventSourceRef.current = null;
      setActiveRunId(null);
      setBusy(false);
      setProgress(null);
      notifyGenerationReady();
    };
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as Record<string, unknown>;
      if (msg.type === "ping") return;
      if (msg.type === "progress" || msg.type === "generation_keepalive") {
        setProgress((prev) => applyProgressEvent(prev, msg));
        return;
      }
      if (msg.type === "clip_started") {
        setProgress({
          phase: "generating",
          message: `Clip ${Number(msg.index ?? 0) + 1}/${Number(msg.total ?? 1)}`,
        });
      }
      if (msg.type === "clip_done" || msg.type === "merged") {
        const clipId = String(msg.clip_id ?? "");
        const videoUrl = String(msg.video_url ?? "");
        fetchClips(runChainId).then((chainClips) => {
          setClips((prev) => replaceChainClips(prev, runChainId, chainClips));
          setSelectedClipId(pickPlaybackClip(chainClips, runChainId) ?? clipId ?? null);
        });
        void videoUrl;
      }
      if (msg.type === "run_cancelled") {
        setProgress({ phase: "cancelled", message: String(msg.message || "Cancelled") });
        finishRun();
      } else if (msg.type === "run_complete" || msg.type === "run_done") {
        finishRun();
      } else if (msg.type === "error" || msg.type === "clip_failed") {
        setError(String(msg.error || msg.message || "Failed"));
        finishRun();
      }
    };
    es.onerror = () => {
      if (closed) return;
      closed = true;
      es.close();
      runEventSourceRef.current = null;
      setBusy(false);
      setProgress(null);
      setError((prev) => prev ?? "Lost connection to server while waiting for progress.");
    };
  }

  async function handleGenerate() {
    if (!canSubmit || !prompt.trim() || busy) return;
    setError(null);
    setBusy(true);
    setProgress({ phase: "starting", message: "Submitting…" });
    const body: Record<string, unknown> = {
      prompt: prompt.trim(),
      mode: isRef2va ? "ref2va" : mode,
      quality,
      width: resolution.width,
      height: resolution.height,
      duration_seconds: durationPreset?.seconds,
      num_frames: durationPreset?.num_frames,
      clip_count: isRef2va ? 1 : clipMultiplier,
      num_steps: numSteps,
      layers,
      reuse,
      autocontinue: isMultiClip,
      autoconcat: isMultiClip,
      ssd_streaming: ssdStreaming,
      token_reduction: tokenReductionLocked ? false : tokenReduction,
      loras: loraPresets
        .filter((p) => loraPresetIds.includes(p.id))
        .map((p) => ({ id: p.id, spec: p.spec, scale: p.scale })),
    };
    if (loraPresetIds.length) body.ssd_streaming = false;
    if (resolution.render_width) body.render_width = resolution.render_width;
    if (resolution.render_height) body.render_height = resolution.render_height;
    if (seed.trim() !== "") body.seed = Number(seed);
    if (isRef2va) {
      body.refs = refs.map((r) => ({
        kind: r.kind,
        path: r.path,
        name: r.name,
        audio_path: r.audioPath || undefined,
      }));
    } else {
      if (needsFirst && imagePath) body.image_path = imagePath;
      if (lastUsesPrimaryUpload && imagePath) body.end_image_path = imagePath;
      if (needsLast && !lastUsesPrimaryUpload && endImagePath) body.end_image_path = endImagePath;
    }
    try {
      const r = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        const detail = (err as { detail?: unknown }).detail;
        throw new Error(typeof detail === "string" ? detail : "Generate failed");
      }
      const data = await r.json();
      setChainId(data.chain_id);
      setSelectedClipId(null);
      setProgress(
        data.started_immediately
          ? { phase: "starting", message: "Starting…" }
          : { phase: "queued", message: "Queued — waiting for current job…" },
      );
      subscribeRun(data.run_id, data.chain_id);
      const chainClips = await fetchClips(data.chain_id);
      setClips((prev) => mergeClips(prev, chainClips));
    } catch (e) {
      setError(String(e));
      setBusy(false);
      setProgress(null);
    }
  }

  const serverOk = config?.server_connected;
  const canSubmit = useMemo(() => {
    if (!prompt.trim() || busy || !serverOk) return false;
    if (isRef2va) return refsAreValid(refs).ok;
    if (needsFirst && !imagePath) return false;
    if (mode === "last_frame" && !imagePath && !endImagePath) return false;
    if (mode === "fl2va" && (!imagePath || !endImagePath)) return false;
    return true;
  }, [prompt, busy, serverOk, isRef2va, refs, needsFirst, imagePath, endImagePath, mode]);

  const fitPromptHeight = useCallback(() => {
    const el = promptRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  useLayoutEffect(() => {
    fitPromptHeight();
  }, [prompt, fitPromptHeight]);

  const endpointLabel = useMemo(() => {
    if (typeof window === "undefined") return config?.server_url ?? "";
    return `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`;
  }, [config?.server_url]);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">H3-WS</span>
          <span className="brand-sub">MiniMax-H3</span>
        </div>
        <div className="header-status">
          {FEATURES.MODELS_PAGE && (
            <button type="button" className="btn-secondary" onClick={toggleModelsOpen}>
              Models
            </button>
          )}
          {FEATURES.CAST_SYSTEM && (
            <CastPicker
              members={castMembers}
              selectedIds={selectedCastIds}
              onToggle={toggleCast}
              onCreate={createCastMember}
              onDelete={deleteCastMember}
              disabled={busy}
            />
          )}
          <button type="button" className="btn-secondary" onClick={() => void startNewProject()}>
            New project
          </button>
          <span className={`status-dot ${serverOk ? "ok" : "off"}`} title={endpointLabel} />
          {serverOk ? "Server connected" : "Server offline"}
        </div>
      </header>

      <div className="app-body">
        <div className="app-main">
          <section className="player-section">
            <div className="player-wrap">
              {activeClip?.video_url ? (
                <video
                  ref={playerVideoRef}
                  className="player"
                  src={activeClip.video_url}
                  controls
                  loop
                  playsInline
                  preload="metadata"
                />
              ) : (
                <div className="player placeholder">
                  {busy ? progress?.message ?? "Generating…" : "Your video will appear here"}
                </div>
              )}
              {busy && (
                <div className="progress-overlay">
                  <div className="progress-bar">
                    {progress?.pct != null ? (
                      <div className="progress-fill" style={{ width: `${Math.min(100, progress.pct)}%` }} />
                    ) : (
                      <div className="progress-pulse" />
                    )}
                  </div>
                  <div className="progress-overlay-row">
                    <span>{progress?.message ?? "Working…"}</span>
                    {activeRunId && progress?.phase !== "cancelled" && (
                      <button type="button" className="btn-cancel" onClick={() => void cancelActiveRun()}>
                        Cancel
                      </button>
                    )}
                  </div>
                </div>
              )}
              {activeClip?.video_url && !busy && (
                <button
                  type="button"
                  className="player-capture-btn"
                  disabled={savingFrame}
                  onClick={() => void saveCurrentFrame()}
                  title="Save frame to library"
                >
                  {savingFrame ? (
                    <span className="player-capture-spinner" aria-hidden />
                  ) : (
                    <svg className="player-capture-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden>
                      <path d="M4 7h2l2-3h8l2 3h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z" />
                      <circle cx="12" cy="13" r="4" />
                    </svg>
                  )}
                </button>
              )}
            </div>
            {error && <div className="error-banner">{error}</div>}
            {showChainPicker && (
              <div className="player-context">
                <div className="player-context-body">
                  <label className="player-context-label">
                    Chain clip
                    <select
                      className="chain-picker-select"
                      value={selectedClipId ?? ""}
                      onChange={(e) => {
                        const c = chainParts.find((x) => x.id === e.target.value);
                        if (c) applyClipSelection(c);
                      }}
                    >
                      {chainParts.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.label}
                          {c.num_frames ? ` · ${formatDuration(c.num_frames, config?.defaults.fps ?? 24)}` : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>
            )}
          </section>

          <section className="composer">
            <div className="prompt-row">
              <div className="prompt-field-wrap">
                {/* Reference chips shown above prompt when in ref2va mode */}
                {FEATURES.REFERENCE_CHIPS && refs.length > 0 && (
                  <ReferenceChips refs={refs} onChange={handleRefsChange} disabled={busy} />
                )}
                {FEATURES.WHAT_MODEL_READS && (
                  <WhatTheModelReads refs={refs} prompt={prompt} disabled={busy} />
                )}
                <textarea
                  ref={promptRef}
                  className="prompt-input"
                  rows={1}
                  placeholder={refs.length > 0
                    ? "Describe the scene using Picture 1, Video 1, Audio 1…"
                    : "Scene, action, camera, look, and audio…"}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void handleGenerate();
                    }
                  }}
                  disabled={busy}
                />
                <div className="prompt-field-actions">
                  <button
                    type="button"
                    className="btn-prompt-action"
                    onClick={() => setPrompt("")}
                    disabled={busy || !prompt}
                  >
                    Clear
                  </button>
                </div>
              </div>
              <button type="button" className="btn-generate gen-submit" onClick={() => void handleGenerate()} disabled={!canSubmit}>
                ↑
              </button>
              {FEATURES.SCENE_QUEUE && (
                <AddToQueueButton onClick={addToSceneQueue} disabled={busy} />
              )}
            </div>

            <button type="button" className="options-toggle" onClick={() => setShowOptions((v) => !v)}>
              {showOptions ? "Hide options" : "Show options"}
            </button>

            {showOptions && config && (
              <div className="options-panel">
                <p className="hint">{config.model_note}</p>
                {config.engine_ok === false && config.engine_error && (
                  <p className="hint hint-inline">{config.engine_error}</p>
                )}

                {/* New Pill UI (Phase 1) - controlled by feature flag */}
                {FEATURES.PILLS_UI && (
                  <>
                    <PillRow className="pills-main-row">
                      <PillSelect
                        label="Mode"
                        options={modeOptions}
                        value={mode}
                        onChange={(next) => {
                          setMode(next);
                          setImagePath(null);
                          setImageName(null);
                          setEndImagePath(null);
                          setEndImageName(null);
                          if (next !== "ref2va") setRefs([]);
                        }}
                        disabled={busy}
                      />
                      <PillSelect
                        label="Quality"
                        options={qualityOptions}
                        value={quality}
                        onChange={(id) => {
                          const preset = config.quality_presets.find((p) => p.id === id);
                          const fields = fieldsFromPreset(preset);
                          setQuality(id);
                          setNumSteps(fields.steps);
                          setLayers(fields.layers);
                          setReuse(fields.reuse);
                          setTokenReduction(fields.tokenReduction);
                          // Disable turbo if switching quality
                          if (turboEnabled) setTurboEnabled(false);
                        }}
                        disabled={busy}
                      />
                      <PillSelect
                        label="Duration"
                        options={durationOptions}
                        value={durationId}
                        onChange={setDurationId}
                        disabled={busy}
                        compact
                      />
                      {FEATURES.TURBO_MODE && (
                        <>
                          <PillDivider />
                          <TurboToggle
                            enabled={turboEnabled}
                            onChange={(enabled) => void handleTurboToggle(enabled)}
                            tier={turboTier}
                            onTierChange={handleTurboTierChange}
                            disabled={busy || loraBusy}
                            loading={turboLoading}
                          />
                        </>
                      )}
                    </PillRow>
                    <PillRow className="pills-params-row">
                      <NumberPill
                        label="Steps"
                        value={numSteps}
                        min={1}
                        max={50}
                        onChange={(next) => {
                          setNumSteps(next);
                          if (next <= 7) setReuse(1);
                        }}
                        disabled={busy}
                        title="h3.c --steps (denoising passes)"
                      />
                      <NumberPill
                        label="Layers"
                        value={layers}
                        min={1}
                        max={50}
                        onChange={setLayers}
                        disabled={busy}
                        title="h3.c --layers (DiT blocks per pass; default 50)"
                      />
                      <NumberPill
                        label="Reuse"
                        value={reuse}
                        min={1}
                        max={8}
                        onChange={setReuse}
                        disabled={busy || numSteps <= 7}
                        title="h3.c --reuse (exclusive with --core-reuse; default 1)"
                      />
                      <PillDivider />
                      <TextPill
                        label="Seed"
                        value={seed}
                        placeholder="random"
                        onChange={setSeed}
                        disabled={busy}
                      />
                    </PillRow>
                    <TurboInfo visible={turboEnabled} tier={turboTier} />
                    {FEATURES.PRESETS && (
                      <PresetManager
                        presets={generationPresets}
                        onSave={savePreset}
                        onLoad={loadPreset}
                        onDelete={deletePreset}
                        disabled={busy}
                      />
                    )}
                  </>
                )}

                {/* Legacy grid UI - shown when PILLS_UI is disabled */}
                {!FEATURES.PILLS_UI && (
                <div className="options-grid options-grid-compact">
                  <label className="opt-mode">
                    Mode
                    <select
                      value={mode}
                      onChange={(e) => {
                        const next = e.target.value;
                        setMode(next);
                        setImagePath(null);
                        setImageName(null);
                        setEndImagePath(null);
                        setEndImageName(null);
                        if (next !== "ref2va") setRefs([]);
                      }}
                    >
                      {(config.generation_modes ?? []).map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Quality
                    <select
                      value={quality}
                      onChange={(e) => {
                        const id = e.target.value;
                        const preset = config.quality_presets.find((p) => p.id === id);
                        const fields = fieldsFromPreset(preset);
                        setQuality(id);
                        setNumSteps(fields.steps);
                        setLayers(fields.layers);
                        setReuse(fields.reuse);
                        setTokenReduction(fields.tokenReduction);
                      }}
                    >
                      {config.quality_presets.map((p) => (
                        <option key={p.id} value={p.id} title={p.guidance ?? undefined}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="opt-resolution">
                    Resolution
                    <select
                      value={resolutionId}
                      onChange={(e) => {
                        const next = e.target.value;
                        setResolutionId(next);
                        if (next === "256x256" || next === "512x512-aggressive") {
                          setTokenReduction(false);
                        }
                      }}
                    >
                      {resolutionGroups(config.resolution_presets).map((g) =>
                        g.group ? (
                          <optgroup key={g.group} label={g.group}>
                            {g.items.map((r) => (
                              <option key={r.id} value={r.id}>
                                {r.label}
                              </option>
                            ))}
                          </optgroup>
                        ) : (
                          g.items.map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.label}
                            </option>
                          ))
                        ),
                      )}
                    </select>
                  </label>
                  <label className="opt-narrow">
                    Duration
                    <select value={durationId} onChange={(e) => setDurationId(e.target.value)}>
                      {config.duration_presets.map((d) => (
                        <option key={d.id} value={d.id} title={d.label}>
                          {d.id}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="opt-narrow">
                    Clips
                    <select
                      value={clipMultiplier}
                      onChange={(e) => setClipMultiplier(Number(e.target.value))}
                      disabled={isRef2va}
                    >
                      {Array.from({ length: config.clip_multiplier_max ?? 10 }, (_, i) => i + 1).map((n) => (
                        <option key={n} value={n}>
                          ×{n}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="opt-narrow">
                    Steps
                    <input
                      type="number"
                      min={1}
                      max={50}
                      title="h3.c --steps (denoising passes)"
                      value={numSteps}
                      onChange={(e) => {
                        const next = Number(e.target.value);
                        setNumSteps(next);
                        if (next <= 7) setReuse(1);
                      }}
                    />
                  </label>
                  <label className="opt-narrow">
                    Layers
                    <input
                      type="number"
                      min={1}
                      max={50}
                      title="h3.c --layers (DiT blocks per pass; default 50)"
                      value={layers}
                      onChange={(e) => setLayers(Number(e.target.value))}
                    />
                  </label>
                  <label className="opt-narrow">
                    Reuse
                    <input
                      type="number"
                      min={1}
                      max={8}
                      title="h3.c --reuse (exclusive with --core-reuse; default 1)"
                      value={reuse}
                      disabled={numSteps <= 7}
                      onChange={(e) => setReuse(Number(e.target.value))}
                    />
                  </label>
                  <label className="opt-seed">
                    Seed
                    <input
                      type="text"
                      placeholder="random"
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                    />
                  </label>
                </div>
                )}

                <div className="lora-row">
                  <div className="lora-row-select">
                    <span className="lora-field-label">LoRA</span>
                    <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                      <LoraMultiSelect
                        presets={loraPresets}
                        selectedIds={loraPresetIds}
                        disabled={loraBusy || addingCustomLora || busy}
                        onToggle={(id, checked) => void toggleLoraPreset(id, checked)}
                        onRemovePreset={(preset) => void removeLoraPreset(preset)}
                      />
                      {FEATURES.LORA_MODAL && (
                        <button
                          type="button"
                          className="btn-secondary btn-compact"
                          onClick={() => setLoraModalOpen(true)}
                          disabled={busy}
                          title="Browse LoRA library"
                        >
                          Browse
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="lora-row-add">
                    <input
                      type="text"
                      className="lora-add-url"
                      placeholder="HF URL or path"
                      aria-label="LoRA URL or file path"
                      value={customLoraUrl}
                      disabled={addingCustomLora}
                      onChange={(e) => setCustomLoraUrl(e.target.value)}
                    />
                    <input
                      type="text"
                      className="lora-add-name"
                      placeholder="Label"
                      aria-label="LoRA display name"
                      value={customLoraLabel}
                      disabled={addingCustomLora}
                      onChange={(e) => setCustomLoraLabel(e.target.value)}
                    />
                    <input
                      type="number"
                      className="lora-add-scale"
                      min={0}
                      max={2}
                      step={0.05}
                      aria-label="LoRA strength"
                      title="Strength (0–2)"
                      value={customLoraScale}
                      disabled={addingCustomLora}
                      onChange={(e) => setCustomLoraScale(e.target.value)}
                    />
                    <button
                      type="button"
                      className="btn-secondary btn-compact lora-add-btn"
                      disabled={!customLoraUrl.trim() || addingCustomLora || loraBusy}
                      onClick={() => void addCustomLora()}
                    >
                      {addingCustomLora ? "…" : "Add"}
                    </button>
                  </div>
                </div>
                {loraActivity && <p className="hint hint-inline">{loraActivity}</p>}
                {loraPresetIds.length > 0 && (
                  <p className="hint hint-inline">
                    {loraPresets.find((p) => loraPresetIds.includes(p.id))?.guidance ??
                      "LoRA is fused into DiT weights at load. SSD streaming stays off."}
                  </p>
                )}

                <div className="options-checks">
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={ssdStreaming && loraPresetIds.length === 0}
                      onChange={(e) => setSsdStreaming(e.target.checked)}
                      disabled={loraPresetIds.length > 0}
                    />
                    SSD streaming (saves RAM, slower denoise)
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={tokenReduction && !tokenReductionLocked}
                      onChange={(e) => setTokenReduction(e.target.checked)}
                      disabled={tokenReductionLocked}
                    />
                    Token reduction
                  </label>
                </div>

                {quality === "close" && (
                  <p className="hint hint-inline">
                    {closePreset?.guidance ??
                      "Close runs 50 complete 50-block denoiser forwards — the oracle when a fast mode changes subject, anatomy, motion, or composition."}
                  </p>
                )}
                {resolution.guidance && <p className="hint hint-inline">{resolution.guidance}</p>}

                {(needsFirst || needsLast || lastUsesPrimaryUpload) && (
                  <div className="options-uploads">
                    <span className="media-panel-title">
                      {mode === "fl2va"
                        ? "First and last frames"
                        : mode === "last_frame"
                          ? "Last frame"
                          : "First frame"}
                    </span>
                    {(needsFirst || lastUsesPrimaryUpload) && (
                      <label className="media-upload">
                        <span className="media-upload-label">
                          {mode === "last_frame" ? "Last frame" : "First frame"}
                        </span>
                        <input
                          ref={imageRef}
                          type="file"
                          accept="image/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setImagePath(await uploadFile(f, "image"));
                              setImageName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">{imageName ?? "Choose image…"}</span>
                      </label>
                    )}
                    {mode === "fl2va" && (
                      <label className="media-upload">
                        <span className="media-upload-label">Last frame</span>
                        <input
                          ref={endImageRef}
                          type="file"
                          accept="image/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setEndImagePath(await uploadFile(f, "image"));
                              setEndImageName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">{endImageName ?? "Choose image…"}</span>
                      </label>
                    )}
                  </div>
                )}

                {mode === "ref2va" && (
                  FEATURES.REFERENCE_CHIPS ? (
                    <RefListEnhanced
                      refs={refs}
                      disabled={busy}
                      frames={frameLibrary}
                      clips={libraryClips}
                      onChange={handleRefsChange}
                      uploadFile={uploadFile}
                    />
                  ) : (
                    <RefList
                      refs={refs}
                      disabled={busy}
                      frames={frameLibrary}
                      clips={libraryClips}
                      onChange={handleRefsChange}
                      uploadFile={uploadFile}
                    />
                  )
                )}

                {isMultiClip && (
                  <p className="hint">
                    ×{clipMultiplier} clips will chain last frame → first frame and merge the result.
                  </p>
                )}

                {activeClip && (
                  <p className="meta">
                    Viewing: {formatDuration(activeClip.num_frames, config.defaults.fps)}
                    {activeClip.width && activeClip.height ? ` · ${activeClip.width}×${activeClip.height}` : ""}
                    {activeClip.bytes ? ` · ${formatBytes(activeClip.bytes)}` : ""}
                  </p>
                )}
              </div>
            )}
          </section>
        </div>

        <aside className="library">
          {/* Timeline view for current chain */}
          {FEATURES.TIMELINE_VIEW && chainId && chainParts.length > 1 && (
            <div style={{ marginBottom: "12px" }}>
              <TimelineStrip
                clips={chainParts}
                selectedClipId={selectedClipId}
                onSelectClip={applyClipSelection}
                lockedClipIds={lockedClipIds}
                onLockToggle={FEATURES.LOCKED_TAKES ? toggleClipLock : undefined}
                disabled={busy}
              />
            </div>
          )}

          <div className="library-header">
            <span className="library-title">Library</span>
            <span className="library-count">{libraryClips.length}</span>
          </div>
          <div className="library-grid">
            {libraryClips.map((clip) => (
              <div
                key={clip.id}
                className={`library-card-wrap ${activeClip?.id === clip.id ? "active" : ""}`}
              >
                <button
                  type="button"
                  className="library-card"
                  onClick={() => applyClipSelection(clip)}
                  title={clip.prompt}
                >
                  {clip.video_url && (
                    <video className="library-thumb" src={clip.video_url} muted playsInline preload="metadata" />
                  )}
                  <span className={`library-label ${clip.label.toLowerCase()}`}>{clip.label}</span>
                  <span className="library-prompt">{clipDisplayPrompt(clip.prompt)}</span>
                </button>
                <button
                  type="button"
                  className="library-delete"
                  title="Delete"
                  disabled={busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    void deleteClip(clip);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          {FEATURES.SCENE_QUEUE && sceneQueue.length > 0 && (
            <SceneQueue
              scenes={sceneQueue}
              onRemove={removeFromSceneQueue}
              onReorder={reorderSceneQueue}
              onEdit={loadSceneToEditor}
              onRunAll={() => void runSceneQueue()}
              onClear={clearSceneQueue}
              disabled={busy}
              running={queueRunning}
            />
          )}

          <div className="library-section">
            <div className="library-header">
              <span className="library-title">Frames</span>
              <span className="library-count">{frameLibrary.length}</span>
            </div>
            {frameLibrary.length === 0 ? (
              <p className="library-empty-hint">
                Pause a video and tap the camera icon to capture stills. Use them as a
                first or last frame, or as a reference image when that mode is selected.
              </p>
            ) : (
              <div className="frame-library-grid">
                {frameLibrary.map((frame) => (
                  <div
                    key={frame.id}
                    className={`frame-card-wrap ${
                      imagePath === frame.path || endImagePath === frame.path ? "active" : ""
                    }`}
                  >
                    <button
                      type="button"
                      className="frame-card"
                      title={`Use as start image: ${frame.label}`}
                      onClick={() => applyFrameAsInput(frame, "start")}
                    >
                      <img className="frame-thumb" src={frame.image_url} alt={frame.label} loading="lazy" />
                      <span className="frame-label">{frame.label}</span>
                    </button>
                    {mode === "fl2va" && (
                      <button
                        type="button"
                        className="frame-use-end"
                        title="Use as end image"
                        disabled={busy}
                        onClick={(e) => {
                          e.stopPropagation();
                          applyFrameAsInput(frame, "end");
                        }}
                      >
                        End
                      </button>
                    )}
                    <button
                      type="button"
                      className="library-delete"
                      title="Delete frame"
                      disabled={busy}
                      onClick={(e) => {
                        e.stopPropagation();
                        void deleteFrame(frame);
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Models panel */}
      {FEATURES.MODELS_PAGE && modelsOpen && (
        <div className="modal-backdrop" onClick={toggleModelsOpen}>
          <div className="modal modal--fullscreen" onClick={(e) => e.stopPropagation()}>
            <ModelsManager
              api={API}
              onClose={() => setModelsOpen(false)}
              onDownloadStateChange={handleModelsDownloadStateChange}
            />
          </div>
        </div>
      )}

      {/* Models close guard — download keeps running in the background */}
      {FEATURES.MODELS_PAGE && modelsCloseWarning && (
        <div className="models-close-toast">
          <span>Download in progress — it will continue in the background.</span>
          <button type="button" className="btn-ghost" onClick={confirmModelsClose}>
            Close anyway
          </button>
          <button type="button" className="btn-ghost" onClick={() => setModelsCloseWarning(false)}>
            Keep open
          </button>
        </div>
      )}

      {/* LoRA Modal */}
      {FEATURES.LORA_MODAL && (
        <LoraModal
          open={loraModalOpen}
          onClose={() => setLoraModalOpen(false)}
          presets={loraPresets}
          selectedIds={loraPresetIds}
          onToggle={(id, checked) => void toggleLoraPreset(id, checked)}
          onRemove={(preset) => void removeLoraPreset(preset)}
          disabled={busy || loraBusy}
        />
      )}
    </div>
  );
}
