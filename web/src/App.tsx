import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clipDisplayPrompt, snapshotFromClip } from "./clipEditor";
import { applyProgressEvent } from "./progress";
import { captureVideoFrame, formatVideoTime } from "./frameCapture";
import { refsAreValid } from "./components/composer/RefListEnhanced";
import { generateId } from "./utils";
import { FEATURES, TURBO_CONFIG, type TurboTier } from "./config";
import { LoraModal } from "./components/lora/LoraModal";
import { TimelineStrip } from "./components/timeline/TimelineStrip";
import { ModelsManager } from "./components/media/ModelsManager";
import { ComposerPanel } from "./components/composer/ComposerPanel";
import { compilePrompt } from "./compile";
import { leadWithStyle } from "./styleAtlas";
import type { CastMediaType, CastMember, CastMedia, Clip, Config, GenerationPreset, LibraryFrame, LoraPreset, PillOption, ProgressState, QualityPreset, ReferenceItem, RoutingMode, SceneQueueItem } from "./types";

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

async function uploadFile(file: File, kind: string): Promise<{ path: string; durationS?: number; filename?: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/api/upload?kind=${encodeURIComponent(kind)}`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error("Upload failed");
  const data = await r.json();
  return {
    path: data.path as string,
    durationS: typeof data.duration_s === "number" ? data.duration_s : undefined,
    filename: data.filename as string | undefined,
  };
}

const IMAGE_MODES = new Set(["first_frame", "fl2va"]);

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [frameLibrary, setFrameLibrary] = useState<LibraryFrame[]>([]);
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState("ref2va");
  const [routing, setRouting] = useState<RoutingMode>("ref2va");
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
  const [addingCustomLora, setAddingCustomLora] = useState(false);
  const [loraBusy, setLoraBusy] = useState(false);
  const [loraActivity, setLoraActivity] = useState<string | null>(null);
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
  const [libraryOpen, setLibraryOpen] = useState(true);
  const [framesOpen, setFramesOpen] = useState(true);
  const [activeStyleId, setActiveStyleId] = useState<string | null>(null);
  const playerVideoRef = useRef<HTMLVideoElement>(null);
  const runEventSourceRef = useRef<EventSource | null>(null);
  const clipsRef = useRef<Clip[]>([]);
  clipsRef.current = clips;

  const libraryClips = useMemo(
    () =>
      clips
        .filter((c) => c.status === "done" && c.video_url)
        .slice()
        .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "")),
    [clips],
  );
  const framesNewest = useMemo(
    () => [...frameLibrary].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "")),
    [frameLibrary],
  );
  const activeClip = useMemo(() => {
    if (selectedClipId) return clips.find((c) => c.id === selectedClipId) ?? null;
    return libraryClips[0] ?? null;
  }, [clips, selectedClipId, libraryClips]);
  const chainParts = useMemo(
    () => (chainId ? clips.filter((c) => c.chain_id === chainId) : []),
    [clips, chainId],
  );
  const showChainPicker = chainParts.filter((c) => c.video_url).length > 1;
  const compiled = useMemo(
    () =>
      compilePrompt({
        prompt,
        refs,
        castMembers,
        selectedCastIds,
        routing,
        imagePath,
        endImagePath,
      }),
    [prompt, refs, castMembers, selectedCastIds, routing, imagePath, endImagePath],
  );
  const effectiveMode = compiled.modeHint;
  const isRef2va = effectiveMode === "ref2va";
  const needsFirst = IMAGE_MODES.has(effectiveMode) && !isRef2va;
  const previewCanvas = resolutionId === "256x256";
  const aggressiveInternal = resolutionId === "512x512-aggressive";
  const tokenReductionLocked =
    previewCanvas ||
    aggressiveInternal ||
    quality === "aggressive" ||
    (layers === 40 && reuse === 3) ||
    loraPresetIds.length > 0;
  const qualityOptions: PillOption[] = useMemo(
    () =>
      (config?.quality_presets ?? []).map((p) => ({
        id: p.id,
        label: p.label,
        shortLabel: ({
          four_step: "4-step",
          aggressive: "aggr",
          fast: "fast",
          balanced: "bal",
          close: "close",
        } as Record<string, string>)[p.id],
        description: p.guidance ?? undefined,
      })),
    [config?.quality_presets],
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
      setRouting(snap.mode === "ref2va" ? "ref2va" : snap.mode === "t2va" ? "auto" : "fl2va");
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
      mode: compiled.modeHint,
      routing,
      selectedCastIds: [...selectedCastIds],
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
    setRouting(scene.routing ?? (scene.mode === "ref2va" ? "ref2va" : scene.mode === "t2va" ? "auto" : "fl2va"));
    setSelectedCastIds(scene.selectedCastIds ?? []);
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
    const pending = sceneQueue.filter((s) => s.status === "pending");
    if (pending.length === 0 || queueRunning || busy) return;

    setQueueRunning(true);
    for (const scene of pending) {
      loadSceneToEditor(scene);
      setSceneQueue((prev) =>
        prev.map((s) => (s.id === scene.id ? { ...s, status: "generating" as const } : s)),
      );
      try {
        await submitScene(scene);
        setSceneQueue((prev) =>
          prev.map((s) => (s.id === scene.id ? { ...s, status: "done" as const } : s)),
        );
      } catch (e) {
        setSceneQueue((prev) =>
          prev.map((s) =>
            s.id === scene.id ? { ...s, status: "failed" as const, error: String(e) } : s,
          ),
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
    setRouting(preset.mode === "ref2va" ? "ref2va" : preset.mode === "t2va" ? "auto" : "fl2va");
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

  async function persistCast(member: CastMember) {
    const r = await fetch(`${API}/api/cast/${member.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(member),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => null);
      throw new Error((body && body.detail) || "Failed to update cast member");
    }
    const data = await r.json();
    const next = data.cast as CastMember;
    setCastMembers((prev) => prev.map((m) => (m.id === next.id ? next : m)));
  }

  async function attachCastMedia(id: string, file: File, type: CastMediaType) {
    const member = castMembers.find((m) => m.id === id);
    if (!member) return;
    const up = await uploadFile(file, type);
    const media: CastMedia = {
      id: generateId(),
      type,
      path: up.path,
      label: file.name,
      thumbnailUrl: type === "image" ? URL.createObjectURL(file) : undefined,
      durationS: up.durationS,
    };
    await persistCast({ ...member, media: [...(member.media ?? []), media] });
  }

  async function removeCastMedia(id: string, mediaId: string) {
    const member = castMembers.find((m) => m.id === id);
    if (!member) return;
    await persistCast({ ...member, media: member.media.filter((m) => m.id !== mediaId) });
  }

  // Header "Models" button: always opens. Never blocks re-opening.
  function openModels() {
    setModelsOpen(true);
  }

  // Backdrop click / close request: if a download is running, surface a
  // confirmation (the server task survives), otherwise close.
  function closeModels() {
    if (modelsDownloadActive) {
      setModelsCloseWarning(true);
      return;
    }
    setModelsOpen(false);
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

  async function addCustomLora(spec: string, label: string, scale: number) {
    if (!spec || addingCustomLora) return;
    setAddingCustomLora(true);
    try {
      const r = await fetch(`${API}/api/loras/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
          label: label || undefined,
          scale: scale || 1,
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
    setSelectedCastIds([]);
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
    setImagePath(null);
    setImageName(null);
    setEndImagePath(null);
    setEndImageName(null);
    setClipMultiplier(1);
  }

  async function addImageRef(file: File) {
    const up = await uploadFile(file, "image");
    handleRefsChange([
      ...refs,
      {
        id: generateId(),
        kind: "image",
        path: up.path,
        name: file.name,
        enabled: true,
        durationS: up.durationS,
        previewUrl: URL.createObjectURL(file),
        refSize: "max",
      },
    ]);
  }

  async function addVideoRef(file: File, kind: "video" | "silent_video" | "video_audio") {
    const up = await uploadFile(file, "video");
    handleRefsChange([
      ...refs,
      {
        id: generateId(),
        kind,
        path: up.path,
        name: file.name,
        enabled: true,
        durationS: up.durationS,
        refSize: "max",
      },
    ]);
  }

  async function addAudioRef(file: File) {
    const up = await uploadFile(file, "audio");
    handleRefsChange([
      ...refs,
      {
        id: generateId(),
        kind: "audio",
        path: up.path,
        name: file.name,
        enabled: true,
        durationS: up.durationS,
        refSize: "max",
      },
    ]);
  }

  async function addVideoAudioRef(video: File, audio: File) {
    const v = await uploadFile(video, "video");
    const a = await uploadFile(audio, "audio");
    handleRefsChange([
      ...refs,
      {
        id: generateId(),
        kind: "video_audio",
        path: v.path,
        name: video.name,
        audioPath: a.path,
        audioName: audio.name,
        enabled: true,
        durationS: a.durationS ?? v.durationS,
        refSize: "max",
      },
    ]);
  }

  async function pickStartFile(file: File) {
    const up = await uploadFile(file, "image");
    setRefs([]);
    setImagePath(up.path);
    setImageName(file.name);
    setRouting("fl2va");
    if (!endImagePath) setMode("first_frame");
    else setMode("fl2va");
  }

  async function pickEndFile(file: File) {
    const up = await uploadFile(file, "image");
    setRefs([]);
    setEndImagePath(up.path);
    setEndImageName(file.name);
    setRouting("fl2va");
    if (!imagePath) setMode("last_frame");
    else setMode("fl2va");
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

  function subscribeRun(runId: string, runChainId: string): Promise<void> {
    runEventSourceRef.current?.close();
    setActiveRunId(runId);
    const es = new EventSource(`${API}/api/runs/${runId}/events`);
    runEventSourceRef.current = es;
    return new Promise((resolve, reject) => {
      let closed = false;
      const finishRun = (err?: string) => {
        if (closed) return;
        closed = true;
        es.close();
        runEventSourceRef.current = null;
        setActiveRunId(null);
        setBusy(false);
        setProgress(null);
        if (err) {
          reject(new Error(err));
          return;
        }
        notifyGenerationReady();
        resolve();
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
          fetchClips(runChainId).then((chainClips) => {
            setClips((prev) => replaceChainClips(prev, runChainId, chainClips));
            setSelectedClipId(pickPlaybackClip(chainClips, runChainId) ?? clipId ?? null);
          });
        }
        if (msg.type === "run_cancelled") {
          setProgress({ phase: "cancelled", message: String(msg.message || "Cancelled") });
          finishRun("Cancelled");
        } else if (msg.type === "run_complete" || msg.type === "run_done") {
          finishRun();
        } else if (msg.type === "error" || msg.type === "clip_failed") {
          const message = String(msg.error || msg.message || "Failed");
          setError(message);
          finishRun(message);
        }
      };
      es.onerror = () => {
        const message = "Lost connection to server while waiting for progress.";
        setError((prev) => prev ?? message);
        finishRun(message);
      };
    });
  }

  async function postGenerate(body: Record<string, unknown>): Promise<void> {
    setError(null);
    setBusy(true);
    setProgress({ phase: "starting", message: "Submitting…" });
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
    const wait = subscribeRun(data.run_id, data.chain_id);
    const chainClips = await fetchClips(data.chain_id);
    setClips((prev) => mergeClips(prev, chainClips));
    await wait;
  }

  function generateBody(opts: {
    compiledPrompt: string;
    fallbackPrompt: string;
    modeHint: string;
    refs: ReferenceItem[];
    quality: string;
    resolutionId: string;
    durationId: string;
    numSteps: number;
    layers: number;
    reuse: number;
    seed: string;
    clipMultiplier: number;
    tokenReduction: boolean;
    ssdStreaming: boolean;
    loraPresetIds: string[];
    imagePath: string | null;
    endImagePath: string | null;
  }): Record<string, unknown> {
    const res = config?.resolution_presets.find((p) => p.id === opts.resolutionId);
    const dur = config?.duration_presets.find((d) => d.id === opts.durationId);
    const ref2va = opts.modeHint === "ref2va";
    const multi = opts.clipMultiplier > 1 && !ref2va;
    const tokenLocked =
      opts.resolutionId === "256x256" ||
      opts.resolutionId === "512x512-aggressive" ||
      opts.quality === "aggressive" ||
      (opts.layers === 40 && opts.reuse === 3) ||
      opts.loraPresetIds.length > 0;
    const body: Record<string, unknown> = {
      prompt: opts.compiledPrompt.trim() || opts.fallbackPrompt.trim(),
      mode: opts.modeHint,
      quality: opts.quality,
      width: res?.width ?? 512,
      height: res?.height ?? 512,
      duration_seconds: dur?.seconds,
      num_frames: dur?.num_frames,
      clip_count: ref2va ? 1 : opts.clipMultiplier,
      num_steps: opts.numSteps,
      layers: opts.layers,
      reuse: opts.reuse,
      autocontinue: multi,
      autoconcat: multi,
      ssd_streaming: opts.loraPresetIds.length ? false : opts.ssdStreaming,
      token_reduction: tokenLocked ? false : opts.tokenReduction,
      loras: loraPresets
        .filter((p) => opts.loraPresetIds.includes(p.id))
        .map((p) => ({ id: p.id, spec: p.spec, scale: p.scale })),
    };
    if (res?.render_width) body.render_width = res.render_width;
    if (res?.render_height) body.render_height = res.render_height;
    if (opts.seed.trim() !== "") body.seed = Number(opts.seed);
    if (ref2va) {
      body.refs = opts.refs.map((r) => ({
        kind: r.kind,
        path: r.path,
        name: r.name,
        audio_path: r.audioPath || undefined,
        ref_size: r.refSize ?? "max",
      }));
    } else {
      if (IMAGE_MODES.has(opts.modeHint) && opts.imagePath) body.image_path = opts.imagePath;
      if (opts.modeHint === "last_frame" && (opts.imagePath || opts.endImagePath)) {
        body.end_image_path = opts.imagePath || opts.endImagePath;
      }
      if (opts.modeHint === "fl2va" && opts.endImagePath) body.end_image_path = opts.endImagePath;
    }
    return body;
  }

  async function submitScene(scene: SceneQueueItem): Promise<void> {
    const sceneCompiled = compilePrompt({
      prompt: scene.prompt,
      refs: scene.refs,
      castMembers,
      selectedCastIds: scene.selectedCastIds ?? [],
      routing: scene.routing ?? (scene.mode === "ref2va" ? "ref2va" : scene.mode === "t2va" ? "auto" : "fl2va"),
      imagePath: scene.imagePath,
      endImagePath: scene.endImagePath,
    });
    if (!scene.prompt.trim()) throw new Error("Empty prompt");
    if (sceneCompiled.modeHint === "ref2va" && !refsAreValid(sceneCompiled.refs).ok) {
      throw new Error("Scene needs a valid image or video reference");
    }
    await postGenerate(
      generateBody({
        compiledPrompt: sceneCompiled.compiledPrompt,
        fallbackPrompt: scene.prompt,
        modeHint: sceneCompiled.modeHint,
        refs: sceneCompiled.refs,
        quality: scene.quality,
        resolutionId: scene.resolutionId,
        durationId: scene.durationId,
        numSteps: scene.numSteps,
        layers: scene.layers,
        reuse: scene.reuse,
        seed: scene.seed,
        clipMultiplier: scene.clipMultiplier,
        tokenReduction: scene.tokenReduction,
        ssdStreaming: scene.ssdStreaming,
        loraPresetIds: scene.loraPresetIds,
        imagePath: scene.imagePath ?? null,
        endImagePath: scene.endImagePath ?? null,
      }),
    );
  }

  async function handleGenerate() {
    if (!canSubmit || !prompt.trim() || busy) return;
    try {
      await postGenerate(
        generateBody({
          compiledPrompt: compiled.compiledPrompt,
          fallbackPrompt: prompt,
          modeHint: compiled.modeHint,
          refs: compiled.refs,
          quality,
          resolutionId,
          durationId,
          numSteps,
          layers,
          reuse,
          seed,
          clipMultiplier,
          tokenReduction,
          ssdStreaming,
          loraPresetIds,
          imagePath,
          endImagePath,
        }),
      );
    } catch (e) {
      setError(String(e));
      setBusy(false);
      setProgress(null);
    }
  }

  const serverOk = config?.server_connected;
  const canSubmit = useMemo(() => {
    if (!prompt.trim() || busy || !serverOk) return false;
    if (isRef2va) return refsAreValid(compiled.refs).ok;
    if (needsFirst && !imagePath) return false;
    if (effectiveMode === "last_frame" && !imagePath && !endImagePath) return false;
    if (effectiveMode === "fl2va" && (!imagePath || !endImagePath)) return false;
    return true;
  }, [prompt, busy, serverOk, isRef2va, compiled.refs, needsFirst, imagePath, endImagePath, effectiveMode]);

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
            <button type="button" className="btn-secondary" onClick={openModels}>
              Models
            </button>
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

          {config && (
            <ComposerPanel
              busy={busy}
              canSubmit={canSubmit}
              prompt={prompt}
              onPromptChange={setPrompt}
              onGenerate={() => void handleGenerate()}
              compiled={compiled}
              refs={routing === "fl2va" ? [] : refs}
              onRefsChange={handleRefsChange}
              onAddImage={(file) => void addImageRef(file)}
              onAddVideo={(file, kind) => void addVideoRef(file, kind)}
              onAddAudio={(file) => void addAudioRef(file)}
              onAddVideoAudio={(video, audio) => void addVideoAudioRef(video, audio)}
              onClearComposer={() => {
                setPrompt("");
                setRefs([]);
                setImagePath(null);
                setImageName(null);
                setEndImagePath(null);
                setEndImageName(null);
                setSelectedCastIds([]);
              }}
              onOpenLora={() => setLoraModalOpen(true)}
              loraCount={loraPresetIds.length}
              loraActivity={loraActivity}
              castMembers={castMembers}
              selectedCastIds={selectedCastIds}
              onToggleCast={toggleCast}
              onCreateCast={createCastMember}
              onDeleteCast={deleteCastMember}
              onAttachCastMedia={attachCastMedia}
              onRemoveCastMedia={removeCastMedia}
              presets={generationPresets}
              onSavePreset={savePreset}
              onLoadPreset={loadPreset}
              onDeletePreset={deletePreset}
              activeStyleId={activeStyleId}
              onApplyStyle={(style) => {
                setPrompt((prev) => leadWithStyle(prev, style.text));
                setActiveStyleId(style.id);
              }}
              routing={routing}
              onRoutingChange={setRouting}
              durationId={durationId}
              onDurationId={setDurationId}
              resolutionId={resolutionId}
              onResolutionId={(id) => {
                setResolutionId(id);
                if (id === "256x256" || id === "512x512-aggressive") setTokenReduction(false);
              }}
              config={config}
              imageName={imageName}
              endImageName={endImageName}
              onPickStartFile={(file) => void pickStartFile(file)}
              onPickEndFile={(file) => void pickEndFile(file)}
              onPickStartFrame={(frame) => {
                setRefs([]);
                setImagePath(frame.path);
                setImageName(frame.label);
                setRouting("fl2va");
              }}
              onPickEndFrame={(frame) => {
                setRefs([]);
                setEndImagePath(frame.path);
                setEndImageName(frame.label);
                setRouting("fl2va");
              }}
              onClearStart={() => { setImagePath(null); setImageName(null); }}
              onClearEnd={() => { setEndImagePath(null); setEndImageName(null); }}
              frames={framesNewest}
              clips={libraryClips}
              onAddFrameRef={(frame) => applyFrameAsInput(frame, "start")}
              onAddClipRef={(clip) => {
                const path = clip.path || clip.filename;
                if (!path) return;
                handleRefsChange([
                  ...refs,
                  { id: generateId(), kind: "silent_video", path, name: clip.label || clip.filename, enabled: true, source: "library" },
                ]);
              }}
              onAddShot={addToSceneQueue}
              seed={seed}
              onSeed={setSeed}
              numSteps={numSteps}
              onNumSteps={setNumSteps}
              layers={layers}
              onLayers={setLayers}
              reuse={reuse}
              onReuse={setReuse}
              quality={quality}
              qualityOptions={qualityOptions}
              onQuality={(id) => {
                const preset = config.quality_presets.find((p) => p.id === id);
                const fields = fieldsFromPreset(preset);
                setQuality(id);
                setNumSteps(fields.steps);
                setLayers(fields.layers);
                setReuse(fields.reuse);
                setTokenReduction(fields.tokenReduction);
                if (turboEnabled) setTurboEnabled(false);
              }}
              turboEnabled={turboEnabled}
              turboTier={turboTier}
              turboLoading={turboLoading}
              loraBusy={loraBusy}
              onTurbo={(enabled) => void handleTurboToggle(enabled)}
              onTurboTier={handleTurboTierChange}
              tokenReduction={tokenReduction}
              tokenReductionLocked={tokenReductionLocked}
              onTokenReduction={setTokenReduction}
              ssdStreaming={ssdStreaming}
              ssdLocked={loraPresetIds.length > 0}
              onSsdStreaming={setSsdStreaming}
              clipMultiplier={clipMultiplier}
              onClipMultiplier={setClipMultiplier}
              sceneQueue={sceneQueue}
              onRemoveScene={removeFromSceneQueue}
              onReorderScenes={reorderSceneQueue}
              onEditScene={loadSceneToEditor}
              onRunQueue={() => void runSceneQueue()}
              onClearQueue={clearSceneQueue}
              queueRunning={queueRunning}
            />
          )}
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

          <button
            type="button"
            className="library-header library-header--toggle"
            onClick={() => setLibraryOpen((v) => !v)}
            aria-expanded={libraryOpen}
          >
            <span className="library-title">Library</span>
            <span className="library-header__meta">
              <span className="library-count">{libraryClips.length}</span>
              <span className={`library-chevron${libraryOpen ? " is-open" : ""}`} aria-hidden>
                ▾
              </span>
            </span>
          </button>
          {libraryOpen && (
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
          )}

          <div className="library-section">
            <button
              type="button"
              className="library-header library-header--toggle"
              onClick={() => setFramesOpen((v) => !v)}
              aria-expanded={framesOpen}
            >
              <span className="library-title">Frames</span>
              <span className="library-header__meta">
                <span className="library-count">{framesNewest.length}</span>
                <span className={`library-chevron${framesOpen ? " is-open" : ""}`} aria-hidden>
                  ▾
                </span>
              </span>
            </button>
            {framesOpen && (framesNewest.length === 0 ? (
              <p className="library-empty-hint">
                Pause a video and tap the camera icon to capture stills. Use them as a
                first or last frame, or as a reference image when that mode is selected.
              </p>
            ) : (
              <div className="frame-library-grid">
                {framesNewest.map((frame) => (
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
            ))}
          </div>
        </aside>
      </div>

      {/* Models panel */}
      {FEATURES.MODELS_PAGE && modelsOpen && (
        <div className="modal-backdrop" onClick={closeModels}>
          <div className="modal modal--fullscreen" onClick={(e) => e.stopPropagation()}>
            <ModelsManager
              api={API}
              onClose={closeModels}
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
          onAddCustom={addCustomLora}
          addingCustom={addingCustomLora}
          disabled={busy || loraBusy}
        />
      )}
    </div>
  );
}
