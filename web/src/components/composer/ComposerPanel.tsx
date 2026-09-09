import { useMemo, useRef, useState } from "react";
import type { CastMediaType, CastMember, Clip, Config, GenerationPreset, LibraryFrame, PillOption, ReferenceItem, RoutingMode, SceneQueueItem } from "../../types";
import type { StyleEntry } from "../../styleAtlas";
import { FEATURES, type TurboTier } from "../../config";
import { mentionCandidates, type CompileResult } from "../../compile";
import { ComposerRail } from "./ComposerRail";
import { AssetCapsules } from "./AssetCapsules";
import { ConfigRow } from "./ConfigRow";
import { SamplerRow } from "./SamplerRow";
import { WhatTheModelReads } from "./WhatTheModelReads";
import { SceneQueue } from "./SceneQueue";
import { MediaPicker } from "./MediaPicker";
import { PresetManager } from "../presets/PresetManager";
import { CastPicker } from "../media/CastPicker";

export type ComposerPanelProps = {
  busy: boolean;
  canSubmit: boolean;
  prompt: string;
  onPromptChange: (v: string) => void;
  onGenerate: () => void;
  compiled: CompileResult;
  refs: ReferenceItem[];
  onRefsChange: (refs: ReferenceItem[]) => void;
  onAddImage: (file: File) => void;
  onAddVideo: (file: File, kind: "video" | "silent_video" | "video_audio") => void;
  onAddAudio: (file: File) => void;
  onAddVideoAudio: (video: File, audio: File) => void;
  onClearComposer: () => void;
  onOpenLora: () => void;
  loraCount: number;
  loraActivity: string | null;
  castMembers: CastMember[];
  selectedCastIds: string[];
  onToggleCast: (id: string) => void;
  onCreateCast: (name: string, description?: string) => Promise<void>;
  onDeleteCast: (id: string) => Promise<void>;
  onAttachCastMedia: (id: string, file: File, type: CastMediaType) => Promise<void>;
  onRemoveCastMedia: (id: string, mediaId: string) => Promise<void>;
  presets: GenerationPreset[];
  onSavePreset: (name: string, description?: string) => void;
  onLoadPreset: (preset: GenerationPreset) => void;
  onDeletePreset: (id: string) => void;
  activeStyleId?: string | null;
  onApplyStyle: (style: StyleEntry) => void;
  routing: RoutingMode;
  onRoutingChange: (next: RoutingMode) => void;
  durationId: string;
  onDurationId: (id: string) => void;
  resolutionId: string;
  onResolutionId: (id: string) => void;
  config: Config;
  imageName: string | null;
  endImageName: string | null;
  onPickStartFile: (file: File) => void;
  onPickEndFile: (file: File) => void;
  onPickStartFrame: (frame: LibraryFrame) => void;
  onPickEndFrame: (frame: LibraryFrame) => void;
  onClearStart: () => void;
  onClearEnd: () => void;
  frames: LibraryFrame[];
  clips: Clip[];
  onAddFrameRef: (frame: LibraryFrame) => void;
  onAddClipRef: (clip: Clip) => void;
  onAddShot: () => void;
  seed: string;
  onSeed: (v: string) => void;
  numSteps: number;
  onNumSteps: (v: number) => void;
  layers: number;
  onLayers: (v: number) => void;
  reuse: number;
  onReuse: (v: number) => void;
  quality: string;
  qualityOptions: PillOption[];
  onQuality: (id: string) => void;
  turboEnabled: boolean;
  turboTier: TurboTier;
  turboLoading?: boolean;
  loraBusy?: boolean;
  onTurbo: (enabled: boolean) => void;
  onTurboTier: (tier: TurboTier) => void;
  tokenReduction: boolean;
  tokenReductionLocked: boolean;
  onTokenReduction: (v: boolean) => void;
  ssdStreaming: boolean;
  ssdLocked: boolean;
  onSsdStreaming: (v: boolean) => void;
  clipMultiplier: number;
  onClipMultiplier: (n: number) => void;
  sceneQueue: SceneQueueItem[];
  onRemoveScene: (id: string) => void;
  onReorderScenes: (scenes: SceneQueueItem[]) => void;
  onEditScene: (scene: SceneQueueItem) => void;
  onRunQueue: () => void;
  onClearQueue: () => void;
  queueRunning?: boolean;
};

export function ComposerPanel(props: ComposerPanelProps) {
  const {
    busy, canSubmit, prompt, onPromptChange, onGenerate, compiled, refs, onRefsChange,
  } = props;
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const [mention, setMention] = useState<{ start: number; filter: string } | null>(null);
  const [mentionIndex, setMentionIndex] = useState(0);

  const candidates = useMemo(() => {
    const all = mentionCandidates(refs, props.castMembers);
    if (!mention) return [];
    const q = mention.filter.toLowerCase();
    return all.filter((c) => c.handle.toLowerCase().includes(q) || c.label.toLowerCase().includes(q));
  }, [refs, props.castMembers, mention]);

  function onPromptInput(value: string, caret: number) {
    onPromptChange(value);
    const before = value.slice(0, caret);
    const at = before.lastIndexOf("@");
    if (at >= 0 && !/\s/.test(before.slice(at))) {
      setMention({ start: at, filter: before.slice(at) });
      setMentionIndex(0);
    } else {
      setMention(null);
    }
  }

  function insertMention(handle: string) {
    if (!mention) return;
    const next = prompt.slice(0, mention.start) + handle + " " + prompt.slice(promptRef.current?.selectionStart ?? prompt.length);
    onPromptChange(next);
    setMention(null);
  }

  const clipItems = props.clips.map((c) => ({ id: c.id, label: c.label, videoUrl: c.video_url }));
  const frameItems = props.frames.map((f) => ({ id: f.id, label: f.label, thumbUrl: f.image_url }));

  return (
    <section className="composer composer-panel">
      {FEATURES.SCENE_QUEUE && (
        <SceneQueue
          scenes={props.sceneQueue}
          onRemove={props.onRemoveScene}
          onReorder={props.onReorderScenes}
          onEdit={props.onEditScene}
          onRunAll={props.onRunQueue}
          onClear={props.onClearQueue}
          disabled={busy}
          running={props.queueRunning}
        />
      )}
      <ComposerRail
        disabled={busy}
        loraCount={props.loraCount}
        onAddImage={props.onAddImage}
        onAddVideo={props.onAddVideo}
        onAddAudio={props.onAddAudio}
        onAddVideoAudio={props.onAddVideoAudio}
        onOpenLora={props.onOpenLora}
        onClear={props.onClearComposer}
        castSlot={
          <CastPicker
            members={props.castMembers}
            selectedIds={props.selectedCastIds}
            onToggle={props.onToggleCast}
            onCreate={props.onCreateCast}
            onDelete={props.onDeleteCast}
            onAttachMedia={props.onAttachCastMedia}
            onRemoveMedia={props.onRemoveCastMedia}
            disabled={busy}
          />
        }
        librarySlot={
          <>
            <MediaPicker
              variant="rail"
              label="Library clip"
              items={clipItems}
              disabled={busy}
              emptyHint="Generate a clip to reuse it as a silent-video reference."
              onPick={(id) => {
                const clip = props.clips.find((c) => c.id === id);
                if (clip) props.onAddClipRef(clip);
              }}
            />
            <MediaPicker
              variant="rail"
              label="Frame as ref"
              items={frameItems}
              disabled={busy || props.routing === "fl2va"}
              emptyHint="Capture a frame from the player first."
              onPick={(id) => {
                const frame = props.frames.find((f) => f.id === id);
                if (frame) props.onAddFrameRef(frame);
              }}
            />
          </>
        }
        presetSlot={
          <PresetManager
            presets={props.presets}
            onSave={props.onSavePreset}
            onLoad={props.onLoadPreset}
            onDelete={props.onDeletePreset}
            disabled={busy}
            activeStyleId={props.activeStyleId}
            onApplyStyle={props.onApplyStyle}
          />
        }
      />

      <AssetCapsules refs={refs} disabled={busy} onChange={onRefsChange} />

      <div className="composer-prompt">
        <div className="prompt-field-wrap">
          <p className="composer-prompt__shot">[Shot 1]</p>
          {mention && candidates.length > 0 && (
            <div className="mention-menu" role="listbox">
              {candidates.map((c, i) => (
                <button
                  key={c.handle}
                  type="button"
                  className={i === mentionIndex ? "is-active" : ""}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    insertMention(c.handle);
                  }}
                >
                  {c.label}
                </button>
              ))}
            </div>
          )}
          <textarea
            ref={promptRef}
            className="prompt-input"
            rows={3}
            placeholder={refs.length > 0
              ? "Describe the scene using @img-1, @vid-1, @aud-1…"
              : "Scene, action, camera, look, and audio…"}
            value={prompt}
            onChange={(e) => onPromptInput(e.target.value, e.target.selectionStart)}
            onKeyDown={(e) => {
              if (mention && candidates.length > 0) {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setMentionIndex((i) => (i + 1) % candidates.length);
                  return;
                }
                if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setMentionIndex((i) => (i - 1 + candidates.length) % candidates.length);
                  return;
                }
                if (e.key === "Enter" || e.key === "Tab") {
                  e.preventDefault();
                  insertMention(candidates[mentionIndex].handle);
                  return;
                }
                if (e.key === "Escape") {
                  setMention(null);
                  return;
                }
              }
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onGenerate();
              }
            }}
            disabled={busy}
          />
          <WhatTheModelReads
            compiledPrompt={compiled.compiledPrompt}
            tokens={compiled.tokens}
            warnings={compiled.warnings}
            disabled={busy}
          />
        </div>
        <div className="composer-prompt__actions">
          <button type="button" className="btn-add-scene" onClick={props.onAddShot} disabled={busy} title="Add scene">
            +
          </button>
          <button type="button" className="btn-generate gen-submit" onClick={onGenerate} disabled={!canSubmit}>
            ↑
          </button>
        </div>
      </div>

      <div className="config-card">
        <ConfigRow
          disabled={busy}
          routing={props.routing}
          onRoutingChange={props.onRoutingChange}
          durationId={props.durationId}
          durationPresets={props.config.duration_presets}
          onDurationId={props.onDurationId}
          resolutionId={props.resolutionId}
          resolutionPresets={props.config.resolution_presets}
          onResolutionId={props.onResolutionId}
          refs={refs}
          imageName={props.imageName}
          endImageName={props.endImageName}
          onPickStartFile={props.onPickStartFile}
          onPickEndFile={props.onPickEndFile}
          onPickStartFrame={props.onPickStartFrame}
          onPickEndFrame={props.onPickEndFrame}
          onClearStart={props.onClearStart}
          onClearEnd={props.onClearEnd}
          frames={props.frames}
          engineNote={props.config.engine_ok === false ? (props.config.engine_error ?? undefined) : undefined}
        />
        <SamplerRow
          disabled={busy}
          seed={props.seed}
          onSeed={props.onSeed}
          numSteps={props.numSteps}
          onNumSteps={props.onNumSteps}
          layers={props.layers}
          onLayers={props.onLayers}
          reuse={props.reuse}
          onReuse={props.onReuse}
          reuseLocked={props.numSteps <= 7}
          quality={props.quality}
          qualityOptions={props.qualityOptions}
          onQuality={props.onQuality}
          turboEnabled={props.turboEnabled}
          turboTier={props.turboTier}
          turboLoading={props.turboLoading}
          loraBusy={props.loraBusy}
          onTurbo={props.onTurbo}
          onTurboTier={props.onTurboTier}
          tokenReduction={props.tokenReduction}
          tokenReductionLocked={props.tokenReductionLocked}
          onTokenReduction={props.onTokenReduction}
          ssdStreaming={props.ssdStreaming}
          ssdLocked={props.ssdLocked}
          onSsdStreaming={props.onSsdStreaming}
          clipMultiplier={props.clipMultiplier}
          clipMultiplierMax={props.config.clip_multiplier_max ?? 10}
          showClips={compiled.modeHint !== "ref2va"}
          onClipMultiplier={props.onClipMultiplier}
        />
      </div>
      {props.loraActivity && <p className="lora-inline-hint">{props.loraActivity}</p>}
    </section>
  );
}
