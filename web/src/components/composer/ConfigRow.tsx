import { useEffect, useMemo, useRef, useState } from "react";
import { activeRefs, handleForRef, nearestDurationId } from "../../compile";
import type { LibraryFrame, PresetOption, ReferenceItem, RoutingMode } from "../../types";
import { MediaPickerGrid } from "./MediaPicker";

type Props = {
  disabled?: boolean;
  routing: RoutingMode;
  onRoutingChange: (next: RoutingMode) => void;
  durationId: string;
  durationPresets: PresetOption[];
  onDurationId: (id: string) => void;
  resolutionId: string;
  resolutionPresets: PresetOption[];
  onResolutionId: (id: string) => void;
  refs: ReferenceItem[];
  imageName: string | null;
  endImageName: string | null;
  onPickStartFile: (file: File) => void;
  onPickEndFile: (file: File) => void;
  onPickStartFrame: (frame: LibraryFrame) => void;
  onPickEndFrame: (frame: LibraryFrame) => void;
  onClearStart: () => void;
  onClearEnd: () => void;
  frames: LibraryFrame[];
  engineNote?: string;
};

const ROUTING_LABEL: Record<RoutingMode, string> = {
  auto: "AUTO",
  fl2va: "FL2VA",
  ref2va: "REF2VA",
};

function cycleRouting(current: RoutingMode): RoutingMode {
  if (current === "auto") return "fl2va";
  if (current === "fl2va") return "ref2va";
  return "auto";
}

export function ConfigRow({
  disabled,
  routing,
  onRoutingChange,
  durationId,
  durationPresets,
  onDurationId,
  resolutionId,
  resolutionPresets,
  onResolutionId,
  refs,
  imageName,
  endImageName,
  onPickStartFile,
  onPickEndFile,
  onPickStartFrame,
  onPickEndFrame,
  onClearStart,
  onClearEnd,
  frames,
  engineNote,
}: Props) {
  const startRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLInputElement>(null);
  const startAnchor = useRef<HTMLDivElement>(null);
  const endAnchor = useRef<HTMLDivElement>(null);
  const aspectAnchor = useRef<HTMLDivElement>(null);
  const resAnchor = useRef<HTMLDivElement>(null);
  const [startOpen, setStartOpen] = useState(false);
  const [endOpen, setEndOpen] = useState(false);
  const [aspectOpen, setAspectOpen] = useState(false);
  const [resOpen, setResOpen] = useState(false);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (startAnchor.current && !startAnchor.current.contains(t)) setStartOpen(false);
      if (endAnchor.current && !endAnchor.current.contains(t)) setEndOpen(false);
      if (aspectAnchor.current && !aspectAnchor.current.contains(t)) setAspectOpen(false);
      if (resAnchor.current && !resAnchor.current.contains(t)) setResOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const resolution = resolutionPresets.find((r) => r.id === resolutionId);
  const aspect = resolution?.aspect || "1:1";
  const aspects = useMemo(() => {
    const seen = new Set<string>();
    const order: string[] = [];
    for (const p of resolutionPresets) {
      const a = p.aspect || "1:1";
      if (!seen.has(a)) {
        seen.add(a);
        order.push(a);
      }
    }
    return order;
  }, [resolutionPresets]);
  const inAspect = resolutionPresets.filter((p) => (p.aspect || "1:1") === aspect);

  const durIndex = Math.max(0, durationPresets.findIndex((d) => d.id === durationId));
  const durLabel = durationPresets[durIndex]?.id ?? durationId;

  const matchRef = activeRefs(refs).find((r) => r.durationS && r.durationS > 0);
  const matchIndex = matchRef ? refs.indexOf(matchRef) : -1;
  const matchHandle = matchRef && matchIndex >= 0 ? handleForRef(refs, matchIndex) : null;

  const resLabel = resolution
    ? resolution.render_width
      ? `${resolution.render_width}→${resolution.width}×${resolution.height}`
      : `${resolution.width}×${resolution.height}`
    : resolutionId;

  const frameItems = frames.map((f) => ({ id: f.id, label: f.label, thumbUrl: f.image_url }));

  return (
    <>
      <div className="config-card__row">
        <button
          type="button"
          className="chip-btn chip-btn--routing"
          disabled={disabled}
          onClick={() => onRoutingChange(cycleRouting(routing))}
          title="Cycle AUTO / always FL2VA / always Ref2VA"
        >
          {ROUTING_LABEL[routing]}
        </button>
        <div className="popover-anchor" ref={startAnchor}>
          <button
            type="button"
            className={`chip-btn${imageName ? " is-on" : ""}`}
            disabled={disabled}
            title={imageName ?? "Start frame"}
            onClick={() => setStartOpen((v) => !v)}
          >
            Start frame
          </button>
          {startOpen && (
            <div className="frame-popover">
              <div className="frame-popover__actions">
                <button type="button" className="chip-btn" onClick={() => startRef.current?.click()}>
                  Upload
                </button>
                {imageName && (
                  <button type="button" className="chip-btn" onClick={() => { onClearStart(); setStartOpen(false); }}>
                    Clear
                  </button>
                )}
              </div>
              <MediaPickerGrid
                items={frameItems}
                emptyHint="Capture a frame from the player first."
                onPick={(id) => {
                  const frame = frames.find((f) => f.id === id);
                  if (frame) {
                    onPickStartFrame(frame);
                    setStartOpen(false);
                  }
                }}
              />
            </div>
          )}
        </div>
        <div className="popover-anchor" ref={endAnchor}>
          <button
            type="button"
            className={`chip-btn${endImageName ? " is-on" : ""}`}
            disabled={disabled}
            title={endImageName ?? "End frame"}
            onClick={() => setEndOpen((v) => !v)}
          >
            End frame
          </button>
          {endOpen && (
            <div className="frame-popover">
              <div className="frame-popover__actions">
                <button type="button" className="chip-btn" onClick={() => endRef.current?.click()}>
                  Upload
                </button>
                {endImageName && (
                  <button type="button" className="chip-btn" onClick={() => { onClearEnd(); setEndOpen(false); }}>
                    Clear
                  </button>
                )}
              </div>
              <MediaPickerGrid
                items={frameItems}
                emptyHint="Capture a frame from the player first."
                onPick={(id) => {
                  const frame = frames.find((f) => f.id === id);
                  if (frame) {
                    onPickEndFrame(frame);
                    setEndOpen(false);
                  }
                }}
              />
            </div>
          )}
        </div>
        <div className="stepper-pill" title="Duration (snapped to H3 5+17n @ 24fps)">
          <button
            type="button"
            disabled={disabled || durIndex <= 0}
            onClick={() => onDurationId(durationPresets[durIndex - 1].id)}
          >
            −
          </button>
          <span className="stepper-pill__value stepper-pill__value--dur">{durLabel}</span>
          <button
            type="button"
            disabled={disabled || durIndex >= durationPresets.length - 1}
            onClick={() => onDurationId(durationPresets[durIndex + 1].id)}
          >
            +
          </button>
        </div>
        {matchRef && matchHandle && matchRef.durationS != null && (
          <button
            type="button"
            className="chip-btn chip-btn--dashed"
            disabled={disabled}
            title={`Snap duration to ${matchHandle} (${matchRef.durationS.toFixed(2)} s)`}
            onClick={() => {
              const id = nearestDurationId(matchRef.durationS ?? 0, durationPresets);
              if (id) onDurationId(id);
            }}
          >
            match {matchHandle}
          </button>
        )}
        <div className="popover-anchor" ref={aspectAnchor}>
          <button
            type="button"
            className="chip-btn chip-btn--aspect"
            disabled={disabled}
            onClick={() => setAspectOpen((v) => !v)}
          >
            {aspect}
          </button>
          {aspectOpen && (
            <div className="aspect-menu">
              {aspects.map((a) => (
                <button
                  key={a}
                  type="button"
                  className={a === aspect ? "is-active" : ""}
                  onClick={() => {
                    const first =
                      resolutionPresets.find((p) => (p.aspect || "1:1") === a && !p.render_width) ??
                      resolutionPresets.find((p) => (p.aspect || "1:1") === a);
                    if (first) onResolutionId(first.id);
                    setAspectOpen(false);
                  }}
                >
                  {a}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="popover-anchor" ref={resAnchor}>
          <button
            type="button"
            className="chip-btn"
            disabled={disabled}
            title={resolution?.label ?? resLabel}
            onClick={() => setResOpen((v) => !v)}
          >
            {resLabel}
          </button>
          {resOpen && (
            <div className="aspect-menu">
              {inAspect.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className={p.id === resolutionId ? "is-active" : ""}
                  onClick={() => {
                    onResolutionId(p.id);
                    setResOpen(false);
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      {engineNote && <p className="engine-note">{engineNote}</p>}
      <input ref={startRef} type="file" accept="image/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (f) {
          onPickStartFile(f);
          setStartOpen(false);
        }
      }} />
      <input ref={endRef} type="file" accept="image/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (f) {
          onPickEndFile(f);
          setEndOpen(false);
        }
      }} />
    </>
  );
}
