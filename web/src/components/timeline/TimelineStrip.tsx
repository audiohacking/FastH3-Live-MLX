import { useCallback, useState } from "react";
import type { Clip } from "../../types";
import { FEATURES } from "../../config";
import { ClipCard } from "./ClipCard";
import { SeamControl, SeamIndicator, type SeamType } from "./SeamControl";

interface SeamState {
  type: SeamType;
  blendDuration: number;
}

interface TimelineStripProps {
  clips: Clip[];
  selectedClipId: string | null;
  onSelectClip: (clip: Clip) => void;
  onReorder?: (clipId: string, newIndex: number) => void;
  onMerge?: () => void;
  seams?: Record<string, SeamState>;
  onSeamChange?: (beforeClipId: string, afterClipId: string, seam: SeamState) => void;
  lockedClipIds?: Set<string>;
  onLockToggle?: (clipId: string) => void;
  disabled?: boolean;
}

function formatTotalDuration(clips: Clip[], fps = 24): string {
  const totalFrames = clips.reduce((sum, c) => sum + (c.num_frames ?? 0), 0);
  if (totalFrames === 0) return "0s";
  const seconds = totalFrames / fps;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function TimelineStrip({
  clips,
  selectedClipId,
  onSelectClip,
  onReorder,
  onMerge,
  seams = {},
  onSeamChange,
  lockedClipIds = new Set(),
  onLockToggle,
  disabled,
}: TimelineStripProps) {
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const [dragOverId, setDragOverId] = useState<string | null>(null);
  const [dragPosition, setDragPosition] = useState<"before" | "after" | null>(null);

  const handleDragStart = useCallback((e: React.DragEvent, clipId: string) => {
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", clipId);
    setDraggedId(clipId);
  }, []);

  const handleDragEnd = useCallback(() => {
    setDraggedId(null);
    setDragOverId(null);
    setDragPosition(null);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, clipId: string) => {
    e.preventDefault();
    if (!draggedId || draggedId === clipId) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const midpoint = rect.left + rect.width / 2;
    const position = e.clientX < midpoint ? "before" : "after";

    setDragOverId(clipId);
    setDragPosition(position);
  }, [draggedId]);

  const handleDrop = useCallback((e: React.DragEvent, targetClipId: string) => {
    e.preventDefault();
    if (!draggedId || !onReorder || draggedId === targetClipId) return;

    const draggedIndex = clips.findIndex((c) => c.id === draggedId);
    const targetIndex = clips.findIndex((c) => c.id === targetClipId);

    if (draggedIndex === -1 || targetIndex === -1) return;

    let newIndex = targetIndex;
    if (dragPosition === "after") {
      newIndex = targetIndex + 1;
    }
    // Adjust if dragging from before the target
    if (draggedIndex < newIndex) {
      newIndex -= 1;
    }

    onReorder(draggedId, newIndex);
    handleDragEnd();
  }, [draggedId, dragPosition, clips, onReorder, handleDragEnd]);

  const doneClips = clips.filter((c) => c.status === "done" && c.video_url);
  const hasMerged = clips.some((c) => c.label === "MERGED");
  const canMerge = doneClips.length > 1 && !hasMerged && onMerge;

  if (clips.length === 0) {
    return (
      <div className="timeline">
        <div className="timeline__empty">
          <svg className="timeline__empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="6" width="20" height="12" rx="2" />
            <path d="M6 6v12M10 6v12M14 6v12M18 6v12" />
          </svg>
          <span className="timeline__empty-text">
            Generate clips to build your timeline
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="timeline">
      <header className="timeline__header">
        <span className="timeline__title">Timeline</span>
        <div className="timeline__actions">
          {canMerge && (
            <button
              type="button"
              className="btn-secondary btn-compact"
              onClick={onMerge}
              disabled={disabled}
            >
              Merge All
            </button>
          )}
        </div>
      </header>

      <div className="timeline__strip">
        {clips.map((clip, index) => {
          const nextClip = clips[index + 1];
          const seamKey = nextClip ? `${clip.id}:${nextClip.id}` : null;
          const seam = seamKey ? seams[seamKey] : null;

          return (
            <div key={clip.id} className="timeline__clip-group">
              <ClipCard
                clip={clip}
                index={index}
                selected={clip.id === selectedClipId}
                locked={lockedClipIds.has(clip.id)}
                onLockToggle={onLockToggle}
                onClick={() => onSelectClip(clip)}
                onDragStart={(e) => handleDragStart(e, clip.id)}
                onDragEnd={handleDragEnd}
                onDragOver={(e) => handleDragOver(e, clip.id)}
                onDrop={(e) => handleDrop(e, clip.id)}
                dragging={clip.id === draggedId}
                dragOverPosition={clip.id === dragOverId ? dragPosition : null}
              />
              {nextClip && FEATURES.TIMELINE_VIEW && onSeamChange && (
                <SeamControl
                  beforeClipId={clip.id}
                  afterClipId={nextClip.id}
                  seamType={seam?.type ?? "cut"}
                  blendDuration={seam?.blendDuration ?? 0.5}
                  onSeamTypeChange={(type) =>
                    onSeamChange(clip.id, nextClip.id, {
                      type,
                      blendDuration: seam?.blendDuration ?? 0.5,
                    })
                  }
                  onBlendDurationChange={(duration) =>
                    onSeamChange(clip.id, nextClip.id, {
                      type: seam?.type ?? "blend",
                      blendDuration: duration,
                    })
                  }
                  disabled={disabled}
                />
              )}
              {nextClip && !onSeamChange && (
                <SeamIndicator seamType={seam?.type ?? "cut"} compact />
              )}
            </div>
          );
        })}
      </div>

      <footer className="timeline__footer">
        <span className="timeline__total">
          Total: <span className="timeline__total-value">{formatTotalDuration(doneClips)}</span>
          {" · "}
          {doneClips.length} clip{doneClips.length !== 1 ? "s" : ""}
        </span>
      </footer>
    </div>
  );
}
