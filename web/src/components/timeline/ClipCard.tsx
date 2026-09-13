import type { Clip } from "../../types";

interface ClipCardProps {
  clip: Clip;
  index: number;
  selected?: boolean;
  locked?: boolean;
  onLockToggle?: (clipId: string) => void;
  onClick?: () => void;
  onDragStart?: (e: React.DragEvent) => void;
  onDragEnd?: (e: React.DragEvent) => void;
  onDragOver?: (e: React.DragEvent) => void;
  onDrop?: (e: React.DragEvent) => void;
  dragging?: boolean;
  dragOverPosition?: "before" | "after" | null;
}

function formatDuration(frames?: number, fps = 24): string {
  if (!frames) return "";
  const seconds = frames / fps;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function truncatePrompt(prompt: string, maxLen = 40): string {
  if (prompt.length <= maxLen) return prompt;
  return prompt.slice(0, maxLen - 1) + "…";
}

export function ClipCard({
  clip,
  index,
  selected,
  locked,
  onLockToggle,
  onClick,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
  dragging,
  dragOverPosition,
}: ClipCardProps) {
  const isGenerating = clip.status === "pending" || clip.status === "generating";
  const isFailed = clip.status === "failed";
  const isDone = clip.status === "done" && clip.video_url;

  const classNames = [
    "timeline-clip",
    selected && "timeline-clip--selected",
    locked && "timeline-clip--locked",
    dragging && "timeline-clip--dragging",
    isGenerating && "timeline-clip--generating",
    isFailed && "timeline-clip--failed",
    dragOverPosition === "before" && "timeline-clip--drag-over-before",
    dragOverPosition === "after" && "timeline-clip--drag-over-after",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={classNames}
      onClick={onClick}
      draggable={!isGenerating}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <span className="timeline-clip__index">{index + 1}</span>

      {clip.video_url ? (
        <video
          className="timeline-clip__thumb"
          src={clip.video_url}
          muted
          playsInline
          preload="metadata"
        />
      ) : (
        <div className="timeline-clip__thumb timeline-clip__thumb--placeholder">
          {isGenerating ? "" : isFailed ? "Failed" : "No video"}
        </div>
      )}

      {isGenerating && (
        <div className="timeline-clip__overlay">
          <span className="timeline-clip__spinner" />
        </div>
      )}

      {clip.num_frames && (
        <span className="timeline-clip__duration">
          {formatDuration(clip.num_frames)}
        </span>
      )}

      <div className="timeline-clip__info">
        <span className={`timeline-clip__label${clip.label === "MERGED" ? " timeline-clip__label--merged" : ""}`}>
          {clip.label}
        </span>
        <span className="timeline-clip__prompt" title={clip.prompt}>
          {truncatePrompt(clip.prompt)}
        </span>
      </div>

      {isDone && onLockToggle && (
        <button
          type="button"
          className={`timeline-clip__lock${locked ? " timeline-clip__lock--active" : ""}`}
          onClick={(e) => {
            e.stopPropagation();
            onLockToggle(clip.id);
          }}
          title={locked ? "Unlock clip (will be included in batch regeneration)" : "Lock clip (exclude from batch regeneration)"}
        >
          {locked ? (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 1C8.676 1 6 3.676 6 7v2H5c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V11c0-1.1-.9-2-2-2h-1V7c0-3.324-2.676-6-6-6zm0 2c2.276 0 4 1.724 4 4v2H8V7c0-2.276 1.724-4 4-4zm0 10c1.1 0 2 .9 2 2s-.9 2-2 2-2-.9-2-2 .9-2 2-2z"/>
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="5" y="11" width="14" height="10" rx="2"/>
              <path d="M8 11V7a4 4 0 018 0"/>
            </svg>
          )}
        </button>
      )}
    </div>
  );
}
