import { useState } from "react";
import type { RefKind, ReferenceItem } from "../../types";

const KIND_ICONS: Record<RefKind, React.ReactNode> = {
  image: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="M21 15l-5-5L5 21" />
    </svg>
  ),
  silent_video: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="M10 9l5 3-5 3V9z" />
      <path d="M2 8h20M2 16h20" strokeOpacity="0.3" />
    </svg>
  ),
  video: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="M10 9l5 3-5 3V9z" />
    </svg>
  ),
  video_audio: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="4" width="14" height="16" rx="2" />
      <path d="M8 9l4 3-4 3V9z" />
      <path d="M19 8v8M22 10v4" />
    </svg>
  ),
  audio: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </svg>
  ),
};

function getTokenForRef(refs: ReferenceItem[], index: number): string {
  const ref = refs[index];
  let count = 0;
  for (let i = 0; i <= index; i++) {
    const r = refs[i];
    if (ref.kind === "image" && r.kind === "image") count++;
    else if (ref.kind === "audio" && r.kind === "audio") count++;
    else if (
      (ref.kind === "video" || ref.kind === "silent_video" || ref.kind === "video_audio") &&
      (r.kind === "video" || r.kind === "silent_video" || r.kind === "video_audio")
    ) count++;
  }
  if (ref.kind === "image") return `Picture ${count}`;
  if (ref.kind === "audio") return `Audio ${count}`;
  return `Video ${count}`;
}

interface ReferenceChipProps {
  ref_: ReferenceItem;
  token: string;
  onRemove: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  disabled?: boolean;
}

function ReferenceChip({ ref_, token, onRemove, disabled }: ReferenceChipProps) {
  const [showPopover, setShowPopover] = useState(false);

  return (
    <div
      className={`ref-chip ref-chip--${ref_.kind}`}
      onMouseEnter={() => setShowPopover(true)}
      onMouseLeave={() => setShowPopover(false)}
    >
      <span className="ref-chip__icon">{KIND_ICONS[ref_.kind]}</span>
      <span className="ref-chip__token">{token}</span>
      <span className="ref-chip__name">{ref_.name}</span>
      <button
        type="button"
        className="ref-chip__remove"
        onClick={onRemove}
        disabled={disabled}
        aria-label={`Remove ${token}`}
      >
        <svg viewBox="0 0 16 16" fill="currentColor">
          <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.75.75 0 1 1 1.06 1.06L9.06 8l3.22 3.22a.75.75 0 1 1-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 0 1-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06z" />
        </svg>
      </button>

      {showPopover && (
        <div className="ref-chip-popover">
          <p className="ref-chip-popover__title">{ref_.name}</p>
          <p className="ref-chip-popover__meta">
            Use <code>{token}</code> in your prompt
            {ref_.audioName && (
              <>
                <br />
                Audio: {ref_.audioName}
              </>
            )}
          </p>
        </div>
      )}
    </div>
  );
}

interface ReferenceChipsProps {
  refs: ReferenceItem[];
  onChange: (refs: ReferenceItem[]) => void;
  disabled?: boolean;
}

export function ReferenceChips({ refs, onChange, disabled }: ReferenceChipsProps) {
  if (refs.length === 0) return null;

  function handleRemove(index: number) {
    onChange(refs.filter((_, i) => i !== index));
  }

  function handleMoveUp(index: number) {
    if (index === 0) return;
    const next = [...refs];
    [next[index - 1], next[index]] = [next[index], next[index - 1]];
    onChange(next);
  }

  function handleMoveDown(index: number) {
    if (index === refs.length - 1) return;
    const next = [...refs];
    [next[index], next[index + 1]] = [next[index + 1], next[index]];
    onChange(next);
  }

  return (
    <div className="chips-container chips-container--above">
      {refs.map((ref, i) => (
        <ReferenceChip
          key={ref.id}
          ref_={ref}
          token={getTokenForRef(refs, i)}
          onRemove={() => handleRemove(i)}
          onMoveUp={i > 0 ? () => handleMoveUp(i) : undefined}
          onMoveDown={i < refs.length - 1 ? () => handleMoveDown(i) : undefined}
          disabled={disabled}
        />
      ))}
    </div>
  );
}

interface AddReferenceButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export function AddReferenceButton({ label, onClick, disabled }: AddReferenceButtonProps) {
  return (
    <button
      type="button"
      className="ref-chip-add"
      onClick={onClick}
      disabled={disabled}
    >
      <svg className="ref-chip-add__icon" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 2a.75.75 0 0 1 .75.75v4.5h4.5a.75.75 0 0 1 0 1.5h-4.5v4.5a.75.75 0 0 1-1.5 0v-4.5h-4.5a.75.75 0 0 1 0-1.5h4.5v-4.5A.75.75 0 0 1 8 2z" />
      </svg>
      {label}
    </button>
  );
}
