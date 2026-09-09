import { useEffect, useRef, useState } from "react";

export type MediaPickItem = {
  id: string;
  label: string;
  thumbUrl?: string;
  videoUrl?: string;
};

type Props = {
  label: string;
  items: MediaPickItem[];
  disabled?: boolean;
  emptyHint?: string;
  variant?: "chip" | "rail";
  onPick: (id: string) => void;
};

export function MediaPickerGrid({
  items,
  emptyHint,
  onPick,
}: {
  items: MediaPickItem[];
  emptyHint?: string;
  onPick: (id: string) => void;
}) {
  if (items.length === 0) {
    return <p className="media-picker__empty">{emptyHint ?? "Nothing in the library yet."}</p>;
  }
  return (
    <div className="media-picker__grid">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          className="media-picker__item"
          title={item.label}
          onClick={() => onPick(item.id)}
        >
          {item.videoUrl ? (
            <video
              className="media-picker__thumb"
              src={item.videoUrl}
              muted
              playsInline
              preload="metadata"
            />
          ) : item.thumbUrl ? (
            <img className="media-picker__thumb" src={item.thumbUrl} alt="" />
          ) : (
            <span className="media-picker__thumb media-picker__thumb--empty" />
          )}
          <span className="media-picker__name">{item.label}</span>
        </button>
      ))}
    </div>
  );
}

export function MediaPicker({ label, items, disabled, emptyHint, variant = "chip", onPick }: Props) {
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

  return (
    <div className="popover-anchor" ref={rootRef}>
      <button
        type="button"
        className={variant === "rail" ? `rail-btn${open ? " is-on" : ""}` : `chip-btn${open ? " is-on" : ""}`}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      >
        {variant === "rail" && (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <path d="M8 12h8M12 8v8" />
          </svg>
        )}
        {label}
      </button>
      {open && (
        <div className="media-picker">
          <MediaPickerGrid
            items={items}
            emptyHint={emptyHint}
            onPick={(id) => {
              onPick(id);
              setOpen(false);
            }}
          />
        </div>
      )}
    </div>
  );
}
