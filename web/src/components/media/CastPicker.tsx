import { useRef, useState } from "react";
import type { CastMediaType, CastMember } from "../../types";

interface CastPickerProps {
  members: CastMember[];
  selectedIds: string[];
  onToggle: (id: string) => void;
  onCreate: (name: string, description?: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onAttachMedia?: (id: string, file: File, type: CastMediaType) => Promise<void>;
  onRemoveMedia?: (id: string, mediaId: string) => Promise<void>;
  disabled?: boolean;
}

export function CastPicker({
  members,
  selectedIds,
  onToggle,
  onCreate,
  onDelete,
  onAttachMedia,
  onRemoveMedia,
  disabled,
}: CastPickerProps) {
  const [expanded, setExpanded] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [attachFor, setAttachFor] = useState<{ id: string; type: CastMediaType } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleCreate() {
    if (!name.trim()) return;
    await onCreate(name.trim(), desc.trim() || undefined);
    setName("");
    setDesc("");
    setShowForm(false);
  }

  async function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    if (confirm("Delete this cast member?")) {
      await onDelete(id);
    }
  }

  return (
    <div className="cast-picker">
      <button type="button" className="cast-picker__toggle rail-btn" onClick={() => setExpanded((v) => !v)} disabled={disabled}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <circle cx="12" cy="8" r="3" />
          <path d="M5 19c0-3.3 3.1-6 7-6s7 2.7 7 6" />
        </svg>
        Cast
        {selectedIds.length > 0 && <span className="cast-picker__count">{selectedIds.length}</span>}
      </button>

      {expanded && (
        <div className="cast-picker__panel">
          {members.length === 0 && !showForm && (
            <p className="cast-picker__empty">No cast members yet. Add one to reuse a character across shots.</p>
          )}

          {members.length > 0 && (
            <div className="cast-picker__list">
              {members.map((m) => {
                const selected = selectedIds.includes(m.id);
                return (
                  <div
                    key={m.id}
                    className={`cast-row ${selected ? "cast-row--selected" : ""}`}
                    onClick={() => onToggle(m.id)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && onToggle(m.id)}
                  >
                    <div className="cast-row__main">
                      <span className="cast-row__name">{m.name}</span>
                      {m.description && <span className="cast-row__desc">{m.description}</span>}
                      {m.media.length > 0 && (
                        <span className="cast-row__meta">
                          {m.media.length} media · @{m.name.replace(/\s+/g, "")}
                        </span>
                      )}
                      {m.media.length > 0 && onRemoveMedia && (
                        <span className="cast-row__meta">
                          {m.media.map((media) => (
                            <button
                              key={media.id}
                              type="button"
                              className="cast-row__delete"
                              title={`Remove ${media.label || media.type}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                void onRemoveMedia(m.id, media.id);
                              }}
                            >
                              {media.type} ×
                            </button>
                          ))}
                        </span>
                      )}
                    </div>
                    {onAttachMedia && (
                      <div className="cast-row__attach" onClick={(e) => e.stopPropagation()}>
                        <button
                          type="button"
                          onClick={() => {
                            setAttachFor({ id: m.id, type: "image" });
                            fileRef.current?.click();
                          }}
                        >
                          + img
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setAttachFor({ id: m.id, type: "video" });
                            fileRef.current?.click();
                          }}
                        >
                          + vid
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setAttachFor({ id: m.id, type: "audio" });
                            fileRef.current?.click();
                          }}
                        >
                          + aud
                        </button>
                      </div>
                    )}
                    <button type="button" className="cast-row__delete" onClick={(e) => void handleDelete(e, m.id)} title="Delete">
                      &times;
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          {showForm ? (
            <div className="cast-form">
              <input
                type="text"
                className="cast-form__input"
                placeholder="Character name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
              />
              <input
                type="text"
                className="cast-form__input"
                placeholder="Description (optional)"
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
              />
              <div className="cast-form__actions">
                <button type="button" className="btn-primary" onClick={() => void handleCreate()} disabled={!name.trim()}>
                  Add
                </button>
                <button type="button" className="btn-ghost" onClick={() => { setShowForm(false); setName(""); setDesc(""); }}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button type="button" className="cast-picker__add" onClick={() => setShowForm(true)}>
              + Add Cast Member
            </button>
          )}
        </div>
      )}
      <input
        ref={fileRef}
        type="file"
        hidden
        accept={attachFor?.type === "video" ? "video/*" : attachFor?.type === "audio" ? "audio/*" : "image/*"}
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          const target = attachFor;
          setAttachFor(null);
          if (f && target && onAttachMedia) void onAttachMedia(target.id, f, target.type);
        }}
      />
    </div>
  );
}
