import { useState } from "react";
import type { CastMember } from "../../types";

interface CastPickerProps {
  members: CastMember[];
  selectedIds: string[];
  onToggle: (id: string) => void;
  onCreate: (name: string, description?: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  disabled?: boolean;
}

/**
 * Named-character cast browser. Lets the user pick cast members whose
 * attached reference media should be injected into the generation, and
 * create/delete cast members. Wired to the persistent /api/cast endpoints.
 */
export function CastPicker({
  members,
  selectedIds,
  onToggle,
  onCreate,
  onDelete,
  disabled,
}: CastPickerProps) {
  const [expanded, setExpanded] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");

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
      <button type="button" className="cast-picker__toggle" onClick={() => setExpanded((v) => !v)} disabled={disabled}>
        Cast
        {selectedIds.length > 0 && <span className="cast-picker__count">{selectedIds.length}</span>}
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`cast-picker__icon ${expanded ? "cast-picker__icon--open" : ""}`}>
          <path d="M19 9l-7 7-7-7" />
        </svg>
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
                        <span className="cast-row__meta">{m.media.length} media item{m.media.length === 1 ? "" : "s"}</span>
                      )}
                    </div>
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
    </div>
  );
}
