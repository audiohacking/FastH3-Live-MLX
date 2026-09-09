import { useState } from "react";
import type { GenerationPreset } from "../../types";
import type { StyleEntry } from "../../styleAtlas";
import { StyleAtlas } from "../media/StyleAtlas";

interface PresetManagerProps {
  presets: GenerationPreset[];
  onSave: (name: string, description?: string) => void;
  onLoad: (preset: GenerationPreset) => void;
  onDelete: (id: string) => void;
  disabled?: boolean;
  activeStyleId?: string | null;
  onApplyStyle: (style: StyleEntry) => void;
}

export function PresetManager({
  presets,
  onSave,
  onLoad,
  onDelete,
  disabled,
  activeStyleId,
  onApplyStyle,
}: PresetManagerProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [tab, setTab] = useState<"looks" | "saved">("looks");
  const [showSaveForm, setShowSaveForm] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [saveDescription, setSaveDescription] = useState("");

  function handleSave() {
    if (!saveName.trim()) return;
    onSave(saveName.trim(), saveDescription.trim() || undefined);
    setSaveName("");
    setSaveDescription("");
    setShowSaveForm(false);
  }

  function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    if (confirm("Delete this preset?")) {
      onDelete(id);
    }
  }

  return (
    <div className="preset-manager">
      <button
        type="button"
        className="preset-manager__toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        disabled={disabled}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`preset-manager__icon ${isExpanded ? "preset-manager__icon--open" : ""}`}
        >
          <path d="M19 9l-7 7-7-7" />
        </svg>
        <span>Presets</span>
        {presets.length > 0 && (
          <span className="preset-manager__count">{presets.length}</span>
        )}
      </button>

      {isExpanded && (
        <div className="preset-manager__panel">
          <div className="preset-manager__tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "looks"}
              className={`preset-manager__tab${tab === "looks" ? " is-on" : ""}`}
              onClick={() => setTab("looks")}
            >
              Looks
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "saved"}
              className={`preset-manager__tab${tab === "saved" ? " is-on" : ""}`}
              onClick={() => setTab("saved")}
            >
              Saved
            </button>
          </div>

          {tab === "looks" && (
            <StyleAtlas disabled={disabled} activeId={activeStyleId} onApply={onApplyStyle} />
          )}

          {tab === "saved" && presets.length === 0 && !showSaveForm && (
            <p className="preset-manager__empty">
              No saved presets. Save your current settings to reuse them later.
            </p>
          )}

          {tab === "saved" && presets.length > 0 && (
            <div className="preset-manager__list">
              {presets.map((preset) => (
                <div
                  key={preset.id}
                  className="preset-card"
                  onClick={() => onLoad(preset)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && onLoad(preset)}
                >
                  <div className="preset-card__content">
                    <span className="preset-card__name">{preset.name}</span>
                    {preset.description && (
                      <span className="preset-card__description">
                        {preset.description}
                      </span>
                    )}
                    <span className="preset-card__meta">
                      {preset.mode} · {preset.resolutionId} · {preset.numSteps} steps
                      {preset.turboEnabled && " · Turbo"}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="preset-card__delete"
                    onClick={(e) => handleDelete(e, preset.id)}
                    title="Delete preset"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}

          {tab === "saved" && (showSaveForm ? (
            <div className="preset-save-form">
              <input
                type="text"
                className="preset-save-form__input"
                placeholder="Preset name"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                autoFocus
              />
              <input
                type="text"
                className="preset-save-form__input"
                placeholder="Description (optional)"
                value={saveDescription}
                onChange={(e) => setSaveDescription(e.target.value)}
              />
              <div className="preset-save-form__actions">
                <button
                  type="button"
                  className="preset-save-form__btn preset-save-form__btn--save"
                  onClick={handleSave}
                  disabled={!saveName.trim()}
                >
                  Save
                </button>
                <button
                  type="button"
                  className="preset-save-form__btn preset-save-form__btn--cancel"
                  onClick={() => {
                    setShowSaveForm(false);
                    setSaveName("");
                    setSaveDescription("");
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              className="preset-manager__save-btn"
              onClick={() => setShowSaveForm(true)}
              disabled={disabled}
            >
              + Save Current Settings
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
