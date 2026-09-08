import type { LoraPreset } from "../../types";

interface LoraCardProps {
  preset: LoraPreset;
  selected: boolean;
  onToggle: (selected: boolean) => void;
  onScaleChange?: (scale: number) => void;
  onRemove?: () => void;
  disabled?: boolean;
  compact?: boolean;
}

export function LoraCard({
  preset,
  selected,
  onToggle,
  onScaleChange,
  onRemove,
  disabled,
  compact,
}: LoraCardProps) {
  const displayLabel = preset.label.replace(/\s*\(default\)\s*$/i, "").trim();

  return (
    <div
      className={`lora-card${selected ? " lora-card--selected" : ""}${compact ? " lora-card--compact" : ""}`}
      role="option"
      aria-selected={selected}
    >
      <button
        type="button"
        className="lora-card__toggle"
        disabled={disabled}
        onClick={() => onToggle(!selected)}
        aria-label={`${selected ? "Deselect" : "Select"} ${displayLabel}`}
      >
        <span className="lora-card__checkbox">
          {selected && (
            <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden>
              <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 1 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0z" />
            </svg>
          )}
        </span>
        <div className="lora-card__content">
          <span className="lora-card__label" title={displayLabel}>
            {displayLabel}
          </span>
          {preset.guidance && (
            <span className="lora-card__guidance" title={preset.guidance}>
              {preset.guidance}
            </span>
          )}
          {preset.steps && (
            <span className="lora-card__meta">
              {preset.steps} steps
              {preset.layers ? `, ${preset.layers} layers` : ""}
            </span>
          )}
        </div>
      </button>

      <div className="lora-card__actions">
        {selected && onScaleChange && (
          <label className="lora-card__scale">
            <span className="lora-card__scale-label">Scale</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.05}
              value={preset.scale}
              disabled={disabled}
              onChange={(e) => onScaleChange(Number(e.target.value))}
              className="lora-card__scale-input"
            />
          </label>
        )}
        {preset.custom && onRemove && (
          <button
            type="button"
            className="lora-card__remove"
            disabled={disabled}
            onClick={onRemove}
            title="Remove custom LoRA"
            aria-label={`Remove ${displayLabel}`}
          >
            <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden>
              <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.75.75 0 1 1 1.06 1.06L9.06 8l3.22 3.22a.75.75 0 1 1-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 0 1-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06z" />
            </svg>
          </button>
        )}
      </div>

      {preset.cached && (
        <span className="lora-card__badge lora-card__badge--cached" title="Downloaded">
          ✓
        </span>
      )}
    </div>
  );
}

interface LoraStackProps {
  presets: LoraPreset[];
  selectedIds: string[];
  onToggle: (id: string, selected: boolean) => void;
  onScaleChange?: (id: string, scale: number) => void;
  onRemove?: (preset: LoraPreset) => void;
  disabled?: boolean;
}

export function LoraStack({
  presets,
  selectedIds,
  onToggle,
  onScaleChange,
  onRemove,
  disabled,
}: LoraStackProps) {
  const selected = presets.filter((p) => selectedIds.includes(p.id));

  if (selected.length === 0) {
    return (
      <div className="lora-stack lora-stack--empty">
        <span className="lora-stack__empty-text">No LoRAs selected</span>
      </div>
    );
  }

  return (
    <div className="lora-stack">
      {selected.map((preset) => (
        <LoraCard
          key={preset.id}
          preset={preset}
          selected
          onToggle={(sel) => onToggle(preset.id, sel)}
          onScaleChange={onScaleChange ? (scale) => onScaleChange(preset.id, scale) : undefined}
          onRemove={onRemove ? () => onRemove(preset) : undefined}
          disabled={disabled}
          compact
        />
      ))}
    </div>
  );
}
