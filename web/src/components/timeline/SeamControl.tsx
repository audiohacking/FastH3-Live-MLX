import { TIMELINE_CONFIG } from "../../config";

export type SeamType = "cut" | "blend";

interface SeamControlProps {
  beforeClipId: string;
  afterClipId: string;
  seamType: SeamType;
  blendDuration: number;
  onSeamTypeChange: (type: SeamType) => void;
  onBlendDurationChange: (duration: number) => void;
  disabled?: boolean;
}

export function SeamControl({
  seamType,
  blendDuration,
  onSeamTypeChange,
  onBlendDurationChange,
  disabled,
}: SeamControlProps) {
  return (
    <div className="seam-control">
      <button
        type="button"
        className="seam-control__toggle"
        onClick={() => onSeamTypeChange(seamType === "cut" ? "blend" : "cut")}
        disabled={disabled}
        title={seamType === "cut" ? "Click to enable blend transition" : "Click to use hard cut"}
      >
        {seamType === "cut" ? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 4v16" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 12c4-4 8-4 8 0s4 4 8 0" />
          </svg>
        )}
      </button>

      {seamType === "blend" && (
        <select
          className="seam-control__duration"
          value={blendDuration}
          onChange={(e) => onBlendDurationChange(Number(e.target.value))}
          disabled={disabled}
          title="Blend duration"
        >
          {TIMELINE_CONFIG.SEAM_DURATIONS.filter((d) => d > 0).map((d) => (
            <option key={d} value={d}>
              {d}s
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

interface SeamIndicatorProps {
  seamType: SeamType;
  compact?: boolean;
}

export function SeamIndicator({ seamType, compact }: SeamIndicatorProps) {
  return (
    <div className={`seam-indicator seam-indicator--${seamType}${compact ? " seam-indicator--compact" : ""}`}>
      {seamType === "cut" ? (
        <div className="seam-indicator__line" />
      ) : (
        <div className="seam-indicator__blend" />
      )}
    </div>
  );
}
