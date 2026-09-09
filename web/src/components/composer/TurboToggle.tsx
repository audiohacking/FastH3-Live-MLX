import { TURBO_CONFIG, type TurboTier } from "../../config";

interface TurboToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  tier: TurboTier;
  onTierChange: (tier: TurboTier) => void;
  disabled?: boolean;
  loading?: boolean;
}

export function TurboToggle({
  enabled,
  onChange,
  tier,
  onTierChange,
  disabled,
  loading,
}: TurboToggleProps) {
  const tiers = Object.entries(TURBO_CONFIG.TIERS) as [TurboTier, typeof TURBO_CONFIG.TIERS[TurboTier]][];

  return (
    <div className="turbo-group">
      <button
        type="button"
        className={`chip-btn chip-btn--turbo${enabled ? " is-on" : ""}${loading ? " is-busy" : ""}`}
        disabled={disabled || loading}
        onClick={() => onChange(!enabled)}
        title={TURBO_CONFIG.GUIDANCE}
        aria-pressed={enabled}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
        </svg>
        {loading ? "Loading…" : enabled ? "turbo" : "turbo off"}
      </button>
      {enabled && !loading && (
        <div className="seg" role="radiogroup" aria-label="Turbo quality">
          {tiers.map(([key, config]) => (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={tier === key}
              className={`seg__btn${tier === key ? " is-on" : ""}`}
              disabled={disabled}
              title={config.description}
              onClick={() => onTierChange(key)}
            >
              {config.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
