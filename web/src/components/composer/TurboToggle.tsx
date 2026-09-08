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
    <div className="turbo-control">
      <button
        type="button"
        className={`turbo-toggle${enabled ? " turbo-toggle--active" : ""}${loading ? " turbo-toggle--loading" : ""}`}
        disabled={disabled || loading}
        onClick={() => onChange(!enabled)}
        title={TURBO_CONFIG.GUIDANCE}
        aria-pressed={enabled}
      >
        <span className="turbo-toggle__icon" aria-hidden>
          {loading ? (
            <span className="turbo-toggle__spinner" />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
            </svg>
          )}
        </span>
        <span className="turbo-toggle__label">
          {loading ? "Loading…" : enabled ? "Turbo ON" : "Turbo"}
        </span>
      </button>

      {enabled && !loading && (
        <div className="turbo-tiers">
          {tiers.map(([key, config]) => (
            <button
              key={key}
              type="button"
              className={`turbo-tier${tier === key ? " turbo-tier--active" : ""}`}
              onClick={() => onTierChange(key)}
              disabled={disabled}
              title={config.description}
            >
              {config.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface TurboInfoProps {
  visible: boolean;
  tier: TurboTier;
}

export function TurboInfo({ visible, tier }: TurboInfoProps) {
  if (!visible) return null;

  const config = TURBO_CONFIG.TIERS[tier];

  return (
    <div className="turbo-info">
      <span className="turbo-info__icon" aria-hidden>⚡</span>
      <span className="turbo-info__text">
        {config.label} mode: {config.steps} steps. {config.description}
      </span>
    </div>
  );
}
