import { TurboToggle } from "./TurboToggle";
import type { PillOption } from "../../types";
import type { TurboTier } from "../../config";

type Props = {
  disabled?: boolean;
  seed: string;
  onSeed: (v: string) => void;
  numSteps: number;
  onNumSteps: (v: number) => void;
  layers: number;
  onLayers: (v: number) => void;
  reuse: number;
  onReuse: (v: number) => void;
  reuseLocked: boolean;
  quality: string;
  qualityOptions: PillOption[];
  onQuality: (id: string) => void;
  turboEnabled: boolean;
  turboTier: TurboTier;
  turboLoading?: boolean;
  loraBusy?: boolean;
  onTurbo: (enabled: boolean) => void;
  onTurboTier: (tier: TurboTier) => void;
  tokenReduction: boolean;
  tokenReductionLocked: boolean;
  onTokenReduction: (v: boolean) => void;
  ssdStreaming: boolean;
  ssdLocked: boolean;
  onSsdStreaming: (v: boolean) => void;
  clipMultiplier: number;
  clipMultiplierMax: number;
  showClips: boolean;
  onClipMultiplier: (n: number) => void;
};

function ParamStepper({
  value,
  min,
  max,
  unit,
  disabled,
  title,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  unit: string;
  disabled?: boolean;
  title?: string;
  onChange: (next: number) => void;
}) {
  return (
    <div className="stepper-pill" title={title}>
      <button type="button" disabled={disabled || value <= min} onClick={() => onChange(value - 1)}>
        −
      </button>
      <span className="stepper-pill__value">
        {value} {unit}
      </span>
      <button type="button" disabled={disabled || value >= max} onClick={() => onChange(value + 1)}>
        +
      </button>
    </div>
  );
}

export function SamplerRow({
  disabled,
  seed,
  onSeed,
  numSteps,
  onNumSteps,
  layers,
  onLayers,
  reuse,
  onReuse,
  reuseLocked,
  quality,
  qualityOptions,
  onQuality,
  turboEnabled,
  turboTier,
  turboLoading,
  loraBusy,
  onTurbo,
  onTurboTier,
  tokenReduction,
  tokenReductionLocked,
  onTokenReduction,
  ssdStreaming,
  ssdLocked,
  onSsdStreaming,
  clipMultiplier,
  clipMultiplierMax,
  showClips,
  onClipMultiplier,
}: Props) {
  const tokenOn = tokenReduction && !tokenReductionLocked;
  const ssdOn = ssdStreaming && !ssdLocked;

  return (
    <>
      <div className="config-card__row">
        <label className="seed-pill">
          <input
            value={seed}
            placeholder="random"
            disabled={disabled}
            onChange={(e) => onSeed(e.target.value)}
            aria-label="Seed"
          />
          <button
            type="button"
            disabled={disabled}
            title="Randomize seed"
            onClick={() => onSeed(String(Math.floor(Math.random() * 2 ** 31)))}
          >
            ↻
          </button>
        </label>
        <ParamStepper
          value={numSteps}
          min={1}
          max={50}
          unit="steps"
          disabled={disabled}
          title="h3.c --steps"
          onChange={(next) => {
            onNumSteps(next);
            if (next <= 7) onReuse(1);
          }}
        />
        <ParamStepper
          value={layers}
          min={1}
          max={50}
          unit="layers"
          disabled={disabled}
          title="h3.c --layers"
          onChange={onLayers}
        />
        <ParamStepper
          value={reuse}
          min={1}
          max={8}
          unit="reuse"
          disabled={disabled || reuseLocked}
          title="h3.c --reuse"
          onChange={onReuse}
        />
        <div className="config-card__tail">
          <TurboToggle
            enabled={turboEnabled}
            onChange={(enabled) => void onTurbo(enabled)}
            tier={turboTier}
            onTierChange={onTurboTier}
            disabled={disabled || loraBusy}
            loading={turboLoading}
          />
        </div>
      </div>
      <div className="config-card__row">
        <div className="seg" role="radiogroup" aria-label="Quality">
          {qualityOptions.map((opt) => (
            <button
              key={opt.id}
              type="button"
              role="radio"
              aria-checked={quality === opt.id}
              className={`seg__btn${quality === opt.id ? " is-on" : ""}`}
              disabled={disabled || opt.disabled}
              title={opt.description}
              onClick={() => onQuality(opt.id)}
            >
              {opt.shortLabel ?? opt.label}
            </button>
          ))}
        </div>
        {showClips && (
          <ParamStepper
            value={clipMultiplier}
            min={1}
            max={clipMultiplierMax}
            unit="clips"
            disabled={disabled}
            title="Chain last frame → first frame"
            onChange={onClipMultiplier}
          />
        )}
        <div className="config-card__tail">
          <button
            type="button"
            className={`chip-btn${tokenOn ? " is-on" : ""}`}
            disabled={disabled || tokenReductionLocked}
            onClick={() => onTokenReduction(!tokenReduction)}
          >
            token {tokenOn ? "on" : "off"}
          </button>
          <button
            type="button"
            className={`chip-btn${ssdOn ? " is-on" : ""}`}
            disabled={disabled || ssdLocked}
            onClick={() => onSsdStreaming(!ssdStreaming)}
          >
            ssd {ssdOn ? "on" : "off"}
          </button>
        </div>
      </div>
    </>
  );
}
