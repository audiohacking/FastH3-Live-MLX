import type { PillOption } from "../../types";

interface PillSelectProps<T extends string = string> {
  label: string;
  options: PillOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  compact?: boolean;
}

export function PillSelect<T extends string = string>({
  label,
  options,
  value,
  onChange,
  disabled,
  compact,
}: PillSelectProps<T>) {
  return (
    <div className={`pill-select${compact ? " pill-select--compact" : ""}`}>
      <span className="pill-select__label">{label}</span>
      <div className="pill-select__options" role="radiogroup" aria-label={label}>
        {options.map((opt, idx) => (
          <button
            key={opt.id}
            type="button"
            role="radio"
            aria-checked={value === opt.id}
            className={`pill-option${value === opt.id ? " pill-option--active" : ""}${idx > 0 ? " pill-option--has-divider" : ""}`}
            disabled={disabled || opt.disabled}
            title={opt.description}
            onClick={() => onChange(opt.id)}
          >
            {compact && opt.shortLabel ? opt.shortLabel : opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface PillGroupProps {
  children: React.ReactNode;
  label?: string;
  className?: string;
}

export function PillGroup({ children, label, className }: PillGroupProps) {
  return (
    <div className={`pill-group${className ? ` ${className}` : ""}`}>
      {label && <span className="pill-group__label">{label}</span>}
      <div className="pill-group__controls">{children}</div>
    </div>
  );
}

interface PillDividerProps {
  vertical?: boolean;
}

export function PillDivider({ vertical }: PillDividerProps) {
  return <div className={`pill-divider${vertical ? " pill-divider--vertical" : ""}`} />;
}

interface PillRowProps {
  children: React.ReactNode;
  className?: string;
}

export function PillRow({ children, className }: PillRowProps) {
  return <div className={`pill-row${className ? ` ${className}` : ""}`}>{children}</div>;
}

interface NumberPillProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  title?: string;
}

export function NumberPill({
  label,
  value,
  min = 1,
  max = 100,
  step = 1,
  onChange,
  disabled,
  title,
}: NumberPillProps) {
  return (
    <label className="pill-number" title={title}>
      <span className="pill-number__label">{label}</span>
      <input
        type="number"
        className="pill-number__input"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

interface TextPillProps {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  title?: string;
}

export function TextPill({
  label,
  value,
  placeholder,
  onChange,
  disabled,
  title,
}: TextPillProps) {
  return (
    <label className="pill-text" title={title}>
      <span className="pill-text__label">{label}</span>
      <input
        type="text"
        className="pill-text__input"
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}
