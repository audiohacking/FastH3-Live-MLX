import { handleForRef, kindHandlePrefix } from "../../compile";
import type { ReferenceItem, RefSize } from "../../types";

type Props = {
  refs: ReferenceItem[];
  disabled?: boolean;
  onChange: (refs: ReferenceItem[]) => void;
};

export function AssetCapsules({ refs, disabled, onChange }: Props) {
  if (refs.length === 0) return null;

  function patch(index: number, next: Partial<ReferenceItem>) {
    onChange(refs.map((r, i) => (i === index ? { ...r, ...next } : r)));
  }

  return (
    <div className="asset-row">
      {refs.map((ref, i) => {
        const handle = handleForRef(refs, i);
        const muted = ref.enabled === false;
        const size: RefSize = ref.refSize ?? "max";
        return (
          <div key={ref.id} className={`asset-capsule${muted ? " asset-capsule--muted" : ""}`}>
            {ref.previewUrl && kindHandlePrefix(ref.kind) === "img" ? (
              <img className="asset-capsule__thumb" src={ref.previewUrl} alt="" />
            ) : (
              <span className="asset-capsule__icon" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  {ref.kind === "audio" ? (
                    <path d="M9 18V6l12-2v12M6 18a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm12-2a2 2 0 1 0 0-4 2 2 0 0 0 0 4z" />
                  ) : (
                    <rect x="3" y="6" width="18" height="12" rx="2" />
                  )}
                </svg>
              </span>
            )}
            <span className="asset-capsule__handle">{handle}</span>
            <span className="asset-capsule__name" title={ref.name}>{ref.name}</span>
            {ref.kind === "image" && (
              <button
                type="button"
                className={`asset-capsule__action${size === "max" ? " asset-capsule__action--active" : ""}`}
                disabled={disabled}
                title={size === "max" ? "Encode at original size" : "Match output canvas"}
                onClick={() => patch(i, { refSize: size === "max" ? "match" : "max" })}
              >
                {size}
              </button>
            )}
            <button
              type="button"
              className="asset-capsule__action"
              disabled={disabled}
              title={muted ? "Enable" : "Mute (skip on generate)"}
              onClick={() => patch(i, { enabled: muted })}
            >
              {muted ? "on" : "mute"}
            </button>
            <button
              type="button"
              className="asset-capsule__action"
              disabled={disabled}
              aria-label={`Remove ${handle}`}
              onClick={() => onChange(refs.filter((_, j) => j !== i))}
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}
