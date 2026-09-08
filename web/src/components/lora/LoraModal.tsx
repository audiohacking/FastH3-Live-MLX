import { useEffect, useMemo, useRef, useState } from "react";
import type { LoraPreset } from "../../types";
import { LoraCard } from "./LoraCard";

interface LoraModalProps {
  open: boolean;
  onClose: () => void;
  presets: LoraPreset[];
  selectedIds: string[];
  onToggle: (id: string, selected: boolean) => void;
  onRemove: (preset: LoraPreset) => void;
  disabled?: boolean;
}

export function LoraModal({
  open,
  onClose,
  presets,
  selectedIds,
  onToggle,
  onRemove,
  disabled,
}: LoraModalProps) {
  const [search, setSearch] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);

  // Focus search input when modal opens
  useEffect(() => {
    if (open) {
      setTimeout(() => searchRef.current?.focus(), 50);
    } else {
      setSearch("");
    }
  }, [open]);

  // Close on escape key
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Filter presets by search
  const filteredPresets = useMemo(() => {
    if (!search.trim()) return presets;
    const q = search.toLowerCase();
    return presets.filter(
      (p) =>
        p.label.toLowerCase().includes(q) ||
        p.spec.toLowerCase().includes(q) ||
        p.guidance?.toLowerCase().includes(q),
    );
  }, [presets, search]);

  // Group presets: selected first, then custom, then built-in
  const groupedPresets = useMemo(() => {
    const selected = filteredPresets.filter((p) => selectedIds.includes(p.id));
    const custom = filteredPresets.filter((p) => p.custom && !selectedIds.includes(p.id));
    const builtin = filteredPresets.filter((p) => !p.custom && !selectedIds.includes(p.id));
    return { selected, custom, builtin };
  }, [filteredPresets, selectedIds]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal modal--fullscreen" onClick={(e) => e.stopPropagation()}>
        <header className="modal__header">
          <h2 className="modal__title">LoRA Library</h2>
          <button
            type="button"
            className="modal__close"
            onClick={onClose}
            aria-label="Close modal"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </header>

        <div className="modal__search">
          <input
            ref={searchRef}
            type="text"
            className="modal__search-input"
            placeholder="Search LoRAs…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="modal__body">
          {groupedPresets.selected.length > 0 && (
            <section className="lora-modal__section">
              <h3 className="lora-modal__section-title">Selected ({groupedPresets.selected.length})</h3>
              <div className="lora-modal__grid">
                {groupedPresets.selected.map((preset) => (
                  <LoraCard
                    key={preset.id}
                    preset={preset}
                    selected
                    onToggle={(sel) => onToggle(preset.id, sel)}
                    onRemove={preset.custom ? () => onRemove(preset) : undefined}
                    disabled={disabled}
                  />
                ))}
              </div>
            </section>
          )}

          {groupedPresets.custom.length > 0 && (
            <section className="lora-modal__section">
              <h3 className="lora-modal__section-title">Custom</h3>
              <div className="lora-modal__grid">
                {groupedPresets.custom.map((preset) => (
                  <LoraCard
                    key={preset.id}
                    preset={preset}
                    selected={false}
                    onToggle={(sel) => onToggle(preset.id, sel)}
                    onRemove={() => onRemove(preset)}
                    disabled={disabled}
                  />
                ))}
              </div>
            </section>
          )}

          {groupedPresets.builtin.length > 0 && (
            <section className="lora-modal__section">
              <h3 className="lora-modal__section-title">Built-in</h3>
              <div className="lora-modal__grid">
                {groupedPresets.builtin.map((preset) => (
                  <LoraCard
                    key={preset.id}
                    preset={preset}
                    selected={false}
                    onToggle={(sel) => onToggle(preset.id, sel)}
                    disabled={disabled}
                  />
                ))}
              </div>
            </section>
          )}

          {filteredPresets.length === 0 && (
            <div className="lora-modal__empty">
              {search ? `No LoRAs matching "${search}"` : "No LoRAs available"}
            </div>
          )}
        </div>

        <footer className="modal__footer">
          <span className="lora-modal__count">
            {selectedIds.length} selected
          </span>
          <button
            type="button"
            className="btn-secondary"
            onClick={onClose}
          >
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}
