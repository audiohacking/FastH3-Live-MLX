import { useEffect, useMemo, useRef, useState } from "react";
import type { LoraPreset } from "../../types";
import { LoraCard, type LoraDownloadProgress } from "./LoraCard";

const CATEGORIES = ["All", "Speed", "Style", "Motion", "Immersive", "Custom"] as const;

interface LoraModalProps {
  open: boolean;
  onClose: () => void;
  presets: LoraPreset[];
  selectedIds: string[];
  onToggle: (id: string, selected: boolean) => void;
  onRemove: (preset: LoraPreset) => void;
  onAddCustom?: (spec: string, label: string, scale: number) => Promise<void>;
  onPresetsChange?: (presets: LoraPreset[]) => void;
  addingCustom?: boolean;
  disabled?: boolean;
  api?: string;
}

export function LoraModal({
  open,
  onClose,
  presets,
  selectedIds,
  onToggle,
  onRemove,
  onAddCustom,
  onPresetsChange,
  addingCustom,
  disabled,
  api = "",
}: LoraModalProps) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]>("All");
  const [customSpec, setCustomSpec] = useState("");
  const [customLabel, setCustomLabel] = useState("");
  const [customScale, setCustomScale] = useState("0.8");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [progress, setProgress] = useState<LoraDownloadProgress | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const selectAfterRef = useRef<string | null>(null);

  useEffect(() => {
    if (open) {
      setTimeout(() => searchRef.current?.focus(), 50);
    } else {
      setSearch("");
      setCategory("All");
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  function closeStream() {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }

  function openProgressStream(loraId: string) {
    closeStream();
    setBusyId(loraId);
    setDownloadError(null);
    const es = new EventSource(`${api}/api/loras/download/stream?lora_id=${encodeURIComponent(loraId)}`);
    eventSourceRef.current = es;

    es.addEventListener("progress", (e) => {
      try {
        const data = JSON.parse((e as MessageEvent).data) as LoraDownloadProgress;
        setProgress(data);
      } catch {
        // ignore
      }
    });

    es.addEventListener("complete", (e) => {
      try {
        const data = JSON.parse((e as MessageEvent).data) as { lora_presets?: LoraPreset[] };
        if (data.lora_presets) onPresetsChange?.(data.lora_presets);
      } catch {
        // ignore
      }
      const selectId = selectAfterRef.current;
      selectAfterRef.current = null;
      closeStream();
      setProgress(null);
      setBusyId(null);
      if (selectId) onToggle(selectId, true);
    });

    es.addEventListener("error", (e) => {
      if (e instanceof MessageEvent && e.data) {
        try {
          const data = JSON.parse(e.data) as { error?: string };
          setDownloadError(data.error || "Download failed");
        } catch {
          setDownloadError("Download failed");
        }
        selectAfterRef.current = null;
        closeStream();
        setProgress(null);
        setBusyId(null);
      }
    });
  }

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void (async () => {
      try {
        const r = await fetch(`${api}/api/loras/download/status`);
        if (!r.ok || cancelled) return;
        const data = (await r.json()) as {
          active?: boolean;
          lora_id?: string | null;
          progress?: { percent: number; downloaded_bytes: number; expected_bytes: number };
        };
        if (data.active && data.lora_id && !cancelled) {
          if (data.progress) {
            setProgress({
              percent: data.progress.percent,
              downloaded_gb: data.progress.downloaded_bytes / 1024 ** 3,
              expected_gb: data.progress.expected_bytes / 1024 ** 3,
              speed: "resuming…",
            });
          }
          openProgressStream(data.lora_id);
        }
      } catch {
        // stay usable
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, api]);

  useEffect(() => () => closeStream(), []);

  const filteredPresets = useMemo(() => {
    const q = search.trim().toLowerCase();
    return presets.filter((p) => {
      if (category !== "All") {
        const cat = p.custom ? "Custom" : p.category || "Style";
        if (cat !== category) return false;
      }
      if (!q) return true;
      return (
        p.label.toLowerCase().includes(q) ||
        (p.spec || "").toLowerCase().includes(q) ||
        (p.guidance || "").toLowerCase().includes(q) ||
        (p.trigger || "").toLowerCase().includes(q) ||
        (p.category || "").toLowerCase().includes(q)
      );
    });
  }, [presets, search, category]);

  const groupedPresets = useMemo(() => {
    const selected = filteredPresets.filter((p) => selectedIds.includes(p.id));
    const custom = filteredPresets.filter((p) => p.custom && !selectedIds.includes(p.id));
    const speed = filteredPresets.filter((p) => !p.custom && !selectedIds.includes(p.id) && p.category === "Speed");
    const style = filteredPresets.filter((p) => !p.custom && !selectedIds.includes(p.id) && (p.category === "Style" || !p.category));
    const motion = filteredPresets.filter((p) => !p.custom && !selectedIds.includes(p.id) && p.category === "Motion");
    const immersive = filteredPresets.filter((p) => !p.custom && !selectedIds.includes(p.id) && p.category === "Immersive");
    return { selected, custom, speed, style, motion, immersive };
  }, [filteredPresets, selectedIds]);

  function handleToggle(preset: LoraPreset, selected: boolean) {
    if (preset.compatible === false) return;
    if (selected && !preset.cached && preset.spec) {
      selectAfterRef.current = preset.id;
      openProgressStream(preset.id);
      return;
    }
    onToggle(preset.id, selected);
  }

  function renderGrid(title: string, rows: LoraPreset[]) {
    if (rows.length === 0) return null;
    return (
      <section className="lora-modal__section">
        <h3 className="lora-modal__section-title">{title}</h3>
        <div className="lora-modal__grid">
          {rows.map((preset) => (
            <LoraCard
              key={preset.id}
              preset={preset}
              selected={selectedIds.includes(preset.id)}
              onToggle={(sel) => handleToggle(preset, sel)}
              onRemove={preset.custom ? () => onRemove(preset) : undefined}
              onDownload={preset.compatible !== false && preset.spec ? () => openProgressStream(preset.id) : undefined}
              downloading={busyId === preset.id}
              progress={busyId === preset.id ? progress : null}
              disabled={disabled || (busyId !== null && busyId !== preset.id)}
            />
          ))}
        </div>
      </section>
    );
  }

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
          <select
            className="lora-modal__category"
            value={category}
            onChange={(e) => setCategory(e.target.value as (typeof CATEGORIES)[number])}
            aria-label="LoRA category"
          >
            {CATEGORIES.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </div>

        {downloadError && <div className="error-banner">{downloadError}</div>}

        <div className="modal__body">
          {renderGrid(`Selected (${groupedPresets.selected.length})`, groupedPresets.selected)}
          {renderGrid("Custom", groupedPresets.custom)}
          {renderGrid("Speed", groupedPresets.speed)}
          {renderGrid("Style", groupedPresets.style)}
          {renderGrid("Motion", groupedPresets.motion)}
          {renderGrid("Immersive", groupedPresets.immersive)}

          {filteredPresets.length === 0 && (
            <div className="lora-modal__empty">
              {search ? `No LoRAs matching "${search}"` : "No LoRAs available"}
            </div>
          )}
        </div>

        <footer className="modal__footer" style={{ flexWrap: "wrap", gap: 8 }}>
          {onAddCustom && (
            <form
              className="lora-row-add"
              onSubmit={(e) => {
                e.preventDefault();
                if (!customSpec.trim() || addingCustom) return;
                void onAddCustom(customSpec.trim(), customLabel.trim(), Number(customScale) || 0.8).then(() => {
                  setCustomSpec("");
                  setCustomLabel("");
                });
              }}
            >
              <input
                type="text"
                className="lora-add-url"
                placeholder="HF URL or path"
                value={customSpec}
                disabled={addingCustom || disabled}
                onChange={(e) => setCustomSpec(e.target.value)}
              />
              <input
                type="text"
                className="lora-add-name"
                placeholder="Label"
                value={customLabel}
                disabled={addingCustom || disabled}
                onChange={(e) => setCustomLabel(e.target.value)}
              />
              <input
                type="number"
                className="lora-add-scale"
                min={0}
                max={2}
                step={0.05}
                value={customScale}
                disabled={addingCustom || disabled}
                onChange={(e) => setCustomScale(e.target.value)}
              />
              <button type="submit" className="btn-secondary btn-compact" disabled={!customSpec.trim() || addingCustom}>
                {addingCustom ? "…" : "Add"}
              </button>
            </form>
          )}
          <span className="lora-modal__count">
            {selectedIds.length} selected
            {busyId ? " · download continues if you close" : ""}
          </span>
          <button type="button" className="btn-secondary" onClick={onClose}>
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}
