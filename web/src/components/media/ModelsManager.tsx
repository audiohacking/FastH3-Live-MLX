import { useEffect, useRef, useState } from "react";

interface ModelComponent {
  id: string;
  label: string;
  present: boolean;
  path: string;
  size_gib: number;
  note: string;
}

interface ModelsStatus {
  ok: boolean;
  model_dir: string;
  components: ModelComponent[];
}

interface DownloadProgress {
  percent: number;
  downloaded_gb: number;
  expected_gb: number;
  speed: string;
  active: boolean;
}

interface DownloadStatusResponse {
  active: boolean;
  component?: string | null;
  error?: string | null;
  progress?: Pick<DownloadProgress, "percent" | "downloaded_gb" | "expected_gb">;
}

interface ModelsManagerProps {
  api: string;
  onClose?: () => void;
  /** Called whenever a server-side download becomes active/inactive (App uses this to guard close). */
  onDownloadStateChange?: (active: boolean) => void;
}

/**
 * Dedicated Models page. Shows what is already on disk and lets the user
 * OPTIONALLY download a missing component. Downloads are never automatic:
 * the user must click "Download". Supports resume via huggingface_hub.
 *
 * The download task lives server-side and survives client disconnects, so the
 * modal is never hard-locked: closing it just detaches the SSE stream, and
 * progress re-attaches on reopen/reload.
 */
export function ModelsManager({ api, onClose, onDownloadStateChange }: ModelsManagerProps) {
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${api}/api/models`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setStatus(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  /** Open (or re-attach to) an SSE stream for the given component. */
  function openProgressStream(component: string) {
    eventSourceRef.current?.close();
    setBusyId(component);
    setError(null);

    const es = new EventSource(`${api}/api/models/download/stream?component=${component}`);
    eventSourceRef.current = es;

    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      es.close();
      if (eventSourceRef.current === es) eventSourceRef.current = null;
    };

    es.addEventListener("progress", (e) => {
      try {
        setProgress(JSON.parse(e.data));
      } catch {
        // ignore malformed payloads
      }
    });

    es.addEventListener("complete", (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.components) {
          setStatus((prev) => (prev ? { ...prev, components: data.components } : prev));
        }
      } catch {
        // ignore
      }
      finish();
      setProgress(null);
      setBusyId(null);
      onDownloadStateChange?.(false);
      void refresh();
    });

    es.addEventListener("error", (e) => {
      // ONLY the server-sent "error" event (a MessageEvent carrying JSON) is a
      // terminal failure. Transport-level errors arrive as ErrorEvent here but
      // have no `data`, so we leave those to EventSource to retry — we must NOT
      // close the stream on the first network blip, or progress dies mid-download.
      if (e instanceof MessageEvent && e.data) {
        try {
          const data = JSON.parse(e.data);
          setError(data.error || "Download failed");
        } catch {
          setError("Download failed");
        }
        finish();
        setProgress(null);
        setBusyId(null);
        onDownloadStateChange?.(false);
      }
    });

    // No es.onerror handler — EventSource auto-reconnects on transient drops.
    onDownloadStateChange?.(true);
  }

  // On mount: load status, then re-attach to any in-flight server download.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      await refresh();
      if (cancelled) return;
      try {
        const r = await fetch(`${api}/api/models/download/status`);
        if (!r.ok) return;
        const data = (await r.json()) as DownloadStatusResponse;
        if (data.active && data.component && !cancelled) {
          if (data.progress) {
            setProgress({ ...data.progress, speed: "resuming…", active: true });
          }
          openProgressStream(data.component);
        }
      } catch {
        // ignore — modal stays usable on transient status failures
      }
    })();
    return () => {
      cancelled = true;
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
      // The download lives server-side; releasing this flag lets App re-open the
      // modal at any time (the flag must never stick "true" on unmount).
      onDownloadStateChange?.(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  function startDownload(component: ModelComponent) {
    if (busyId !== null) return;
    if (!component.present && !window.confirm(
      `Download ${component.label}?\n\n` +
      `This can be large (${component.id === "fl2va" ? "~134" : "~62"} GB).\n` +
      `Downloads resume automatically if interrupted.\n\nProceed?`
    )) {
      return;
    }
    setError(null);
    setProgress({ percent: 0, downloaded_gb: 0, expected_gb: 0, speed: "starting...", active: true });
    openProgressStream(component.id);
  }

  /** Close is always enabled; a running download continues in the background. */
  function handleClose() {
    if (busyId !== null) {
      if (!window.confirm(
        "Download is still running. It will continue in the background.\n\nClose this panel?"
      )) {
        return;
      }
    }
    onClose?.();
  }

  return (
    <div className="models-page">
      <div className="models-page__head">
        <div>
          <h2 className="models-page__title">Models</h2>
          <p className="models-page__dir">
            {status ? status.model_dir : "Loading model directory…"}
          </p>
        </div>
        <div className="models-page__actions">
          <button type="button" className="btn-ghost" onClick={() => void refresh()} disabled={loading}>
            Refresh
          </button>
          {onClose && (
            <button type="button" className="btn-ghost" onClick={handleClose}>
              Close
            </button>
          )}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {loading && !status && <p className="models-page__hint">Checking local model layout…</p>}

      {status && (
        <div className="models-page__list">
          {status.components.map((c) => (
            <div key={c.id} className={`model-card ${c.present ? "model-card--present" : "model-card--missing"}`}>
              <div className="model-card__body">
                <div className="model-card__row">
                  <span className="model-card__label">{c.label}</span>
                  <span className={`model-card__badge ${c.present ? "model-card__badge--ok" : "model-card__badge--missing"}`}>
                    {c.present ? "Present" : "Missing"}
                  </span>
                </div>
                {c.size_gib > 0 && <p className="model-card__size">{c.size_gib.toFixed(1)} GB on disk</p>}
                <p className="model-card__note">{c.note}</p>
                <p className="model-card__path">{c.path}</p>

                {/* Download progress bar */}
                {busyId === c.id && progress && (
                  <div className="download-progress">
                    <div className="download-progress__bar-container">
                      <div
                        className="download-progress__bar"
                        style={{ width: `${progress.percent}%` }}
                      />
                    </div>
                    <div className="download-progress__text">
                      <span>
                        {progress.downloaded_gb > 0 && progress.speed.startsWith("resuming") ? (
                          <span className="download-progress__percent">Resuming {progress.downloaded_gb.toFixed(1)} GB…</span>
                        ) : (
                          <>
                            <span className="download-progress__percent">{progress.percent}%</span>
                            {" · "}
                            {progress.downloaded_gb.toFixed(1)} / {progress.expected_gb.toFixed(0)} GB
                          </>
                        )}
                      </span>
                      <span className="download-progress__speed">{progress.speed}</span>
                    </div>
                  </div>
                )}
              </div>
              <div className="model-card__actions">
                {c.present ? (
                  <span className="model-card__ok">Ready</span>
                ) : (
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={() => startDownload(c)}
                    disabled={busyId !== null}
                  >
                    {busyId === c.id ? "Downloading…" : "Download"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && !status && !error && (
        <p className="models-page__hint">No model status available.</p>
      )}

      {busyId && (
        <p className="models-page__hint" style={{ marginTop: 8 }}>
          💡 Downloads resume automatically if interrupted. You can close this window.
        </p>
      )}
    </div>
  );
}
