import type { SceneQueueItem, SceneStatus } from "../../types";

interface SceneQueueProps {
  scenes: SceneQueueItem[];
  onRemove: (id: string) => void;
  onReorder: (scenes: SceneQueueItem[]) => void;
  onEdit: (scene: SceneQueueItem) => void;
  onRunAll: () => void;
  onClear: () => void;
  disabled?: boolean;
  running?: boolean;
}

const STATUS_LABEL: Record<SceneStatus, string> = {
  pending: "pending",
  generating: "running",
  done: "done",
  failed: "failed",
  cancelled: "cancelled",
};

function moveScene(scenes: SceneQueueItem[], index: number, delta: number): SceneQueueItem[] | null {
  const next = index + delta;
  if (next < 0 || next >= scenes.length) return null;
  if (scenes[index].status !== "pending" || scenes[next].status !== "pending") return null;
  const copy = [...scenes];
  const [item] = copy.splice(index, 1);
  copy.splice(next, 0, item);
  return copy;
}

export function SceneQueue({
  scenes,
  onRemove,
  onReorder,
  onEdit,
  onRunAll,
  onClear,
  disabled,
  running,
}: SceneQueueProps) {
  if (scenes.length === 0) return null;

  const pendingCount = scenes.filter((s) => s.status === "pending").length;
  const canRun = pendingCount > 0 && !running && !disabled;

  return (
    <div className="scene-reel">
      <div className="scene-reel__bar">
        <span className="scene-reel__title">Scenes</span>
        <span className="scene-reel__count">
          {pendingCount} pending · {scenes.length}
        </span>
        <button type="button" className="chip-btn chip-btn--run" onClick={onRunAll} disabled={!canRun}>
          {running ? "Running…" : `Run ${pendingCount}`}
        </button>
        <button type="button" className="chip-btn" onClick={onClear} disabled={disabled || running}>
          Clear
        </button>
      </div>
      <div className="scene-reel__lane">
        {scenes.map((scene, index) => (
          <article key={scene.id} className={`scene-reel__card scene-reel__card--${scene.status}`}>
            <header className="scene-reel__card-head">
              <span className="scene-reel__num">{index + 1}</span>
              <span className="scene-reel__dur">{scene.durationId}</span>
              <span className="scene-reel__mode">{scene.mode}</span>
              <span className="scene-reel__status">{STATUS_LABEL[scene.status]}</span>
            </header>
            <p className="scene-reel__prompt">{scene.prompt.trim() || "(no prompt)"}</p>
            <footer className="scene-reel__card-foot">
              <button
                type="button"
                className="scene-reel__icon"
                disabled={disabled || running || index === 0 || scene.status !== "pending"}
                title="Move left"
                onClick={() => {
                  const next = moveScene(scenes, index, -1);
                  if (next) onReorder(next);
                }}
              >
                ◀
              </button>
              <button
                type="button"
                className="scene-reel__icon"
                disabled={disabled || running || index === scenes.length - 1 || scene.status !== "pending"}
                title="Move right"
                onClick={() => {
                  const next = moveScene(scenes, index, 1);
                  if (next) onReorder(next);
                }}
              >
                ▶
              </button>
              {scene.status === "pending" && (
                <>
                  <button
                    type="button"
                    className="scene-reel__text"
                    disabled={disabled || running}
                    onClick={() => onEdit(scene)}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    className="scene-reel__text"
                    disabled={disabled || running}
                    title="Remove"
                    onClick={() => onRemove(scene.id)}
                  >
                    ×
                  </button>
                </>
              )}
              {scene.status === "failed" && scene.error && (
                <span className="scene-reel__error" title={scene.error}>
                  error
                </span>
              )}
            </footer>
          </article>
        ))}
      </div>
    </div>
  );
}
