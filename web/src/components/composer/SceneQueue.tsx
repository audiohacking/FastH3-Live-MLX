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

const STATUS_ICON: Record<SceneStatus, string> = {
  pending: "",
  generating: "",
  done: "",
  failed: "",
  cancelled: "",
};

const STATUS_LABEL: Record<SceneStatus, string> = {
  pending: "Pending",
  generating: "Generating...",
  done: "Complete",
  failed: "Failed",
  cancelled: "Cancelled",
};

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
  const pendingCount = scenes.filter((s) => s.status === "pending").length;
  const hasScenes = scenes.length > 0;
  const canRun = pendingCount > 0 && !running && !disabled;

  function handleDragStart(e: React.DragEvent, index: number) {
    e.dataTransfer.setData("text/plain", String(index));
    e.dataTransfer.effectAllowed = "move";
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }

  function handleDrop(e: React.DragEvent, dropIndex: number) {
    e.preventDefault();
    const dragIndex = parseInt(e.dataTransfer.getData("text/plain"), 10);
    if (dragIndex === dropIndex || isNaN(dragIndex)) return;

    const reordered = [...scenes];
    const [moved] = reordered.splice(dragIndex, 1);
    reordered.splice(dropIndex, 0, moved);
    onReorder(reordered);
  }

  function truncatePrompt(prompt: string, max = 60): string {
    if (prompt.length <= max) return prompt;
    return prompt.slice(0, max - 1) + "…";
  }

  return (
    <div className="scene-queue">
      <div className="scene-queue__header">
        <span className="scene-queue__title">Scene Queue</span>
        {hasScenes && (
          <span className="scene-queue__count">
            {pendingCount} pending / {scenes.length} total
          </span>
        )}
      </div>

      {!hasScenes && (
        <p className="scene-queue__empty">
          Queue scenes to generate them in sequence. Click "Add to Queue" to add the current configuration.
        </p>
      )}

      {hasScenes && (
        <div className="scene-queue__list">
          {scenes.map((scene, index) => (
            <div
              key={scene.id}
              className={`scene-queue__item scene-queue__item--${scene.status}`}
              draggable={scene.status === "pending" && !disabled}
              onDragStart={(e) => handleDragStart(e, index)}
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, index)}
            >
              <span className="scene-queue__item-number">{index + 1}</span>
              <div className="scene-queue__item-content">
                <span className="scene-queue__item-prompt">
                  {truncatePrompt(scene.prompt || "(no prompt)")}
                </span>
                <span className="scene-queue__item-meta">
                  {scene.mode} · {scene.resolutionId} · {scene.numSteps} steps
                  {scene.turboEnabled && " · Turbo"}
                </span>
              </div>
              <span className="scene-queue__item-status" title={STATUS_LABEL[scene.status]}>
                {STATUS_ICON[scene.status]}
              </span>
              <div className="scene-queue__item-actions">
                {scene.status === "pending" && (
                  <>
                    <button
                      type="button"
                      className="scene-queue__btn scene-queue__btn--edit"
                      onClick={() => onEdit(scene)}
                      disabled={disabled}
                      title="Load into editor"
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      className="scene-queue__btn scene-queue__btn--remove"
                      onClick={() => onRemove(scene.id)}
                      disabled={disabled}
                      title="Remove from queue"
                    >
                      &times;
                    </button>
                  </>
                )}
                {scene.status === "failed" && scene.error && (
                  <span className="scene-queue__item-error" title={scene.error}>
                    Error
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {hasScenes && (
        <div className="scene-queue__actions">
          <button
            type="button"
            className="scene-queue__btn scene-queue__btn--run"
            onClick={onRunAll}
            disabled={!canRun}
          >
            {running ? "Running..." : `Run All (${pendingCount})`}
          </button>
          <button
            type="button"
            className="scene-queue__btn scene-queue__btn--clear"
            onClick={onClear}
            disabled={disabled || running}
          >
            Clear Queue
          </button>
        </div>
      )}
    </div>
  );
}

interface AddToQueueButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export function AddToQueueButton({ onClick, disabled }: AddToQueueButtonProps) {
  return (
    <button
      type="button"
      className="add-to-queue-btn"
      onClick={onClick}
      disabled={disabled}
      title="Add current configuration to scene queue"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
        <path d="M12 5v14M5 12h14" />
      </svg>
      <span>Add to Queue</span>
    </button>
  );
}
