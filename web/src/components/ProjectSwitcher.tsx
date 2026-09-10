import { useEffect, useRef, useState } from "react";

export type Project = {
  id: string;
  name: string;
  created_at?: string;
  updated_at?: string;
  clip_count?: number;
  active?: boolean;
};

type Props = {
  projects: Project[];
  activeProjectId: string | null;
  disabled?: boolean;
  onSelect: (id: string) => void | Promise<void>;
  onCreate: () => void | Promise<void>;
  onRename: (id: string, name: string) => void | Promise<void>;
  onDelete: (id: string) => void | Promise<void>;
};

export function ProjectSwitcher({
  projects,
  activeProjectId,
  disabled,
  onSelect,
  onCreate,
  onRename,
  onDelete,
}: Props) {
  const [open, setOpen] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const active = projects.find((p) => p.id === activeProjectId) ?? projects[0];

  useEffect(() => {
    if (!open) return;
    const onDoc = (ev: MouseEvent) => {
      if (!rootRef.current?.contains(ev.target as Node)) {
        setOpen(false);
        setRenamingId(null);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  async function commitRename(id: string) {
    const name = renameValue.trim();
    setRenamingId(null);
    if (!name) return;
    const current = projects.find((p) => p.id === id);
    if (current && current.name === name) return;
    await onRename(id, name);
  }

  return (
    <div className={`project-switcher${open ? " is-open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="project-switcher__trigger"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        title="Switch project"
      >
        <span className="project-switcher__label">{active?.name ?? "Project"}</span>
        <span className="project-switcher__chev" aria-hidden>
          ▾
        </span>
      </button>
      {open && (
        <div className="project-switcher__menu" role="listbox">
          {projects.map((project) => {
            const isActive = project.id === activeProjectId;
            const isRenaming = renamingId === project.id;
            return (
              <div
                key={project.id}
                className={`project-switcher__row${isActive ? " is-active" : ""}`}
              >
                {isRenaming ? (
                  <input
                    className="project-switcher__rename"
                    value={renameValue}
                    autoFocus
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void commitRename(project.id);
                      if (e.key === "Escape") setRenamingId(null);
                    }}
                    onBlur={() => void commitRename(project.id)}
                  />
                ) : (
                  <button
                    type="button"
                    className="project-switcher__item"
                    role="option"
                    aria-selected={isActive}
                    onClick={() => {
                      setOpen(false);
                      if (!isActive) void onSelect(project.id);
                    }}
                  >
                    <span className="project-switcher__name">{project.name}</span>
                    <span className="project-switcher__meta">
                      {project.clip_count ?? 0}
                    </span>
                  </button>
                )}
                {!isRenaming && (
                  <div className="project-switcher__actions">
                    <button
                      type="button"
                      className="project-switcher__icon"
                      title="Rename"
                      onClick={(e) => {
                        e.stopPropagation();
                        setRenamingId(project.id);
                        setRenameValue(project.name);
                      }}
                    >
                      ✎
                    </button>
                    <button
                      type="button"
                      className="project-switcher__icon project-switcher__icon--danger"
                      title={projects.length <= 1 ? "Keep at least one project" : "Delete project"}
                      disabled={projects.length <= 1}
                      onClick={(e) => {
                        e.stopPropagation();
                        const n = project.clip_count ?? 0;
                        const msg =
                          n > 0
                            ? `Delete “${project.name}” and its ${n} video${n === 1 ? "" : "s"}?`
                            : `Delete empty project “${project.name}”?`;
                        if (!confirm(msg)) return;
                        setOpen(false);
                        void onDelete(project.id);
                      }}
                    >
                      ×
                    </button>
                  </div>
                )}
              </div>
            );
          })}
          <button
            type="button"
            className="project-switcher__new"
            onClick={() => {
              setOpen(false);
              void onCreate();
            }}
          >
            + New project
          </button>
        </div>
      )}
    </div>
  );
}
