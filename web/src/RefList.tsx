import { useRef } from "react";
import type { Clip, LibraryFrame, ReferenceItem, RefKind } from "./types";
import { generateId } from "./utils";

const KIND_LABEL: Record<RefKind, string> = {
  image: "Picture",
  silent_video: "Silent video",
  video: "Video",
  video_audio: "Video + audio",
  audio: "Audio",
};

function promptToken(kind: RefKind, indexAmongKind: number): string {
  if (kind === "image") return `Picture ${indexAmongKind}`;
  if (kind === "audio") return `Audio ${indexAmongKind}`;
  return `Video ${indexAmongKind}`;
}

function tokensFor(refs: ReferenceItem[]): string[] {
  let pictures = 0;
  let videos = 0;
  let audios = 0;
  return refs.map((ref) => {
    if (ref.kind === "image") {
      pictures += 1;
      return promptToken("image", pictures);
    }
    if (ref.kind === "audio") {
      audios += 1;
      return promptToken("audio", audios);
    }
    videos += 1;
    return promptToken("video", videos);
  });
}

export function refsAreValid(refs: ReferenceItem[]): { ok: boolean; error?: string } {
  if (refs.length === 0) return { ok: false, error: "Add at least one reference" };
  const images = refs.filter((r) => r.kind === "image").length;
  const videos = refs.filter((r) =>
    r.kind === "silent_video" || r.kind === "video" || r.kind === "video_audio",
  ).length;
  const audios = refs.filter((r) => r.kind === "audio" || r.kind === "video_audio").length;
  const files = refs.reduce((n, r) => n + (r.kind === "video_audio" ? 2 : 1), 0);
  if (images > 9) return { ok: false, error: "At most 9 reference images" };
  if (videos > 3) return { ok: false, error: "At most 3 reference videos" };
  if (audios > 3) return { ok: false, error: "At most 3 audio references" };
  if (files > 12) return { ok: false, error: "At most 12 mixed reference files" };
  const hasVisual = images + videos > 0;
  if (audios > 0 && !hasVisual) {
    return { ok: false, error: "Audio must accompany an image or video reference" };
  }
  if (refs.some((r) => r.kind === "video_audio" && !r.audioPath)) {
    return { ok: false, error: "Video + replacement audio needs both files" };
  }
  return { ok: true };
}

type Props = {
  refs: ReferenceItem[];
  disabled?: boolean;
  frames: LibraryFrame[];
  clips: Clip[];
  onChange: (next: ReferenceItem[]) => void;
  uploadFile: (file: File, kind: string) => Promise<string>;
};

export function RefList({ refs, disabled, frames, clips, onChange, uploadFile }: Props) {
  const imageRef = useRef<HTMLInputElement>(null);
  const silentVideoRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);
  const videoAudioVideoRef = useRef<HTMLInputElement>(null);
  const videoAudioAudioRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);
  const pendingVideoAudio = useRef<Partial<ReferenceItem> | null>(null);
  const tokens = tokensFor(refs);
  const validity = refs.length ? refsAreValid(refs) : { ok: true };

  function add(item: Omit<ReferenceItem, "id">) {
    onChange([...refs, { ...item, id: generateId() }]);
  }

  function move(index: number, dir: -1 | 1) {
    const next = [...refs];
    const j = index + dir;
    if (j < 0 || j >= next.length) return;
    const tmp = next[index];
    next[index] = next[j];
    next[j] = tmp;
    onChange(next);
  }

  function remove(index: number) {
    onChange(refs.filter((_, i) => i !== index));
  }

  return (
    <div className="ref-panel">
      <div className="ref-panel-header">
        <span className="media-panel-title">Reference clips</span>
        {refs.length > 0 && <span className="ref-panel-count">{refs.length}</span>}
      </div>
      <p className="hint hint-inline">
        Add image, video, or audio files in the order they should appear. In the prompt,
        name them <code>Picture 1</code>, <code>Video 1</code>, <code>Audio 1</code>.
        Audio clips must be 2–15 s (max 3, 15 s total) and need an image or video too.
      </p>

      {refs.length > 0 && (
        <ol className="ref-list">
          {refs.map((ref, i) => (
            <li key={ref.id} className="ref-item">
              <span className="ref-token">{tokens[i]}</span>
              <span className="ref-kind">{KIND_LABEL[ref.kind]}</span>
              <span className="ref-name" title={ref.path}>
                {ref.name}
                {ref.audioName ? ` + ${ref.audioName}` : ""}
              </span>
              <span className="ref-actions">
                <button type="button" className="btn-prompt-action" disabled={disabled || i === 0} onClick={() => move(i, -1)}>
                  Up
                </button>
                <button
                  type="button"
                  className="btn-prompt-action"
                  disabled={disabled || i === refs.length - 1}
                  onClick={() => move(i, 1)}
                >
                  Down
                </button>
                <button type="button" className="btn-prompt-action" disabled={disabled} onClick={() => remove(i)}>
                  Remove
                </button>
              </span>
            </li>
          ))}
        </ol>
      )}

      {!validity.ok && <p className="hint hint-inline">{validity.error}</p>}

      <div className="ref-add-row">
        <button type="button" className="btn-secondary btn-compact" disabled={disabled} onClick={() => imageRef.current?.click()}>
          Add image
        </button>
        <button type="button" className="btn-secondary btn-compact" disabled={disabled} onClick={() => silentVideoRef.current?.click()}>
          Add silent video
        </button>
        <button type="button" className="btn-secondary btn-compact" disabled={disabled} onClick={() => videoRef.current?.click()}>
          Add video
        </button>
        <button
          type="button"
          className="btn-secondary btn-compact"
          disabled={disabled}
          onClick={() => videoAudioVideoRef.current?.click()}
        >
          Add video + audio
        </button>
        <button type="button" className="btn-secondary btn-compact" disabled={disabled} onClick={() => audioRef.current?.click()}>
          Add audio
        </button>
      </div>

      {frames.length > 0 && (
        <label className="clip-source-picker">
          <span className="media-upload-label">Add saved frame as image</span>
          <select
            disabled={disabled}
            defaultValue=""
            onChange={(e) => {
              const frame = frames.find((f) => f.id === e.target.value);
              e.currentTarget.value = "";
              if (!frame) return;
              add({ kind: "image", path: frame.path, name: frame.label });
            }}
          >
            <option value="">Select a frame…</option>
            {frames.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
      )}

      {clips.length > 0 && (
        <label className="clip-source-picker">
          <span className="media-upload-label">Add library clip as silent video</span>
          <select
            disabled={disabled}
            defaultValue=""
            onChange={(e) => {
              const clip = clips.find((c) => c.id === e.target.value);
              e.currentTarget.value = "";
              if (!clip) return;
              const path = clip.path || clip.filename;
              if (!path) return;
              add({
                kind: "silent_video",
                path,
                name: clip.label || clip.filename,
              });
            }}
          >
            <option value="">Select a clip…</option>
            {clips.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label} · {c.filename}
              </option>
            ))}
          </select>
        </label>
      )}

      <input
        ref={imageRef}
        type="file"
        accept="image/*"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          const path = await uploadFile(f, "image");
          add({ kind: "image", path, name: f.name });
        }}
      />
      <input
        ref={silentVideoRef}
        type="file"
        accept="video/*"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          const path = await uploadFile(f, "video");
          add({ kind: "silent_video", path, name: f.name });
        }}
      />
      <input
        ref={videoRef}
        type="file"
        accept="video/*"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          const path = await uploadFile(f, "video");
          add({ kind: "video", path, name: f.name });
        }}
      />
      <input
        ref={videoAudioVideoRef}
        type="file"
        accept="video/*"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          const path = await uploadFile(f, "video");
          pendingVideoAudio.current = { kind: "video_audio", path, name: f.name };
          videoAudioAudioRef.current?.click();
        }}
      />
      <input
        ref={videoAudioAudioRef}
        type="file"
        accept="audio/*"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          const pending = pendingVideoAudio.current;
          pendingVideoAudio.current = null;
          if (!f || !pending?.path || !pending.name) return;
          const audioPath = await uploadFile(f, "audio");
          add({
            kind: "video_audio",
            path: pending.path,
            name: pending.name,
            audioPath,
            audioName: f.name,
          });
        }}
      />
      <input
        ref={audioRef}
        type="file"
        accept="audio/*"
        hidden
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          const path = await uploadFile(f, "audio");
          add({ kind: "audio", path, name: f.name });
        }}
      />
    </div>
  );
}
