import { useRef } from "react";
import type { Clip, LibraryFrame, ReferenceItem } from "../../types";
import { generateId } from "../../utils";
import { ReferenceChips, AddReferenceButton } from "./ReferenceChips";

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

export function RefListEnhanced({ refs, disabled, frames, clips, onChange, uploadFile }: Props) {
  const imageRef = useRef<HTMLInputElement>(null);
  const silentVideoRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);
  const videoAudioVideoRef = useRef<HTMLInputElement>(null);
  const videoAudioAudioRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);
  const pendingVideoAudio = useRef<Partial<ReferenceItem> | null>(null);
  const validity = refs.length ? refsAreValid(refs) : { ok: true };

  function add(item: Omit<ReferenceItem, "id">) {
    onChange([...refs, { ...item, id: generateId() }]);
  }

  return (
    <div className="ref-list-enhanced">
      <div className="ref-list-enhanced__header">
        <span className="ref-list-enhanced__title">Reference clips</span>
        {refs.length > 0 && (
          <span className="ref-list-enhanced__count">{refs.length}</span>
        )}
      </div>

      <p className="ref-list-enhanced__help">
        Add image, video, or audio files in order. Reference them as{" "}
        <code>Picture 1</code>, <code>Video 1</code>, <code>Audio 1</code> in your prompt.
        Audio clips must be 2–15 s (max 3).
      </p>

      {refs.length > 0 && (
        <div className="ref-list-enhanced__chips">
          <ReferenceChips refs={refs} onChange={onChange} disabled={disabled} />
        </div>
      )}

      {!validity.ok && validity.error && (
        <p className="ref-list-enhanced__error">{validity.error}</p>
      )}

      <div className="ref-list-enhanced__buttons">
        <AddReferenceButton
          label="Image"
          onClick={() => imageRef.current?.click()}
          disabled={disabled}
        />
        <AddReferenceButton
          label="Silent video"
          onClick={() => silentVideoRef.current?.click()}
          disabled={disabled}
        />
        <AddReferenceButton
          label="Video"
          onClick={() => videoRef.current?.click()}
          disabled={disabled}
        />
        <AddReferenceButton
          label="Video + audio"
          onClick={() => videoAudioVideoRef.current?.click()}
          disabled={disabled}
        />
        <AddReferenceButton
          label="Audio"
          onClick={() => audioRef.current?.click()}
          disabled={disabled}
        />
      </div>

      {frames.length > 0 && (
        <div className="ref-source-picker">
          <span className="ref-source-picker__label">Add saved frame as image</span>
          <select
            className="ref-source-picker__select"
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
        </div>
      )}

      {clips.length > 0 && (
        <div className="ref-source-picker">
          <span className="ref-source-picker__label">Add library clip as silent video</span>
          <select
            className="ref-source-picker__select"
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
        </div>
      )}

      {/* Hidden file inputs */}
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
