import { useRef, useState } from "react";

type VideoKind = "video" | "silent_video" | "video_audio";

type Props = {
  disabled?: boolean;
  loraCount: number;
  castSlot: React.ReactNode;
  librarySlot?: React.ReactNode;
  presetSlot: React.ReactNode;
  onAddImage: (file: File) => void;
  onAddVideo: (file: File, kind: VideoKind) => void;
  onAddAudio: (file: File) => void;
  onAddVideoAudio: (video: File, audio: File) => void;
  onOpenLora: () => void;
  onClear: () => void;
};

export function ComposerRail({
  disabled,
  loraCount,
  castSlot,
  librarySlot,
  presetSlot,
  onAddImage,
  onAddVideo,
  onAddAudio,
  onAddVideoAudio,
  onOpenLora,
  onClear,
}: Props) {
  const imageRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);
  const silentRef = useRef<HTMLInputElement>(null);
  const videoAudioVideoRef = useRef<HTMLInputElement>(null);
  const videoAudioAudioRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);
  const pendingVideo = useRef<File | null>(null);
  const [videoMenu, setVideoMenu] = useState(false);
  const videoKind = useRef<VideoKind>("video");

  return (
    <div className="composer-rail">
      <div className="composer-rail__group">
        <button type="button" className="rail-btn" disabled={disabled} onClick={() => imageRef.current?.click()}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <circle cx="8.5" cy="10" r="1.5" />
            <path d="M21 16l-5-5-11 8" />
          </svg>
          Add image
        </button>
        <div className="popover-anchor">
          <button
            type="button"
            className="rail-btn rail-btn--menu"
            disabled={disabled}
            onClick={() => setVideoMenu((v) => !v)}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <rect x="3" y="6" width="14" height="12" rx="2" />
              <path d="M17 10l5-3v10l-5-3" />
            </svg>
            Add video
          </button>
          {videoMenu && (
            <div className="rail-menu">
              <button
                type="button"
                onClick={() => {
                  videoKind.current = "video";
                  setVideoMenu(false);
                  videoRef.current?.click();
                }}
              >
                Video (keep audio)
              </button>
              <button
                type="button"
                onClick={() => {
                  videoKind.current = "silent_video";
                  setVideoMenu(false);
                  silentRef.current?.click();
                }}
              >
                Silent video
              </button>
              <button
                type="button"
                onClick={() => {
                  setVideoMenu(false);
                  videoAudioVideoRef.current?.click();
                }}
              >
                Video + replacement audio
              </button>
            </div>
          )}
        </div>
        <button type="button" className="rail-btn" disabled={disabled} onClick={() => audioRef.current?.click()}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M9 18V6l12-2v12" />
            <circle cx="6" cy="18" r="2.5" />
            <circle cx="18" cy="16" r="2.5" />
          </svg>
          Add audio
        </button>
        <button type="button" className="rail-btn" disabled={disabled} onClick={onOpenLora}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M12 3l1.8 5.4H19l-4.4 3.2 1.7 5.4L12 14.8 7.7 17l1.7-5.4L5 8.4h5.2z" />
          </svg>
          Add LoRA
          {loraCount > 0 && <span className="rail-btn__badge">{loraCount}</span>}
        </button>
        {castSlot}
        {librarySlot}
        <button type="button" className="rail-btn" disabled={disabled} onClick={onClear}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M4 7h16M9 7V5h6v2M8 7l1 12h6l1-12" />
          </svg>
          Clear
        </button>
      </div>
      <div className="composer-rail__group">{presetSlot}</div>

      <input ref={imageRef} type="file" accept="image/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (f) onAddImage(f);
      }} />
      <input ref={videoRef} type="file" accept="video/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (f) onAddVideo(f, "video");
      }} />
      <input ref={silentRef} type="file" accept="video/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (f) onAddVideo(f, "silent_video");
      }} />
      <input ref={videoAudioVideoRef} type="file" accept="video/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (!f) return;
        pendingVideo.current = f;
        videoAudioAudioRef.current?.click();
      }} />
      <input ref={videoAudioAudioRef} type="file" accept="audio/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        const video = pendingVideo.current;
        pendingVideo.current = null;
        if (f && video) onAddVideoAudio(video, f);
      }} />
      <input ref={audioRef} type="file" accept="audio/*" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (f) onAddAudio(f);
      }} />
    </div>
  );
}
