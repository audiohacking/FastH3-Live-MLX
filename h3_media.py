"""H3 canvas / duration snap helpers and PyAV media I/O.

All Python-side media (concat, last-frame, duration, and the h3.c ffmpeg CLI
shim in ``h3_av.py``) goes through PyAV. h3.c posix_spawns ``scripts/h3-av``
via ``H3_AV`` instead of a system ffmpeg install.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

FPS = 24
SPATIAL_ALIGN = 32
MIN_SPATIAL = 32
MAX_PIXELS = 768 * 1344  # released 768p-class cap from h3.c
MIN_FRAMES = 22  # 5 + 17*1


def snap_frames(raw: int) -> int:
    """Snap *up* to the next legal H3 temporal shape ``5 + 17n`` (n ≥ 1)."""
    n = max(1, int(raw))
    if n <= MIN_FRAMES:
        return MIN_FRAMES
    k = math.ceil((n - 5) / 17)
    k = max(1, k)
    return 5 + 17 * k


def frames_to_seconds(num_frames: int, fps: int = FPS) -> float:
    return float(num_frames) / float(fps)


# Rounded UI lengths → legal ``5+17n`` frame counts (not the exact 24 fps math).
UI_DURATION_FRAMES: dict[int, int] = {
    1: 22,
    2: 56,
    5: 107,
    10: 243,
    15: 362,
}


def seconds_to_frames(seconds: float, fps: int = FPS) -> int:
    sec = float(seconds)
    for rounded, frames in UI_DURATION_FRAMES.items():
        actual = frames_to_seconds(frames, fps=fps)
        if abs(sec - rounded) < 0.05 or abs(sec - actual) < 0.05:
            return frames
    raw = int(math.ceil(sec * fps))
    return snap_frames(raw)


def snap_spatial(n: int, align: int = SPATIAL_ALIGN) -> int:
    n = int(n)
    if n < MIN_SPATIAL:
        return MIN_SPATIAL
    return max(MIN_SPATIAL, int(round(n / align) * align))


def validate_canvas(width: int, height: int) -> tuple[int, int]:
    w = snap_spatial(width)
    h = snap_spatial(height)
    if w * h > MAX_PIXELS:
        raise ValueError(
            f"canvas {w}×{h} ({w * h} px) exceeds H3 limit {MAX_PIXELS} "
            f"(768×1344)"
        )
    return w, h


def duration_preset(
    rounded_s: int, *, num_frames: int, note: str = ""
) -> dict[str, Any]:
    nf = snap_frames(int(num_frames))
    label = f"{rounded_s}s ({nf} frames @ {FPS} fps)"
    if note:
        label = f"{label} — {note}"
    return {
        "id": f"{rounded_s}s",
        "seconds": float(rounded_s),
        "num_frames": nf,
        "label": label,
    }


DURATION_PRESETS = [
    duration_preset(1, num_frames=UI_DURATION_FRAMES[1], note="dev"),
    duration_preset(2, num_frames=UI_DURATION_FRAMES[2]),
    duration_preset(5, num_frames=UI_DURATION_FRAMES[5]),
    duration_preset(10, num_frames=UI_DURATION_FRAMES[10]),
    duration_preset(15, num_frames=UI_DURATION_FRAMES[15]),
]


def _canvas(
    preset_id: str,
    width: int,
    height: int,
    *,
    aspect: str,
    group: str,
    label: str,
    guidance: str,
    render: tuple[int, int] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": preset_id,
        "width": width,
        "height": height,
        "aspect": aspect,
        "group": group,
        "label": label,
        "guidance": guidance,
    }
    if render is not None:
        item["render_width"], item["render_height"] = render
    return item


# Output canvases: each side ×32, product ≤ 768×1344. Exact 16:9 at 768 short
# side is 1365×768 and does not fit; 1024×576 is the exact 16:9 under the cap,
# 1248×704 is the largest near-16:9. Same for 9:16. Internal render sizes are
# same-aspect DiT/VAE canvases that vImage-upscale (512 square only).
RESOLUTION_PRESETS = [
    _canvas(
        "256x256",
        256,
        256,
        aspect="1:1",
        group="Square",
        label="1:1 · 256 × 256 — preview",
        guidance="Native 8×8 token grid with automatic low-resolution RoPE. Keep token reduction off.",
    ),
    _canvas(
        "448x448",
        448,
        448,
        aspect="1:1",
        group="Square",
        label="1:1 · 448 × 448 — FastH3 Live",
        guidance=(
            "FastH3 Live default canvas (~200k px). Used by liveserver.py with "
            "4-step FastH3 LoRA and 362 frames."
        ),
    ),
    _canvas(
        "512x512",
        512,
        512,
        aspect="1:1",
        group="Square",
        label="1:1 · 512 × 512 — default",
        guidance="Balanced default resolution.",
    ),
    _canvas(
        "512x512-fast",
        512,
        512,
        aspect="1:1",
        group="Square",
        label="1:1 · 512 × 512 · 384 internal",
        guidance="Validated fast-quality scaling point: DiT/VAE at 384, output 512.",
        render=(384, 384),
    ),
    _canvas(
        "512x512-aggressive",
        512,
        512,
        aspect="1:1",
        group="Square",
        label="1:1 · 512 × 512 · 320 internal",
        guidance="Validated aggressive scaling point. Do not add token-reduction with layers 40 + reuse 3.",
        render=(320, 320),
    ),
    _canvas(
        "768x768",
        768,
        768,
        aspect="1:1",
        group="Square",
        label="1:1 · 768 × 768 — 768p",
        guidance="Validated close-quality square; substantially more expensive.",
    ),
    _canvas(
        "1024x576",
        1024,
        576,
        aspect="16:9",
        group="Landscape",
        label="16:9 · 1024 × 576 — YouTube / landscape",
        guidance="Exact 16:9. Largest exact 16:9 under the ×32 / 768×1344 cap (next step 1536×864 overflows).",
    ),
    _canvas(
        "1248x704",
        1248,
        704,
        aspect="16:9",
        group="Landscape",
        label="16:9 · 1248 × 704 — largest",
        guidance="Closest 16:9 under the pixel cap (1248/704 ≈ 1.773 vs 1.778). True 768p 16:9 would be 1365×768.",
    ),
    _canvas(
        "1344x768",
        1344,
        768,
        aspect="7:4",
        group="Landscape",
        label="7:4 · 1344 × 768 — max landscape",
        guidance="Released 768p-class landscape pixel-cap. 7:4, not 16:9.",
    ),
    _canvas(
        "1024x768",
        1024,
        768,
        aspect="4:3",
        group="Landscape",
        label="4:3 · 1024 × 768",
        guidance="Exact 4:3 768p-class canvas.",
    ),
    _canvas(
        "960x768",
        960,
        768,
        aspect="5:4",
        group="Landscape",
        label="5:4 · 960 × 768",
        guidance="Exact 5:4 (landscape complement of 4:5 social).",
    ),
    _canvas(
        "768x960",
        768,
        960,
        aspect="4:5",
        group="Portrait",
        label="4:5 · 768 × 960 — Instagram / social",
        guidance="Exact 4:5 with 768 short side. Feed-style portrait (not Stories/Reels 9:16).",
    ),
    _canvas(
        "768x1024",
        768,
        1024,
        aspect="3:4",
        group="Portrait",
        label="3:4 · 768 × 1024",
        guidance="Exact 3:4 768p-class canvas.",
    ),
    _canvas(
        "576x1024",
        576,
        1024,
        aspect="9:16",
        group="Portrait",
        label="9:16 · 576 × 1024 — Stories / Reels / TikTok",
        guidance="Exact 9:16. Largest exact 9:16 under the cap (next step 864×1536 overflows).",
    ),
    _canvas(
        "704x1248",
        704,
        1248,
        aspect="9:16",
        group="Portrait",
        label="9:16 · 704 × 1248 — largest",
        guidance="Closest 9:16 under the pixel cap. True 768p 9:16 would be 768×1365.",
    ),
    _canvas(
        "768x1344",
        768,
        1344,
        aspect="4:7",
        group="Portrait",
        label="4:7 · 768 × 1344 — max portrait",
        guidance="Released 768p-class portrait pixel-cap. 4:7, not 9:16.",
    ),
]

ALLOWED_OUTPUT_SIZES = frozenset(
    (int(p["width"]), int(p["height"])) for p in RESOLUTION_PRESETS
)


def require_ui_canvas(width: int, height: int) -> tuple[int, int]:
    """Accept only documented H3 output canvases (mechanical ×32 cap still applies)."""
    w, h = validate_canvas(width, height)
    if (w, h) not in ALLOWED_OUTPUT_SIZES:
        allowed = ", ".join(f"{a}×{b}" for a, b in sorted(ALLOWED_OUTPUT_SIZES))
        raise ValueError(f"canvas {w}×{h} is not a documented H3 size ({allowed})")
    return w, h


def sanitize_filename(prompt: str, maxlen: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (prompt or "").strip())[:maxlen].strip("_")
    return slug.lower() or "clip"


def media_available() -> bool:
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


MIN_REF_AUDIO_S = 2.0
MAX_REF_AUDIO_S = 15.0
MAX_REF_AUDIO_TOTAL_S = 15.0
MAX_REF_AUDIO_CLIPS = 3
_DURATION_SLACK_S = 0.05


def media_shim_ok() -> tuple[bool, str]:
    """PyAV + h3-av shim (the tools h3.c will posix_spawn)."""
    from h3_paths import default_h3_av

    if not media_available():
        return False, "PyAV is required (pip install av) — h3.c media goes through h3-av"
    shim = default_h3_av()
    if not shim.is_file():
        return False, f"h3-av shim not found at {shim}"
    return True, str(shim)


def ffmpeg_ok() -> tuple[bool, str]:
    """Backward-compatible alias: the media stack is the PyAV shim."""
    return media_shim_ok()


def probe_duration_seconds(path: Path | str) -> float | None:
    """Duration in seconds via PyAV. No ffprobe CLI."""
    src = Path(path)
    if not src.is_file() or not media_available():
        return None
    try:
        import av

        container = av.open(str(src))
        try:
            if container.duration and container.duration > 0:
                return float(container.duration) / float(av.time_base)
            best = 0.0
            for stream in container.streams:
                if stream.duration and stream.time_base and stream.duration > 0:
                    seconds = float(stream.duration * stream.time_base)
                    if seconds > best:
                        best = seconds
            return best or None
        finally:
            container.close()
    except Exception:
        return None


def resize_still_to_canvas(src: Path | str, dest: Path | str, width: int, height: int) -> Path:
    """Resize an image still to the output canvas. Images only."""
    import av

    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    inp = av.open(str(source))
    try:
        frame = next(inp.decode(video=0))
    finally:
        inp.close()
    frame = frame.reformat(width=int(width), height=int(height), format="rgb24")
    out = av.open(str(target), mode="w", format="image2")
    try:
        stream = out.add_stream("png", rate=1)
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "rgb24"
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return target


def assert_audio_durations(seconds: list[float]) -> None:
    """Enforce h3.c audio-reference limits (2–15 s each, ≤3 clips, total ≤15 s)."""
    if len(seconds) > MAX_REF_AUDIO_CLIPS:
        raise ValueError(f"at most {MAX_REF_AUDIO_CLIPS} audio references")
    total = 0.0
    for duration in seconds:
        if duration + _DURATION_SLACK_S < MIN_REF_AUDIO_S:
            raise ValueError(
                f"audio reference is {duration:.2f}s; h3.c requires at least "
                f"{MIN_REF_AUDIO_S:.0f}s"
            )
        if duration - _DURATION_SLACK_S > MAX_REF_AUDIO_S:
            raise ValueError(
                f"audio reference is {duration:.2f}s; h3.c allows at most "
                f"{MAX_REF_AUDIO_S:.0f}s"
            )
        total += duration
    if total - _DURATION_SLACK_S > MAX_REF_AUDIO_TOTAL_S:
        raise ValueError(
            f"audio references total {total:.2f}s; h3.c caps the sum at "
            f"{MAX_REF_AUDIO_TOTAL_S:.0f}s"
        )


def extract_last_frame(video_path: str | Path, dest: str | Path) -> Path:
    """Write the last decoded frame of ``video_path`` as PNG to ``dest``."""
    import av
    from PIL import Image

    src = Path(video_path)
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(src))
    stream = container.streams.video[0]
    last = None
    for frame in container.decode(stream):
        last = frame
    container.close()
    if last is None:
        raise RuntimeError(f"no video frames in {src}")
    img = last.to_ndarray(format="rgb24")
    Image.fromarray(img).save(out)
    return out


def concat_mp4s(paths: list[Path], dest: Path) -> Path:
    """Stream-copy concatenate MP4s with PyAV. Falls back to re-encode if needed."""
    import av

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        import shutil

        shutil.copy2(paths[0], dest)
        return dest

    first = av.open(str(paths[0]))
    out = av.open(str(dest), mode="w")
    stream_map: dict[int, Any] = {}
    for i, stream in enumerate(first.streams):
        if stream.type not in ("video", "audio"):
            continue
        out_stream = out.add_stream(template=stream)
        stream_map[i] = out_stream
    first.close()

    try:
        for path in paths:
            src = av.open(str(path))
            for packet in src.demux():
                if packet.stream_index not in stream_map:
                    continue
                if packet.dts is None:
                    continue
                packet.stream = stream_map[packet.stream_index]
                out.mux(packet)
            src.close()
    finally:
        out.close()
    return dest
