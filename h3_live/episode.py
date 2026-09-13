"""Offline episode batch — generate N scenes, concat to one MP4 for download.

Uses the FastH3 student ladder (4 steps) but drops Live stream tricks: full
448 canvas, no token-reduction, native 24 fps export (no retime stretch).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from h3_live import LIVE_FRAMES, LIVE_HEIGHT, LIVE_WIDTH

# Higher-quality offline recipe vs continuous Live (384/noTR stream path).
EPISODE_WIDTH = LIVE_WIDTH
EPISODE_HEIGHT = LIVE_HEIGHT
EPISODE_RENDER_WIDTH = LIVE_WIDTH  # full canvas — no vImage upscale
EPISODE_RENDER_HEIGHT = LIVE_HEIGHT
EPISODE_FRAMES = LIVE_FRAMES
EPISODE_STEPS = 4
EPISODE_LAYERS = 50
EPISODE_REUSE = 1
EPISODE_TOKEN_REDUCTION = False
EPISODE_MIN_SCENES = 1
EPISODE_MAX_SCENES = 10

EPISODE_RECIPE_NOTE = (
    f"episode out={EPISODE_WIDTH}x{EPISODE_HEIGHT} "
    f"render={EPISODE_RENDER_WIDTH}x{EPISODE_RENDER_HEIGHT} "
    f"frames={EPISODE_FRAMES} steps={EPISODE_STEPS} noTR · native 24 fps"
)


def clamp_scene_count(n: int | str | None) -> int:
    try:
        value = int(n)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("scenes must be an integer 1–10") from exc
    if value < EPISODE_MIN_SCENES or value > EPISODE_MAX_SCENES:
        raise ValueError(
            f"scenes must be {EPISODE_MIN_SCENES}–{EPISODE_MAX_SCENES}, got {value}"
        )
    return value


@dataclass
class EpisodeJob:
    """Mutable job status shared with the dashboard (guard with LiveState.lock)."""

    status: str = "idle"  # idle | running | ready | error | cancelled
    scenes_requested: int = 0
    scenes_done: int = 0
    current_scene: int | None = None
    error: str | None = None
    output_path: Path | None = None
    started_at: float | None = None
    finished_at: float | None = None
    last_cast: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    # Scene clip paths kept until next job clears them.
    clip_paths: list[Path] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.status == "running"

    def reset_for_start(self, scenes: int) -> None:
        self.cancel.clear()
        self.status = "running"
        self.scenes_requested = scenes
        self.scenes_done = 0
        self.current_scene = None
        self.error = None
        self.output_path = None
        self.started_at = time.time()
        self.finished_at = None
        self.last_cast = None
        self.clip_paths = []

    def as_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            elapsed = round(end - self.started_at, 1)
        ready = self.status == "ready" and self.output_path is not None
        size = None
        if ready and self.output_path is not None and self.output_path.is_file():
            size = self.output_path.stat().st_size
        return {
            "status": self.status,
            "scenes_requested": self.scenes_requested,
            "scenes_done": self.scenes_done,
            "current_scene": self.current_scene,
            "error": self.error,
            "recipe": EPISODE_RECIPE_NOTE,
            "download_ready": ready,
            "download_url": "/api/episode.mp4" if ready else None,
            "bytes": size,
            "elapsed_s": elapsed,
            "last_cast": self.last_cast,
        }


def concat_mp4s(paths: list[Path], dest: Path) -> Path:
    """Concatenate H.264+AAC MP4 clips into one file (packet remux, PTS rebase)."""
    import av

    if not paths:
        raise ValueError("no clips to concatenate")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    dest.parent.mkdir(parents=True, exist_ok=True)
    out = av.open(str(dest), mode="w")
    try:
        in0 = av.open(str(paths[0]))
        try:
            stream_map: dict[int, object] = {}
            for stream in in0.streams:
                if stream.type not in ("video", "audio"):
                    continue
                out_stream = out.add_stream_from_template(stream)
                stream_map[stream.index] = out_stream
        finally:
            in0.close()

        if not stream_map:
            raise RuntimeError(f"no video/audio streams in {paths[0]}")

        # Next PTS/DTS per output stream index (in that stream's time_base).
        next_ts: dict[int, int] = {s.index: 0 for s in out.streams}

        for path in paths:
            inp = av.open(str(path))
            try:
                # Map this file's streams by type/order onto out streams.
                by_type: dict[str, list] = {"video": [], "audio": []}
                for stream in inp.streams:
                    if stream.type in by_type:
                        by_type[stream.type].append(stream)
                out_by_type: dict[str, list] = {"video": [], "audio": []}
                for stream in out.streams:
                    if stream.type in out_by_type:
                        out_by_type[stream.type].append(stream)

                local_map: dict[int, object] = {}
                for kind in ("video", "audio"):
                    for src, dst in zip(by_type[kind], out_by_type[kind]):
                        local_map[src.index] = dst

                # Track first/last dts per src stream to compute duration gap.
                first_dts: dict[int, int | None] = {i: None for i in local_map}
                last_dts: dict[int, int] = {}
                last_dur: dict[int, int] = {}

                for packet in inp.demux():
                    if packet.stream is None or packet.stream.index not in local_map:
                        continue
                    if packet.dts is None and packet.pts is None:
                        continue
                    src_i = packet.stream.index
                    dst = local_map[src_i]
                    dts = packet.dts if packet.dts is not None else packet.pts
                    assert dts is not None
                    if first_dts[src_i] is None:
                        first_dts[src_i] = dts
                    base = first_dts[src_i] or 0
                    offset = next_ts[dst.index]
                    packet.stream = dst
                    packet.dts = offset + (dts - base)
                    if packet.pts is not None:
                        packet.pts = offset + (packet.pts - base)
                    out.mux(packet)
                    last_dts[src_i] = dts
                    last_dur[src_i] = int(packet.duration or 0)

                for src_i, dst in local_map.items():
                    if first_dts[src_i] is None or src_i not in last_dts:
                        continue
                    span = last_dts[src_i] - (first_dts[src_i] or 0) + max(
                        last_dur.get(src_i, 0), 1
                    )
                    next_ts[dst.index] += max(span, 1)
            finally:
                inp.close()
    finally:
        out.close()

    if not dest.is_file() or dest.stat().st_size < 100:
        raise RuntimeError(f"concat produced empty file: {dest}")
    return dest
