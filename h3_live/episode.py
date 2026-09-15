"""Offline episode batch — generate N scenes, concat to one MP4 for download.

Exclusive with Live: full quality knobs (no stream sustain constraints),
native 24 fps export (no retime stretch). Defaults stay sharp 448 / ~5 s.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from h3_live import LIVE_HEIGHT, LIVE_WIDTH
from h3_media import UI_DURATION_FRAMES, frames_to_seconds, snap_frames

# Defaults — sharp offline recipe (overridable per job from the dashboard).
EPISODE_WIDTH = LIVE_WIDTH
EPISODE_HEIGHT = LIVE_HEIGHT
EPISODE_RENDER_WIDTH = LIVE_WIDTH  # full canvas — no vImage upscale
EPISODE_RENDER_HEIGHT = LIVE_HEIGHT
EPISODE_FRAMES = UI_DURATION_FRAMES[5]  # ~5 s authored; 10→243, 15→362
EPISODE_STEPS = 4
EPISODE_LAYERS = 50
EPISODE_REUSE = 1
EPISODE_TOKEN_REDUCTION = False
EPISODE_MIN_SCENES = 1
EPISODE_MAX_SCENES = 10
EPISODE_MIN_STEPS = 1
EPISODE_MAX_STEPS = 50
EPISODE_MIN_LAYERS = 35
EPISODE_MAX_LAYERS = 50
EPISODE_MIN_REUSE = 1
EPISODE_MAX_REUSE = 3
EPISODE_RENDER_CHOICES = (320, 384, 448)
# Clip length presets (authored seconds → legal 5+17n frames). Cap 15 s.
EPISODE_DURATION_CHOICES = (5, 10, 15)


def episode_frames_for_seconds(seconds: int | float) -> int:
    """Map a UI duration (5 / 10 / 15) to H3-legal frames."""
    sec = int(round(float(seconds)))
    if sec in UI_DURATION_FRAMES:
        return int(UI_DURATION_FRAMES[sec])
    if sec in (5, 10, 15):
        # Fallback if UI map changes — snap ceil(sec * 24).
        return snap_frames(int(sec * 24))
    raise ValueError(f"duration must be one of {EPISODE_DURATION_CHOICES}, got {seconds}")


def episode_recipe_label(
    *,
    width: int,
    height: int,
    render_width: int,
    render_height: int,
    frames: int,
    steps: int,
    layers: int,
    reuse: int,
    token_reduction: bool,
) -> str:
    tr = "TR" if token_reduction else "noTR"
    secs = frames_to_seconds(frames)
    return (
        f"episode out={width}x{height} render={render_width}x{render_height} "
        f"frames={frames} (~{secs:.1f}s) steps={steps} layers={layers} "
        f"reuse={reuse} {tr} · native 24 fps"
    )


EPISODE_RECIPE_NOTE = episode_recipe_label(
    width=EPISODE_WIDTH,
    height=EPISODE_HEIGHT,
    render_width=EPISODE_RENDER_WIDTH,
    render_height=EPISODE_RENDER_HEIGHT,
    frames=EPISODE_FRAMES,
    steps=EPISODE_STEPS,
    layers=EPISODE_LAYERS,
    reuse=EPISODE_REUSE,
    token_reduction=EPISODE_TOKEN_REDUCTION,
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


@dataclass(frozen=True)
class EpisodeRecipe:
    """Per-batch generation knobs (set at start; fixed for all scenes)."""

    width: int = EPISODE_WIDTH
    height: int = EPISODE_HEIGHT
    render_width: int = EPISODE_RENDER_WIDTH
    render_height: int = EPISODE_RENDER_HEIGHT
    frames: int = EPISODE_FRAMES
    steps: int = EPISODE_STEPS
    layers: int = EPISODE_LAYERS
    reuse: int = EPISODE_REUSE
    token_reduction: bool = EPISODE_TOKEN_REDUCTION
    duration_s: int = 5

    def label(self) -> str:
        return episode_recipe_label(
            width=self.width,
            height=self.height,
            render_width=self.render_width,
            render_height=self.render_height,
            frames=self.frames,
            steps=self.steps,
            layers=self.layers,
            reuse=self.reuse,
            token_reduction=self.token_reduction,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "frames": self.frames,
            "steps": self.steps,
            "layers": self.layers,
            "reuse": self.reuse,
            "token_reduction": self.token_reduction,
            "duration_s": self.duration_s,
            "label": self.label(),
        }


def parse_episode_recipe(body: dict[str, Any] | None) -> EpisodeRecipe:
    """Validate dashboard / API episode knobs. Missing fields keep sharp defaults."""
    raw = body or {}
    duration_s = int(raw.get("duration_s") or 5)
    if duration_s not in EPISODE_DURATION_CHOICES:
        raise ValueError(
            f"duration_s must be one of {EPISODE_DURATION_CHOICES}, got {duration_s}"
        )
    frames = episode_frames_for_seconds(duration_s)

    render = int(raw.get("render_width") or raw.get("render") or EPISODE_RENDER_WIDTH)
    if render not in EPISODE_RENDER_CHOICES:
        raise ValueError(
            f"render must be one of {EPISODE_RENDER_CHOICES}, got {render}"
        )

    steps = int(raw.get("steps") if raw.get("steps") is not None else EPISODE_STEPS)
    if not (EPISODE_MIN_STEPS <= steps <= EPISODE_MAX_STEPS):
        raise ValueError(
            f"steps must be {EPISODE_MIN_STEPS}–{EPISODE_MAX_STEPS}, got {steps}"
        )

    layers = int(raw.get("layers") if raw.get("layers") is not None else EPISODE_LAYERS)
    if not (EPISODE_MIN_LAYERS <= layers <= EPISODE_MAX_LAYERS):
        raise ValueError(
            f"layers must be {EPISODE_MIN_LAYERS}–{EPISODE_MAX_LAYERS}, got {layers}"
        )

    reuse = int(raw.get("reuse") if raw.get("reuse") is not None else EPISODE_REUSE)
    if not (EPISODE_MIN_REUSE <= reuse <= EPISODE_MAX_REUSE):
        raise ValueError(
            f"reuse must be {EPISODE_MIN_REUSE}–{EPISODE_MAX_REUSE}, got {reuse}"
        )

    token_reduction = bool(raw.get("token_reduction", EPISODE_TOKEN_REDUCTION))
    # h3.c: do not combine TR with layers 40 + reuse 3
    if token_reduction and layers == 40 and reuse == 3:
        raise ValueError("token-reduction cannot be combined with layers 40 and reuse 3")

    return EpisodeRecipe(
        width=EPISODE_WIDTH,
        height=EPISODE_HEIGHT,
        render_width=render,
        render_height=render,
        frames=frames,
        steps=steps,
        layers=layers,
        reuse=reuse,
        token_reduction=token_reduction,
        duration_s=duration_s,
    )


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
    recipe: EpisodeRecipe = field(default_factory=EpisodeRecipe)
    cancel: threading.Event = field(default_factory=threading.Event)
    # Scene clip paths kept until next job clears them.
    clip_paths: list[Path] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.status == "running"

    def reset_for_start(self, scenes: int, recipe: EpisodeRecipe | None = None) -> None:
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
        self.recipe = recipe or EpisodeRecipe()

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
            "recipe": self.recipe.label(),
            "recipe_detail": self.recipe.as_dict(),
            "download_ready": ready,
            "download_url": "/api/episode.mp4" if ready else None,
            "bytes": size,
            "elapsed_s": elapsed,
            "last_cast": self.last_cast,
            "duration_choices": list(EPISODE_DURATION_CHOICES),
            "render_choices": list(EPISODE_RENDER_CHOICES),
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
