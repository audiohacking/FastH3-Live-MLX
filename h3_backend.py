"""One-shot h3.c process manager plus a warm FL2VA interactive session."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from h3_media import (
    assert_audio_durations,
    ffmpeg_ok,
    probe_duration_seconds,
    require_ui_canvas,
    snap_frames,
)
from h3_paths import (
    console_h3,
    default_h3_bin,
    default_model_dir,
    h3_media_env,
    h3_process_cwd,
    mk_scratch_file,
)

log = logging.getLogger("h3-backend")

ProgressCallback = Callable[[dict[str, Any]], None]


class GenerationCancelledError(RuntimeError):
    """Raised when the user or SIGINT cancelled an in-flight ``h3`` process."""


# h3.c CLI defaults. Close keeps --steps 50 explicit: 50 complete 50-block
# denoiser forwards, vs 20×50 for the default path.
H3_DEFAULT_STEPS = 20
H3_DEFAULT_LAYERS = 50
H3_DEFAULT_REUSE = 1

QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "four_step": {
        "id": "four_step",
        "label": "Four-step",
        "steps": 4,
        "layers": H3_DEFAULT_LAYERS,
        "reuse": H3_DEFAULT_REUSE,
        "core_reuse": None,
        "token_reduction": False,
        "render": None,
    },
    "aggressive": {
        "id": "aggressive",
        "label": "Aggressive preview",
        "steps": H3_DEFAULT_STEPS,
        "layers": 40,
        "reuse": 3,
        "core_reuse": None,
        "token_reduction": False,
        "render": (320, 320),  # only applied when output is 512×512
    },
    "fast": {
        "id": "fast",
        "label": "Fast",
        "steps": H3_DEFAULT_STEPS,
        "layers": 45,
        "reuse": 2,
        "core_reuse": None,
        "token_reduction": True,
        "render": (384, 384),  # only applied when output is 512×512
    },
    "balanced": {
        "id": "balanced",
        "label": "Balanced",
        "steps": H3_DEFAULT_STEPS,
        "layers": H3_DEFAULT_LAYERS,
        "reuse": H3_DEFAULT_REUSE,
        "core_reuse": None,
        "token_reduction": False,
        "render": None,
    },
    "close": {
        "id": "close",
        "label": "Close / reference",
        "steps": 50,
        "layers": H3_DEFAULT_LAYERS,
        "reuse": H3_DEFAULT_REUSE,
        "core_reuse": None,
        "token_reduction": False,
        "render": None,
        "guidance": (
            "50 complete 50-block denoiser forwards — much more expensive than "
            "the default 20×50, but the right oracle when a fast mode changes "
            "subject, anatomy, motion, or composition."
        ),
    },
}

QUALITY_PRESET_LIST = [
    {
        "id": p["id"],
        "label": p["label"],
        "steps": p["steps"],
        "layers": p["layers"],
        "reuse": p["reuse"],
        "token_reduction": p["token_reduction"],
        "guidance": p.get("guidance"),
    }
    for p in QUALITY_PRESETS.values()
]

GENERATION_MODES = [
    {"id": "t2va", "label": "Text to video+audio"},
    {"id": "first_frame", "label": "First frame → video"},
    {"id": "last_frame", "label": "Last frame → video"},
    {"id": "fl2va", "label": "First and last frame"},
    {"id": "ref2va", "label": "Ordered references (Ref2VA)"},
]

_STEP_RE = re.compile(
    r"(?:step|pass|denois\w*)[^\d]{0,12}(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)


def _format_mmss(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    return f"{s // 60:02d}:{s % 60:02d}"


class ProgressEta:
    """tqdm-style remaining time from h3.c ``phase completed/total`` lines.

    h3.c does not print ETA. ``--profile`` is after-the-fact Metal timings.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stage: str | None = None
        self.t0: float = 0.0

    def enrich(self, mp: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
        clock = time.time() if now is None else now
        stage = str(mp.get("stage") or "")
        step = mp.get("step")
        total = mp.get("total")
        if stage != self.stage:
            self.stage = stage
            self.t0 = clock
        if not isinstance(step, int) or not isinstance(total, int) or total <= 0:
            return mp
        if "pct" not in mp:
            mp["pct"] = round(100.0 * step / total, 1)
        elapsed = clock - self.t0
        if step <= 0 or elapsed <= 0:
            return mp
        rate = step / elapsed
        mp["avg_step_s"] = round(1.0 / rate, 2)
        mp["eta_s"] = round(max(0, total - step) / rate, 1)
        return mp


def progress_console_line(mp: dict[str, Any]) -> str:
    label = str(mp.get("label") or mp.get("stage") or "h3")
    parts = [label]
    if mp.get("eta_s") is not None:
        parts.append(f"{_format_mmss(float(mp['eta_s']))} remaining")
    if mp.get("avg_step_s") is not None:
        parts.append(f"{mp['avg_step_s']}s/it")
    return "  ".join(parts)


def physical_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None
    return None


def ram_gb() -> float | None:
    n = physical_memory_bytes()
    return None if n is None else n / (1024**3)


def recommend_ssd_streaming(ram: float | None = None) -> bool:
    """True when RAM is tight. Do not auto-enable: SSD streaming is much slower."""
    gb = ram if ram is not None else ram_gb()
    if gb is None:
        return False
    return gb < 64.0


def apple_chip_brand() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def metal4_gpu(brand: str | None = None) -> bool:
    """True when h3.c TensorOps / ``--use-int8-row-fc2`` class is expected.

    Historically M5-only. M3 Ultra / M4 report Metal 4; the local h3 fork now
    enables TensorOps on those chips too, so treat them as eligible.
    """
    text = brand if brand is not None else apple_chip_brand()
    if re.search(r"\bM(?:4|5|6|7)\b", text):
        return True
    if re.search(r"M3\s*Ultra", text, re.I):
        return True
    return False


def resolve_int8_row_fc2(req: "GenerateRequest", *, metal4: bool) -> bool:
    if req.ssd_streaming:
        return False
    if req.int8_row_fc2 is not None:
        return bool(req.int8_row_fc2)
    return bool(metal4)


def fl2va_dir(model_dir: Path) -> Path:
    return model_dir / "FL2VA"


def ref2va_dir(model_dir: Path) -> Path:
    return model_dir / "Ref2VA"


def _has_safetensors(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.safetensors"))


def model_layout_ok(model_dir: Path, *, need_ref2va: bool = False) -> tuple[bool, str]:
    root = Path(model_dir)
    if not root.is_dir():
        return False, f"model dir not found: {root}"
    fl = fl2va_dir(root)
    if not fl.is_dir():
        return False, f"FL2VA checkpoint missing under {root}"
    required = (
        fl / "transformer" / "config.json",
        fl / "transformer" / "model.safetensors.index.json",
        fl / "tokenizer" / "tokenizer.json",
        fl / "text_encoder" / "model.safetensors.index.json",
    )
    for path in required:
        if not path.is_file():
            return False, f"incomplete FL2VA checkpoint: missing {path}"
    if not _has_safetensors(fl / "transformer"):
        return False, f"incomplete FL2VA transformer shards under {fl / 'transformer'}"
    if not _has_safetensors(fl / "text_encoder"):
        return False, f"incomplete FL2VA text encoder under {fl / 'text_encoder'}"
    if not _has_safetensors(fl / "video_vae" / "source"):
        return False, f"incomplete FL2VA video VAE under {fl / 'video_vae' / 'source'}"
    if not _has_safetensors(fl / "audio_vae"):
        return False, f"incomplete FL2VA audio VAE under {fl / 'audio_vae'}"
    if need_ref2va:
        rf = ref2va_dir(root)
        if not _has_safetensors(rf / "transformer"):
            return False, f"Ref2VA transformer missing under {rf / 'transformer'}"
    return True, str(root)


def expand_quality(
    quality: str,
    *,
    steps: int | None = None,
    layers: int | None = None,
    reuse: int | None = None,
    core_reuse: int | None = None,
    token_reduction: bool | None = None,
    width: int = 512,
    height: int = 512,
    render_width: int | None = None,
    render_height: int | None = None,
) -> dict[str, Any]:
    preset = QUALITY_PRESETS.get((quality or "balanced").strip().lower())
    if preset is None:
        preset = QUALITY_PRESETS["balanced"]
    out = dict(preset)
    if steps is not None:
        out["steps"] = max(1, int(steps))
    if layers is not None:
        out["layers"] = max(1, min(H3_DEFAULT_LAYERS, int(layers)))
    if reuse is not None:
        out["reuse"] = max(1, int(reuse))
    if core_reuse is not None:
        out["core_reuse"] = max(1, int(core_reuse))
        out["reuse"] = None
    if token_reduction is not None:
        out["token_reduction"] = bool(token_reduction)
    if width == 256 and height == 256:
        out["token_reduction"] = False
    rw, rh = render_width, render_height
    suggested = out.get("render")
    if rw is None and rh is None and suggested and width == 512 and height == 512:
        rw, rh = suggested
    if (rw is None) != (rh is None):
        raise ValueError("--render-width and --render-height must be set together")
    if rw is not None and rh is not None:
        if rw > width or rh > height:
            raise ValueError("internal render size cannot exceed output canvas")
        out["render"] = (int(rw), int(rh))
    else:
        out["render"] = None
    # h3.c: do not combine token-reduction with layers 40 + reuse 3
    if out.get("token_reduction") and out.get("layers") == 40 and out.get("reuse") == 3:
        raise ValueError(
            "token-reduction cannot be combined with --layers 40 and --reuse 3"
        )
    if out.get("reuse") and out.get("core_reuse"):
        raise ValueError("--reuse and --core-reuse are mutually exclusive")
    if int(out["steps"]) <= 7 and (out.get("reuse") or 1) > 1:
        out["reuse"] = 1
    return out


REF_KINDS = ("image", "silent_video", "video", "video_audio", "audio")
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIO = 3
MAX_REF_FILES = 12


@dataclass
class RefItem:
    """One ordered Ref2VA input. ``kind`` maps 1:1 to an h3.c flag."""

    kind: str
    path: Path
    audio_path: Path | None = None
    name: str = ""
    ref_size: str = "max"

    def file_count(self) -> int:
        if self.kind == "video_audio":
            return 2
        return 1


def refs_from_legacy(
    *,
    ref_images: list[Path] | None = None,
    ref_silent_videos: list[Path] | None = None,
    ref_videos: list[Path] | None = None,
    ref_audio: list[Path] | None = None,
) -> list[RefItem]:
    items: list[RefItem] = []
    for p in ref_images or []:
        items.append(RefItem(kind="image", path=Path(p)))
    for p in ref_silent_videos or []:
        items.append(RefItem(kind="silent_video", path=Path(p)))
    for p in ref_videos or []:
        items.append(RefItem(kind="video", path=Path(p)))
    for p in ref_audio or []:
        items.append(RefItem(kind="audio", path=Path(p)))
    return items


def validate_refs(refs: list[RefItem]) -> None:
    n_img = n_vid = n_aud = n_files = 0
    has_visual = False
    for item in refs:
        kind = (item.kind or "").strip().lower()
        if kind not in REF_KINDS:
            raise ValueError(f"unknown reference kind: {item.kind}")
        if not item.path:
            raise ValueError("reference path is required")
        n_files += item.file_count()
        if kind == "image":
            n_img += 1
            has_visual = True
        elif kind in ("silent_video", "video", "video_audio"):
            n_vid += 1
            has_visual = True
            if kind == "video_audio":
                n_aud += 1
                if item.audio_path is None:
                    raise ValueError("--ref-video-audio requires a replacement audio file")
        elif kind == "audio":
            n_aud += 1
    if n_img > MAX_REF_IMAGES:
        raise ValueError(f"at most {MAX_REF_IMAGES} reference images")
    if n_vid > MAX_REF_VIDEOS:
        raise ValueError(f"at most {MAX_REF_VIDEOS} reference videos")
    if n_aud > MAX_REF_AUDIO:
        raise ValueError(f"at most {MAX_REF_AUDIO} audio references")
    if n_files > MAX_REF_FILES:
        raise ValueError(f"at most {MAX_REF_FILES} mixed reference files")
    if n_aud and not has_visual:
        raise ValueError("standalone audio must accompany an image or video reference")
    _validate_ref_audio_durations(refs)


def parse_refs_payload(raw: Any) -> list[RefItem]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("refs must be a list")
    items: list[RefItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each ref must be an object")
        kind = str(entry.get("kind") or "").strip().lower()
        path = str(entry.get("path") or "").strip()
        if not kind or not path:
            raise ValueError("each ref needs kind and path")
        audio = str(entry.get("audio_path") or "").strip()
        items.append(
            RefItem(
                kind=kind,
                path=Path(path),
                audio_path=Path(audio) if audio else None,
                name=str(entry.get("name") or Path(path).name),
                ref_size=str(entry.get("ref_size") or "max").strip().lower() or "max",
            )
        )
    validate_refs(items)
    return items


def _audio_paths_for_ref(item: RefItem) -> list[Path]:
    if item.kind == "audio":
        return [item.path]
    if item.kind == "video_audio" and item.audio_path is not None:
        return [item.audio_path]
    if item.kind == "video":
        return [item.path]
    return []


def _validate_ref_audio_durations(refs: list[RefItem]) -> None:
    seconds: list[float] = []
    for item in refs:
        for path in _audio_paths_for_ref(item):
            duration = probe_duration_seconds(path)
            if duration is None:
                continue
            seconds.append(duration)
    if seconds:
        assert_audio_durations(seconds)


def append_ref_flags(cmd: list[str], refs: list[RefItem]) -> None:
    """Emit flags in list order so the model sees Picture 1 / Video 1 correctly."""
    for item in refs:
        if item.kind == "image":
            cmd.extend(["--ref-image", str(item.path)])
        elif item.kind == "silent_video":
            cmd.extend(["--ref-silent-video", str(item.path)])
        elif item.kind == "video":
            cmd.extend(["--ref-video", str(item.path)])
        elif item.kind == "video_audio":
            cmd.extend(["--ref-video-audio", str(item.path), str(item.audio_path)])
        elif item.kind == "audio":
            cmd.extend(["--ref-audio", str(item.path)])


@dataclass
class LoraRef:
    spec: str
    path: Path
    scale: float = 1.0


@dataclass
class GenerateRequest:
    prompt: str
    output_path: Path
    width: int = 512
    height: int = 512
    num_frames: int = 22
    quality: str = "balanced"
    steps: int | None = None
    layers: int | None = None
    reuse: int | None = None
    core_reuse: int | None = None
    token_reduction: bool | None = None
    render_width: int | None = None
    render_height: int | None = None
    seed: int | None = None
    ssd_streaming: bool = False
    int8_row_fc2: bool | None = None
    first_frame: Path | None = None
    last_frame: Path | None = None
    refs: list[RefItem] = field(default_factory=list)
    loras: list[LoraRef] = field(default_factory=list)
    mode: str = "t2va"
    profile: bool = True
    # Skip interactive PTY — linenoise deadlocks on long FastH3 Live prompts.
    oneshot: bool = False


def uses_ref2va(req: GenerateRequest) -> bool:
    return bool(req.refs) or req.mode == "ref2va"


def uses_fl2va_anchors(req: GenerateRequest) -> bool:
    return req.first_frame is not None or req.last_frame is not None


def build_h3_argv(
    *,
    h3_bin: Path,
    model_dir: Path,
    req: GenerateRequest,
) -> list[str]:
    width, height = require_ui_canvas(req.width, req.height)
    frames = snap_frames(req.num_frames)
    q = expand_quality(
        req.quality,
        steps=req.steps,
        layers=req.layers,
        reuse=req.reuse,
        core_reuse=req.core_reuse,
        token_reduction=req.token_reduction,
        width=width,
        height=height,
        render_width=req.render_width,
        render_height=req.render_height,
    )
    if req.loras and req.ssd_streaming:
        raise ValueError("LoRA cannot be combined with --ssd-streaming")
    if req.refs:
        validate_refs(req.refs)
    if uses_ref2va(req) and uses_fl2va_anchors(req):
        raise ValueError("Ref2VA references cannot be mixed with first/last-frame anchors")
    if req.mode == "ref2va" and not req.refs:
        raise ValueError("ref2va requires at least one image, video, or audio reference")

    cmd: list[str] = [
        str(h3_bin),
        "-d",
        str(model_dir),
        "-p",
        req.prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--frames",
        str(frames),
        "--steps",
        str(q["steps"]),
        "--layers",
        str(q["layers"]),
        "-o",
        str(req.output_path),
    ]
    if q.get("core_reuse"):
        cmd.extend(["--core-reuse", str(q["core_reuse"])])
    elif q.get("reuse"):
        cmd.extend(["--reuse", str(q["reuse"])])
    if q.get("token_reduction"):
        cmd.append("--token-reduction")
    render = q.get("render")
    if render:
        cmd.extend(["--render-width", str(render[0]), "--render-height", str(render[1])])
    if req.seed is not None and int(req.seed) >= 0:
        cmd.extend(["--seed", str(int(req.seed))])
    if req.ssd_streaming:
        cmd.append("--ssd-streaming")
    elif req.int8_row_fc2:
        cmd.append("--use-int8-row-fc2")
    for lora in req.loras:
        cmd.extend(["--lora", f"{lora.path}:{lora.scale:.4g}"])
    if req.profile:
        cmd.append("--profile")
    if req.first_frame:
        cmd.extend(["--first-frame", str(req.first_frame)])
    if req.last_frame:
        cmd.extend(["--last-frame", str(req.last_frame)])
    append_ref_flags(cmd, req.refs)
    return cmd


class H3Engine:
    """Spawns ``./h3`` per job. One generation at a time is enforced by the caller."""

    def __init__(
        self,
        h3_bin: Path | None = None,
        model_dir: Path | None = None,
        *,
        default_ssd_streaming: bool | None = None,
    ) -> None:
        self.h3_bin = Path(h3_bin) if h3_bin else default_h3_bin()
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        if default_ssd_streaming is None:
            default_ssd_streaming = False
        self.default_ssd_streaming = bool(default_ssd_streaming)
        self.metal4 = metal4_gpu()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._cancel = threading.Event()
        self._progress: dict[str, Any] = {}
        self._t0 = 0.0
        self._session: Any | None = None
        self._session_lora_key: tuple[tuple[str, float], ...] = ()
        self._eta = ProgressEta()

    def model_progress_for_ws(self) -> dict[str, Any] | None:
        p = dict(self._progress)
        return p or None

    def request_cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        session = self._session
        if session is not None:
            try:
                session.stop()
            except Exception:
                pass
            self._session = None

    def shutdown(self, wait: bool = True) -> None:
        self.request_cancel()
        session = self._session
        self._session = None
        if session is not None:
            try:
                session.stop()
            except Exception:
                pass
        proc = self._proc
        if proc is None:
            return
        if wait:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._proc = None

    def info(self) -> dict[str, Any]:
        if not self.h3_bin.is_file():
            return {
                "ok": False,
                "error": f"h3 binary not found at {self.h3_bin} (run scripts/build_h3.sh)",
            }
        ok, note = model_layout_ok(self.model_dir)
        if not ok:
            return {"ok": False, "error": note, "h3_bin": str(self.h3_bin)}
        try:
            proc = subprocess.run(
                [str(self.h3_bin), "--info", "-d", str(self.model_dir)],
                capture_output=True,
                text=True,
                timeout=60,
                env=h3_media_env(),
                cwd=str(h3_process_cwd(self.h3_bin)),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc), "h3_bin": str(self.h3_bin)}
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        media_ok, media_note = ffmpeg_ok()
        if proc.returncode == 0 and not media_ok:
            return {
                "ok": False,
                "error": media_note,
                "h3_bin": str(self.h3_bin),
                "model_dir": str(self.model_dir),
            }
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "output": text.strip()[-4000:],
            "h3_bin": str(self.h3_bin),
            "model_dir": str(self.model_dir),
            "ram_gb": ram_gb(),
            "recommend_ssd_streaming": recommend_ssd_streaming(),
            "metal4": self.metal4,
            "warm_session": bool(self._session and getattr(self._session, "alive", False)),
        }

    def _parse_line(self, line: str, steps: int) -> None:
        line = line.strip()
        if not line:
            return
        elapsed = round(time.time() - self._t0, 1)
        mp: dict[str, Any] = {
            "stage": "generating",
            "elapsed_s": elapsed,
            "label": line[:160],
        }
        m = _STEP_RE.search(line)
        phase_m = re.match(r"(.{1,40}?)\s+(\d+)/(\d+)\s*$", line)
        if phase_m:
            phase = phase_m.group(1).strip()
            step, total = int(phase_m.group(2)), int(phase_m.group(3))
            mp["stage"] = phase or "generating"
            mp["step"] = step
            mp["total"] = total
            mp["label"] = f"{phase} {step}/{total}"
            if total > 0:
                mp["pct"] = round(100.0 * step / total, 1)
        elif m:
            step, total = int(m.group(1)), int(m.group(2))
            mp["step"] = step
            mp["total"] = total
            if total > 0:
                mp["pct"] = round(100.0 * step / total, 1)
        elif steps > 0:
            mp["total"] = steps
        self._eta.enrich(mp)
        self._progress = mp

    def generate(
        self,
        req: GenerateRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        if not self.h3_bin.is_file():
            raise FileNotFoundError(
                f"h3 binary not found at {self.h3_bin}. Run scripts/build_h3.sh"
            )
        need_ref = uses_ref2va(req)
        ok, note = model_layout_ok(self.model_dir, need_ref2va=need_ref)
        if not ok:
            raise FileNotFoundError(note)
        media_ok, media_note = ffmpeg_ok()
        if not media_ok:
            raise FileNotFoundError(media_note)

        req.output_path = Path(req.output_path)
        req.output_path.parent.mkdir(parents=True, exist_ok=True)
        req.int8_row_fc2 = resolve_int8_row_fc2(req, metal4=self.metal4)

        if need_ref or req.oneshot:
            self._stop_session()
            return self._generate_oneshot(req, on_progress=on_progress)

        from h3_session import SessionError

        try:
            return self._generate_session(req, on_progress=on_progress)
        except GenerationCancelledError:
            self._stop_session()
            raise
        except SessionError as exc:
            log.warning("warm session unavailable (%s) — one-shot fallback", exc)
            self._stop_session()
            return self._generate_oneshot(req, on_progress=on_progress)

    def _stop_session(self) -> None:
        session = self._session
        self._session = None
        self._session_lora_key = ()
        if session is None:
            return
        try:
            session.stop()
        except Exception as exc:
            log.debug("session stop: %s", exc)

    def _generate_session(
        self,
        req: GenerateRequest,
        *,
        on_progress: ProgressCallback | None,
    ) -> str:
        from h3_session import (
            H3InteractiveSession,
            build_session_argv,
            copy_session_output,
        )

        q = expand_quality(
            req.quality,
            steps=req.steps,
            layers=req.layers,
            reuse=req.reuse,
            core_reuse=req.core_reuse,
            token_reduction=req.token_reduction,
            width=req.width,
            height=req.height,
            render_width=req.render_width,
            render_height=req.render_height,
        )
        self._cancel.clear()
        self._t0 = time.time()
        self._eta.reset()
        self._progress = {"stage": "starting", "elapsed_s": 0, "total": q["steps"]}
        if on_progress:
            on_progress(self._progress)

        def _on_progress(mp: dict[str, Any]) -> None:
            mp = dict(mp)
            mp["elapsed_s"] = round(time.time() - self._t0, 1)
            self._eta.enrich(mp)
            self._progress = mp
            console_h3("%s", progress_console_line(mp))
            if on_progress:
                on_progress(mp)

        lora_key = tuple((str(item.path), float(item.scale)) for item in req.loras)
        session = self._session
        if (
            session is None
            or not getattr(session, "alive", False)
            or self._session_lora_key != lora_key
        ):
            self._stop_session()
            argv = build_session_argv(
                h3_bin=self.h3_bin, model_dir=self.model_dir, req=req
            )
            log.info("h3 session argv: %s", " ".join(argv[:8]) + " …")
            env = h3_media_env()
            log.info("h3 muxer: %s", env.get("H3_FFMPEG") or env.get("H3_AV"))
            if req.loras:
                log.info(
                    "h3 lora: %s",
                    ", ".join(f"{item.path.name}@{item.scale}" for item in req.loras),
                )
            if req.profile:
                env["H3_PROFILE"] = "1"
            session = H3InteractiveSession(cancel=self._cancel)
            session.start(argv, env=env, cwd=h3_process_cwd(self.h3_bin))
            self._session = session
            self._session_lora_key = lora_key
        session.apply_request(req)
        produced = session.generate(req.prompt, on_progress=_on_progress)
        copy_session_output(produced, req.output_path)
        if not req.output_path.is_file() or req.output_path.stat().st_size < 32:
            raise RuntimeError(f"h3 did not write an MP4 at {req.output_path}")
        return str(req.output_path)

    def _generate_oneshot(
        self,
        req: GenerateRequest,
        *,
        on_progress: ProgressCallback | None,
    ) -> str:
        q = expand_quality(
            req.quality,
            steps=req.steps,
            layers=req.layers,
            reuse=req.reuse,
            core_reuse=req.core_reuse,
            token_reduction=req.token_reduction,
            width=req.width,
            height=req.height,
            render_width=req.render_width,
            render_height=req.render_height,
        )
        argv = build_h3_argv(h3_bin=self.h3_bin, model_dir=self.model_dir, req=req)
        log.info("h3 argv: %s", " ".join(argv[:8]) + " …")

        self._cancel.clear()
        self._t0 = time.time()
        self._eta.reset()
        self._progress = {"stage": "starting", "elapsed_s": 0, "total": q["steps"]}
        if on_progress:
            on_progress(self._progress)

        env = h3_media_env()
        log.info("h3 muxer: %s", env.get("H3_FFMPEG") or env.get("H3_AV"))
        try:
            with self._lock:
                proc = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    cwd=str(h3_process_cwd(self.h3_bin)),
                )
                self._proc = proc
        except OSError as exc:
            raise RuntimeError(f"failed to spawn h3: {exc}") from exc

        assert proc.stdout is not None
        tail: list[str] = []
        try:
            for line in proc.stdout:
                if self._cancel.is_set():
                    proc.terminate()
                    raise GenerationCancelledError("cancelled")
                self._parse_line(line, int(q["steps"]))
                if on_progress:
                    on_progress(self._progress)
                stripped = line.rstrip()
                if stripped:
                    tail.append(stripped)
                    if len(tail) > 80:
                        del tail[:-80]
                    if self._progress.get("step") is not None:
                        console_h3("%s", progress_console_line(self._progress))
                    else:
                        console_h3("%s", stripped)
            rc = proc.wait()
        finally:
            self._proc = None

        if self._cancel.is_set():
            raise GenerationCancelledError("cancelled")
        if rc != 0:
            detail = "\n".join(tail[-40:]) if tail else "(h3 printed nothing)"
            raise RuntimeError(f"h3 exited {rc}\n{detail}")
        if not req.output_path.is_file() or req.output_path.stat().st_size < 32:
            raise RuntimeError(f"h3 did not write an MP4 at {req.output_path}")
        return str(req.output_path)


def scratch_output(prefix: str = "h3_") -> Path:
    fd, path = mk_scratch_file(prefix, ".mp4")
    os.close(fd)
    return Path(path)
