"""HTTP API, library, and job orchestration for h3-ws."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from h3_bootstrap import ensure_python_requirements

ensure_python_requirements()
from starlette.requests import Request  # noqa: E402

from h3_backend import (
    GENERATION_MODES,
    H3_DEFAULT_LAYERS,
    H3_DEFAULT_REUSE,
    H3_DEFAULT_STEPS,
    QUALITY_PRESET_LIST,
    GenerateRequest,
    GenerationCancelledError,
    H3Engine,
    LoraRef,
    fl2va_dir,
    model_layout_ok,
    parse_refs_payload,
    ram_gb,
    recommend_ssd_streaming,
    ref2va_dir,
)
from h3_media import (
    DURATION_PRESETS,
    FPS,
    RESOLUTION_PRESETS,
    concat_mp4s,
    extract_last_frame,
    media_available,
    require_ui_canvas,
    sanitize_filename,
    seconds_to_frames,
    snap_frames,
    validate_canvas,
)
from h3_paths import REPO_ROOT, configure_scratch_root, mk_scratch_dir
from h3_lora import (
    ensure_lora,
    lora_catalog,
    normalize_lora_spec,
    read_custom_loras,
    write_custom_loras,
    _label_for_spec,
)

log = logging.getLogger("h3-web")

INDEX_FILE = "index.json"
SETTINGS_FILE = "settings.json"
USER_DATA_FILE = "user_data.json"
CLIP_MULTIPLIER_MAX = 10
DEFAULT_OUTPUT_DIR = REPO_ROOT / "web_outputs"
DEFAULT_UPLOAD_DIR = REPO_ROOT / "web_uploads"
PROGRESS_KEEPALIVE_INTERVAL_S = 1.0

_RUN_BODIES: dict[str, dict[str, Any]] = {}


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ClipRecord:
    id: str
    prompt: str
    label: str
    video_url: str
    filename: str
    chain_id: str
    clip_index: int
    mode: str
    status: str
    created_at: str
    elapsed_s: Optional[float] = None
    bytes: Optional[int] = None
    error: Optional[str] = None
    num_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    num_steps: Optional[int] = None
    layers: Optional[int] = None
    reuse: Optional[int] = None
    duration_seconds: Optional[float] = None
    clip_count: Optional[int] = None
    autocontinue: Optional[bool] = None
    autoconcat: Optional[bool] = None
    quality: Optional[str] = None
    loras: Optional[list[dict[str, Any]]] = None


@dataclass
class RunRecord:
    id: str
    status: str
    prompts: list[str]
    chain_id: str
    clip_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    error: Optional[str] = None
    autocontinue: bool = False
    autoconcat: bool = False
    merged_url: Optional[str] = None
    merged_clip_id: Optional[str] = None


class AppState:
    def __init__(
        self,
        output_dir: Path,
        upload_dir: Path,
        engine: H3Engine,
        *,
        embedded: bool = True,
        http_url: str = "",
        server_url: str = "",
        runtime_defaults: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.upload_dir = Path(upload_dir)
        self.engine = engine
        self.embedded = embedded
        self.http_url = http_url
        self.server_url = server_url
        self.runtime_defaults = runtime_defaults or {}
        configure_scratch_root(self.output_dir / ".scratch")
        self.runs: dict[str, RunRecord] = {}
        self.clips: dict[str, ClipRecord] = {}
        self.event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._submit_lock = asyncio.Lock()
        self._worker_started = False
        self._worker_task: asyncio.Task[None] | None = None
        self._cancelled_runs: set[str] = set()
        self._active_run_id: str | None = None
        self._sigint_count = 0
        self._sigint_last_ts = 0.0
        self._uvicorn_server: Any = None
        self.user_data_path = self.output_dir / USER_DATA_FILE
        self.presets: list[dict[str, Any]] = []
        self.cast_members: list[dict[str, Any]] = []
        self._load_user_data()

    # ── User data (presets + cast) persistence ───────────────────────────────
    # Stored separately from the generation index so presets/cast survive
    # server restarts and are not wiped by session clear.

    def _load_user_data(self) -> None:
        try:
            if not self.user_data_path.is_file():
                return
            data = json.loads(self.user_data_path.read_text(encoding="utf-8"))
            self.presets = data.get("presets", []) or []
            self.cast_members = data.get("cast", []) or []
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read user data at %s", self.user_data_path)

    def _save_user_data(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            data = {"presets": self.presets, "cast": self.cast_members}
            self.user_data_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Could not save user data: %s", exc)

    def is_generation_active(self) -> bool:
        return self._active_run_id is not None

    def is_pipeline_idle(self) -> bool:
        return self._active_run_id is None and self._pending.qsize() == 0

    async def enqueue_generation_run(self, run_id: str) -> bool:
        async with self._submit_lock:
            idle = self.is_pipeline_idle()
            run = self.runs.get(run_id)
            if run is not None:
                run.status = RunStatus.RUNNING.value if idle else RunStatus.QUEUED.value
            await self._pending.put(run_id)
            return idle

    def request_shutdown(self) -> None:
        uv = self._uvicorn_server
        if uv is not None:
            uv.should_exit = True

    def on_console_interrupt(self) -> None:
        if not self.is_generation_active():
            log.info("Shutting down…")
            self.request_shutdown()
            return
        now = time.monotonic()
        if now - self._sigint_last_ts > 2.0:
            self._sigint_count = 0
        self._sigint_last_ts = now
        self._sigint_count += 1
        self.engine.request_cancel()
        if self._active_run_id:
            self._cancelled_runs.add(self._active_run_id)
        if self._sigint_count == 1:
            log.warning(
                "Interrupt received — cancelling generation "
                "(press Ctrl+C again within 2s to force quit)"
            )
        else:
            log.warning("Force quit")
            self.engine.shutdown(wait=False)
            os._exit(130)

    def is_run_cancelled(self, run_id: str) -> bool:
        return run_id in self._cancelled_runs

    def request_cancel_run(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run:
            return False
        if run.status in (
            RunStatus.DONE.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        ):
            return False
        self._cancelled_runs.add(run_id)
        if self._active_run_id == run_id:
            self.engine.request_cancel()
        return True

    def ensure_worker(self) -> None:
        task = self._worker_task
        if self._worker_started and task is not None and not task.done():
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.load_index()
        self._worker_task = asyncio.create_task(_worker_loop(self))
        self._worker_started = True

    def load_index(self) -> None:
        path = self.output_dir / INDEX_FILE
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for c in data.get("clips", []):
                self.clips[c["id"]] = ClipRecord(
                    **{k: v for k, v in c.items() if k in ClipRecord.__dataclass_fields__}
                )
            for r in data.get("runs", []):
                self.runs[r["id"]] = RunRecord(
                    **{k: v for k, v in r.items() if k in RunRecord.__dataclass_fields__}
                )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def save_index(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / INDEX_FILE
        data = {
            "clips": [asdict(c) for c in self.clips.values()],
            "runs": [asdict(r) for r in self.runs.values()],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clip_url(self, filename: str) -> str:
        return f"/api/videos/{filename}"

    def delete_clip_record(self, clip_id: str) -> bool:
        clip = self.clips.get(clip_id)
        if not clip:
            return False
        path = self.output_dir / clip.filename
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                log.warning("Could not delete clip file %s: %s", path, exc)
        del self.clips[clip_id]
        for run in list(self.runs.values()):
            if clip_id in run.clip_ids:
                run.clip_ids = [cid for cid in run.clip_ids if cid != clip_id]
        return True

    def delete_chain(self, chain_id: str) -> int:
        removed = 0
        for clip_id, clip in list(self.clips.items()):
            if clip.chain_id == chain_id:
                if self.delete_clip_record(clip_id):
                    removed += 1
        for run_id, run in list(self.runs.items()):
            if run.chain_id == chain_id:
                del self.runs[run_id]
        self.save_index()
        return removed

    def clear_session(self) -> dict[str, int]:
        deleted_files = 0
        seen: set[Path] = set()
        for clip in list(self.clips.values()):
            if clip.filename:
                path = self.output_dir / clip.filename
                if path.is_file() and path not in seen:
                    try:
                        path.unlink()
                        deleted_files += 1
                        seen.add(path)
                    except OSError as exc:
                        log.warning("Could not delete clip file %s: %s", path, exc)
        for path in self.output_dir.glob("*.mp4"):
            if path.is_file() and path not in seen:
                try:
                    path.unlink()
                    deleted_files += 1
                except OSError:
                    pass
        clip_count = len(self.clips)
        self.clips.clear()
        self.runs.clear()
        self.event_queues.clear()
        self.save_index()
        return {"deleted_clips": clip_count, "deleted_files": deleted_files}

    async def emit(self, run_id: str, event: dict[str, Any]) -> None:
        q = self.event_queues.get(run_id)
        if q:
            await q.put(event)


def _clip_for_api(state: AppState, clip: ClipRecord) -> dict[str, Any]:
    data = asdict(clip)
    filename = str(data.get("filename") or "").strip()
    if filename:
        file_path = state.output_dir / filename
        if file_path.is_file():
            data["path"] = str(file_path)
            if not data.get("video_url"):
                data["video_url"] = state.clip_url(filename)
        else:
            data["video_url"] = ""
    return data


def read_web_settings(output_dir: Path) -> dict[str, Any]:
    path = output_dir / SETTINGS_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def write_web_settings(output_dir: Path, data: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SETTINGS_FILE).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _frames_dir(output_dir: Path) -> Path:
    return output_dir / "frames"


def _read_frame_library(output_dir: Path) -> list[dict[str, Any]]:
    raw = read_web_settings(output_dir).get("frame_library")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        fid = str(item.get("id") or "").strip() or f"frame_{uuid.uuid4().hex[:8]}"
        filename = str(item.get("filename") or Path(path).name)
        entry: dict[str, Any] = {
            "id": fid,
            "label": str(item.get("label") or "Frame"),
            "path": path,
            "filename": filename,
            "created_at": str(item.get("created_at") or datetime.now().isoformat()),
        }
        for key in ("width", "height", "source_clip_id", "time_s"):
            if item.get(key) is not None:
                entry[key] = item[key]
        out.append(entry)
    return out


def _write_frame_library(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    data = read_web_settings(output_dir)
    data["frame_library"] = entries
    write_web_settings(output_dir, data)


def _frame_for_api(entry: dict[str, Any]) -> dict[str, Any]:
    filename = str(entry.get("filename") or Path(str(entry.get("path") or "")).name)
    out = dict(entry)
    out["image_url"] = f"/api/frames/files/{filename}"
    return out


def resolve_web_dist() -> Path:
    return REPO_ROOT / "web" / "dist"


def web_dist_stale() -> bool:
    dist = resolve_web_dist()
    if not dist.is_dir():
        return True
    assets = dist / "assets"
    js_files = list(assets.glob("index-*.js")) if assets.is_dir() else []
    if not js_files:
        return True
    newest_js = max(js_files, key=lambda path: path.stat().st_mtime)
    src_root = REPO_ROOT / "web" / "src"
    if not src_root.is_dir():
        return False
    try:
        newest_src = max(
            path.stat().st_mtime for path in src_root.rglob("*") if path.is_file()
        )
    except ValueError:
        return False
    return newest_src > newest_js.stat().st_mtime


def ensure_web_dist_built(*, auto_build: bool = True) -> bool:
    dist = resolve_web_dist()
    if dist.is_dir() and not web_dist_stale():
        return True
    if not auto_build:
        return dist.is_dir() and not web_dist_stale()
    web_dir = REPO_ROOT / "web"
    if not (web_dir / "package.json").is_file():
        return False
    npm = shutil.which("npm")
    if not npm:
        log.warning("web/dist missing and npm not found — run: cd web && npm run build")
        return False
    if not (web_dir / "node_modules").is_dir():
        log.info("Installing Web UI deps…")
        try:
            subprocess.run(
                [npm, "install", "--no-fund", "--no-audit"],
                cwd=str(web_dir),
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            log.warning("Web UI npm install failed: %s", exc)
            return False
    log.info("Building Web UI…")
    try:
        subprocess.run([npm, "run", "build"], cwd=str(web_dir), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("Web UI build failed: %s", exc)
        return False
    return dist.is_dir() and not web_dist_stale()


def local_hostname() -> str:
    try:
        name = socket.gethostname().strip().split(".")[0]
        if name:
            return name
    except OSError:
        pass
    return "localhost"


def public_host(bind_host: str) -> str:
    host = (bind_host or "").strip()
    if not host or host in ("0.0.0.0", "::", "[::]"):
        return local_hostname()
    return host


def build_server_urls(bind_host: str, port: int) -> tuple[str, str]:
    host = public_host(bind_host)
    return f"ws://{host}:{port}/ws", f"http://{host}:{port}/"


def urls_from_request(request: Any) -> tuple[str, str]:
    try:
        host = request.headers.get("host") or ""
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    except Exception:
        return "", ""
    if not host:
        return "", ""
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{host}/ws", f"{scheme}://{host}/"


def _upload_extension(kind: str, filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    allowed: dict[str, set[str]] = {
        "image": {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"},
        "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"},
        "video": {".mp4", ".mov", ".webm", ".mkv", ".avi"},
    }
    if ext and ext in allowed.get(kind, set()):
        return ext
    defaults = {"image": ".jpg", "audio": ".mp3", "video": ".mp4"}
    return defaults.get(kind, ".bin")


async def _save_upload_file(
    request: Request, upload_dir: Path, *, kind: str = "image"
) -> dict[str, str]:
    form = await request.form()
    upload_file = form.get("file")
    if upload_file is None:
        raise ValueError("file is required")
    read = getattr(upload_file, "read", None)
    if read is None:
        raise ValueError("file is required")
    filename = getattr(upload_file, "filename", None) or "upload.bin"
    ext = _upload_extension(kind, filename)
    dest = upload_dir / f"{uuid.uuid4()}{ext}"
    content = await read()
    dest.write_bytes(content)
    return {"path": str(dest), "filename": filename, "kind": kind}


def _clip_settings_from_body(body: dict[str, Any]) -> dict[str, Any]:
    width = int(body.get("width") or 512)
    height = int(body.get("height") or 512)
    try:
        width, height = validate_canvas(width, height)
    except ValueError:
        pass
    num_frames = body.get("num_frames")
    if num_frames is None and body.get("duration_seconds") is not None:
        num_frames = seconds_to_frames(float(body["duration_seconds"]))
    num_frames = snap_frames(int(num_frames or 22))
    seed = body.get("seed")
    try:
        seed_i = int(seed) if seed is not None and str(seed).strip() != "" else None
    except (TypeError, ValueError):
        seed_i = None
    return {
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "seed": seed_i,
        "num_steps": int(body.get("num_steps") or body.get("steps") or H3_DEFAULT_STEPS),
        "layers": int(body["layers"]) if body.get("layers") is not None else H3_DEFAULT_LAYERS,
        "reuse": int(body["reuse"]) if body.get("reuse") is not None else H3_DEFAULT_REUSE,
        "duration_seconds": float(body.get("duration_seconds") or num_frames / FPS),
        "clip_count": int(body.get("clip_count") or 1),
        "autocontinue": bool(body.get("autocontinue")),
        "autoconcat": bool(body.get("autoconcat")),
        "quality": str(body.get("quality") or "fast"),
        "render_width": int(body["render_width"]) if body.get("render_width") else None,
        "render_height": int(body["render_height"]) if body.get("render_height") else None,
        "loras": body.get("loras") or body.get("lora_specs") or [],
    }


def _loras_from_body(state: AppState, body: dict[str, Any]) -> list[LoraRef]:
    from h3_lora import lora_catalog, parse_lora_specs, resolve_lora_path

    specs = parse_lora_specs(
        body.get("loras") or body.get("lora_specs"),
        lora_catalog(state.output_dir),
    )
    return [
        LoraRef(spec=spec, path=resolve_lora_path(spec), scale=scale)
        for spec, scale in specs
    ]


def _resolve_existing_media(state: AppState, raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("reference path is required")
    p = Path(text)
    if p.is_file():
        return p.resolve()
    name = Path(text).name
    for cand in (
        state.output_dir / name,
        state.upload_dir / name,
        _frames_dir(state.output_dir) / name,
    ):
        if cand.is_file():
            return cand.resolve()
    raise ValueError(f"reference file not found: {text}")


def _resolve_refs(state: AppState, refs: list[Any]) -> list[Any]:
    for item in refs:
        item.path = _resolve_existing_media(state, str(item.path))
        if item.audio_path is not None:
            item.audio_path = _resolve_existing_media(state, str(item.audio_path))
    return refs


def _request_from_body(
    body: dict[str, Any],
    prompt: str,
    output: Path,
    *,
    state: AppState | None = None,
    first_frame: Path | None = None,
    last_frame: Path | None = None,
) -> GenerateRequest:
    settings = _clip_settings_from_body(body)
    mode = str(body.get("mode") or "t2va").strip().lower()
    first = first_frame
    last = last_frame
    if first is None and body.get("image_path"):
        first = Path(str(body["image_path"]))
    if last is None and body.get("end_image_path"):
        last = Path(str(body["end_image_path"]))
    if mode == "last_frame" and last is None and first is not None:
        last, first = first, None
    if mode == "first_frame" and first is None:
        raise ValueError("first_frame mode requires an image")
    if mode == "last_frame" and last is None:
        raise ValueError("last_frame mode requires an image")
    if mode == "fl2va" and (first is None or last is None):
        raise ValueError("fl2va mode requires first and last images")
    refs = parse_refs_payload(body.get("refs"))
    if refs:
        mode = "ref2va"
        first = None
        last = None
    elif mode == "ref2va":
        raise ValueError("ref2va requires at least one image, video, or audio reference")
    rw = settings.get("render_width")
    rh = settings.get("render_height")
    loras = _loras_from_body(state, body) if state is not None else []
    return GenerateRequest(
        prompt=prompt,
        output_path=output,
        width=int(settings["width"] or 512),
        height=int(settings["height"] or 512),
        num_frames=int(settings["num_frames"] or 22),
        quality=str(settings.get("quality") or "fast"),
        steps=int(settings["num_steps"] or H3_DEFAULT_STEPS),
        layers=int(settings["layers"]) if settings.get("layers") is not None else None,
        reuse=int(settings["reuse"]) if settings.get("reuse") is not None else None,
        core_reuse=int(body["core_reuse"]) if body.get("core_reuse") is not None else None,
        token_reduction=bool(body["token_reduction"])
        if body.get("token_reduction") is not None
        else None,
        render_width=int(rw) if rw else None,
        render_height=int(rh) if rh else None,
        seed=settings.get("seed"),
        ssd_streaming=bool(body.get("ssd_streaming")) and not loras,
        first_frame=first,
        last_frame=last,
        refs=refs,
        loras=loras,
        mode=mode,
    )


async def _fail_run(state: AppState, run_id: str, message: str) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    run.status = RunStatus.FAILED.value
    run.error = message
    for cid in run.clip_ids:
        clip = state.clips.get(cid)
        if clip and clip.status != RunStatus.DONE.value:
            clip.status = RunStatus.FAILED.value
            clip.error = message
    state.save_index()
    await state.emit(run_id, {"type": "error", "error": message, "run_id": run_id})


async def _abort_run_cancelled(state: AppState, run_id: str) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    run.status = RunStatus.CANCELLED.value
    run.error = "cancelled"
    for cid in run.clip_ids:
        clip = state.clips.get(cid)
        if clip and clip.status not in (RunStatus.DONE.value,):
            clip.status = RunStatus.CANCELLED.value
            clip.error = "cancelled"
    state.save_index()
    await state.emit(
        run_id,
        {"type": "run_cancelled", "run_id": run_id, "message": "Generation cancelled"},
    )


async def _execute_run(state: AppState, run_id: str) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    if state.is_run_cancelled(run_id):
        await _abort_run_cancelled(state, run_id)
        return
    state._active_run_id = run_id
    run.status = RunStatus.RUNNING.value
    body = dict(_RUN_BODIES.get(run_id, {}))
    await state.emit(
        run_id,
        {
            "type": "run_started",
            "run_id": run_id,
            "clip_count": len(run.prompts),
            "autoconcat": run.autoconcat,
            "autocontinue": run.autocontinue,
        },
    )
    done_paths: list[Path] = []
    continue_from = body.get("continue_from")
    prev_frame: Path | None = None
    if continue_from:
        parent = state.clips.get(str(continue_from))
        if parent and parent.filename:
            parent_path = state.output_dir / parent.filename
            if parent_path.is_file() and media_available():
                tmp = mk_scratch_dir("h3_cont_")
                prev_frame = extract_last_frame(parent_path, tmp / "last.png")

    try:
        for i, (clip_id, prompt) in enumerate(zip(run.clip_ids, run.prompts)):
            if state.is_run_cancelled(run_id):
                await _abort_run_cancelled(state, run_id)
                return
            clip = state.clips[clip_id]
            clip.status = RunStatus.RUNNING.value
            dest = state.output_dir / clip.filename
            await state.emit(
                run_id,
                {
                    "type": "clip_started",
                    "clip_id": clip_id,
                    "index": i,
                    "total": len(run.prompts),
                    "prompt": prompt,
                },
            )
            first = prev_frame
            last = Path(str(body["end_image_path"])) if body.get("end_image_path") else None
            if i == 0 and first is None and body.get("image_path"):
                first = Path(str(body["image_path"]))
            loop = asyncio.get_running_loop()

            def _progress(mp: dict[str, Any], *, _rid=run_id) -> None:
                asyncio.run_coroutine_threadsafe(
                    state.emit(
                        _rid,
                        {
                            "type": "progress",
                            "phase": mp.get("stage") or "generating",
                            "elapsed_s": mp.get("elapsed_s"),
                            "model_progress": mp,
                        },
                    ),
                    loop,
                )

            t0 = time.time()
            req = _request_from_body(
                body,
                prompt,
                dest,
                state=state,
                first_frame=first,
                last_frame=last if i == 0 else None,
            )
            if req.loras:
                req.ssd_streaming = False
            if req.refs:
                req.first_frame = None
                req.last_frame = None
                req.mode = "ref2va"
            elif run.autocontinue and i > 0 and first is not None:
                req.mode = "first_frame"
            try:
                await asyncio.to_thread(state.engine.generate, req, on_progress=_progress)
            except GenerationCancelledError:
                await _abort_run_cancelled(state, run_id)
                return
            elapsed = round(time.time() - t0, 2)
            size = dest.stat().st_size if dest.is_file() else 0
            clip.status = RunStatus.DONE.value
            clip.elapsed_s = elapsed
            clip.bytes = size
            clip.video_url = state.clip_url(clip.filename)
            clip.label = "CURRENT" if i == len(run.prompts) - 1 else f"CLIP {i + 1}"
            if i == 0:
                clip.label = "ORIGINAL" if len(run.prompts) > 1 else "CURRENT"
            state.save_index()
            await state.emit(
                run_id,
                {
                    "type": "clip_done",
                    "clip_id": clip.id,
                    "video_url": clip.video_url,
                    "bytes": clip.bytes,
                    "filename": clip.filename,
                    "chain_id": clip.chain_id,
                },
            )
            done_paths.append(dest)
            if run.autocontinue and i < len(run.prompts) - 1 and media_available():
                tmp = mk_scratch_dir("h3_chain_")
                prev_frame = extract_last_frame(dest, tmp / "last.png")

        if run.autoconcat and len(done_paths) > 1 and media_available():
            merged_name = f"web_{sanitize_filename(run.prompts[0])}_merged.mp4"
            merged_path = state.output_dir / merged_name
            concat_mp4s(done_paths, merged_path)
            mid = str(uuid.uuid4())
            mclip = ClipRecord(
                id=mid,
                prompt=run.prompts[0] + f" (×{len(done_paths)} merged)",
                label="MERGED",
                video_url=state.clip_url(merged_name),
                filename=merged_name,
                chain_id=run.chain_id,
                clip_index=len(run.clip_ids),
                mode=str(body.get("mode") or "t2va"),
                status=RunStatus.DONE.value,
                created_at=datetime.now().isoformat(),
                bytes=merged_path.stat().st_size if merged_path.is_file() else None,
                **{
                    k: v
                    for k, v in _clip_settings_from_body(body).items()
                    if k in ClipRecord.__dataclass_fields__
                },
            )
            state.clips[mid] = mclip
            run.merged_clip_id = mid
            run.merged_url = mclip.video_url
            await state.emit(
                run_id,
                {
                    "type": "merged",
                    "video_url": mclip.video_url,
                    "clip_id": mid,
                    "filename": merged_name,
                    "chain_id": run.chain_id,
                },
            )

        run.status = RunStatus.DONE.value
        state.save_index()
        await state.emit(
            run_id, {"type": "run_complete", "run_id": run_id, "chain_id": run.chain_id}
        )
    except Exception as exc:
        log.exception("run %s failed", run_id)
        await _fail_run(state, run_id, str(exc))
    finally:
        state._active_run_id = None
        _RUN_BODIES.pop(run_id, None)


async def _worker_loop(state: AppState) -> None:
    while True:
        run_id = await state._pending.get()
        try:
            await _execute_run(state, run_id)
        except Exception:
            log.exception("worker crashed on run %s", run_id)
        finally:
            state._pending.task_done()


def _ensure_web_deps() -> None:
    ensure_python_requirements()


def create_app(
    state: AppState,
    mount_static: bool = True,
    ws_handler: Callable[..., Any] | None = None,
) -> Any:
    _ensure_web_deps()
    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.ensure_worker()
        loop = asyncio.get_running_loop()

        def _on_interrupt() -> None:
            state.on_console_interrupt()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_interrupt)
            except (NotImplementedError, RuntimeError):
                pass
        yield
        state.engine.shutdown(wait=True)

    app = FastAPI(title="h3-ws", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if ws_handler is not None:

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            from server import WsProtocolAdapter

            await ws_handler(WsProtocolAdapter(websocket, websocket.client))

    def _defaults() -> dict[str, Any]:
        if state.runtime_defaults:
            return dict(state.runtime_defaults)
        return {
            "num_frames": 22,
            "width": 512,
            "height": 512,
            "num_steps": H3_DEFAULT_STEPS,
            "layers": H3_DEFAULT_LAYERS,
            "reuse": H3_DEFAULT_REUSE,
            "fps": FPS,
            "quality": "fast",
        }

    @app.get("/api/health")
    async def api_health(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        info = state.engine.info()
        return {
            "ok": True,
            "engine_ok": bool(info.get("ok")),
            "server_url": state.server_url,
            "web_url": state.http_url,
            "engine": info,
        }

    @app.get("/api/config")
    async def api_config(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        info = state.engine.info()
        gb = ram_gb()
        ssd = recommend_ssd_streaming(gb)
        note = (
            f"Native h3.c Metal. Model dir: {state.engine.model_dir}. "
            f"Binary: {state.engine.h3_bin}."
        )
        if gb is not None:
            note += f" This Mac reports ~{gb:.0f} GB unified memory."
        if ssd:
            note += " Under 64 GB RAM: leave SSD streaming off unless a run is killed for memory — it makes denoise much slower."
        if info.get("metal4"):
            note += " Metal 4 GPU: int8-row-fc2 is on."
        if not info.get("ok"):
            note += f" Engine: {info.get('error') or 'not ready'}."
        return {
            "server_connected": True,
            "embedded": state.embedded,
            "server_url": state.server_url,
            "web_url": state.http_url,
            "engine_ok": bool(info.get("ok")),
            "engine_error": None if info.get("ok") else info.get("error"),
            "h3_bin": str(state.engine.h3_bin),
            "model_dir": str(state.engine.model_dir),
            "ram_gb": gb,
            "recommend_ssd_streaming": ssd,
            "metal4": bool(info.get("metal4")),
            "quality_presets": QUALITY_PRESET_LIST,
            "lora_presets": lora_catalog(state.output_dir),
            "resolution_presets": RESOLUTION_PRESETS,
            "duration_presets": DURATION_PRESETS,
            "generation_modes": GENERATION_MODES,
            "ref_kinds": [
                {"id": "image", "label": "Image", "flag": "--ref-image"},
                {"id": "silent_video", "label": "Silent video", "flag": "--ref-silent-video"},
                {"id": "video", "label": "Video (keep audio)", "flag": "--ref-video"},
                {"id": "video_audio", "label": "Video + replacement audio", "flag": "--ref-video-audio"},
                {"id": "audio", "label": "Audio (with image or video)", "flag": "--ref-audio"},
            ],
            "clip_multiplier_max": CLIP_MULTIPLIER_MAX,
            "defaults": _defaults(),
            "model_note": note,
            "pyav_available": media_available(),
        }

    # ── Models management (status + user-confirmed download) ─────────────────

    def _component_status(model_dir: Path) -> list[dict[str, Any]]:
        """Return present/missing status for FL2VA and Ref2VA components."""
        def _dir_gib(path: Path) -> float:
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
            return total / (1024 ** 3)

        def _has_safetensors(path: Path) -> bool:
            return path.is_dir() and any(path.glob("*.safetensors"))

        fl = fl2va_dir(model_dir)
        r2 = ref2va_dir(model_dir)
        fl_ok, _ = model_layout_ok(model_dir)
        r2_ok, _ = model_layout_ok(model_dir, need_ref2va=True)
        return [
            {
                "id": "fl2va",
                "label": "FL2VA (core)",
                "present": fl_ok,
                "path": str(fl),
                "size_gib": round(_dir_gib(fl), 1),
                "note": "Required for t2va / first / last frame generation.",
            },
            {
                "id": "ref2va",
                "label": "Ref2VA (references)",
                "present": r2_ok,
                "path": str(r2),
                "size_gib": round(_dir_gib(r2), 1),
                "note": "Required for reference (image/video) modes.",
            },
        ]

    @app.get("/api/models")
    async def api_models_status():
        model_dir = state.engine.model_dir
        return {
            "ok": True,
            "model_dir": str(model_dir),
            "components": _component_status(model_dir),
        }

    # ── Download state for SSE progress ───────────────────────────────────────
    # Owned server-side so the task survives client disconnects. Stores the
    # asyncio.Task handle so a reconnect re-attaches instead of double-starting.
    _download_state: dict[str, Any] = {
        "active": False,
        "component": None,
        "error": None,
        "task": None,
    }

    # Approximate expected sizes in bytes (matching scripts/download_model.py).
    EXPECTED_BYTES = {
        "fl2va": 134.1 * 1024**3,   # ~134 GB
        "ref2va": 61.7 * 1024**3,   # ~62 GB (transformer only)
    }

    _COMPONENT_DIR = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}

    def _component_progress(model_dir: Path, component: str) -> tuple[int, int]:
        """Sum bytes of .incomplete partials + already-relocated final files.

        ``snapshot_download --local-dir=models/MiniMax-H3`` writes partials into
        the component's PRIVATE cache dir:
            models/MiniMax-H3/.cache/huggingface/download/{FL2VA|Ref2VA}/...incomplete
        The global hub cache (~/.cache/huggingface/hub/blobs) is a DIFFERENT, stale
        repo and must not drive the progress bar.
        """
        comp_dir = _COMPONENT_DIR.get(component)
        if not comp_dir:
            return 0, 0
        root = model_dir / ".cache" / "huggingface" / "download" / comp_dir
        if not root.is_dir():
            return 0, 0
        incomplete = 0
        complete = 0
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if p.suffix == ".incomplete":
                incomplete += size
            else:
                complete += size
        return complete, incomplete

    async def _run_download(component: str) -> None:
        """Run download in the background; update _download_state on finish."""
        script = REPO_ROOT / "scripts" / "download_model.py"
        cmd = [sys.executable, str(script), "--local-dir", str(state.engine.model_dir)]
        if component == "ref2va":
            cmd.append("--with-ref2va")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await proc.communicate()
            text = output.decode("utf-8", "replace")
            if proc.returncode != 0:
                _download_state["error"] = text.strip()[-2000:] or "download failed"
            else:
                _download_state["error"] = None
        except Exception as exc:
            _download_state["error"] = str(exc)
        finally:
            _download_state["active"] = False
            _download_state["task"] = None
            _download_state["component"] = None

    @app.get("/api/models/download/status")
    async def api_models_download_status():
        """Return whether a download is in flight and, if so, its live progress."""
        st = _download_state
        task = st.get("task")
        active = bool(st.get("active")) and task is not None and not task.done()
        out: dict[str, Any] = {
            "active": active,
            "component": st.get("component"),
            "error": st.get("error"),
        }
        if active and st.get("component"):
            comp = st["component"]
            complete, incomplete = await asyncio.to_thread(
                _component_progress, state.engine.model_dir, comp
            )
            current = complete + incomplete
            expected = EXPECTED_BYTES.get(comp, 0)
            pct = min(99, int(100 * current / expected)) if expected > 0 else 0
            out["progress"] = {
                "percent": pct,
                "downloaded_gb": round(current / 1024**3, 2),
                "expected_gb": round(expected / 1024**3, 1),
            }
        return out

    @app.get("/api/models/download/stream")
    async def api_models_download_stream(component: str):
        """SSE endpoint for download progress.

        The download task is owned server-side and survives client disconnects:
        a reconnect re-attaches to the same in-flight task. Only ONE terminal
        event fires when the task finishes; state is then cleared so a later
        manual re-download works.
        """
        from sse_starlette.sse import EventSourceResponse
        import time

        component = component.strip().lower()
        if component not in ("fl2va", "ref2va"):
            raise HTTPException(400, "component must be 'fl2va' or 'ref2va'")

        st = _download_state
        task = st.get("task")
        running = task is not None and not task.done()

        if running and st.get("component") != component:
            raise HTTPException(409, f"another component ({st['component']}) is downloading")

        # Spawn the download only if none is running; otherwise re-attach.
        if not running:
            st["active"] = True
            st["component"] = component
            st["error"] = None
            st["task"] = asyncio.create_task(_run_download(component))

        expected = EXPECTED_BYTES.get(component, 0)
        last_total = 0
        last_time = time.time()

        async def event_generator():
            try:
                while True:
                    cur_task = st.get("task")
                    if cur_task is None or cur_task.done():
                        break
                    complete, incomplete = await asyncio.to_thread(
                        _component_progress, state.engine.model_dir, component
                    )
                    current = complete + incomplete
                    now = time.time()
                    speed = 0.0
                    if now - last_time >= 0.5:
                        speed = (current - last_total) / (now - last_time)
                        last_total = current
                        last_time = now
                    pct = min(99, int(100 * current / expected)) if expected > 0 else 0
                    if speed > 1024**2:
                        speed_str = f"{speed / 1024**2:.1f} MB/s"
                    elif speed > 1024:
                        speed_str = f"{speed / 1024:.0f} KB/s"
                    else:
                        speed_str = f"{speed:.0f} B/s" if speed > 0 else "starting..."
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "percent": pct,
                            "downloaded_gb": round(current / 1024**3, 2),
                            "expected_gb": round(expected / 1024**3, 1),
                            "speed": speed_str,
                            "active": True,
                        }),
                    }
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                # Client disconnected — the download task continues server-side.
                raise

            # Terminal event — the task has finished.
            if st["error"]:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": st["error"]}),
                }
            else:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "ok": True,
                        "components": _component_status(state.engine.model_dir),
                    }),
                }

        return EventSourceResponse(event_generator())

    @app.post("/api/models/download")
    async def api_models_download(body: dict[str, Any]):
        """User-confirmed download of a missing model component (legacy blocking).

        Prefer /api/models/download/stream for progress tracking.
        huggingface_hub natively resumes partial downloads.
        """
        component = str(body.get("component") or "").strip().lower()
        if component not in ("fl2va", "ref2va"):
            raise HTTPException(400, "component must be 'fl2va' or 'ref2va'")
        script = REPO_ROOT / "scripts" / "download_model.py"
        if not script.is_file():
            raise HTTPException(500, f"Download script not found at {script}")
        cmd = [sys.executable, str(script), "--local-dir", str(state.engine.model_dir)]
        if component == "ref2va":
            cmd.append("--with-ref2va")

        async def _run() -> dict[str, Any]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                output, _ = await proc.communicate()
                text = output.decode("utf-8", "replace")
                if proc.returncode != 0:
                    return {"ok": False, "error": text.strip()[-2000:] or "download failed"}
                return {
                    "ok": True,
                    "components": _component_status(state.engine.model_dir),
                }
            except Exception as exc:  # pragma: no cover - os-level failures
                return {"ok": False, "error": str(exc)}

        result = await _run()
        if not result["ok"]:
            raise HTTPException(500, result.get("error", "download failed"))
        return result

    @app.post("/api/loras/ensure")
    async def api_lora_ensure(body: dict[str, Any]):
        spec = normalize_lora_spec(str(body.get("spec") or body.get("url") or ""))
        if not spec:
            raise HTTPException(400, "spec or url is required")
        try:
            return await asyncio.to_thread(ensure_lora, spec)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/loras/custom")
    async def api_lora_custom(body: dict[str, Any]):
        spec = normalize_lora_spec(str(body.get("spec") or body.get("url") or ""))
        if not spec:
            raise HTTPException(400, "spec or url is required")
        try:
            scale = float(body.get("scale", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "scale must be a number")
        label = str(body.get("label") or "").strip() or _label_for_spec(spec)
        entries = read_custom_loras(state.output_dir)
        existing = next((e for e in entries if e.get("spec") == spec), None)
        if existing is not None:
            lid = str(existing["id"])
            existing["label"] = label
            existing["scale"] = scale
        else:
            lid = f"custom_{uuid.uuid4().hex[:8]}"
            entries.append(
                {"id": lid, "label": label, "spec": spec, "scale": scale, "custom": True}
            )
        write_custom_loras(state.output_dir, entries)
        try:
            await asyncio.to_thread(ensure_lora, spec)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        catalog = lora_catalog(state.output_dir)
        preset = next((p for p in catalog if p.get("id") == lid), None)
        return {
            "ok": True,
            "id": lid,
            "reused": existing is not None,
            "preset": preset,
            "lora_presets": catalog,
        }

    @app.delete("/api/loras/custom/{lora_id}")
    async def api_lora_delete(lora_id: str):
        entries = [e for e in read_custom_loras(state.output_dir) if e["id"] != lora_id]
        write_custom_loras(state.output_dir, entries)
        return {"ok": True, "lora_presets": lora_catalog(state.output_dir)}

    # ── Presets (persistent, survive restarts) ───────────────────────────────

    @app.get("/api/presets")
    async def api_presets_list():
        state.presets.sort(key=lambda p: p.get("updatedAt", ""), reverse=True)
        return {"presets": state.presets}

    @app.post("/api/presets")
    async def api_presets_save(body: dict[str, Any]):
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        preset = dict(body)
        now = datetime.now().isoformat()
        if not preset.get("id"):
            preset["id"] = f"preset_{uuid.uuid4().hex[:8]}"
        preset["name"] = name
        preset["createdAt"] = preset.get("createdAt") or now
        preset["updatedAt"] = now
        existing = next((p for p in state.presets if p.get("id") == preset["id"]), None)
        if existing is not None:
            state.presets.remove(existing)
        state.presets.append(preset)
        state._save_user_data()
        return {"ok": True, "preset": preset}

    @app.delete("/api/presets/{preset_id}")
    async def api_presets_delete(preset_id: str):
        before = len(state.presets)
        state.presets = [p for p in state.presets if p.get("id") != preset_id]
        if len(state.presets) == before:
            raise HTTPException(404, "Preset not found")
        state._save_user_data()
        return {"ok": True, "deleted": preset_id}

    # ── Cast members (persistent, survive restarts) ──────────────────────────

    @app.get("/api/cast")
    async def api_cast_list():
        state.cast_members.sort(key=lambda c: c.get("name", "").lower())
        return {"cast": state.cast_members}

    @app.post("/api/cast")
    async def api_cast_create(body: dict[str, Any]):
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        now = datetime.now().isoformat()
        member = {
            "id": f"cast_{uuid.uuid4().hex[:8]}",
            "name": name,
            "description": str(body.get("description") or "").strip() or None,
            "media": body.get("media") or [],
            "createdAt": now,
            "updatedAt": now,
        }
        state.cast_members.append(member)
        state._save_user_data()
        return {"ok": True, "cast": member}

    @app.put("/api/cast/{cast_id}")
    async def api_cast_update(cast_id: str, body: dict[str, Any]):
        member = next((c for c in state.cast_members if c.get("id") == cast_id), None)
        if member is None:
            raise HTTPException(404, "Cast member not found")
        name = str(body.get("name") or "").strip()
        if name:
            member["name"] = name
        if "description" in body:
            member["description"] = str(body.get("description") or "").strip() or None
        if "media" in body:
            member["media"] = body.get("media") or []
        member["updatedAt"] = datetime.now().isoformat()
        state._save_user_data()
        return {"ok": True, "cast": member}

    @app.delete("/api/cast/{cast_id}")
    async def api_cast_delete(cast_id: str):
        before = len(state.cast_members)
        state.cast_members = [c for c in state.cast_members if c.get("id") != cast_id]
        if len(state.cast_members) == before:
            raise HTTPException(404, "Cast member not found")
        state._save_user_data()
        return {"ok": True, "deleted": cast_id}

    @app.get("/api/clips")
    async def list_clips(chain_id: Optional[str] = None):
        clips = list(state.clips.values())
        if chain_id:
            clips = [c for c in clips if c.chain_id == chain_id]
        clips.sort(key=lambda c: c.created_at)
        return {"clips": [_clip_for_api(state, c) for c in clips]}

    @app.post("/api/session/clear")
    async def clear_session():
        return {"ok": True, **state.clear_session()}

    @app.delete("/api/clips/{clip_id}")
    async def delete_clip(clip_id: str):
        if not state.delete_clip_record(clip_id):
            raise HTTPException(404, "Clip not found")
        state.save_index()
        return {"ok": True, "deleted": clip_id}

    @app.delete("/api/chains/{chain_id}")
    async def delete_chain(chain_id: str):
        count = state.delete_chain(chain_id)
        if count == 0:
            raise HTTPException(404, "Chain not found")
        return {"ok": True, "deleted": count, "chain_id": chain_id}

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        if run_id not in state.runs:
            raise HTTPException(404, "Run not found")
        if not state.request_cancel_run(run_id):
            raise HTTPException(409, f"Cannot cancel run in state {state.runs[run_id].status}")
        return {"ok": True, "status": "cancelling"}

    @app.post("/api/generate")
    async def generate(body: dict[str, Any]):
        state.ensure_worker()
        prompt = str(body.get("prompt") or "").strip()
        prompts = body.get("prompts") or []
        if prompt:
            prompts = [prompt] + [p for p in prompts if str(p).strip()]
        prompts = [str(p).strip() for p in prompts if p and str(p).strip()]
        if not prompts:
            raise HTTPException(400, "prompt is required")

        ui_mode = (body.get("mode") or "t2va").strip().lower()
        try:
            refs = _resolve_refs(state, parse_refs_payload(body.get("refs")))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if refs:
            ui_mode = "ref2va"
            body = dict(body)
            body["mode"] = "ref2va"
            body["refs"] = [
                {
                    "kind": item.kind,
                    "path": str(item.path),
                    "audio_path": str(item.audio_path) if item.audio_path else "",
                    "name": item.name,
                }
                for item in refs
            ]
            if body.get("image_path") or body.get("end_image_path"):
                raise HTTPException(
                    400,
                    "Ref2VA references cannot be mixed with first/last-frame anchors",
                )
        elif ui_mode == "ref2va":
            raise HTTPException(
                400,
                "ref2va requires at least one reference (image, silent video, video, or audio)",
            )

        clip_count = max(1, min(CLIP_MULTIPLIER_MAX, int(body.get("clip_count") or 1)))
        if ui_mode == "ref2va":
            clip_count = 1
        continue_from = None if ui_mode == "ref2va" else body.get("continue_from")
        if clip_count > 1:
            continue_from = None
            chain_id = str(uuid.uuid4())
            prompts = [prompts[0]] * clip_count if len(prompts) == 1 else prompts
        else:
            chain_id = str(body.get("chain_id") or uuid.uuid4())

        if ui_mode == "first_frame" and not body.get("image_path") and not continue_from:
            raise HTTPException(400, "first_frame mode requires an image")
        if ui_mode == "last_frame" and not body.get("end_image_path") and not body.get("image_path"):
            raise HTTPException(400, "last_frame mode requires an image")
        if ui_mode == "fl2va" and (
            not body.get("image_path") or not body.get("end_image_path")
        ):
            raise HTTPException(400, "fl2va requires first and last images")

        try:
            settings = _clip_settings_from_body(body)
            require_ui_canvas(int(settings["width"]), int(settings["height"]))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        autocontinue = bool(body.get("autocontinue")) or clip_count > 1 or bool(continue_from)
        autoconcat = bool(body.get("autoconcat")) or clip_count > 1
        body = dict(body)
        body["autocontinue"] = autocontinue
        body["autoconcat"] = autoconcat
        if continue_from:
            body["continue_from"] = continue_from

        existing = [c for c in state.clips.values() if c.chain_id == chain_id]
        base_index = len(existing)
        run_id = str(uuid.uuid4())
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_ids: list[str] = []
        for i, p in enumerate(prompts):
            clip_id = str(uuid.uuid4())
            slug = sanitize_filename(p) or "clip"
            filename = f"web_{slug}_{ts}_{i}.mp4"
            clip = ClipRecord(
                id=clip_id,
                prompt=p,
                label="ORIGINAL" if base_index == 0 and i == 0 and not continue_from else "CURRENT",
                video_url="",
                filename=filename,
                chain_id=chain_id,
                clip_index=base_index + i,
                mode=ui_mode,
                status=RunStatus.QUEUED.value,
                created_at=datetime.now().isoformat(),
                **{
                    k: v
                    for k, v in settings.items()
                    if k in ClipRecord.__dataclass_fields__
                },
            )
            state.clips[clip_id] = clip
            clip_ids.append(clip_id)

        run = RunRecord(
            id=run_id,
            status=RunStatus.QUEUED.value,
            prompts=prompts,
            chain_id=chain_id,
            clip_ids=clip_ids,
            created_at=datetime.now().isoformat(),
            autocontinue=autocontinue,
            autoconcat=autoconcat,
        )
        state.runs[run_id] = run
        _RUN_BODIES[run_id] = body
        state.save_index()
        state.event_queues[run_id] = asyncio.Queue()
        started = await state.enqueue_generation_run(run_id)
        state.save_index()
        log.info(
            "Web UI: %s run %s  clips=%d  mode=%s  %sx%s  frames=%s  quality=%s  steps=%s  layers=%s  reuse=%s",
            "starting" if started else "queued",
            run_id,
            len(clip_ids),
            ui_mode,
            settings.get("width"),
            settings.get("height"),
            settings.get("num_frames"),
            settings.get("quality"),
            settings.get("num_steps"),
            settings.get("layers"),
            settings.get("reuse"),
        )
        return {
            "run_id": run_id,
            "chain_id": chain_id,
            "clip_ids": clip_ids,
            "status": state.runs[run_id].status,
            "started_immediately": started,
        }

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        if run_id not in state.runs:
            raise HTTPException(404, "Run not found")
        if run_id not in state.event_queues:
            state.event_queues[run_id] = asyncio.Queue()

        async def stream() -> AsyncIterator[str]:
            q = state.event_queues[run_id]
            run = state.runs[run_id]
            if run.status in (RunStatus.DONE.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value):
                if run.status == RunStatus.FAILED.value:
                    yield f"data: {json.dumps({'type': 'error', 'error': run.error or 'Generation failed', 'run_id': run_id})}\n\n"
                    return
                if run.status == RunStatus.CANCELLED.value:
                    yield f"data: {json.dumps({'type': 'run_cancelled', 'run_id': run_id, 'message': 'Generation cancelled'})}\n\n"
                    return
                if run.status == RunStatus.DONE.value and run.merged_clip_id:
                    merged = state.clips.get(run.merged_clip_id)
                    if merged and merged.video_url:
                        yield f"data: {json.dumps({'type': 'merged', 'video_url': merged.video_url, 'clip_id': merged.id, 'filename': merged.filename, 'chain_id': merged.chain_id})}\n\n"
                yield f"data: {json.dumps({'type': 'run_complete', 'run_id': run_id, 'chain_id': run.chain_id})}\n\n"
                return
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=120.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("run_complete", "run_done", "error", "run_cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/upload")
    async def upload(request: Request, kind: str = "image"):
        kind = (kind or "image").strip().lower()
        if kind not in ("image", "audio", "video"):
            raise HTTPException(400, f"unsupported upload kind: {kind}")
        try:
            return await _save_upload_file(request, state.upload_dir, kind=kind)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/frames")
    async def list_frames():
        entries = _read_frame_library(state.output_dir)
        entries.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        return {"frames": [_frame_for_api(e) for e in entries]}

    @app.post("/api/frames")
    async def save_frame(request: Request):
        form = await request.form()
        upload_file = form.get("file")
        if upload_file is None:
            raise HTTPException(400, "file is required")
        read = getattr(upload_file, "read", None)
        if read is None:
            raise HTTPException(400, "file is required")
        content = await read()
        if not content:
            raise HTTPException(400, "empty frame file")
        frames_root = _frames_dir(state.output_dir)
        frames_root.mkdir(parents=True, exist_ok=True)
        fid = f"frame_{uuid.uuid4().hex[:8]}"
        filename = f"{fid}.png"
        dest = frames_root / filename
        dest.write_bytes(content)
        time_raw = form.get("time_s")
        time_s: float | None = None
        if time_raw is not None and str(time_raw).strip():
            try:
                time_s = float(str(time_raw))
            except (TypeError, ValueError):
                raise HTTPException(400, "time_s must be a number") from None
        label = str(form.get("label") or "").strip() or (
            f"Frame @ {time_s:.1f}s" if time_s is not None else "Saved frame"
        )
        entry: dict[str, Any] = {
            "id": fid,
            "label": label,
            "path": str(dest.resolve()),
            "filename": filename,
            "created_at": datetime.now().isoformat(),
        }
        source_clip_id = str(form.get("source_clip_id") or "").strip() or None
        if source_clip_id:
            entry["source_clip_id"] = source_clip_id
        if time_s is not None:
            entry["time_s"] = round(time_s, 3)
        entries = _read_frame_library(state.output_dir)
        entries.append(entry)
        _write_frame_library(state.output_dir, entries)
        return {"ok": True, "frame": _frame_for_api(entry)}

    @app.delete("/api/frames/{frame_id}")
    async def delete_frame(frame_id: str):
        fid = (frame_id or "").strip()
        entries = _read_frame_library(state.output_dir)
        kept: list[dict[str, Any]] = []
        removed = None
        for entry in entries:
            if entry.get("id") == fid:
                removed = entry
            else:
                kept.append(entry)
        if removed is None:
            raise HTTPException(404, "Frame not found")
        path = Path(str(removed.get("path") or ""))
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        _write_frame_library(state.output_dir, kept)
        return {"ok": True, "deleted": fid, "frames": [_frame_for_api(e) for e in kept]}

    @app.get("/api/frames/files/{filename}")
    async def frame_file(filename: str):
        path = _frames_dir(state.output_dir) / Path(filename).name
        if not path.is_file():
            raise HTTPException(404, "Frame file not found")
        return FileResponse(path)

    @app.get("/api/videos/{filename}")
    async def video_file(filename: str):
        path = state.output_dir / Path(filename).name
        if not path.is_file():
            raise HTTPException(404, "Video not found")
        return FileResponse(path, media_type="video/mp4")

    if mount_static:
        dist = resolve_web_dist()
        if dist.is_dir():
            app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")

    return app


def build_combined_application(ws_handler: Callable[..., Any], state: AppState) -> Any:
    return create_app(state, mount_static=True, ws_handler=ws_handler)


async def run_uvicorn(app: Any, host: str, port: int, state: AppState | None = None) -> None:
    _ensure_web_deps()
    import uvicorn

    from h3_paths import debug_console

    # Keep our console logger; uvicorn's default log_config would replace it.
    log_level = "debug" if debug_console() else "info"
    config = uvicorn.Config(
        app, host=host, port=port, log_level=log_level, log_config=None
    )
    server = uvicorn.Server(config)
    if state is not None:
        state._uvicorn_server = server
    await server.serve()
