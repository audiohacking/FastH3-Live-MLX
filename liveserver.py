#!/usr/bin/env python3
"""FastH3 Live on Metal h3.c — standalone continuous stream (no Web UI).

  python liveserver.py
  # → http://127.0.0.1:9000  (VLC / ffplay)

Generates 448×448 × 362 clips with the **FastH3 Dense-DataFree student DiT**
(converted Diffusers → native layout for ``./h3``), retimes them below 24 fps
so wall-clock can keep up, and live-wires MPEG-TS to reconnectable HTTP viewers.

Prep once (after ``hf download`` of the FastVideo transformer):

  ./scripts/prepare_fasth3_native_tree.sh

Does not boot server.py / the React app. Engine stays antirez h3.c.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import random
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3_backend import (  # noqa: E402
    GenerateRequest,
    H3Engine,
    LoraRef,
)
from h3_live import (  # noqa: E402
    DEFAULT_CHARACTERS,
    DEFAULT_SCENES,
    LIVE_CURATED_SHARE,
    LIVE_ENSEMBLE_BIAS,
    LIVE_FRAMES,
    LIVE_HEIGHT,
    LIVE_LAYERS,
    LIVE_LORA_ID,
    LIVE_MARGIN_RATIO,
    LIVE_MAX_CAST,
    LIVE_MAX_PLAY_FPS,
    LIVE_MIN_PLAY_FPS,
    LIVE_MODEL_DIR_FUSED_TURBO_INT8_NAME,
    LIVE_MODEL_DIR_FUSED_TURBO_NAME,
    LIVE_MODEL_DIR_INT8_NAME,
    LIVE_MODEL_DIR_NAME,
    LIVE_PLAY_FPS,
    LIVE_QUALITY_PRESET,
    LIVE_RENDER_HEIGHT,
    LIVE_RENDER_WIDTH,
    LIVE_REUSE,
    LIVE_STEPS,
    LIVE_TOKEN_REDUCTION,
    LIVE_WIDTH,
)
from h3_live.dashboard import (  # noqa: E402
    attach_ring_logger,
    host_metrics,
    load_dashboard_html,
    logs_after,
)
from h3_live.episode import (  # noqa: E402
    EPISODE_FRAMES,
    EPISODE_HEIGHT,
    EPISODE_LAYERS,
    EPISODE_RECIPE_NOTE,
    EPISODE_RENDER_HEIGHT,
    EPISODE_RENDER_WIDTH,
    EPISODE_REUSE,
    EPISODE_TOKEN_REDUCTION,
    EPISODE_WIDTH,
    EpisodeJob,
    clamp_scene_count,
    concat_mp4s,
)
from h3_live.pace import adaptive_play_fps, ema  # noqa: E402
from h3_live.presets import (  # noqa: E402
    DEFAULT_PRESET,
    PRESETS,
    apply_preset_to_args,
    get_preset,
    presets_public,
    recipe_label,
)
from h3_live.retime import feed_mpegts_to_sink, retime_to_mpegts  # noqa: E402
from h3_live.scenes import PromptPool  # noqa: E402
from h3_live.timing import PhaseTimer  # noqa: E402
from h3_lora import catalog_entry, ensure_lora  # noqa: E402
from h3_paths import default_model_dir, mk_scratch_dir  # noqa: E402

log = logging.getLogger("h3-live")


def _recipe_edit_blocked(broadcast: "Broadcast") -> str | None:
    """Recipe knobs are stop-only — no mid-stream Apply."""
    with broadcast.state.lock:
        generating = broadcast.state.generating
        episode = broadcast.state.episode.is_active()
    if episode:
        return "stop/cancel the episode before changing recipe"
    if generating or broadcast.demand_count() > 0:
        return "stop the stream before changing recipe"
    return None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    attach_ring_logger("h3-live")


# Soft keepalive for players that need bytes before the first clip (~70s).
# Null PID packets — not a decoded video frame (avoids VLC "snow").
_TS_NULL = bytes([0x47, 0x1F, 0xFF, 0x10]) + bytes(184)


class LiveState:
    """Shared status / control for the dashboard + producer."""

    # Dashboard must heartbeat within this window or the watch session expires.
    WATCH_TTL_S = 25.0

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.paused = False
        self.generating = False
        self.media_ready = False  # at least one real TS clip has been pushed
        # session_id → last heartbeat (monotonic). Replaces fragile ±1 token counts.
        self.watch_sessions: dict[str, float] = {}
        self.last_clip: int | None = None
        self.last_scene: int | None = None
        self.last_cast: str | None = None
        self.last_gen_s: float | None = None
        self.margin_s: float | None = None
        self.width = 0
        self.height = 0
        self.frames = 0
        self.steps = 0
        self.layers = LIVE_LAYERS
        self.reuse = LIVE_REUSE
        self.play_fps = 0.0
        self.cancel_fn = None  # set to engine.request_cancel
        # Prompt control applies to the *next* clip (in-flight job keeps its prompt).
        self.prompt_mode = "random"  # random | custom
        self.custom_prompt = ""
        self.pool_scenes = 0
        self.next_prompt_note = "random pool"
        # Quality recipe (Apply → next clip).
        self.quality_preset = DEFAULT_PRESET
        self.render_width = LIVE_RENDER_WIDTH
        self.render_height = LIVE_RENDER_HEIGHT
        self.token_reduction = LIVE_TOKEN_REDUCTION
        self.ensemble_only = False
        self.curated_share = LIVE_CURATED_SHARE
        self.recipe_note = ""
        # Offline episode batch (pauses Live producer while running).
        self.episode = EpisodeJob()
        self.episode_hold = False
        # Active LoRA (None when student-only). Scale Apply rebuilds the list.
        self.lora_id: str | None = None
        self.lora_label: str | None = None
        self.lora_scale: float | None = None
        self.loras: list = []
        # Live generate progress (from h3.c stage lines + ProgressEta).
        self.progress: dict | None = None
        self.last_phases: str | None = None
        self.gen_started_at: float | None = None

    def set_progress(self, mp: dict | None) -> None:
        with self.lock:
            if mp is None:
                self.progress = None
                return
            self.progress = {
                "stage": mp.get("stage"),
                "label": mp.get("label"),
                "step": mp.get("step"),
                "total": mp.get("total"),
                "pct": mp.get("pct"),
                "eta_s": mp.get("eta_s"),
                "avg_step_s": mp.get("avg_step_s"),
                "elapsed_s": mp.get("elapsed_s"),
            }

    def _prune_watch_sessions_locked(self) -> list[str]:
        now = time.monotonic()
        dead = [
            sid
            for sid, seen in self.watch_sessions.items()
            if now - seen > self.WATCH_TTL_S
        ]
        for sid in dead:
            del self.watch_sessions[sid]
        return dead

    def touch_watch(self, session: str) -> int:
        """Register / refresh a dashboard watch session. Returns active count."""
        session = (session or "").strip()
        with self.lock:
            expired = self._prune_watch_sessions_locked()
            if session:
                self.watch_sessions[session] = time.monotonic()
            n = len(self.watch_sessions)
        if expired:
            log.info("expired %d stale watch session(s)", len(expired))
        return n

    def drop_watch(self, session: str) -> int:
        """Remove one dashboard watch session. Returns active count."""
        session = (session or "").strip()
        with self.lock:
            if session:
                self.watch_sessions.pop(session, None)
            self._prune_watch_sessions_locked()
            return len(self.watch_sessions)

    def watcher_count(self) -> int:
        with self.lock:
            self._prune_watch_sessions_locked()
            return len(self.watch_sessions)

    def snapshot(self, viewers: int) -> dict:
        host = host_metrics()
        with self.lock:
            self._prune_watch_sessions_locked()
            custom = self.custom_prompt
            preview = custom.strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "…"
            return {
                **host,
                "viewers": viewers,
                "watchers": len(self.watch_sessions),
                "paused": self.paused,
                "generating": self.generating,
                "media_ready": self.media_ready,
                "last_clip": self.last_clip,
                "last_scene": self.last_scene,
                "last_cast": self.last_cast,
                "last_gen_s": self.last_gen_s,
                "margin_s": self.margin_s,
                "width": self.width,
                "height": self.height,
                "frames": self.frames,
                "steps": self.steps,
                "layers": self.layers,
                "reuse": self.reuse,
                "play_fps": self.play_fps,
                "prompt_mode": self.prompt_mode,
                "custom_prompt": custom,
                "custom_preview": preview,
                "pool_scenes": self.pool_scenes,
                "next_prompt_note": self.next_prompt_note,
                "quality_preset": self.quality_preset,
                "render_width": self.render_width,
                "render_height": self.render_height,
                "token_reduction": self.token_reduction,
                "ensemble_only": self.ensemble_only,
                "curated_share": self.curated_share,
                "recipe_note": self.recipe_note,
                "presets": presets_public(),
                "episode": self.episode.as_dict(),
                "episode_hold": self.episode_hold,
                "lora": (
                    {
                        "id": self.lora_id,
                        "label": self.lora_label or self.lora_id,
                        "scale": self.lora_scale,
                    }
                    if self.lora_id
                    else None
                ),
                "progress": dict(self.progress) if self.progress else None,
                "last_phases": self.last_phases,
                "gen_elapsed_s": (
                    round(time.time() - self.gen_started_at, 1)
                    if self.generating and self.gen_started_at
                    else None
                ),
            }

    def demand_count(self, ts_viewers: int) -> int:
        return ts_viewers + self.watcher_count()


class Broadcast:
    """HTTP dashboard + MPEG-TS fan-out on one port."""

    def __init__(self, host: str, port: int, state: LiveState) -> None:
        self.clients: set[queue.Queue[bytes]] = set()
        self.lock = threading.Lock()
        self.state = state
        broadcast = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _json(self, code: int, payload: dict | list) -> None:
                import json

                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _html(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):  # noqa: N802
                path = self.path.split("?", 1)[0]
                ua = (self.headers.get("User-Agent") or "").lower()
                accept = (self.headers.get("Accept") or "").lower()
                # VLC / ffplay / mpv historically open the root URL. Browsers get
                # the dashboard; media players get MPEG-TS on `/` too.
                media_ua = any(
                    tok in ua
                    for tok in (
                        "vlc",
                        "libvlc",
                        "ffmpeg",
                        "ffplay",
                        "mpv",
                        "lavf",
                        "gstreamer",
                        "quicktime",
                    )
                )
                wants_ts = (
                    path in ("/stream.ts", "/stream", "/live.ts")
                    or (path in ("/", "/index.html") and media_ua)
                    or "video/mp2t" in accept
                )
                if path in ("/", "/index.html") and not wants_ts:
                    self._html(load_dashboard_html())
                    return
                if path == "/api/status":
                    self._json(200, broadcast.state.snapshot(broadcast.viewer_count()))
                    return
                if path.startswith("/api/logs"):
                    from urllib.parse import parse_qs, urlparse

                    qs = parse_qs(urlparse(self.path).query)
                    after = int((qs.get("after") or ["0"])[0])
                    self._json(200, {"lines": logs_after(after)})
                    return
                if path in ("/api/episode.mp4", "/api/episode"):
                    self._episode_download()
                    return
                if wants_ts:
                    with broadcast.state.lock:
                        episode_busy = (
                            broadcast.state.episode_hold
                            or broadcast.state.episode.is_active()
                        )
                    if episode_busy:
                        self.send_error(
                            409, "episode batch running — live stream is stopped"
                        )
                        return
                    self._stream()
                    return
                self.send_error(404, "use / for dashboard, /stream.ts for MPEG-TS (VLC)")

            def _episode_download(self) -> None:
                with broadcast.state.lock:
                    job = broadcast.state.episode
                    path = job.output_path if job.status == "ready" else None
                if path is None or not path.is_file():
                    self._json(
                        404,
                        {"ok": False, "error": "episode not ready"},
                    )
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="fasth3-episode.mp4"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802
                import json

                path = self.path.split("?", 1)[0]
                if path != "/api/control":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode() or "{}")
                except json.JSONDecodeError:
                    self._json(400, {"ok": False, "error": "bad json"})
                    return
                action = str(body.get("action") or "")
                session = str(body.get("session") or "")
                if action == "pause":
                    with broadcast.state.lock:
                        broadcast.state.paused = True
                    log.info("generation paused from dashboard")
                elif action == "resume":
                    with broadcast.state.lock:
                        broadcast.state.paused = False
                    log.info("generation resumed from dashboard")
                elif action in ("watch", "heartbeat"):
                    if not session:
                        self._json(400, {"ok": False, "error": "session required"})
                        return
                    with broadcast.state.lock:
                        episode_busy = (
                            broadcast.state.episode_hold
                            or broadcast.state.episode.is_active()
                        )
                    if action == "watch" and episode_busy:
                        self._json(
                            409,
                            {
                                "ok": False,
                                "error": "episode batch is running — live is stopped until it finishes",
                            },
                        )
                        return
                    n = broadcast.state.touch_watch(session)
                    if action == "watch":
                        log.info(
                            "dashboard watch session=%s… (%d web, %d ts)",
                            session[:8],
                            n,
                            broadcast.viewer_count(),
                        )
                elif action == "unwatch":
                    if not session:
                        # Bare unwatch must not clear other browsers / agent curls.
                        self._json(200, {"ok": True, "action": action, "ignored": True})
                        return
                    n = broadcast.state.drop_watch(session)
                    log.info(
                        "dashboard unwatch session=%s… (%d web, %d ts)",
                        session[:8],
                        n,
                        broadcast.viewer_count(),
                    )
                    if broadcast.demand_count() == 0:
                        broadcast.cancel_generation("no viewers")
                elif action == "cancel":
                    broadcast.cancel_all("dashboard cancel")
                elif action == "episode_start":
                    try:
                        scenes = clamp_scene_count(body.get("scenes", 1))
                    except ValueError as exc:
                        self._json(400, {"ok": False, "error": str(exc)})
                        return
                    try:
                        broadcast.start_episode(scenes)
                    except RuntimeError as exc:
                        self._json(409, {"ok": False, "error": str(exc)})
                        return
                    log.info(
                        "episode start — %d scene(s); live stopped\n%s",
                        scenes,
                        EPISODE_RECIPE_NOTE,
                    )
                elif action == "episode_cancel":
                    broadcast.cancel_all("episode cancel")
                    log.info("episode cancel requested")
                elif action == "set_prompt":
                    mode = str(body.get("mode") or "random").strip().lower()
                    text = str(body.get("text") or "")
                    if mode not in ("random", "custom"):
                        self._json(400, {"ok": False, "error": "mode must be random|custom"})
                        return
                    if mode == "custom" and not text.strip():
                        self._json(400, {"ok": False, "error": "custom text required"})
                        return
                    when = (
                        "applies to next clip"
                        if broadcast.state.generating
                        else "ready for next clip"
                    )
                    with broadcast.state.lock:
                        broadcast.state.prompt_mode = mode
                        if mode == "custom":
                            broadcast.state.custom_prompt = text
                            note = text.strip().replace("\n", " ")
                            if len(note) > 80:
                                note = note[:77] + "…"
                            if PromptPool.looks_like_context_ir(text):
                                broadcast.state.next_prompt_note = f"custom IR: {note}"
                            else:
                                broadcast.state.next_prompt_note = f"custom idea→IR: {note}"
                        else:
                            broadcast.state.next_prompt_note = (
                                f"random pool ({broadcast.state.pool_scenes} scenes)"
                            )
                    if mode == "custom":
                        raw = text.strip()
                        if PromptPool.looks_like_context_ir(raw):
                            log.info(
                                "prompt applied: custom Context-IR (%d chars, %s)\n"
                                "—— applied prompt ——\n%s\n—— end prompt ——",
                                len(raw),
                                when,
                                raw,
                            )
                        else:
                            preview = PromptPool.wrap_idea_as_live_prompt(raw)
                            log.info(
                                "prompt applied: short idea → Context-IR wrap (%s)\n"
                                "—— idea ——\n%s\n"
                                "—— wrapped preview (cast filled at generate) ——\n%s\n"
                                "—— end prompt ——",
                                when,
                                raw,
                                preview,
                            )
                    else:
                        log.info(
                            "prompt applied: random pool (%d scenes, %s)",
                            broadcast.state.pool_scenes,
                            when,
                        )
                elif action in ("set_quality", "set_lora_scale", "set_recipe"):
                    blocked = _recipe_edit_blocked(broadcast)
                    if blocked:
                        self._json(409, {"ok": False, "error": blocked})
                        return
                    try:
                        # --- LoRA scale (optional) ---
                        if "scale" in body and body["scale"] is not None:
                            if not broadcast.state.lora_id:
                                if action == "set_lora_scale":
                                    self._json(
                                        400, {"ok": False, "error": "no LoRA loaded"}
                                    )
                                    return
                            else:
                                scale = float(body["scale"])
                                if not (0.05 <= scale <= 2.0):
                                    self._json(
                                        400,
                                        {
                                            "ok": False,
                                            "error": "scale must be in [0.05, 2.0]",
                                        },
                                    )
                                    return
                                ref = resolve_lora(broadcast.state.lora_id, scale)
                                with broadcast.state.lock:
                                    broadcast.state.lora_scale = float(ref.scale)
                                    broadcast.state.loras = [ref]
                                log.info(
                                    "LoRA %s scale → %.2f (stream stopped)",
                                    broadcast.state.lora_id,
                                    ref.scale,
                                )

                        # --- Named preset fills canvas fields unless overridden ---
                        preset_name = str(body.get("preset") or "").strip().lower()
                        if preset_name:
                            preset = get_preset(preset_name)
                            with broadcast.state.lock:
                                broadcast.state.quality_preset = preset.name
                                broadcast.state.render_width = preset.render_width
                                broadcast.state.render_height = preset.render_height
                                broadcast.state.token_reduction = preset.token_reduction
                                broadcast.state.frames = preset.frames

                        with broadcast.state.lock:
                            if "render_width" in body and body["render_width"] is not None:
                                broadcast.state.render_width = int(body["render_width"])
                            if "render_height" in body and body["render_height"] is not None:
                                broadcast.state.render_height = int(body["render_height"])
                            if "token_reduction" in body and body["token_reduction"] is not None:
                                broadcast.state.token_reduction = bool(
                                    body["token_reduction"]
                                )
                            if "frames" in body and body["frames"] is not None:
                                frames = int(body["frames"])
                                if frames < 5:
                                    raise ValueError("frames must be >= 5")
                                broadcast.state.frames = frames
                            if "steps" in body and body["steps"] is not None:
                                steps = int(body["steps"])
                                if not (1 <= steps <= 50):
                                    raise ValueError("steps must be in [1, 50]")
                                broadcast.state.steps = steps
                            if "layers" in body and body["layers"] is not None:
                                layers = int(body["layers"])
                                if not (35 <= layers <= 50):
                                    raise ValueError("layers must be in [35, 50]")
                                broadcast.state.layers = layers
                            if "reuse" in body and body["reuse"] is not None:
                                reuse = int(body["reuse"])
                                if not (1 <= reuse <= 3):
                                    raise ValueError("reuse must be in [1, 3]")
                                broadcast.state.reuse = reuse
                            if "ensemble_only" in body and body["ensemble_only"] is not None:
                                broadcast.state.ensemble_only = bool(body["ensemble_only"])
                            if "curated_share" in body and body["curated_share"] is not None:
                                broadcast.state.curated_share = float(body["curated_share"])
                            # Explicit overrides without a named preset → custom
                            if not preset_name and any(
                                k in body and body[k] is not None
                                for k in (
                                    "render_width",
                                    "render_height",
                                    "frames",
                                    "token_reduction",
                                    "steps",
                                    "layers",
                                    "reuse",
                                )
                            ):
                                broadcast.state.quality_preset = "custom"
                            broadcast.state.recipe_note = recipe_label(
                                preset=broadcast.state.quality_preset,
                                width=broadcast.state.width,
                                height=broadcast.state.height,
                                render_width=broadcast.state.render_width,
                                render_height=broadcast.state.render_height,
                                frames=broadcast.state.frames,
                                token_reduction=broadcast.state.token_reduction,
                                steps=broadcast.state.steps,
                                layers=broadcast.state.layers,
                                reuse=broadcast.state.reuse,
                            )
                            note = broadcast.state.recipe_note
                        log.info("recipe applied (stream stopped)\n%s", note)
                    except (KeyError, TypeError, ValueError) as exc:
                        self._json(400, {"ok": False, "error": str(exc)})
                        return
                    except SystemExit as exc:
                        self._json(400, {"ok": False, "error": str(exc)})
                        return
                else:
                    self._json(400, {"ok": False, "error": f"unknown action {action}"})
                    return
                self._json(200, {"ok": True, "action": action})

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp2t")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.end_headers()
                # Tiny probe so MSE players attach; not a video PES (no snow).
                try:
                    self.wfile.write(_TS_NULL * 3)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                q: queue.Queue[bytes] = queue.Queue(maxsize=512)
                with broadcast.lock:
                    broadcast.clients.add(q)
                log.info("viewer connected (%d now)", len(broadcast.clients))
                try:
                    while True:
                        try:
                            chunk = q.get(timeout=1.0)
                        except queue.Empty:
                            # Keepalive only between packets — never mid-chunk.
                            chunk = _TS_NULL * 2
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with broadcast.lock:
                        broadcast.clients.discard(q)
                        left = len(broadcast.clients)
                    log.info("viewer left (%d now)", left)
                    if broadcast.demand_count() == 0:
                        broadcast.cancel_generation("no viewers")

            def log_message(self, *_args) -> None:
                return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self._episode_runner = None  # set via set_episode_runner

    def set_episode_runner(self, runner) -> None:
        """Bound from main() so HTTP can start the exclusive batch worker."""
        self._episode_runner = runner

    def viewer_count(self) -> int:
        with self.lock:
            return len(self.clients)

    def demand_count(self) -> int:
        """TS stream clients + dashboard watch tokens (Play before media)."""
        return self.state.demand_count(self.viewer_count())

    def stop_live(self, reason: str) -> None:
        """Hard-stop Live so batch can own the engine exclusively."""
        with self.state.lock:
            self.state.episode_hold = True
            self.state.watch_sessions.clear()
            self.state.media_ready = False
        with self.lock:
            clients = list(self.clients)
            self.clients.clear()
        for q in clients:
            try:
                q.put_nowait(b"")
            except queue.Full:
                pass
        self.cancel_generation(reason, force=True)
        log.info("live stopped — %s", reason)

    def start_episode(self, scenes: int) -> None:
        with self.state.lock:
            if self.state.episode.is_active():
                raise RuntimeError("an episode is already generating")
            if self._episode_runner is None:
                raise RuntimeError("episode runner not ready")
            self.state.episode.reset_for_start(scenes)
        self.stop_live("episode batch started")
        threading.Thread(
            target=self._episode_runner,
            args=(scenes,),
            name="h3live-episode",
            daemon=True,
        ).start()

    def cancel_all(self, reason: str) -> None:
        with self.state.lock:
            if self.state.episode.is_active():
                self.state.episode.cancel.set()
        self.cancel_generation(reason, force=True)

    def cancel_generation(self, reason: str, *, force: bool = False) -> None:
        """Stop the in-flight ./h3 job immediately (e.g. last viewer disconnected)."""
        with self.state.lock:
            fn = self.state.cancel_fn
            was_generating = self.state.generating
            episode_busy = self.state.episode.is_active()
        # Viewer drop must not kill an exclusive episode batch.
        if episode_busy and not force:
            log.info("skip cancel (%s) — episode owns the engine", reason)
            return
        if not force and not was_generating:
            return
        log.info("cancel generation — %s", reason)
        if fn:
            try:
                fn()
            except Exception as exc:
                log.warning("cancel failed: %s", exc)

    def wait_for_viewer(self, stop: threading.Event, *, poll_s: float = 0.5) -> bool:
        """Block until Live is allowed and a viewer/watch is active (not paused)."""
        idle_logged = False
        while not stop.is_set():
            with self.state.lock:
                paused = self.state.paused
                hold = self.state.episode_hold or self.state.episode.is_active()
            if hold:
                if not idle_logged:
                    log.info("idle — episode batch owns the engine (live paused)")
                    idle_logged = True
                time.sleep(poll_s)
                continue
            if self.demand_count() > 0 and not paused:
                return True
            if not idle_logged:
                log.info(
                    "idle — waiting for Play on the dashboard or VLC → /stream.ts"
                )
                idle_logged = True
            time.sleep(poll_s)
        return False

    def push(self, chunk: bytes) -> None:
        with self.lock:
            targets = list(self.clients)
        for q in targets:
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass

    def push_file(self, path: Path, *, duration_s: float | None = None) -> None:
        """Push a .ts file; if ``duration_s`` is set, pace roughly in realtime.

        Chunk size must be a multiple of 188 (MPEG-TS packet). Keepalive null
        packets are also 188 bytes; inserting them between misaligned chunks
        permanently desyncs every browser/VLC parser.
        """
        data = Path(path).read_bytes()
        if not data:
            return
        with self.state.lock:
            if self.state.episode_hold or self.state.episode.is_active():
                log.info("drop paced clip — episode owns the engine")
                return
            self.state.media_ready = True
        # Dashboard Play holds a watch token first, then opens MSE after media_ready.
        # Wait briefly so the first TS bytes aren't discarded into an empty fan-out.
        deadline = time.time() + 20.0
        while self.viewer_count() == 0 and time.time() < deadline:
            if self.demand_count() == 0:
                log.info("drop clip — no demand before TS attach")
                return
            time.sleep(0.05)
        if self.viewer_count() == 0:
            log.warning("no TS client attached — dropping paced clip")
            return
        if not duration_s or duration_s <= 0:
            self.push(data)
            return
        # 174 packets ≈ 32 KiB, exact multiple of 188
        chunk = 188 * 174
        n = max(1, (len(data) + chunk - 1) // chunk)
        interval = duration_s / n
        t0 = time.time()
        for i in range(n):
            if self.viewer_count() == 0:
                if self.demand_count() == 0:
                    log.info("stop pacing — no viewers")
                    return
                time.sleep(0.05)
                continue
            self.push(data[i * chunk : (i + 1) * chunk])
            target = t0 + (i + 1) * interval
            delay = target - time.time()
            if delay > 0.001:
                time.sleep(delay)


class PaceMux:
    """Optional ``ffmpeg -re`` pace layer; otherwise push TS bytes directly."""

    def __init__(self, broadcast: Broadcast, use_re: bool) -> None:
        self.broadcast = broadcast
        self.proc = None
        self.stdin = None
        if not use_re:
            return
        from h3_paths import real_ffmpeg

        ffmpeg = real_ffmpeg()
        if not ffmpeg:
            log.warning("no ffmpeg for -re pacing — pushing TS directly (may burst)")
            return
        import subprocess

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-fflags",
            "+genpts",
            "-f",
            "mpegts",
            "-i",
            "pipe:0",
            "-c",
            "copy",
            "-mpegts_flags",
            "+resend_headers",
            "-pat_period",
            "0.1",
            "-f",
            "mpegts",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self.stdin = self.proc.stdin
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            chunk = self.proc.stdout.read(1 << 15)
            if not chunk:
                break
            self.broadcast.push(chunk)

    def write_ts(self, ts_path: Path, *, duration_s: float) -> None:
        if self.stdin is not None:
            feed_mpegts_to_sink(ts_path, self.stdin)
            self.stdin.flush()
        else:
            self.broadcast.push_file(ts_path, duration_s=duration_s)

    def close(self) -> None:
        if self.stdin is not None:
            try:
                self.stdin.close()
            except OSError:
                pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


def resolve_lora(lora_id: str, scale: float | None) -> LoraRef:
    entry = catalog_entry(lora_id)
    if entry is None:
        raise SystemExit(f"unknown LoRA id {lora_id!r} (see h3_lora.BUILTIN_LORAS)")
    spec = str(entry["spec"])
    info = ensure_lora(spec)
    sc = float(scale if scale is not None else entry.get("scale") or 1.0)
    return LoraRef(spec=spec, path=Path(info["path"]), scale=sc)


def resolve_live_model_dir(
    explicit: Path | None,
    *,
    prefer_stock: bool = False,
) -> Path:
    """Prefer fused-turbo / FastH3 student, else stock MiniMax-H3.

    ``prefer_stock`` forces the official FL2VA tree (needed for base+LoRA
    recipes like TaoMate 3-step — do not stack on the FastH3 student DiT).
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    stock = default_model_dir().resolve()
    if prefer_stock:
        return stock
    root = stock.parent  # …/models

    def usable(candidate: Path) -> bool:
        tr = candidate / "FL2VA" / "transformer"
        return tr.is_dir() and any(tr.glob("*.safetensors"))

    for name in (
        LIVE_MODEL_DIR_FUSED_TURBO_INT8_NAME,
        LIVE_MODEL_DIR_FUSED_TURBO_NAME,
        LIVE_MODEL_DIR_INT8_NAME,
        LIVE_MODEL_DIR_NAME,
    ):
        candidate = root / name
        if usable(candidate):
            return candidate.resolve()
    return stock


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FastH3 Live stream via Metal h3.c")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="h3 -d root (default: FusedTurbo / FastH3 student if present; "
        "stock MiniMax-H3 when --lora taomate_h3_3step)",
    )
    p.add_argument("--width", type=int, default=LIVE_WIDTH)
    p.add_argument("--height", type=int, default=LIVE_HEIGHT)
    p.add_argument(
        "--quality-preset",
        default=LIVE_QUALITY_PRESET,
        choices=sorted(PRESETS.keys()),
        help="draft|live|sharp|long — sets render/TR/frames (default: live)",
    )
    p.add_argument(
        "--render-width",
        type=int,
        default=None,
        help="internal DiT/VAE width (0 = same as --width; overrides preset)",
    )
    p.add_argument(
        "--render-height",
        type=int,
        default=None,
        help="internal DiT/VAE height (0 = same as --height; overrides preset)",
    )
    p.add_argument("--frames", type=int, default=None, help="overrides quality preset")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"denoising steps (default: {LIVE_STEPS}, or LoRA catalog steps)",
    )
    p.add_argument("--layers", type=int, default=LIVE_LAYERS)
    p.add_argument("--reuse", type=int, default=LIVE_REUSE)
    p.add_argument(
        "--fps",
        type=float,
        default=LIVE_PLAY_FPS,
        help="playback fps for retime; 0 = adaptive from measured gen time",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=LIVE_MARGIN_RATIO,
        help="adaptive only: play_budget = gen_s * margin (default 1.08)",
    )
    p.add_argument("--min-fps", type=float, default=LIVE_MIN_PLAY_FPS)
    p.add_argument("--max-fps", type=float, default=LIVE_MAX_PLAY_FPS)
    p.add_argument(
        "--token-reduction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pair middle-block video tokens (overrides preset)",
    )
    p.add_argument(
        "--int8-row-fc2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use int8 row-FC2 when TensorOps is available (default on)",
    )
    p.add_argument(
        "--lora",
        default=None,
        help="builtin LoRA id (e.g. taomate_h3_3step). Default: none with FastH3 student. "
        "TaoMate forces stock MiniMax-H3 FL2VA + 3 steps.",
    )
    p.add_argument("--lora-scale", type=float, default=None)
    p.add_argument(
        "--profile",
        action="store_true",
        help="pass h3 --profile (Metal phase lines on stderr; Live also logs "
        "progress-stage wall buckets by default)",
    )
    p.add_argument(
        "--fallback-lora",
        action="store_true",
        help=f"on stock FL2VA only: fuse {LIVE_LORA_ID} (not the student DiT)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--scenes",
        action="append",
        default=None,
        help="scene file (repeatable; replaces defaults when set)",
    )
    p.add_argument("--characters", type=Path, default=DEFAULT_CHARACTERS)
    p.add_argument("--curated-share", type=float, default=LIVE_CURATED_SHARE)
    p.add_argument(
        "--ensemble-only",
        action="store_true",
        help="draw only 3-character ensemble scenes",
    )
    p.add_argument(
        "--ensemble-bias",
        type=float,
        default=LIVE_ENSEMBLE_BIAS,
        help="probability of preferring an ensemble scene when available (0–1)",
    )
    p.add_argument(
        "--max-cast",
        type=int,
        default=LIVE_MAX_CAST,
        help="drop scenes with more than N character slots (default 3)",
    )
    p.add_argument("--prefill", type=int, default=2, help="clips to buffer before URL")
    p.add_argument("--stretch", default="rubberband")
    p.add_argument("--vbitrate", default="4M")
    p.add_argument("--max-clips", type=int, default=0, help="0 = forever")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--pace", action="store_true", help="ffmpeg -re pace if available")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    # Capture explicit overrides (None = use preset).
    ov_rw, ov_rh = args.render_width, args.render_height
    ov_frames, ov_tr = args.frames, args.token_reduction
    ov_steps = args.steps
    apply_preset_to_args(args, get_preset(args.quality_preset))
    if ov_rw is not None:
        args.render_width = ov_rw
    if ov_rh is not None:
        args.render_height = ov_rh
    if ov_frames is not None:
        args.frames = ov_frames
    if ov_tr is not None:
        args.token_reduction = ov_tr
    # Steps: explicit --steps wins; else LoRA catalog recipe; else Live default.
    if ov_steps is not None:
        args.steps = ov_steps
    elif args.lora:
        entry = catalog_entry(args.lora)
        cat_steps = entry.get("steps") if entry else None
        args.steps = int(cat_steps) if cat_steps is not None else LIVE_STEPS
    else:
        args.steps = LIVE_STEPS
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    scene_paths = (
        [Path(s) for s in args.scenes]
        if args.scenes
        else [Path(p) for p in DEFAULT_SCENES]
    )
    pool = PromptPool(
        scene_paths,
        args.characters,
        curated_share=args.curated_share,
        ensemble_only=bool(args.ensemble_only),
        ensemble_bias=float(args.ensemble_bias),
        max_cast=int(args.max_cast),
        explicit=bool(args.scenes),
    )
    counts = pool.counts()
    log.info(
        "prompt pool: %s = %d scenes × %d characters (%d curated, %.0f%% curated draws, "
        "ensemble_bias=%.0f%%%s)",
        " + ".join(f"{n} from {Path(p).name}" for p, n in counts.items()),
        sum(counts.values()),
        len(pool.full),
        len(pool.curated),
        args.curated_share * 100,
        args.ensemble_bias * 100,
        ", ensemble-only" if args.ensemble_only else "",
    )

    lora_entry = catalog_entry(args.lora) if args.lora else None
    prefer_stock = bool(
        args.lora
        and (
            str(args.lora).startswith("taomate")
            or (lora_entry and "taomate" in str(lora_entry.get("id") or "").lower())
        )
    )
    model_dir = resolve_live_model_dir(args.model_dir, prefer_stock=prefer_stock)
    student_names = (
        LIVE_MODEL_DIR_NAME,
        LIVE_MODEL_DIR_INT8_NAME,
        LIVE_MODEL_DIR_FUSED_TURBO_NAME,
        LIVE_MODEL_DIR_FUSED_TURBO_INT8_NAME,
    )
    using_student = any(name in model_dir.parts for name in student_names)
    using_fused = any(
        name in model_dir.parts
        for name in (LIVE_MODEL_DIR_FUSED_TURBO_NAME, LIVE_MODEL_DIR_FUSED_TURBO_INT8_NAME)
    )
    loras: list[LoraRef] = []
    lora_id: str | None = None
    lora_label: str | None = None
    if args.lora:
        if prefer_stock and using_student:
            log.error(
                "TaoMate/base LoRA cannot run on FastH3 student tree %s — "
                "pass --model-dir models/MiniMax-H3 or omit student trees",
                model_dir,
            )
            return 1
        lora_id = str(args.lora)
        entry = catalog_entry(lora_id) or {}
        lora_label = str(entry.get("label") or lora_id)
        loras = [resolve_lora(lora_id, args.lora_scale)]
        log.info(
            "LoRA %s @ %.2f → %s (steps=%d)",
            lora_id,
            loras[0].scale,
            loras[0].path,
            args.steps,
        )
    elif not using_student and args.fallback_lora:
        lora_id = LIVE_LORA_ID
        entry = catalog_entry(lora_id) or {}
        lora_label = str(entry.get("label") or lora_id)
        loras = [resolve_lora(LIVE_LORA_ID, args.lora_scale)]
        log.warning(
            "no FastH3 student tree at models/%s — fusing LoRA %s (slower/wrong recipe)",
            LIVE_MODEL_DIR_NAME,
            LIVE_LORA_ID,
        )
    elif not using_student:
        log.warning(
            "model_dir=%s is not the FastH3 student; run "
            "./scripts/prepare_fasth3_native_tree.sh after "
            "hf download FastVideo/…-Dense-DataFree --include 'transformer/*'",
            model_dir,
        )
    elif using_fused:
        log.info("Fused-turbo DiT at %s (no LoRA)", model_dir)
    else:
        log.info("FastH3 student DiT at %s (no LoRA)", model_dir)

    # Overlap DiT command encoding with GPU work on Metal (M3 Ultra inherits the
    # M5 GPU-sampler gate via TensorOps, which otherwise disables the split).
    os.environ.setdefault("H3_DIT_COMMAND_BLOCKS", "30")
    # Fixed text width so prepared DiT stays resident across Live prompts
    # (natural token counts rarely match; A/B: pad-160 warm −5.7s/clip @56f,
    # cold pad overhead ~0). 192 covers observed pool max ~185.
    os.environ.setdefault("H3_PAD_TEXT_TOKENS", "192")

    engine = H3Engine(model_dir=model_dir)
    info = engine.info()
    if not info.get("ok"):
        log.error("engine not ready: %s", info.get("error"))
        return 1

    out_root = Path(args.out_dir) if args.out_dir else Path(mk_scratch_dir("h3live_"))
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("clip scratch: %s", out_root)

    state = LiveState()
    state.width = args.width
    state.height = args.height
    state.frames = args.frames
    state.steps = args.steps
    state.layers = args.layers
    state.reuse = args.reuse
    state.play_fps = float(args.fps) if args.fps > 0 else 0.0
    state.cancel_fn = engine.request_cancel
    state.pool_scenes = sum(counts.values())
    state.next_prompt_note = f"random pool ({state.pool_scenes} scenes)"
    state.quality_preset = args.quality_preset
    state.render_width = args.render_width
    state.render_height = args.render_height
    state.token_reduction = bool(args.token_reduction)
    state.ensemble_only = bool(args.ensemble_only)
    state.curated_share = float(args.curated_share)
    state.recipe_note = recipe_label(
        preset=state.quality_preset,
        width=state.width,
        height=state.height,
        render_width=state.render_width,
        render_height=state.render_height,
        frames=state.frames,
        token_reduction=state.token_reduction,
        steps=state.steps,
        layers=state.layers,
        reuse=state.reuse,
    )
    if lora_id and loras:
        state.lora_id = lora_id
        state.lora_label = lora_label
        state.lora_scale = float(loras[0].scale)
        state.loras = list(loras)

    broadcast = Broadcast(args.host, args.port, state)
    mux = PaceMux(broadcast, use_re=args.pace)

    url_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{url_host}:{args.port}/"
    stop = threading.Event()

    def _sig(_signum, _frame) -> None:
        log.info("stopping…")
        stop.set()
        engine.request_cancel()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    adaptive = args.fps <= 0
    log.info(
        "recipe %s",
        state.recipe_note,
    )
    log.info(
        "play_fps=%s margin=%.2f",
        "adaptive" if adaptive else f"{args.fps:.2f}",
        args.margin,
    )
    log.info(
        "dashboard %s  |  VLC/ffplay → %sstream.ts  (also / if User-Agent is VLC)",
        url,
        url,
    )

    # Producer fills this; feeder paces to viewers. Prefill before advertising URL.
    clip_q: queue.Queue[tuple[Path, float, dict] | None] = queue.Queue(
        maxsize=max(2, args.prefill + 1)
    )
    gen_times: list[float] = []
    gen_ema: float | None = None
    ready = threading.Event()

    def run_episode(scenes: int) -> None:
        """Exclusive batch: sharp canvas, native 24 fps concat, no stream retime."""
        episode_dir = out_root / "episode"
        episode_dir.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []
        job = state.episode
        try:
            with state.lock:
                pool.ensemble_only = state.ensemble_only
                pool.curated_share = state.curated_share
                mode = state.prompt_mode
                custom = state.custom_prompt
            for i in range(scenes):
                if stop.is_set() or job.cancel.is_set():
                    raise RuntimeError("cancelled")
                with state.lock:
                    job.current_scene = i + 1
                if mode == "custom" and custom.strip() and i == 0:
                    prompt, cast = pool.normalize_custom(custom)
                    scene_idx = -1
                    source = "episode · custom first scene"
                else:
                    prompt, cast, scene_idx = pool.draw()
                    source = f"episode · pool scene {scene_idx:03d}"
                seed = (
                    args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
                )
                mp4 = episode_dir / f"scene_{i + 1:02d}.mp4"
                log.info(
                    "episode scene %d/%d  %s  seed %s  cast=%s\n%s\n"
                    "—— prompt (%d chars) ——\n%s\n—— end ——",
                    i + 1,
                    scenes,
                    source,
                    seed,
                    cast,
                    EPISODE_RECIPE_NOTE,
                    len(prompt),
                    prompt,
                )
                with state.lock:
                    state.generating = True
                    state.gen_started_at = time.time()
                    state.progress = None
                    job.last_cast = cast[:120]
                    active_loras = list(state.loras)
                t0 = time.time()
                req = GenerateRequest(
                    prompt=prompt,
                    output_path=mp4,
                    width=EPISODE_WIDTH,
                    height=EPISODE_HEIGHT,
                    render_width=EPISODE_RENDER_WIDTH,
                    render_height=EPISODE_RENDER_HEIGHT,
                    num_frames=EPISODE_FRAMES,
                    quality="four_step",
                    steps=args.steps,
                    layers=EPISODE_LAYERS,
                    reuse=EPISODE_REUSE,
                    token_reduction=EPISODE_TOKEN_REDUCTION,
                    seed=seed,
                    ssd_streaming=False,
                    int8_row_fc2=bool(args.int8_row_fc2),
                    profile=bool(args.profile),
                    oneshot=False,
                    loras=active_loras,
                    mode="t2va",
                )
                phases = PhaseTimer()

                def _ep_progress(mp: dict) -> None:
                    phases.on_progress(mp)
                    state.set_progress(mp)

                try:
                    engine.generate(req, on_progress=_ep_progress)
                finally:
                    with state.lock:
                        state.generating = False
                        state.progress = None
                        state.gen_started_at = None
                if stop.is_set() or job.cancel.is_set():
                    raise RuntimeError("cancelled")
                if not mp4.is_file():
                    raise RuntimeError(f"missing output {mp4}")
                gen_s = time.time() - t0
                phase_line = phases.summary()
                log.info(
                    "episode scene %d/%d done in %.1fs  phases %s",
                    i + 1,
                    scenes,
                    gen_s,
                    phase_line,
                )
                with state.lock:
                    state.last_phases = phase_line
                clips.append(mp4)
                with state.lock:
                    job.scenes_done = i + 1
                    job.clip_paths = list(clips)

            dest = episode_dir / "episode.mp4"
            log.info("episode concat %d scenes → %s", len(clips), dest)
            concat_mp4s(clips, dest)
            with state.lock:
                job.output_path = dest
                job.status = "ready"
                job.finished_at = time.time()
                job.current_scene = None
                state.episode_hold = False
            log.info(
                "episode ready — download %s (%.1f MiB)",
                dest,
                dest.stat().st_size / (1024 * 1024),
            )
        except Exception as exc:
            cancelled = job.cancel.is_set() or str(exc) == "cancelled"
            with state.lock:
                job.status = "cancelled" if cancelled else "error"
                job.error = None if cancelled else str(exc)
                job.finished_at = time.time()
                job.current_scene = None
                state.episode_hold = False
                state.generating = False
            if cancelled:
                log.info("episode cancelled")
            else:
                log.error("episode failed: %s", exc)

    broadcast.set_episode_runner(run_episode)

    def producer() -> None:
        nonlocal gen_ema
        clip_i = 0
        ts_offset = 0.0
        while not stop.is_set():
            if args.max_clips and clip_i >= args.max_clips:
                break
            # Do not burn Metal while nobody is watching (or while episode owns GPU).
            if not broadcast.wait_for_viewer(stop):
                break
            with state.lock:
                if state.episode_hold or state.episode.is_active():
                    continue
                mode = state.prompt_mode
                custom = state.custom_prompt
                render_w = state.render_width
                render_h = state.render_height
                frames = state.frames
                steps = state.steps
                layers = state.layers
                reuse = state.reuse
                token_reduction = state.token_reduction
                recipe = state.recipe_note
                pool.ensemble_only = state.ensemble_only
                pool.curated_share = state.curated_share
            if mode == "custom" and custom.strip():
                was_idea = not PromptPool.looks_like_context_ir(custom)
                prompt, cast = pool.normalize_custom(custom)
                scene_idx = -1
                source = (
                    "custom · idea→Context-IR wrap"
                    if was_idea
                    else "custom · Context-IR as pasted"
                )
            else:
                prompt, cast, scene_idx = pool.draw()
                source = f"pool scene {scene_idx:03d}"
            seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
            mp4 = out_root / f"clip_{clip_i:04d}.mp4"
            log.info(
                "clip %04d  %s  seed %s  cast=%s\n"
                "%s\n"
                "—— prompt sent to h3 (%d chars) ——\n%s\n—— end prompt ——",
                clip_i,
                source,
                seed,
                cast,
                recipe,
                len(prompt),
                prompt,
            )
            with state.lock:
                state.generating = True
                state.gen_started_at = time.time()
                state.progress = None
                active_loras = list(state.loras)
            t0 = time.time()
            req = GenerateRequest(
                prompt=prompt,
                output_path=mp4,
                width=args.width,
                height=args.height,
                render_width=render_w or None,
                render_height=render_h or None,
                num_frames=frames,
                quality="four_step",
                steps=steps,
                layers=layers,
                reuse=reuse,
                token_reduction=bool(token_reduction),
                seed=seed,
                ssd_streaming=False,
                int8_row_fc2=bool(args.int8_row_fc2),
                profile=bool(args.profile),
                oneshot=False,  # warm FL2VA via !prompt-file (avoids linenoise deadlock)
                loras=active_loras,
                mode="t2va",
            )
            phases = PhaseTimer()

            def _live_progress(mp: dict) -> None:
                phases.on_progress(mp)
                state.set_progress(mp)

            try:
                engine.generate(req, on_progress=_live_progress)
            except Exception as exc:
                log.error("generate failed: %s", exc)
                with state.lock:
                    state.generating = False
                    state.progress = None
                    state.gen_started_at = None
                if stop.is_set():
                    break
                time.sleep(2)
                continue

            gen_s = time.time() - t0
            phase_line = phases.summary()
            log.info("phases %s", phase_line)
            with state.lock:
                state.generating = False
                state.progress = None
                state.gen_started_at = None
                state.last_phases = phase_line

            # Last viewer dropped mid-job — do not retime/queue a partial clip.
            if broadcast.demand_count() == 0:
                log.info("clip %04d discarded — no viewers after cancel (%.1fs)", clip_i, gen_s)
                continue

            gen_times.append(gen_s)
            gen_ema = ema(gen_ema, gen_s)
            avg = sum(gen_times[-5:]) / len(gen_times[-5:])

            if adaptive:
                play_fps = adaptive_play_fps(
                    frames,
                    gen_ema,
                    margin_ratio=args.margin,
                    min_fps=args.min_fps,
                    max_fps=args.max_fps,
                )
            else:
                play_fps = float(args.fps)
            play_seconds = frames / play_fps
            margin = play_seconds - gen_s
            sustain = frames / avg if avg > 0 else 0.0
            status = "OK" if margin >= 0 else "BEHIND"
            log.info(
                "gen %.1fs (avg5 %.1f ema %.1f)  play_fps %.2f (%.1fs)  "
                "margin %+.1fs/clip  sustains %.1ffps  %s%s",
                gen_s,
                avg,
                gen_ema,
                play_fps,
                play_seconds,
                margin,
                sustain,
                status,
                " [adaptive]" if adaptive else "",
            )
            with state.lock:
                state.play_fps = play_fps
                state.last_gen_s = gen_s
                state.margin_s = margin
                state.last_clip = clip_i
                state.last_scene = scene_idx
                state.last_cast = cast[:120]
                state.frames = frames

            try:
                ts = retime_to_mpegts(
                    mp4,
                    out_root / f"clip_{clip_i:04d}.ts",
                    play_fps=play_fps,
                    stretch=args.stretch,
                    vbitrate=args.vbitrate,
                    ts_offset=ts_offset,
                )
            except Exception as exc:
                log.error("retime failed: %s", exc)
                continue

            meta = {
                "clip": clip_i,
                "scene": scene_idx,
                "seed": seed,
                "cast": cast,
                "gen_s": gen_s,
                "play_fps": play_fps,
            }
            while not stop.is_set():
                try:
                    clip_q.put((ts, play_seconds, meta), timeout=0.5)
                    break
                except queue.Full:
                    continue
            ts_offset += play_seconds
            clip_i += 1
        clip_q.put(None)

    def feeder() -> None:
        streamed = 0
        while not stop.is_set():
            try:
                item = clip_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            ts_path, dur, meta = item
            with state.lock:
                if state.episode_hold or state.episode.is_active():
                    log.info(
                        "drop live clip %04d — episode mode (exclusive)",
                        meta.get("clip", -1),
                    )
                    continue
            streamed += 1
            if not ready.is_set() and streamed >= max(1, args.prefill):
                ready.set()
                log.info("stream ready → %s  (open in VLC)", url)
            # Prefill: dump fast. After ready: realtime pace so gen overlaps play.
            pace = dur if ready.is_set() else 0.0
            log.info(
                "streamed %04d  scene %03d  seed %s  buf=%d  gen %.1fs  play_fps %.2f",
                meta["clip"],
                meta["scene"],
                meta["seed"],
                clip_q.qsize(),
                meta["gen_s"],
                meta.get("play_fps", 0.0),
            )
            mux.write_ts(ts_path, duration_s=pace)

    prod_t = threading.Thread(target=producer, name="h3live-prod", daemon=True)
    feed_t = threading.Thread(target=feeder, name="h3live-feed", daemon=True)
    prod_t.start()
    feed_t.start()

    try:
        while prod_t.is_alive() and not stop.is_set():
            prod_t.join(timeout=0.5)
        stop.set()
        with state.lock:
            join_s = (args.frames / state.play_fps) if state.play_fps > 0 else 90.0
        feed_t.join(timeout=join_s + 5)
    finally:
        mux.close()
        engine.shutdown(wait=False)
        log.info("stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
