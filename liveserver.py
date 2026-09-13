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
    LIVE_FRAMES,
    LIVE_HEIGHT,
    LIVE_LAYERS,
    LIVE_LORA_ID,
    LIVE_MARGIN_RATIO,
    LIVE_MAX_PLAY_FPS,
    LIVE_MIN_PLAY_FPS,
    LIVE_MODEL_DIR_INT8_NAME,
    LIVE_MODEL_DIR_NAME,
    LIVE_PLAY_FPS,
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
from h3_live.pace import adaptive_play_fps, ema  # noqa: E402
from h3_live.retime import feed_mpegts_to_sink, retime_to_mpegts  # noqa: E402
from h3_live.scenes import PromptPool  # noqa: E402
from h3_lora import catalog_entry, ensure_lora  # noqa: E402
from h3_paths import default_model_dir, mk_scratch_dir  # noqa: E402

log = logging.getLogger("h3-live")


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
        self.play_fps = 0.0
        self.cancel_fn = None  # set to engine.request_cancel
        # Prompt control applies to the *next* clip (in-flight job keeps its prompt).
        self.prompt_mode = "random"  # random | custom
        self.custom_prompt = ""
        self.pool_scenes = 0
        self.next_prompt_note = "random pool"

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
                "play_fps": self.play_fps,
                "prompt_mode": self.prompt_mode,
                "custom_prompt": custom,
                "custom_preview": preview,
                "pool_scenes": self.pool_scenes,
                "next_prompt_note": self.next_prompt_note,
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
                if wants_ts:
                    self._stream()
                    return
                self.send_error(404, "use / for dashboard, /stream.ts for MPEG-TS (VLC)")

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
                    broadcast.cancel_generation("dashboard cancel", force=True)
                elif action == "set_prompt":
                    mode = str(body.get("mode") or "random").strip().lower()
                    text = str(body.get("text") or "")
                    if mode not in ("random", "custom"):
                        self._json(400, {"ok": False, "error": "mode must be random|custom"})
                        return
                    if mode == "custom" and not text.strip():
                        self._json(400, {"ok": False, "error": "custom text required"})
                        return
                    with broadcast.state.lock:
                        broadcast.state.prompt_mode = mode
                        if mode == "custom":
                            broadcast.state.custom_prompt = text
                            note = text.strip().replace("\n", " ")
                            if len(note) > 80:
                                note = note[:77] + "…"
                            broadcast.state.next_prompt_note = f"custom: {note}"
                        else:
                            broadcast.state.next_prompt_note = (
                                f"random pool ({broadcast.state.pool_scenes} scenes)"
                            )
                    log.info(
                        "prompt mode → %s (%s)",
                        mode,
                        "applies to next clip" if broadcast.state.generating else "ready",
                    )
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

    def viewer_count(self) -> int:
        with self.lock:
            return len(self.clients)

    def demand_count(self) -> int:
        """TS stream clients + dashboard watch tokens (Play before media)."""
        return self.state.demand_count(self.viewer_count())

    def cancel_generation(self, reason: str, *, force: bool = False) -> None:
        """Stop the in-flight ./h3 job immediately (e.g. last viewer disconnected)."""
        with self.state.lock:
            fn = self.state.cancel_fn
            was_generating = self.state.generating
        if not force and not was_generating:
            return
        log.info("cancel generation — %s", reason)
        if fn:
            try:
                fn()
            except Exception as exc:
                log.warning("cancel failed: %s", exc)

    def wait_for_viewer(self, stop: threading.Event, *, poll_s: float = 0.5) -> bool:
        """Block until a stream viewer or dashboard watch is active and not paused."""
        idle_logged = False
        while not stop.is_set():
            with self.state.lock:
                paused = self.state.paused
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


def resolve_live_model_dir(explicit: Path | None) -> Path:
    """Prefer INT8 FastH3 student, then BF16, else stock MiniMax-H3."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    root = default_model_dir().resolve().parent  # …/models

    def usable(candidate: Path) -> bool:
        tr = candidate / "FL2VA" / "transformer"
        return tr.is_dir() and any(tr.glob("*.safetensors"))

    for name in (LIVE_MODEL_DIR_INT8_NAME, LIVE_MODEL_DIR_NAME):
        candidate = root / name
        if usable(candidate):
            return candidate.resolve()
    return default_model_dir().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FastH3 Live stream via Metal h3.c")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="h3 -d root (default: models/MiniMax-H3-FastH3 if present)",
    )
    p.add_argument("--width", type=int, default=LIVE_WIDTH)
    p.add_argument("--height", type=int, default=LIVE_HEIGHT)
    p.add_argument(
        "--render-width",
        type=int,
        default=LIVE_RENDER_WIDTH,
        help="internal DiT/VAE width (0 = same as --width)",
    )
    p.add_argument(
        "--render-height",
        type=int,
        default=LIVE_RENDER_HEIGHT,
        help="internal DiT/VAE height (0 = same as --height)",
    )
    p.add_argument("--frames", type=int, default=LIVE_FRAMES)
    p.add_argument("--steps", type=int, default=LIVE_STEPS)
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
        default=LIVE_TOKEN_REDUCTION,
        help="pair middle-block video tokens (Metal speed)",
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
        help="builtin LoRA id (default: none when FastH3 student tree is used)",
    )
    p.add_argument("--lora-scale", type=float, default=None)
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
    p.add_argument("--curated-share", type=float, default=0.30)
    p.add_argument("--prefill", type=int, default=2, help="clips to buffer before URL")
    p.add_argument("--stretch", default="rubberband")
    p.add_argument("--vbitrate", default="4M")
    p.add_argument("--max-clips", type=int, default=0, help="0 = forever")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--pace", action="store_true", help="ffmpeg -re pace if available")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


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
        explicit=bool(args.scenes),
    )
    counts = pool.counts()
    log.info(
        "prompt pool: %s = %d scenes × %d characters (%d curated, %.0f%% curated draws)",
        " + ".join(f"{n} from {Path(p).name}" for p, n in counts.items()),
        sum(counts.values()),
        len(pool.full),
        len(pool.curated),
        args.curated_share * 100,
    )

    model_dir = resolve_live_model_dir(args.model_dir)
    using_student = any(
        name in model_dir.parts
        for name in (LIVE_MODEL_DIR_NAME, LIVE_MODEL_DIR_INT8_NAME)
    )
    loras: list[LoraRef] = []
    if args.lora:
        loras = [resolve_lora(args.lora, args.lora_scale)]
        log.info("LoRA %s @ %.2f → %s", args.lora, loras[0].scale, loras[0].path)
    elif not using_student and args.fallback_lora:
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
    else:
        log.info("FastH3 student DiT at %s (no LoRA)", model_dir)

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
    state.play_fps = float(args.fps) if args.fps > 0 else 0.0
    state.cancel_fn = engine.request_cancel
    state.pool_scenes = sum(counts.values())
    state.next_prompt_note = f"random pool ({state.pool_scenes} scenes)"

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
        "recipe %dx%d (render %dx%d) frames=%d steps=%d play_fps=%s margin=%.2f",
        args.width,
        args.height,
        args.render_width or args.width,
        args.render_height or args.height,
        args.frames,
        args.steps,
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

    def producer() -> None:
        nonlocal gen_ema
        clip_i = 0
        ts_offset = 0.0
        while not stop.is_set():
            if args.max_clips and clip_i >= args.max_clips:
                break
            # Do not burn Metal while nobody is watching.
            if not broadcast.wait_for_viewer(stop):
                break
            with state.lock:
                mode = state.prompt_mode
                custom = state.custom_prompt
            if mode == "custom" and custom.strip():
                prompt, cast = pool.fill_names(custom)
                scene_idx = -1
            else:
                prompt, cast, scene_idx = pool.draw()
            seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
            mp4 = out_root / f"clip_{clip_i:04d}.mp4"
            log.info(
                "clip %04d  %s  seed %s  %s",
                clip_i,
                f"custom" if scene_idx < 0 else f"scene {scene_idx:03d}",
                seed,
                cast[:80],
            )
            with state.lock:
                state.generating = True
            t0 = time.time()
            req = GenerateRequest(
                prompt=prompt,
                output_path=mp4,
                width=args.width,
                height=args.height,
                render_width=args.render_width or None,
                render_height=args.render_height or None,
                num_frames=args.frames,
                quality="four_step",
                steps=args.steps,
                layers=args.layers,
                reuse=args.reuse,
                token_reduction=bool(args.token_reduction),
                seed=seed,
                ssd_streaming=False,
                int8_row_fc2=bool(args.int8_row_fc2),
                profile=False,
                oneshot=True,  # long Live prompts deadlock linenoise on PTY
                loras=loras,
                mode="t2va",
            )
            try:
                engine.generate(req)
            except Exception as exc:
                log.error("generate failed: %s", exc)
                with state.lock:
                    state.generating = False
                if stop.is_set():
                    break
                time.sleep(2)
                continue

            gen_s = time.time() - t0
            with state.lock:
                state.generating = False

            # Last viewer dropped mid-job — do not retime/queue a partial clip.
            if broadcast.demand_count() == 0:
                log.info("clip %04d discarded — no viewers after cancel (%.1fs)", clip_i, gen_s)
                continue

            gen_times.append(gen_s)
            gen_ema = ema(gen_ema, gen_s)
            avg = sum(gen_times[-5:]) / len(gen_times[-5:])

            if adaptive:
                play_fps = adaptive_play_fps(
                    args.frames,
                    gen_ema,
                    margin_ratio=args.margin,
                    min_fps=args.min_fps,
                    max_fps=args.max_fps,
                )
            else:
                play_fps = float(args.fps)
            play_seconds = args.frames / play_fps
            margin = play_seconds - gen_s
            sustain = args.frames / avg if avg > 0 else 0.0
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
