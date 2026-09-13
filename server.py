#!/usr/bin/env python3
"""
server.py — Local MiniMax-H3 WebSocket + Web UI server (h3.c / Metal)
=====================================================================
Wraps the native ``h3`` binary. One generation at a time; streams MP4 over
WebSocket. Embeds the React UI on the same port by default.

  python server.py
  python server.py --host 127.0.0.1 --port 8765 --width 512 --height 512
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

NotifyJson = Callable[..., Awaitable[None]]

from h3_bootstrap import ensure_python_requirements

ensure_python_requirements()
import websockets  # noqa: E402

from h3_backend import (  # noqa: E402
    H3_DEFAULT_LAYERS,
    H3_DEFAULT_REUSE,
    H3_DEFAULT_STEPS,
    QUALITY_PRESETS,
    GenerateRequest,
    GenerationCancelledError,
    H3Engine,
    parse_refs_payload,
    ram_gb,
)
from h3_media import require_ui_canvas, snap_frames  # noqa: E402
from h3_paths import REPO_ROOT, debug_console, default_h3_bin, default_model_dir, real_ffmpeg  # noqa: E402

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_NUM_FRAMES = 22
DEFAULT_QUALITY = "fast"
DEFAULT_CHUNK_SIZE = 64 * 1024
GENERATION_KEEPALIVE_INTERVAL_S = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("h3server")


class WsProtocolAdapter:
    """Wrap a Starlette WebSocket so RequestHandler can use send/recv like websockets."""

    def __init__(self, websocket: Any, remote_address: Any = None) -> None:
        self._ws = websocket
        self.remote_address = remote_address or getattr(websocket, "client", ("?",))

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, bytes):
            await self._ws.send_bytes(data)
        else:
            await self._ws.send_text(data)

    async def recv(self) -> str | bytes:
        msg = await self._ws.receive()
        if msg.get("type") == "websocket.disconnect":
            raise websockets.exceptions.ConnectionClosed(msg.get("code", 1000), "")
        if msg.get("bytes") is not None:
            return msg["bytes"]
        return msg.get("text", "")

    def __aiter__(self) -> WsProtocolAdapter:
        return self

    async def __anext__(self) -> str | bytes:
        try:
            return await self.recv()
        except websockets.exceptions.ConnectionClosed:
            raise StopAsyncIteration from None


class GenerationScheduler:
    __slots__ = ("_gen_lock", "_meta", "_n_waiters", "_running_id")

    def __init__(self) -> None:
        self._gen_lock = asyncio.Lock()
        self._meta = asyncio.Lock()
        self._n_waiters = 0
        self._running_id: str | None = None

    @property
    def running_generation_id(self) -> str | None:
        return self._running_id

    @contextlib.asynccontextmanager
    async def generation_slot(self, notify: NotifyJson):
        async with self._meta:
            self._n_waiters += 1
            ahead = self._n_waiters - 1
        try:
            if self._gen_lock.locked():
                async with self._meta:
                    active = self._running_id
                await notify(
                    type="queue_status",
                    position=max(1, ahead),
                    available_gpus=0,
                    total_gpus=1,
                    active_generation_id=active,
                )
            async with self._gen_lock:
                gid = str(uuid.uuid4())
                async with self._meta:
                    self._running_id = gid
                try:
                    yield gid
                finally:
                    async with self._meta:
                        self._running_id = None
        finally:
            async with self._meta:
                self._n_waiters = max(0, self._n_waiters - 1)


class RequestHandler:
    def __init__(
        self,
        ws: Any,
        engine: H3Engine,
        defaults: dict[str, Any],
        chunk_size: int,
        scheduler: GenerationScheduler,
        spill_dir: Path,
    ) -> None:
        self.ws = ws
        self.engine = engine
        self.defaults = defaults
        self.chunk_size = chunk_size
        self.scheduler = scheduler
        self.spill_dir = spill_dir
        self._session: dict = {}

    async def _send_json(self, **kwargs: Any) -> None:
        await self.ws.send(json.dumps(kwargs))

    async def handle(self) -> None:
        async for frame in self.ws:
            if isinstance(frame, bytes):
                continue
            try:
                msg = json.loads(frame)
            except (json.JSONDecodeError, ValueError):
                continue
            t = msg.get("type", "")
            if t == "session_init_v2":
                self._session = msg
            elif t == "simple_generate":
                await self._handle_generate(msg)
                return

    def _msg_int(self, msg: dict, name: str, default: int) -> int:
        raw = msg.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    async def _handle_generate(self, msg: dict) -> None:
        prompt = str(msg.get("prompt") or "").strip()
        if not prompt:
            await self._send_json(type="error", error_code="invalid_prompt", message="Empty prompt")
            return

        async def _notify_queue(**kwargs: Any) -> None:
            await self._send_json(**kwargs)

        async with self.scheduler.generation_slot(_notify_queue) as generation_id:
            t_start = time.time()
            width = self._msg_int(msg, "width", int(self.defaults["width"]))
            height = self._msg_int(msg, "height", int(self.defaults["height"]))
            try:
                width, height = require_ui_canvas(width, height)
            except ValueError as exc:
                await self._send_json(type="error", error_code="invalid_canvas", message=str(exc))
                return
            frames = snap_frames(self._msg_int(msg, "num_frames", int(self.defaults["num_frames"])))
            steps = self._msg_int(
                msg, "num_steps", int(self.defaults.get("num_steps") or H3_DEFAULT_STEPS)
            )
            layers = (
                self._msg_int(msg, "layers", H3_DEFAULT_LAYERS)
                if "layers" in msg
                else None
            )
            reuse = (
                self._msg_int(msg, "reuse", H3_DEFAULT_REUSE) if "reuse" in msg else None
            )
            seed = self._msg_int(msg, "seed", -1)
            quality = str(msg.get("quality") or self.defaults.get("quality") or "fast")
            if "ssd_streaming" in msg:
                ssd = bool(msg.get("ssd_streaming"))
            else:
                ssd = bool(self.defaults.get("ssd_streaming"))
            image = msg.get("initial_image") or self._session.get("initial_image")
            end_image = msg.get("end_image") or self._session.get("end_image")
            first = Path(str(image)) if isinstance(image, str) and image else None
            last = Path(str(end_image)) if isinstance(end_image, str) and end_image else None
            try:
                refs = parse_refs_payload(msg.get("refs"))
            except ValueError as exc:
                await self._send_json(type="error", error_code="invalid_refs", message=str(exc))
                return
            if refs and (first or last):
                await self._send_json(
                    type="error",
                    error_code="mixed_checkpoints",
                    message="Ref2VA references cannot be mixed with first/last-frame anchors",
                )
                return
            if refs:
                first = None
                last = None
            rw = msg.get("render_width")
            rh = msg.get("render_height")
            try:
                render_width = int(rw) if rw else None
                render_height = int(rh) if rh else None
            except (TypeError, ValueError):
                await self._send_json(
                    type="error",
                    error_code="invalid_render",
                    message="render_width and render_height must be integers",
                )
                return

            from h3_backend import scratch_output

            out = scratch_output("ws_")
            req = GenerateRequest(
                prompt=prompt,
                output_path=out,
                width=width,
                height=height,
                num_frames=frames,
                quality=quality,
                steps=steps,
                layers=layers,
                reuse=reuse,
                seed=None if seed < 0 else seed,
                ssd_streaming=ssd,
                first_frame=first,
                last_frame=last,
                refs=refs,
                render_width=render_width,
                render_height=render_height,
                token_reduction=bool(msg["token_reduction"])
                if "token_reduction" in msg
                else None,
                mode="ref2va" if refs else str(msg.get("mode") or "t2va"),
            )
            log.info(
                "  ▶ generation %s  %s  %sx%s  frames=%s  quality=%s",
                generation_id[:8],
                prompt[:72],
                width,
                height,
                frames,
                quality,
            )
            await self._send_json(
                type="gpu_assigned",
                gpu_id="metal:0",
                session_timeout=7200,
                generation_id=generation_id,
            )
            await self._send_json(type="ltx2_stream_start", total_segments=1, stream_mode="single")
            await self._send_json(type="ltx2_segment_start", segment_idx=0, total_segments=1)

            async def _keepalive() -> None:
                while True:
                    await asyncio.sleep(GENERATION_KEEPALIVE_INTERVAL_S)
                    try:
                        mp = self.engine.model_progress_for_ws()
                        payload: dict[str, Any] = {
                            "type": "generation_keepalive",
                            "elapsed_s": round(time.time() - t_start, 1),
                            "phase": "generating",
                            "generation_id": generation_id,
                        }
                        if mp:
                            payload["model_progress"] = mp
                        await self._send_json(**payload)
                    except Exception:
                        break

            ka_task = asyncio.create_task(_keepalive())
            video_path: str | None = None
            try:
                video_path = await asyncio.to_thread(self.engine.generate, req)
            except GenerationCancelledError as exc:
                await self._send_json(type="error", error_code="cancelled", message=str(exc))
                return
            except Exception as exc:
                log.error("Generation %s failed: %s", generation_id, exc)
                await self._send_json(type="error", error_code="generation_failed", message=str(exc))
                return
            finally:
                ka_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ka_task

            data = Path(video_path).read_bytes()
            offset = 0
            while offset < len(data):
                chunk = data[offset : offset + self.chunk_size]
                await self.ws.send(chunk)
                offset += len(chunk)
            elapsed_ms = int((time.time() - t_start) * 1000)
            await self._send_json(type="ltx2_segment_complete", segment_idx=0, bytes=len(data))
            await self._send_json(type="ltx2_stream_complete", total_bytes=len(data))
            await self._send_json(
                type="latency",
                generation_id=generation_id,
                e2e_ms=elapsed_ms,
                generation_ms=elapsed_ms,
            )
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass


class VideoServer:
    def __init__(
        self,
        host: str,
        port: int,
        engine: H3Engine,
        defaults: dict[str, Any],
        chunk_size: int,
        spill_dir: Path,
    ) -> None:
        self.host = host
        self.port = port
        self.engine = engine
        self.defaults = defaults
        self.chunk_size = chunk_size
        self.spill_dir = spill_dir
        self.scheduler = GenerationScheduler()

    async def handle_ws_connection(self, ws: Any) -> None:
        addr = getattr(ws, "remote_address", "?")
        log.info("  ┌ connect  %s", addr)
        try:
            handler = RequestHandler(
                ws=ws,
                engine=self.engine,
                defaults=self.defaults,
                chunk_size=self.chunk_size,
                scheduler=self.scheduler,
                spill_dir=self.spill_dir,
            )
            await handler.handle()
        except Exception as exc:
            log.error("  ✗ error  %s  %s", addr, exc)
        finally:
            log.info("  └ disconnect  %s", addr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="h3-ws — MiniMax-H3 WebSocket + Web UI (h3.c)")
    net = p.add_argument_group("network")
    net.add_argument("--host", default=os.environ.get("H3_WS_HOST", DEFAULT_HOST))
    net.add_argument("--port", type=int, default=int(os.environ.get("H3_WS_PORT", DEFAULT_PORT)))
    mdl = p.add_argument_group("model")
    mdl.add_argument("--h3-bin", type=Path, default=default_h3_bin())
    mdl.add_argument("--model-dir", type=Path, default=default_model_dir())
    vid = p.add_argument_group("video")
    vid.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    vid.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    vid.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    vid.add_argument("--quality", default=DEFAULT_QUALITY, choices=sorted(QUALITY_PRESETS))
    vid.add_argument(
        "--ssd-streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream DiT weights from SSD (saves RAM, much slower denoise). Default: off.",
    )
    ui = p.add_argument_group("web ui")
    ui.add_argument("--web-ui", action=argparse.BooleanOptionalAction, default=True)
    ui.add_argument("--web-output-dir", type=Path, default=REPO_ROOT / "web_outputs")
    misc = p.add_argument_group("misc")
    misc.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.verbose or debug_console():
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("h3").setLevel(logging.DEBUG)
        logging.getLogger("h3-session").setLevel(logging.DEBUG)
        logging.getLogger("h3-backend").setLevel(logging.DEBUG)
    if not debug_console() and not args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    try:
        width, height = require_ui_canvas(args.width, args.height)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    frames = snap_frames(args.num_frames)
    ssd = bool(args.ssd_streaming)

    engine = H3Engine(
        h3_bin=args.h3_bin,
        model_dir=args.model_dir,
        default_ssd_streaming=bool(ssd),
    )
    defaults = {
        "width": width,
        "height": height,
        "num_frames": frames,
        "num_steps": H3_DEFAULT_STEPS,
        "layers": H3_DEFAULT_LAYERS,
        "reuse": H3_DEFAULT_REUSE,
        "fps": 24,
        "quality": args.quality,
        "ssd_streaming": bool(ssd),
    }
    spill_dir = Path(args.web_output_dir) / ".spill"
    video_server = VideoServer(
        host=args.host,
        port=args.port,
        engine=engine,
        defaults=defaults,
        chunk_size=DEFAULT_CHUNK_SIZE,
        spill_dir=spill_dir,
    )

    from web_ui import (
        AppState,
        build_combined_application,
        build_server_urls,
        ensure_web_dist_built,
        run_uvicorn,
    )

    ws_url, http_url = build_server_urls(args.host, args.port)
    gb = ram_gb()
    print(f"\n{'═' * 60}")
    print("  h3-ws  —  MiniMax-H3 on Apple Silicon (h3.c)")
    print(f"  Binary   : {engine.h3_bin}")
    print(f"  Model    : {engine.model_dir}")
    print(f"  Canvas   : {width}×{height}  frames={frames}  quality={args.quality}")
    if gb is not None:
        print(f"  RAM      : ~{gb:.0f} GB unified" + ("  (SSD streaming on — slower denoise)" if ssd else ""))
    if engine.metal4:
        print("  GPU      : Metal 4 — --use-int8-row-fc2 on")
    mux = real_ffmpeg()
    print(f"  Muxer    : {mux or 'h3-av (PyAV shim)'}")
    print(f"  Endpoint : {ws_url}")
    if args.web_ui:
        print(f"  Web UI   : {http_url}")
        ensure_web_dist_built(auto_build=True)
    print(
        f"  Console  : h3 progress {'on (DEBUG=false to quiet)' if debug_console() else 'quiet'}"
    )
    print(f"{'═' * 60}\n", flush=True)

    info = engine.info()
    if not info.get("ok"):
        log.warning("Engine not ready: %s", info.get("error"))
        log.warning("Build with scripts/build_h3.sh and download weights with scripts/download_model.py")

    state = AppState(
        output_dir=Path(args.web_output_dir).resolve(),
        upload_dir=(REPO_ROOT / "web_uploads").resolve(),
        engine=engine,
        embedded=True,
        http_url=http_url,
        server_url=ws_url,
        runtime_defaults=defaults,
    )

    async def _ws(ws: Any) -> None:
        await video_server.handle_ws_connection(ws)

    app = build_combined_application(_ws, state)
    asyncio.run(run_uvicorn(app, args.host, args.port, state))


if __name__ == "__main__":
    main()
