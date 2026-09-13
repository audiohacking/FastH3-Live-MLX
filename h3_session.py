"""Resident interactive ``./h3`` session (PTY). One-shot spawn remains the fallback.

h3.c's REPL uses linenoise, which needs a TTY. We drive it over a PTY, apply
``!`` commands, then send the prompt. Ref2VA video/audio refs are not exposed as
interactive commands (only ``!ref-image``), so those jobs stay one-shot.
"""

from __future__ import annotations

import fcntl
import logging
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import termios
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from h3_backend import GenerateRequest, GenerationCancelledError, expand_quality
from h3_media import require_ui_canvas, snap_frames
from h3_paths import console_h3

log = logging.getLogger("h3-session")

ProgressCallback = Callable[[dict[str, Any]], None]

SESSION_START_TIMEOUT_S = 600.0
GENERATE_TIMEOUT_S = 4 * 3600.0

_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|].*?(?:\x07|\x1b\\))")
_PROMPT_RE = re.compile(r"h3>\s*$", re.MULTILINE)
_DONE_RE = re.compile(r"Done -> (.+?) \[[0-9.]+s\]")
_OUTPUTS_RE = re.compile(r"Outputs:\s+(\S+)")
_PROGRESS_RE = re.compile(r"(.{1,40}?)\s+(\d+)/(\d+)\s*$")
_ERROR_RE = re.compile(r"(?:^|\r)h3:\s+(.+)", re.MULTILINE)


class SessionError(RuntimeError):
    """Interactive session is unusable; caller should fall back to one-shot."""


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def normalize_pty(text: str) -> str:
    return strip_ansi(text).replace("\r", "\n")


def has_repl_prompt(text: str) -> bool:
    return bool(_PROMPT_RE.search(normalize_pty(text)))


def parse_done_path(text: str) -> Path | None:
    match = _DONE_RE.search(strip_ansi(text))
    if not match:
        return None
    return Path(match.group(1).strip())


def parse_outputs_dir(text: str) -> Path | None:
    match = _OUTPUTS_RE.search(strip_ansi(text))
    if not match:
        return None
    return Path(match.group(1).strip())


def _echo_pty(chunk: str) -> None:
    for line in strip_ansi(chunk).replace("\r", "\n").splitlines():
        text = line.strip()
        if not text or text == "h3>" or _PROGRESS_RE.search(text):
            continue
        console_h3("%s", text)


def parse_cli_progress(chunk: str) -> dict[str, Any] | None:
    cleaned = strip_ansi(chunk).replace("\r", "\n")
    last: dict[str, Any] | None = None
    for line in cleaned.splitlines():
        line = line.strip()
        match = _PROGRESS_RE.search(line)
        if not match:
            continue
        phase = match.group(1).strip()
        if phase in {"h3>", "Seed:", "Done"}:
            continue
        step, total = int(match.group(2)), int(match.group(3))
        last = {
            "stage": phase or "generating",
            "step": step,
            "total": total,
            "label": f"{phase} {step}/{total}",
        }
        if total > 0:
            last["pct"] = round(100.0 * step / total, 1)
    return last


def session_commands_for_request(req: GenerateRequest) -> list[str]:
    """``!`` commands that align a warm session with this job (FL2VA only)."""
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
    cmds = [
        f"!size {width}x{height}",
        f"!frames {frames}",
        f"!steps {q['steps']}",
        f"!layers {q['layers']}",
        "!show off",
        "!open off",
        "!refs clear",
    ]
    if q.get("core_reuse"):
        cmds.append("!reuse 1")
        cmds.append(f"!core-reuse {int(q['core_reuse'])}")
    else:
        cmds.append("!core-reuse 1")
        cmds.append(f"!reuse {int(q.get('reuse') or 1)}")
    cmds.append("!token-reduction on" if q.get("token_reduction") else "!token-reduction off")
    if req.ssd_streaming:
        cmds.append("!int8-row-fc2 off")
        cmds.append("!ssd-streaming on")
    else:
        cmds.append("!ssd-streaming off")
        cmds.append("!int8-row-fc2 on" if req.int8_row_fc2 else "!int8-row-fc2 off")
    render = q.get("render")
    if render:
        cmds.append(f"!render-size {int(render[0])}x{int(render[1])}")
    else:
        cmds.append("!render-size native")
    if req.seed is not None and int(req.seed) >= 0:
        cmds.append(f"!seed {int(req.seed)}")
    else:
        cmds.append("!seed random")
    if req.first_frame:
        cmds.append(f"!first {req.first_frame}")
    else:
        cmds.append("!first clear")
    if req.last_frame:
        cmds.append(f"!last {req.last_frame}")
    else:
        cmds.append("!last clear")
    return cmds


def build_session_argv(
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
    cmd = [
        str(h3_bin),
        "-d",
        str(model_dir),
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
    if req.ssd_streaming:
        cmd.append("--ssd-streaming")
    elif req.int8_row_fc2:
        cmd.append("--use-int8-row-fc2")
    for lora in req.loras:
        cmd.extend(["--lora", f"{lora.path}:{lora.scale:.4g}"])
    if req.profile:
        cmd.append("--profile")
    return cmd


def session_prompt_line(prompt: str) -> str:
    text = " ".join((prompt or "").split())
    if text.startswith("!"):
        text = " " + text
    return text


class H3InteractiveSession:
    def __init__(self, cancel: Any | None = None) -> None:
        self._cancel = cancel
        self._master: int | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self.output_dir: Path | None = None
        self.alive = False

    def start(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
        timeout_s: float = SESSION_START_TIMEOUT_S,
    ) -> None:
        master, slave = pty.openpty()
        try:
            winsz = struct.pack("HHHH", 24, 160, 0, 0)
            fcntl.ioctl(slave, termios.TIOCSWINSZ, winsz)
        except OSError:
            pass
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        run_env.setdefault("TERM", "xterm-256color")
        workdir = str(cwd) if cwd is not None else str(Path(argv[0]).resolve().parent)
        try:
            proc = subprocess.Popen(
                argv,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                close_fds=True,
                env=run_env,
                cwd=workdir,
            )
        except OSError as exc:
            os.close(master)
            os.close(slave)
            raise SessionError(f"failed to spawn interactive h3: {exc}") from exc
        os.close(slave)
        self._master = master
        self._proc = proc
        try:
            boot = self._read_until(
                has_repl_prompt,
                timeout_s,
                on_chunk=lambda chunk: _echo_pty(chunk),
            )
        except Exception:
            self.stop()
            raise
        self.output_dir = parse_outputs_dir(boot)
        self.alive = True
        log.info("interactive h3 ready  pid=%s  outputs=%s", proc.pid, self.output_dir)

    def apply_request(self, req: GenerateRequest, timeout_s: float = 60.0) -> None:
        for cmd in session_commands_for_request(req):
            self._send_line(cmd)
            self._read_until(
                has_repl_prompt,
                timeout_s,
                on_chunk=lambda chunk: _echo_pty(chunk),
            )

    def generate(
        self,
        prompt: str,
        *,
        on_progress: ProgressCallback | None = None,
        timeout_s: float = GENERATE_TIMEOUT_S,
    ) -> Path:
        line = session_prompt_line(prompt)
        if not line:
            raise ValueError("prompt is required")
        self._send_line(line)
        collected: list[str] = []

        def ready(buf: str) -> bool:
            text = normalize_pty(buf)
            return bool(_DONE_RE.search(text) and _PROMPT_RE.search(text)) or bool(
                _ERROR_RE.search(text) and _PROMPT_RE.search(text) and "unknown command" not in text
            )

        def on_chunk(chunk: str) -> None:
            collected.append(chunk)
            _echo_pty(chunk)
            progress = parse_cli_progress(chunk)
            if progress and on_progress:
                on_progress(progress)

        buf = self._read_until(ready, timeout_s, on_chunk=on_chunk)
        text = strip_ansi(buf)
        done = parse_done_path(text)
        if done and done.is_file():
            return done
        err = _ERROR_RE.findall(text)
        if err:
            raise RuntimeError(err[-1].strip())
        if done:
            raise RuntimeError(f"h3 reported {done} but the file is missing")
        raise SessionError("interactive generate finished without Done -> path")

    def stop(self) -> None:
        self.alive = False
        proc = self._proc
        master = self._master
        self._proc = None
        self._master = None
        if proc is not None and proc.poll() is None:
            try:
                if master is not None:
                    os.write(master, b"!quit\n")
            except OSError:
                pass
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if master is not None:
            try:
                os.close(master)
            except OSError:
                pass

    def _send_line(self, line: str) -> None:
        if self._master is None or self._proc is None or self._proc.poll() is not None:
            self.alive = False
            raise SessionError("interactive h3 is not running")
        payload = (line.rstrip("\r\n") + "\r").encode("utf-8")
        console_h3("h3> %s", line.rstrip("\r\n")[:200])
        try:
            os.write(self._master, payload)
        except OSError as exc:
            self.alive = False
            raise SessionError(f"interactive h3 stdin closed: {exc}") from exc

    def _read_until(
        self,
        predicate: Callable[[str], bool],
        timeout_s: float,
        *,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        if self._master is None:
            raise SessionError("interactive h3 is not running")
        deadline = time.time() + timeout_s
        buf = ""
        while time.time() < deadline:
            if self._cancel is not None and self._cancel.is_set():
                raise GenerationCancelledError("cancelled")
            if self._proc is not None and self._proc.poll() is not None:
                self.alive = False
                raise SessionError(f"interactive h3 exited {self._proc.returncode}")
            ready, _, _ = select.select([self._master], [], [], 0.25)
            if not ready:
                continue
            try:
                data = os.read(self._master, 8192)
            except OSError as exc:
                self.alive = False
                raise SessionError(f"interactive h3 pty closed: {exc}") from exc
            if not data:
                self.alive = False
                raise SessionError("interactive h3 pty EOF")
            chunk = data.decode("utf-8", errors="replace")
            buf += chunk
            if on_chunk:
                on_chunk(chunk)
            if predicate(buf):
                return buf
        snippet = normalize_pty(buf).strip()[-500:]
        raise SessionError(
            f"timed out after {timeout_s:.0f}s waiting for h3"
            + (f": {snippet!r}" if snippet else "")
        )


def copy_session_output(src: Path, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest
