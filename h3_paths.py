"""Scratch paths for temp media — never macOS TMPDIR (/var/folders/...)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

_configured_root: Path | None = None


def configure_scratch_root(root: Path | str | None) -> None:
    """Prefer ``output_dir/.scratch``, else ``/tmp/h3-ws``."""
    global _configured_root
    if root is None:
        _configured_root = None
        return
    _configured_root = Path(root).expanduser().resolve()


def scratch_root() -> Path:
    if _configured_root is not None:
        return _configured_root
    env = os.environ.get("H3_SCRATCH_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path("/tmp/h3-ws")


def ensure_scratch_root() -> Path:
    root = scratch_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def mk_scratch_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(ensure_scratch_root())))


def mk_scratch_file(prefix: str, suffix: str) -> tuple[int, str]:
    return tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(ensure_scratch_root()))


def default_h3_bin() -> Path:
    env = os.environ.get("H3_BIN", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "third_party" / "h3.c" / "h3"


def h3_process_cwd(h3_bin: Path | str | None = None) -> Path:
    """h3.c opens ``h3_shaders.metal`` from the process working directory."""
    binary = Path(h3_bin) if h3_bin is not None else default_h3_bin()
    return binary.expanduser().resolve().parent


def debug_console() -> bool:
    """Dump h3 progress to the server console unless ``DEBUG=false``."""
    raw = os.environ.get("DEBUG")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


_last_console_line = ""


def console_h3(message: str, *args: object) -> None:
    """Print an h3 line to the server log when console debug is on."""
    if not debug_console():
        return
    import logging

    text = message % args if args else message
    text = " ".join(str(text).split())
    if not text:
        return
    global _last_console_line
    if text == _last_console_line:
        return
    _last_console_line = text
    logging.getLogger("h3").info("%s", text[:300])


def default_model_dir() -> Path:
    env = os.environ.get("H3_MODEL_DIR", "").strip() or os.environ.get(
        "H3_WS_MODEL_DIR", ""
    ).strip()
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "models" / "MiniMax-H3"


def default_h3_av() -> Path:
    env = os.environ.get("H3_AV", "").strip()
    if env:
        return Path(env).expanduser()
    return REPO_ROOT / "scripts" / "h3-av"


_SHIM_NAMES = {"h3-av", "h3-ffmpeg", "h3-ffprobe", "h3_av.py"}


def _looks_like_media_shim(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if resolved.name in _SHIM_NAMES:
        return True
    try:
        if resolved == (REPO_ROOT / "h3_av.py").resolve():
            return True
        if resolved == (REPO_ROOT / "scripts" / "h3-av").resolve():
            return True
    except OSError:
        pass
    return False


def _force_pyav() -> bool:
    return os.environ.get("H3_AV_FORCE_PYAV", "").strip().lower() in {"1", "true", "yes"}


def real_ffmpeg() -> str | None:
    """System ffmpeg that actually starts, never the PyAV shim."""
    return _real_tool("ffmpeg")


def real_ffprobe() -> str | None:
    return _real_tool("ffprobe")


_tool_ok: dict[str, bool] = {}


def _tool_runs(path: str) -> bool:
    """Reject Homebrew binaries that crash on missing dylibs (dyld)."""
    cached = _tool_ok.get(path)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=8,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        _tool_ok[path] = False
        return False
    text = f"{proc.stdout or ''}{proc.stderr or ''}"
    ok = proc.returncode == 0 and "Library not loaded" not in text and "dyld[" not in text
    if not ok:
        logging.getLogger("h3").warning(
            "skipping broken %s (%s)",
            path,
            "dyld" if "Library not loaded" in text or "dyld[" in text else f"exit {proc.returncode}",
        )
    _tool_ok[path] = ok
    return ok


def _real_tool(name: str) -> str | None:
    if _force_pyav():
        return None
    folders: list[str] = []
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            folders.append(part)
    folders.extend(["/opt/homebrew/bin", "/usr/local/bin"])
    seen: set[str] = set()
    for folder in folders:
        candidate = Path(folder) / name
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            continue
        if _looks_like_media_shim(candidate):
            continue
        resolved = str(candidate)
        if not _tool_runs(resolved):
            continue
        return resolved
    return None


def ffmpeg_shim_bindir() -> Path:
    """Directory with ``ffmpeg`` / ``ffprobe`` names pointing at h3-av.

    Unpatched h3.c looks up those names on PATH. Patched builds prefer H3_AV.
    """
    bindir = REPO_ROOT / "scripts" / ".h3-av-bin"
    bindir.mkdir(parents=True, exist_ok=True)
    shim = default_h3_av().resolve()
    for name in ("ffmpeg", "ffprobe"):
        link = bindir / name
        try:
            if link.is_symlink() and link.resolve() == shim:
                continue
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(shim)
        except OSError:
            continue
    return bindir


def h3_media_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """h3.c mux/decode: a working system ffmpeg, otherwise the PyAV shim.

    Homebrew binaries that crash on missing dylibs (``Library not loaded``)
    are skipped. ``H3_AV`` overrides both tools in patched h3.c, so it must
    be unset when the real binaries are used. A warm session captures this
    env at spawn — restart the server after muxer changes.
    """
    import sys

    env = dict(base if base is not None else os.environ)
    env["H3_PYTHON"] = sys.executable
    ffmpeg = real_ffmpeg()
    ffprobe = real_ffprobe()
    shim = str(default_h3_av())
    if ffmpeg and not _force_pyav():
        env.pop("H3_AV", None)
        env["H3_FFMPEG"] = ffmpeg
        env["H3_FFPROBE"] = ffprobe or shim
        return env
    env["H3_AV"] = shim
    env["H3_FFMPEG"] = shim
    env["H3_FFPROBE"] = shim
    bindir = str(ffmpeg_shim_bindir())
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    return env
