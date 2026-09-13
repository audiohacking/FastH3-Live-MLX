"""Install the repo's ``requirements.txt`` into the current interpreter."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from h3_paths import REPO_ROOT

# Import names that must load after ``requirements.txt`` is applied.
# Install itself always uses that file — never this list as a package set.
_IMPORT_CHECKS = (
    "websockets",
    "av",
    "PIL",
    "numpy",
    "huggingface_hub",
    "fastapi",
    "starlette",
    "uvicorn",
    "multipart",
    "mcp",
)

_installed_this_process = False


def requirements_file() -> Path:
    return REPO_ROOT / "requirements.txt"


def _stamp_path() -> Path:
    return Path(sys.prefix) / ".h3-ws-requirements.sha256"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _imports_ok() -> bool:
    return all(_can_import(name) for name in _IMPORT_CHECKS)


def _stamp_matches(req: Path) -> bool:
    stamp = _stamp_path()
    if not stamp.is_file():
        return False
    try:
        return stamp.read_text(encoding="utf-8").strip() == _digest(req)
    except OSError:
        return False


def _write_stamp(req: Path) -> None:
    try:
        _stamp_path().write_text(_digest(req) + "\n", encoding="utf-8")
    except OSError:
        pass


def _install_argv(req: Path) -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, "-r", str(req)]
    return [sys.executable, "-m", "pip", "install", "-r", str(req)]


def _forced() -> bool:
    return os.environ.get("H3_INSTALL_REQUIREMENTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def ensure_python_requirements() -> None:
    """Install ``requirements.txt`` when imports are missing or it changed."""
    global _installed_this_process
    if _installed_this_process:
        return
    req = requirements_file()
    if not req.is_file():
        raise SystemExit(f"requirements.txt not found at {req}")
    if not _forced() and _imports_ok() and _stamp_matches(req):
        _installed_this_process = True
        return
    print(f"Installing Python packages from {req}…", flush=True)
    subprocess.check_call(_install_argv(req))
    importlib.invalidate_caches()
    missing = [name for name in _IMPORT_CHECKS if not _can_import(name)]
    if missing:
        raise SystemExit(
            "Still missing after installing requirements.txt: " + ", ".join(missing)
        )
    _write_stamp(req)
    _installed_this_process = True
