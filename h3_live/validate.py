"""Thin wrapper around the upstream scene grammar validator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from h3_live import DATA_DIR


def load_upstream_validator():
    path = DATA_DIR / "validate_scenes.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/sync_fasth3_live_bucket.sh"
        )
    spec = importlib.util.spec_from_file_location("fasth3_validate_scenes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_scene_file(
    path: Path,
    *,
    frames: int = 362,
    authored_fps: float = 24.0,
) -> tuple[int, int]:
    """Validate all blocks. Returns ``(checked, failures)``.

    Scenes are authored for the 362-frame / ~15.08 s timeline even when Live
    generates shorter clips, so validation uses 362 by default.
    """
    mod = load_upstream_validator()
    text = Path(path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n---\n") if b.strip()]
    duration = frames / authored_fps
    failures = 0
    for i, block in enumerate(blocks):
        bad = mod.check(block, i, duration, None)
        if bad:
            failures += 1
    return len(blocks), failures
