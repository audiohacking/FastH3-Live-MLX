"""Aggregate h3.c progress stages into a per-clip wall-time breakdown."""

from __future__ import annotations

import time
from typing import Any

# Collapse verbose progress labels into a few Live-relevant buckets.
_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("text", ("text encoder", "refine text")),
    ("prep", ("precompute adaln", "load transformer", "prepared")),
    ("denoise", ("denoise",)),
    ("audio_vae", ("audio vae",)),
    ("video_vae", ("video vae",)),
    ("mux", ("ffmpeg",)),
)


def _bucket(stage: str) -> str:
    key = stage.strip().lower()
    for name, prefixes in _BUCKETS:
        if any(key.startswith(p) for p in prefixes):
            return name
    return "other"


class PhaseTimer:
    """Accumulate wall time per progress stage while a generate() runs."""

    def __init__(self) -> None:
        self._t0 = time.time()
        self._bucket: str | None = None
        self._bucket_t0 = self._t0
        self.totals: dict[str, float] = {}
        self._done = False

    def on_progress(self, mp: dict[str, Any]) -> None:
        if self._done:
            return
        stage = str(mp.get("stage") or "").strip()
        if not stage:
            return
        bucket = _bucket(stage)
        now = time.time()
        if bucket != self._bucket:
            if self._bucket is not None:
                self.totals[self._bucket] = self.totals.get(self._bucket, 0.0) + (
                    now - self._bucket_t0
                )
            self._bucket = bucket
            self._bucket_t0 = now

    def finish(self) -> dict[str, float]:
        if not self._done:
            now = time.time()
            if self._bucket is not None:
                self.totals[self._bucket] = self.totals.get(self._bucket, 0.0) + (
                    now - self._bucket_t0
                )
            self.totals["wall"] = now - self._t0
            self._done = True
            self._bucket = None
        return {k: round(v, 2) for k, v in self.totals.items()}

    def summary(self) -> str:
        finished = self.finish()
        wall = finished.get("wall", 0.0)
        order = ("text", "prep", "denoise", "audio_vae", "video_vae", "mux", "other")
        parts = [f"{k}={finished[k]:.1f}s" for k in order if k in finished]
        for k, v in sorted(finished.items()):
            if k not in order and k != "wall":
                parts.append(f"{k}={v:.1f}s")
        parts.append(f"wall={wall:.1f}s")
        return " ".join(parts)
