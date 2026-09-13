"""Play-fps pacing for FastH3 Live across machines of different speeds."""

from __future__ import annotations

NATIVE_FPS = 24.0


def adaptive_play_fps(
    frames: int,
    gen_s: float,
    *,
    margin_ratio: float = 1.08,
    min_fps: float = 0.5,
    max_fps: float = NATIVE_FPS,
) -> float:
    """Choose play fps so playback lasts at least ``gen_s * margin_ratio``.

    Faster hosts (lower gen_s) get higher play_fps automatically; slower hosts
    drop toward ``min_fps``. Cap at authored ``NATIVE_FPS``.
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if gen_s <= 0:
        return max_fps
    if margin_ratio < 1.0:
        margin_ratio = 1.0
    budget = gen_s * margin_ratio
    fps = float(frames) / budget
    if fps < min_fps:
        return min_fps
    if fps > max_fps:
        return max_fps
    return fps


def ema(prev: float | None, value: float, *, alpha: float = 0.4) -> float:
    """Exponential moving average for stable adaptive fps."""
    if prev is None:
        return value
    a = min(1.0, max(0.05, alpha))
    return a * value + (1.0 - a) * prev
