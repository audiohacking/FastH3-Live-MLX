"""Live quality presets — render canvas × token-reduction × frames.

jacokon/fasth3-live denoises at full 448². Our draft path (320 + token-reduction)
was the main soft-picture cause. Presets climb that ladder; adaptive play fps
absorbs gen time while margin stays ≥ 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from h3_live import LIVE_FRAMES, LIVE_HEIGHT, LIVE_WIDTH


@dataclass(frozen=True)
class QualityPreset:
    name: str
    render_width: int
    render_height: int
    token_reduction: bool
    frames: int
    note: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "token_reduction": self.token_reduction,
            "frames": self.frames,
            "note": self.note,
        }


# Locked after ladder A/B on M3 Ultra (see LIVE.md): 384 noTR sustains continuous
# Live. 384+TR was +22% warm but doubled/striped subjects — do not ship.
# Full 448 noTR is the jacokon-matched sharp preset when headroom allows.
PRESETS: dict[str, QualityPreset] = {
    "draft": QualityPreset(
        name="draft",
        render_width=320,
        render_height=320,
        token_reduction=True,
        frames=LIVE_FRAMES,
        note="speed path — soft picture (legacy)",
    ),
    "live": QualityPreset(
        name="live",
        render_width=384,
        render_height=384,
        token_reduction=False,
        frames=LIVE_FRAMES,
        note="default — 384 render, no TR (TR doubled subjects)",
    ),
    "sharp": QualityPreset(
        name="sharp",
        render_width=LIVE_WIDTH,
        render_height=LIVE_HEIGHT,
        token_reduction=False,
        frames=LIVE_FRAMES,
        note="jacokon-matched full 448² internal canvas",
    ),
    "long": QualityPreset(
        name="long",
        render_width=384,
        render_height=384,
        token_reduction=False,
        frames=243,
        note="live canvas with longer clip (243f ≈ 10.1 s authored)",
    ),
}

DEFAULT_PRESET = "live"


def get_preset(name: str) -> QualityPreset:
    key = (name or "").strip().lower()
    if key not in PRESETS:
        raise KeyError(f"unknown quality preset {name!r}; choose {', '.join(PRESETS)}")
    return PRESETS[key]


def apply_preset_to_args(ns: object, preset: QualityPreset) -> None:
    """Mutate an argparse Namespace (or similar) with preset knobs."""
    ns.render_width = preset.render_width  # type: ignore[attr-defined]
    ns.render_height = preset.render_height  # type: ignore[attr-defined]
    ns.token_reduction = preset.token_reduction  # type: ignore[attr-defined]
    ns.frames = preset.frames  # type: ignore[attr-defined]
    ns.quality_preset = preset.name  # type: ignore[attr-defined]


def recipe_label(
    *,
    preset: str,
    width: int,
    height: int,
    render_width: int,
    render_height: int,
    frames: int,
    token_reduction: bool,
    steps: int,
) -> str:
    rw = render_width or width
    rh = render_height or height
    tr = "TR" if token_reduction else "noTR"
    return (
        f"preset={preset} out={width}x{height} render={rw}x{rh} "
        f"frames={frames} steps={steps} {tr}"
    )


def presets_public() -> list[Mapping[str, object]]:
    return [p.as_dict() for p in PRESETS.values()]
