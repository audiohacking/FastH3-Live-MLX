"""FastH3 Live helpers for Metal h3.c (scenes, cast, retime)."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fasth3_live"
# Original general pools kept as prompts_scenes.txt / prompts_scenes_2.txt +
# h3_characters.json. Live defaults are The Office–only (slow, unsettling).
DEFAULT_SCENES = (
    DATA_DIR / "prompts_scenes_office.txt",
)
DEFAULT_CHARACTERS = DATA_DIR / "h3_characters_office.json"

# liveserver defaults — FastH3 Live on the distilled student DiT (not base+LoRA)
LIVE_WIDTH = 448
LIVE_HEIGHT = 448
# Internal DiT/VAE canvas. Default is the ``live`` quality preset (384, no TR):
# 320+token-reduction was a soft draft below jacokon's ~200k-px flare floor.
# Use preset ``draft`` (320+TR) for max speed, ``sharp`` (448 no TR) for full canvas.
LIVE_RENDER_WIDTH = 384
LIVE_RENDER_HEIGHT = 384
LIVE_FRAMES = 124  # ~5.2s authored; 243 via preset ``long``; 362 once sustain rises
LIVE_STEPS = 4
LIVE_LAYERS = 50  # keep all blocks: 4-step student is off-distribution if thinned
LIVE_REUSE = 1  # every step is a trained DMD rung — do not skip evals
LIVE_PLAY_FPS = 0.0  # 0 = adaptive from measured gen time (see --fps / --margin)
LIVE_MARGIN_RATIO = 1.08  # play budget = gen_s * margin when adaptive
LIVE_MIN_PLAY_FPS = 0.5
LIVE_MAX_PLAY_FPS = 24.0
LIVE_TOKEN_REDUCTION = False  # do not stack with render-down (h3.c guidance)
LIVE_CURATED_SHARE = 0.50  # recognizable faces (was 0.30)
LIVE_MAX_CAST = 3  # Office Live: never fill more than 1–3 characters per scene
LIVE_ENSEMBLE_BIAS = 0.70  # prefer 3-hand scenes when available (face-early)
LIVE_QUALITY_PRESET = "live"
# Prefer models/MiniMax-H3-FastH3 (see scripts/prepare_fasth3_native_tree.sh).
# LoRA id kept only as fallback when the student tree is absent.
LIVE_LORA_ID = "fasth3_v1_dense_datafree"
LIVE_MODEL_DIR_NAME = "MiniMax-H3-FastH3"
LIVE_MODEL_DIR_INT8_NAME = "MiniMax-H3-FastH3-INT8"
# Optional fused-turbo native tree (converted from MATLOWAI Comfy bake when possible).
LIVE_MODEL_DIR_FUSED_TURBO_NAME = "MiniMax-H3-FusedTurbo"
LIVE_MODEL_DIR_FUSED_TURBO_INT8_NAME = "MiniMax-H3-FusedTurbo-INT8"
