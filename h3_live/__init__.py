"""FastH3 Live helpers for Metal h3.c (scenes, cast, retime)."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fasth3_live"
DEFAULT_SCENES = (
    DATA_DIR / "prompts_scenes.txt",
    DATA_DIR / "prompts_scenes_2.txt",
)
DEFAULT_CHARACTERS = DATA_DIR / "h3_characters.json"

# liveserver defaults — FastH3 Live on the distilled student DiT (not base+LoRA)
LIVE_WIDTH = 448
LIVE_HEIGHT = 448
# Internal DiT/VAE canvas — h3.c scales RGB up to LIVE_WIDTH/HEIGHT. On M3 Ultra
# 320→448 cut 22f denoise ~8.4s→~5.4s (~36%) with coherent picture (same knob as
# the upstream FastH3 Live / h3.c README fast-quality path).
LIVE_RENDER_WIDTH = 320
LIVE_RENDER_HEIGHT = 320
LIVE_FRAMES = 124  # ~5.2s authored; 362 still OK once sustain rises
LIVE_STEPS = 4
LIVE_LAYERS = 50  # keep all blocks: 4-step student is off-distribution if thinned
LIVE_REUSE = 1  # every step is a trained DMD rung — do not skip evals
LIVE_PLAY_FPS = 0.0  # 0 = adaptive from measured gen time (see --fps / --margin)
LIVE_MARGIN_RATIO = 1.08  # play budget = gen_s * margin when adaptive
LIVE_MIN_PLAY_FPS = 0.5
LIVE_MAX_PLAY_FPS = 24.0
LIVE_TOKEN_REDUCTION = True  # Metal win; Live canvas is square so pairing is safe-ish
# Prefer models/MiniMax-H3-FastH3 (see scripts/prepare_fasth3_native_tree.sh).
# LoRA id kept only as fallback when the student tree is absent.
LIVE_LORA_ID = "fasth3_v1_dense_datafree"
LIVE_MODEL_DIR_NAME = "MiniMax-H3-FastH3"
LIVE_MODEL_DIR_INT8_NAME = "MiniMax-H3-FastH3-INT8"
