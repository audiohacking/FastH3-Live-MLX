#!/usr/bin/env python3
"""Inspect MATLOWAI fused-turbo Comfy INT8 ConvRot for a native h3.c path.

jacokon recommends this bake over the FastH3 student (same speed, cleaner
picture). It ships as a *single* ComfyUI INT8 ConvRot safetensors — not the
Diffusers BF16 shards our ``convert_fasth3_diffusers_to_native.py`` consumes.

This script reports whether a downloaded file can be mapped, and writes a stub
tree layout note. It does **not** invent a ConvRot→native dequant converter.

Usage (on the Apple Silicon host, after accepting MiniMax territory terms):

  hf download MATLOWAI/minimax-h3-fused-turbo-int8-convrot \\
    diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors \\
    --local-dir models/comfy_fused_turbo

  python scripts/inspect_fused_turbo_for_native.py \\
    --src models/comfy_fused_turbo/diffusion_models/*.safetensors

If/when a native BF16 (or Diffusers) source appears, place the converted DiT at:

  models/MiniMax-H3-FusedTurbo/FL2VA/transformer/

and symlink TE/VAEs from MiniMax-H3 (same as prepare_fasth3_native_tree.sh).
``liveserver.py`` already prefers that tree when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("need safetensors", file=sys.stderr)
    sys.exit(1)

# Keys that distinguish Comfy INT8 ConvRot from native fused BF16/I8 shards.
CONVROT_HINTS = ("scale_weight", "weight_scale", "convrot", "input_scale")
NATIVE_HINTS = ("blocks.0.attn.qkv_proj.weight", "blocks.0.attn.qkv.weight")
DIFFUSERS_HINTS = ("transformer_blocks.0.attn.to_q.weight",)


def inspect(path: Path) -> dict:
    info: dict = {"path": str(path), "size_gb": path.stat().st_size / (1 << 30)}
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
    info["n_tensors"] = len(keys)
    sample = keys[:40]
    info["sample_keys"] = sample
    joined = "\n".join(keys).lower()
    info["looks_convrot_int8"] = any(h in joined for h in CONVROT_HINTS) or any(
        k.endswith(".scale") for k in keys[:200]
    )
    info["looks_native"] = any(h in keys for h in NATIVE_HINTS) or any(
        k.startswith("blocks.0.") for k in keys
    )
    info["looks_diffusers"] = any(h in keys for h in DIFFUSERS_HINTS) or any(
        k.startswith("transformer_blocks.") for k in keys
    )
    if info["looks_native"] and not info["looks_convrot_int8"]:
        info["verdict"] = (
            "native-like — copy/symlink into models/MiniMax-H3-FusedTurbo/FL2VA/transformer/"
        )
    elif info["looks_diffusers"]:
        info["verdict"] = (
            "Diffusers layout — run convert_fasth3_diffusers_to_native.py with "
            "--dst models/MiniMax-H3-FusedTurbo/FL2VA/transformer"
        )
    else:
        info["verdict"] = (
            "Comfy INT8 ConvRot (or unknown) — no Metal convert path yet. "
            "Keep FastH3 student; do not fuse 8-step turbo LoRA onto the 4-step Live path."
        )
    return info


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="safetensors file to inspect")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()
    src = args.src.expanduser().resolve()
    if not src.is_file():
        # allow glob-ish single match via shell already expanded
        print(f"missing {src}", file=sys.stderr)
        return 1
    report = inspect(src)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print("\n" + report["verdict"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
