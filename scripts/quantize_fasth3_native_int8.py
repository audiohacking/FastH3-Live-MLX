#!/usr/bin/env python3
"""Offline per-row INT8 quantize of FastH3 DiT linears for native h3.c.

h3.c loads pre-quantized weights when ``blocks.N.attn.qkv_proj.weight`` is I8
and ``…weight_scale`` is F32 with shape [rows, 1] (absmax/127 per output row).

This skips the Metal runtime BF16→INT8 weight pass at DiT load — faster cold
start and less peak RAM. Matches ``h3_gpu_quantize_weight_int8``.

  python scripts/quantize_fasth3_native_int8.py \\
    --src models/MiniMax-H3-FastH3/FL2VA/transformer \\
    --dst models/MiniMax-H3-FastH3-INT8/FL2VA/transformer
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LINEAR_SUFFIXES = (
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric INT8, scale = absmax/127. Returns (i8, scale[rows,1])."""
    w = weight.detach().float()
    if w.ndim != 2:
        raise SystemExit(f"expected 2D weight, got {tuple(w.shape)}")
    absmax = w.abs().amax(dim=1)
    scale = torch.where(
        absmax > 0,
        absmax / 127.0,
        torch.full_like(absmax, 1.0 / 127.0),
    )
    inv = torch.where(absmax > 0, 127.0 / absmax, torch.full_like(absmax, 127.0))
    q = torch.clamp(torch.round(w * inv.unsqueeze(1)), -127, 127).to(torch.int8)
    return q.contiguous(), scale.unsqueeze(1).contiguous()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    args.dst.mkdir(parents=True, exist_ok=True)

    # Group keys by shard
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    new_map: dict[str, str] = {}
    total = 0
    quantized = 0

    for shard in sorted(by_shard):
        path = args.src / shard
        print(f"reading {shard}…", flush=True)
        with safe_open(str(path), framework="pt") as f:
            payload: dict[str, torch.Tensor] = {}
            for name in by_shard[shard]:
                t = f.get_tensor(name)
                linear = False
                for suf in LINEAR_SUFFIXES:
                    if name.endswith(suf):
                        linear = True
                        break
                if linear and name.startswith("blocks."):
                    q, scale = quantize_rows(t)
                    payload[name] = q
                    payload[name + "_scale"] = scale
                    quantized += 1
                else:
                    # Keep token_refiner / heads in BF16 — refiner path is BF16-only.
                    payload[name] = t.contiguous()

        out_path = args.dst / shard
        print(
            f"writing {out_path.name} ({len(payload)} tensors, "
            f"{quantized} linears so far)…",
            flush=True,
        )
        save_file(payload, str(out_path))
        for k, t in payload.items():
            new_map[k] = shard
            total += t.nbytes if t.is_floating_point() or t.dtype == torch.int8 else t.nbytes

    meta = {"metadata": {"total_size": total}, "weight_map": new_map}
    (args.dst / "model.safetensors.index.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    for name in ("config.json", "config.diffusers.json"):
        src = args.src / name
        if src.is_file():
            shutil.copy2(src, args.dst / name)
    print(f"done: quantized {quantized} linears → {args.dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
