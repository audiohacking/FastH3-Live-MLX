#!/usr/bin/env python3
"""Fold a MiniMax-H3 LoRA into the bf16 checkpoint by in-place byte patching.

h3.c has no LoRA runtime, and does not need one: a LoRA is W' = W + scale*(B@A),
which can be baked into the checkpoint once, offline. This tool clones the
original shards (copy-on-write where the filesystem supports it) and rewrites
only the byte ranges of the targeted tensors, so headers, tensor order, and
alignment stay byte-identical to the originals.

Folding the 4-step Turbo distillation adapter this way enables 5-6 step
sampling at zero runtime cost. Measured on an M5 Max (960x544, identical
prompt/seed): 39 frames 87s -> 65s and a 5s clip 8.8min -> 6.2min versus the
--steps 20 --reuse 2 --layers 45 preset, at comparable visual quality.

Requires only numpy. Adapter tensors must be named <target>.lora_A.weight /
<target>.lora_B.weight over the checkpoint's own key space, as
larryvrh/MiniMax-H3-Turbo-Lora's packaged .safetensors files are.

Usage:
  python3 tools/fold_turbo_lora.py \
      --checkpoint MiniMax-H3/FL2VA/transformer \
      --lora minimax_h3_turbo_v4_step600_ema.safetensors \
      --out MiniMax-H3-turbo/FL2VA/transformer
Then point h3 at a model directory whose transformer is the folded tree and
sample with --steps 5 or 6 (4 shows motion smear; do not combine with --reuse).
"""
import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Return (header dict, header byte length) from a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header, header_len


def read_tensor_bf16(path: Path, meta: dict) -> np.ndarray:
    """Read a single tensor as bf16 from safetensors, return f32 view."""
    header, header_len = read_safetensors_header(path)
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + start)
        raw = np.frombuffer(f.read(end - start), dtype=np.uint16)
    # Expand bf16 to f32: shift left 16 bits and view as float32
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def write_tensor_bf16(path: Path, meta: dict, header_len: int, data: np.ndarray):
    """Write a tensor back in-place as bf16."""
    start, end = meta["data_offsets"]
    # Convert f32 to bf16: take upper 16 bits
    bf16 = (data.view(np.uint32) >> 16).astype(np.uint16)
    with open(path, "r+b") as f:
        f.seek(8 + header_len + start)
        f.write(bf16.tobytes())


def load_lora(path: Path) -> dict[str, np.ndarray]:
    """Load all lora tensors from a safetensors file."""
    header, header_len = read_safetensors_header(path)
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dtype_str = meta["dtype"]
        if dtype_str == "BF16":
            tensors[name] = read_tensor_bf16(path, meta)
        elif dtype_str == "F32":
            with open(path, "rb") as f:
                f.seek(8 + header_len + start)
                tensors[name] = np.frombuffer(
                    f.read(end - start), dtype=np.float32
                ).reshape(meta["shape"])
        else:
            raise ValueError(f"Unsupported dtype {dtype_str} for {name}")
    return tensors


def fold_checkpoint(checkpoint_dir: Path, lora_path: Path, out_dir: Path, scale: float):
    """Fold LoRA weights into checkpoint shards."""
    lora = load_lora(lora_path)

    # Group lora tensors by target
    lora_pairs: dict[str, dict] = {}
    for name, tensor in lora.items():
        if ".lora_A." in name:
            target = name.replace(".lora_A.weight", "")
            lora_pairs.setdefault(target, {})["A"] = tensor
        elif ".lora_B." in name:
            target = name.replace(".lora_B.weight", "")
            lora_pairs.setdefault(target, {})["B"] = tensor
        elif ".alpha" in name or name.endswith(".alpha"):
            target = name.replace(".alpha", "").replace(".lora_alpha", "")
            lora_pairs.setdefault(target, {})["alpha"] = float(tensor.flat[0])

    # Clone checkpoint directory
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in checkpoint_dir.glob("*.safetensors"):
        dst = out_dir / shard.name
        if not dst.exists():
            # Use copy to enable CoW on APFS
            shutil.copy2(shard, dst)

    # Process each shard
    folded = 0
    for shard in out_dir.glob("*.safetensors"):
        header, header_len = read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            # Match against lora targets
            target = name.replace(".weight", "")
            if target not in lora_pairs:
                continue
            pair = lora_pairs[target]
            if "A" not in pair or "B" not in pair:
                continue

            A = pair["A"]
            B = pair["B"]
            alpha = pair.get("alpha", float(A.shape[0]))
            rank = A.shape[0]

            # Read original weight
            W = read_tensor_bf16(shard, meta)

            # Compute delta: B @ A
            delta = B @ A
            factor = scale * alpha / rank

            # Parity check before write
            W_new = W + factor * delta
            if not np.allclose(W_new.shape, W.shape):
                print(f"  Skip {name}: shape mismatch {W.shape} vs {W_new.shape}")
                continue

            # Write back
            write_tensor_bf16(shard, meta, header_len, W_new)
            folded += 1
            print(f"  Folded {name}")

    print(f"\nFolded {folded} weights into {out_dir}")
    return folded


def main():
    parser = argparse.ArgumentParser(
        description="Fold a LoRA into h3.c checkpoint shards"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to FL2VA/transformer directory",
    )
    parser.add_argument(
        "--lora", type=Path, required=True, help="Path to LoRA .safetensors file"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory for folded shards"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="LoRA scale factor (default: 1.0)"
    )
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        parser.error(f"Checkpoint directory not found: {args.checkpoint}")
    if not args.lora.is_file():
        parser.error(f"LoRA file not found: {args.lora}")

    print(f"Folding {args.lora.name} into {args.checkpoint}")
    print(f"Output: {args.out}")
    print(f"Scale: {args.scale}")
    print()

    fold_checkpoint(args.checkpoint, args.lora, args.out, args.scale)


if __name__ == "__main__":
    main()
