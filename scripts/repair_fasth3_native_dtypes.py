#!/usr/bin/env python3
"""In-place cast of FastH3 native shards to match FL2VA reference dtypes.

Only rewrites shards that contain mismatched tensors (the 12 F32 heads), so it
finishes in seconds instead of re-converting ~66 GB.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("models/MiniMax-H3-FastH3/FL2VA/transformer"),
    )
    p.add_argument(
        "--ref",
        type=Path,
        default=Path("models/MiniMax-H3/FL2VA/transformer"),
    )
    args = p.parse_args()

    ref_map = json.loads((args.ref / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    dst_map = json.loads((args.dst / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    # Discover mismatches.
    fixes: dict[str, list[str]] = defaultdict(list)  # dst_shard -> [names]
    for name, ref_shard in ref_map.items():
        dst_shard = dst_map[name]
        with safe_open(str(args.ref / ref_shard), framework="pt") as rf:
            ref_t = rf.get_tensor(name)
        with safe_open(str(args.dst / dst_shard), framework="pt") as sf:
            stu_t = sf.get_tensor(name)
        if ref_t.dtype != stu_t.dtype:
            if tuple(ref_t.shape) != tuple(stu_t.shape):
                raise SystemExit(f"shape mismatch {name}")
            fixes[dst_shard].append(name)

    if not fixes:
        print("all dtypes already match")
        return 0

    print(f"rewriting {len(fixes)} shards ({sum(len(v) for v in fixes.values())} tensors)")
    for shard, names in sorted(fixes.items()):
        path = args.dst / shard
        with safe_open(str(path), framework="pt") as f:
            payload = {k: f.get_tensor(k) for k in f.keys()}
        for name in names:
            ref_shard = ref_map[name]
            with safe_open(str(args.ref / ref_shard), framework="pt") as rf:
                want = rf.get_tensor(name).dtype
            old = payload[name]
            payload[name] = old.to(dtype=want).contiguous()
            print(f"  {name}: {old.dtype} → {want}")
        save_file(payload, str(path))
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
