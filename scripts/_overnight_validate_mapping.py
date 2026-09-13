#!/usr/bin/env python3
"""Overnight helper: cosine-check FastH3 Diffusers→native QKV/SwiGLU vs base FL2VA."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DIFF = ROOT / "models/FastH3-Dense-DataFree/transformer"
BASE = ROOT / "models/MiniMax-H3/FL2VA/transformer"
STU = ROOT / "models/MiniMax-H3-FastH3/FL2VA/transformer"


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main() -> int:
    diff_map = json.loads(
        (DIFF / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    base_map = json.loads((BASE / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    stu_map = json.loads((STU / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    bi = 0
    q_key = f"transformer_blocks.{bi}.attn.to_q.weight"
    k_key = f"transformer_blocks.{bi}.attn.to_k.weight"
    v_key = f"transformer_blocks.{bi}.attn.to_v.weight"
    fc_key = f"transformer_blocks.{bi}.ff.net.0.proj.weight"
    shards = {diff_map[k] for k in (q_key, k_key, v_key, fc_key)}
    handles = {s: safe_open(str(DIFF / s), framework="pt") for s in shards}
    q = handles[diff_map[q_key]].get_tensor(q_key).float()
    k = handles[diff_map[k_key]].get_tensor(k_key).float()
    v = handles[diff_map[v_key]].get_tensor(v_key).float()
    fc1 = handles[diff_map[fc_key]].get_tensor(fc_key).float()

    with safe_open(
        str(BASE / base_map[f"blocks.{bi}.attn.qkv_proj.weight"]), framework="pt"
    ) as f:
        ref_qkv = f.get_tensor(f"blocks.{bi}.attn.qkv_proj.weight").float()
        ref_fc1 = f.get_tensor(f"blocks.{bi}.mlp.fc1.weight").float()
    with safe_open(
        str(STU / stu_map[f"blocks.{bi}.attn.qkv_proj.weight"]), framework="pt"
    ) as f:
        stu_qkv = f.get_tensor(f"blocks.{bi}.attn.qkv_proj.weight").float()
        stu_fc1 = f.get_tensor(f"blocks.{bi}.mlp.fc1.weight").float()

    cands = {
        "QKV": torch.cat([q, k, v], 0),
        "QVK": torch.cat([q, v, k], 0),
        "KQV": torch.cat([k, q, v], 0),
        "VQK": torch.cat([v, q, k], 0),
        "VKQ": torch.cat([v, k, q], 0),
    }
    print("qkv vs BASE:")
    best_name, best_cos = max(
        ((n, cos(t, ref_qkv)) for n, t in cands.items()), key=lambda x: x[1]
    )
    for name, t in cands.items():
        mark = " <-- best" if name == best_name else ""
        print(f"  {name}: {cos(t, ref_qkv):+.5f}{mark}")
    print(f"  STUDENT vs BASE: {cos(stu_qkv, ref_qkv):+.5f}")
    print(f"  STUDENT vs QKV:  {cos(stu_qkv, cands['QKV']):+.5f}")

    first, second = fc1.chunk(2, 0)
    asis = cos(fc1, ref_fc1)
    swapped = cos(torch.cat([second, first], 0), ref_fc1)
    print("swiglu vs BASE:")
    print(f"  as-is (first,second):   {asis:+.5f}")
    print(f"  swapped (second,first): {swapped:+.5f}")
    print(f"  STUDENT vs BASE:        {cos(stu_fc1, ref_fc1):+.5f}")
    print(f"  STUDENT vs swapped:     {cos(stu_fc1, torch.cat([second, first], 0)):+.5f}")
    print(f"  STUDENT vs as-is:       {cos(stu_fc1, fc1):+.5f}")

    # Decision hints
    if best_name != "QKV":
        print(
            f"WARN: convert uses QKV but best vs base is {best_name} "
            f"({best_cos:+.5f})",
            file=sys.stderr,
        )
    if swapped > asis + 0.05:
        want = "swapped"
    elif asis > swapped + 0.05:
        want = "as-is"
    else:
        want = "inconclusive"
    stu_matches_swap = abs(cos(stu_fc1, torch.cat([second, first], 0)) - 1.0) < 1e-5
    stu_matches_asis = abs(cos(stu_fc1, fc1) - 1.0) < 1e-5
    print(f"base prefers: {want}; student is swapped={stu_matches_swap} as-is={stu_matches_asis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
