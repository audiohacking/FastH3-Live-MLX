#!/usr/bin/env python3
"""Convert FastH3 Diffusers transformer shards → native MiniMax / h3.c layout.

FastVideo ships the DMD2 student as Diffusers keys
(``transformer_blocks.*.attn.to_{q,k,v}``, value-first SwiGLU). Metal ``./h3``
loads the fused Comfy/native tree (``blocks.*.attn.qkv_proj``, gate-first
SwiGLU) that official ``MiniMax-H3/FL2VA/transformer`` already uses.

This is the layout half of what jacokon/fasth3-live did before INT8 ConvRot —
enough for h3.c to open the distilled student on Apple Silicon.

  python scripts/convert_fasth3_diffusers_to_native.py \\
    --src models/FastH3-Dense-DataFree/transformer \\
    --ref models/MiniMax-H3/FL2VA/transformer \\
    --dst models/MiniMax-H3-FastH3/FL2VA/transformer

Then symlink the rest of FL2VA (text encoder, VAEs, tokenizer) from the base
tree — see ``scripts/prepare_fasth3_native_tree.sh``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

DIFF_INDEX = "diffusion_pytorch_model.safetensors.index.json"
NATIVE_INDEX = "model.safetensors.index.json"
NATIVE_CONFIG = "config.json"

# Diffusers top-level → native (non-block) renames.
TOP_RENAMES = {
    "proj_in.weight": "video_patch_proj.weight",
    "proj_in.bias": "video_patch_proj.bias",
    "audio_proj_in.weight": "audio_patch_proj.weight",
    "audio_proj_in.bias": "audio_patch_proj.bias",
    "context_embedder.weight": "condition_proj.weight",
    "context_embedder.bias": "condition_proj.bias",
    "time_embedder.linear_1.weight": "time_embedder.proj_in.weight",
    "time_embedder.linear_1.bias": "time_embedder.proj_in.bias",
    "time_embedder.linear_2.weight": "time_embedder.proj_out.weight",
    "time_embedder.linear_2.bias": "time_embedder.proj_out.bias",
    "norm_out.norm.weight": "final_layer.norm.weight",
    "norm_out.linear.weight": "final_layer.adaln_proj.linear.weight",
    "norm_out.linear.bias": "final_layer.adaln_proj.linear.bias",
    "proj_out.weight": "final_layer.video_out.weight",
    "proj_out.bias": "final_layer.video_out.bias",
    "audio_proj_out.weight": "final_layer.audio_out.weight",
    "audio_proj_out.bias": "final_layer.audio_out.bias",
    "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
}


def _open_shards(src: Path) -> tuple[dict[str, str], dict[str, object]]:
    index_path = src / DIFF_INDEX
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    handles: dict[str, object] = {}
    for shard in sorted(set(weight_map.values())):
        handles[shard] = safe_open(str(src / shard), framework="pt")
    return weight_map, handles


def _get(weight_map: dict[str, str], handles: dict[str, object], key: str) -> torch.Tensor:
    return handles[weight_map[key]].get_tensor(key)


def _swap_swiglu(fc1: torch.Tensor) -> torch.Tensor:
    """Diffusers is value-first; native/Comfy is gate-first."""
    first, second = fc1.chunk(2, dim=0)
    return torch.cat([second, first], dim=0)


def _fuse_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int = 56,
    head_dim: int = 128,
) -> torch.Tensor:
    """Fuse Diffusers Q/K/V into native interleaved-per-head rows.

    Official MiniMax-H3 / h3.c store QKV as
    ``[Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, …]`` (see h3.c README). A plain
    ``cat([Q,K,V])`` chunked layout is what Diffusers trains, but it produces
    muddy near-constant frames on Metal.
    """
    hidden_out = num_heads * head_dim
    if q.shape[0] != hidden_out or k.shape[0] != hidden_out or v.shape[0] != hidden_out:
        raise SystemExit(
            f"QKV out rows want {hidden_out}, got q={tuple(q.shape)} "
            f"k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    qq = q.reshape(num_heads, head_dim, -1)
    kk = k.reshape(num_heads, head_dim, -1)
    vv = v.reshape(num_heads, head_dim, -1)
    return torch.stack([qq, kk, vv], dim=1).reshape(num_heads * 3 * head_dim, -1)


_BLOCK_RE = re.compile(
    r"^(?P<prefix>transformer_blocks|token_refiner\.refiner_blocks)\.(?P<i>\d+)\."
)


def _match_ref_dtype(
    tensor: torch.Tensor, ref: torch.Tensor | None, name: str
) -> torch.Tensor:
    """Native FL2VA keeps patch/time/final heads in F32; Diffusers is often BF16."""
    if ref is None:
        return tensor.contiguous()
    if tensor.dtype == ref.dtype and tuple(tensor.shape) == tuple(ref.shape):
        return tensor.contiguous()
    if tuple(tensor.shape) != tuple(ref.shape):
        raise SystemExit(
            f"shape mismatch {name}: got {tuple(tensor.shape)} want {tuple(ref.shape)}"
        )
    return tensor.to(dtype=ref.dtype).contiguous()


def convert_tensors(
    weight_map: dict[str, str],
    handles: dict[str, object],
    *,
    rope_inv_freq: torch.Tensor | None,
    ref_dtypes: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    keys = set(weight_map)
    ref_dtypes = ref_dtypes or {}

    # --- plain renames ---
    for src_key, dst_key in TOP_RENAMES.items():
        if src_key not in keys:
            raise SystemExit(f"expected Diffusers key missing: {src_key}")
        out[dst_key] = _match_ref_dtype(
            _get(weight_map, handles, src_key), ref_dtypes.get(dst_key), dst_key
        )

    # --- rope (Diffusers computes it; h3.c requires the tensor) ---
    if rope_inv_freq is None:
        raise SystemExit("rope.inv_freq required from --ref FL2VA transformer")
    out["rope.inv_freq"] = rope_inv_freq.contiguous()

    # --- blocks + token refiner ---
    block_ids: dict[str, set[int]] = defaultdict(set)
    for key in keys:
        m = _BLOCK_RE.match(key)
        if m:
            block_ids[m.group("prefix")].add(int(m.group("i")))

    for prefix, ids in sorted(block_ids.items()):
        native_prefix = (
            "blocks"
            if prefix == "transformer_blocks"
            else "token_refiner.blocks"
        )
        for i in sorted(ids):
            s = f"{prefix}.{i}."
            d = f"{native_prefix}.{i}."

            def need(suffix: str) -> torch.Tensor:
                k = s + suffix
                if k not in keys:
                    raise SystemExit(f"missing {k}")
                return _get(weight_map, handles, k)

            # AdaLN + norms (unchanged names under new prefix)
            for suffix in (
                "adaln_proj.linear.weight",
                "adaln_proj.linear.bias",
                "norm1.weight",
                "norm2.weight",
            ):
                if (s + suffix) in keys:
                    dst = d + suffix
                    out[dst] = _match_ref_dtype(
                        need(suffix), ref_dtypes.get(dst), dst
                    )

            # Q/K norms
            for src_suf, dst_suf in (
                ("attn.norm_q.weight", "attn.q_norm.weight"),
                ("attn.norm_k.weight", "attn.k_norm.weight"),
            ):
                dst = d + dst_suf
                out[dst] = _match_ref_dtype(
                    need(src_suf), ref_dtypes.get(dst), dst
                )

            # Fused QKV
            q, k, v = (need(f"attn.to_{x}.weight") for x in "qkv")
            dst = d + "attn.qkv_proj.weight"
            out[dst] = _match_ref_dtype(
                _fuse_qkv(q, k, v), ref_dtypes.get(dst), dst
            )

            # Out proj
            dst = d + "attn.out_proj.weight"
            out[dst] = _match_ref_dtype(
                need("attn.to_out.0.weight"), ref_dtypes.get(dst), dst
            )

            # MLP — swap SwiGLU halves
            dst = d + "mlp.fc1.weight"
            out[dst] = _match_ref_dtype(
                _swap_swiglu(need("ff.net.0.proj.weight")),
                ref_dtypes.get(dst),
                dst,
            )
            dst = d + "mlp.fc2.weight"
            out[dst] = _match_ref_dtype(
                need("ff.net.2.weight"), ref_dtypes.get(dst), dst
            )

    return out


def _shard_like_ref(
    tensors: dict[str, torch.Tensor],
    ref_index: dict[str, str],
) -> dict[str, dict[str, torch.Tensor]]:
    """Place converted tensors into the same shard filenames as the FL2VA ref."""
    buckets: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    missing_in_ref: list[str] = []
    for name, tensor in tensors.items():
        shard = ref_index.get(name)
        if shard is None:
            missing_in_ref.append(name)
            # Fall back: last shard (usually final_layer lives there)
            shard = sorted(set(ref_index.values()))[-1]
        buckets[shard][name] = tensor
    if missing_in_ref:
        print(
            f"note: {len(missing_in_ref)} keys not in ref index "
            f"(wrote to last shard): {missing_in_ref[:5]}…",
            file=sys.stderr,
        )
    return buckets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Diffusers transformer/")
    p.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="native FL2VA transformer/ (config + rope + shard map)",
    )
    p.add_argument("--dst", type=Path, required=True, help="output native transformer/")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="map keys and print plan without writing shards",
    )
    args = p.parse_args()

    ref_index_path = args.ref / NATIVE_INDEX
    ref_config_path = args.ref / NATIVE_CONFIG
    if not ref_index_path.is_file() or not ref_config_path.is_file():
        raise SystemExit(f"--ref must contain {NATIVE_INDEX} and {NATIVE_CONFIG}")

    ref_index = json.loads(ref_index_path.read_text())["weight_map"]
    rope_shard = ref_index["rope.inv_freq"]
    with safe_open(str(args.ref / rope_shard), framework="pt") as f:
        rope = f.get_tensor("rope.inv_freq")

    # Load reference tensors once for dtype/shape contracts (esp. F32 heads).
    ref_handles: dict[str, object] = {}
    ref_dtypes: dict[str, torch.Tensor] = {}
    for name, shard in ref_index.items():
        if shard not in ref_handles:
            ref_handles[shard] = safe_open(str(args.ref / shard), framework="pt")
        ref_dtypes[name] = ref_handles[shard].get_tensor(name)

    weight_map, handles = _open_shards(args.src)
    print(f"src keys={len(weight_map)}  ref keys={len(ref_index)}")
    tensors = convert_tensors(
        weight_map, handles, rope_inv_freq=rope, ref_dtypes=ref_dtypes
    )
    print(f"converted tensors={len(tensors)}")

    # Sanity: every native key present
    missing = sorted(set(ref_index) - set(tensors))
    extra = sorted(set(tensors) - set(ref_index))
    if missing:
        print(f"ERROR missing native keys ({len(missing)}):", *missing[:20], sep="\n  ")
        return 1
    if extra:
        print(f"WARN extra keys ({len(extra)}):", *extra[:20], sep="\n  ")

    # Spot-check shapes against ref
    sample = [
        "video_patch_proj.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.adaln_proj.linear.weight",
        "final_layer.video_out.weight",
        "rope.inv_freq",
    ]
    for name in sample:
        shard = ref_index[name]
        with safe_open(str(args.ref / shard), framework="pt") as f:
            ref_t = f.get_tensor(name)
        got = tensors[name]
        if tuple(got.shape) != tuple(ref_t.shape):
            print(f"ERROR shape {name}: got {tuple(got.shape)} want {tuple(ref_t.shape)}")
            return 1
        print(f"  ok {name} {tuple(got.shape)} {got.dtype}")

    if args.dry_run:
        print("dry-run: not writing")
        return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    buckets = _shard_like_ref(tensors, ref_index)
    new_map: dict[str, str] = {}
    total = 0
    for shard_name in sorted(buckets):
        payload = buckets[shard_name]
        out_path = args.dst / shard_name
        print(f"writing {out_path.name} ({len(payload)} tensors)…")
        save_file(payload, str(out_path))
        for k, t in payload.items():
            new_map[k] = shard_name
            total += t.nbytes
    index = {"metadata": {"total_size": total}, "weight_map": new_map}
    (args.dst / NATIVE_INDEX).write_text(json.dumps(index, indent=2) + "\n")
    shutil.copy2(ref_config_path, args.dst / NATIVE_CONFIG)
    # Keep Diffusers config alongside for provenance
    src_cfg = args.src / "config.json"
    if src_cfg.is_file():
        shutil.copy2(src_cfg, args.dst / "config.diffusers.json")
    print(f"done → {args.dst}  ({total / 1e9:.2f} GB logical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
