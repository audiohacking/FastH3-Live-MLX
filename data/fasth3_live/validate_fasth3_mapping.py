# SPDX-License-Identifier: Apache-2.0
"""Empirically decide the ambiguous parts of the FastH3 -> ComfyUI mapping.

FastH3 is a fine-tune of the same MiniMax-H3 the reference ComfyUI checkpoint
holds, so every corresponding weight should be highly correlated. Two mapping
choices are silent if wrong -- they still produce a runnable file that generates
a wrong video:

  * the concat order inside the fused ``attn.qkv_proj``
  * which half of ``ff.net.0.proj`` is the SwiGLU gate

This script scores each candidate against the reference checkpoint. The correct
choice wins by a wide margin; if two candidates score the same, the test is
inconclusive and nothing should be assumed.

Run under the ComfyUI venv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from st_read import SafeTensorsFile

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"


def dequantize_reference(f, key: str) -> torch.Tensor:
    """Rebuild a BF16 view of a possibly int8+convrot reference weight."""
    w = f.get_tensor(key)
    scale_key = key + "_scale"
    quant_key = key.replace(".weight", ".comfy_quant")
    keys = set(f.keys())
    if scale_key not in keys or quant_key not in keys:
        return w.float()

    conf = json.loads(bytes(f.get_tensor(quant_key).cpu().numpy().tolist()))
    scale = f.get_tensor(scale_key)

    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
    params = TensorWiseINT8Layout.Params(
        scale=scale.cuda(),
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(w.shape),
        is_weight=True,
        convrot=bool(conf.get("convrot", False)),
        convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
    )
    qt = QuantizedTensor(w.cuda(), "TensorWiseINT8Layout", params)
    return qt.dequantize().float()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="diffusers transformer/ dir")
    p.add_argument("--base", required=True, help="reference ComfyUI minimax_h3 checkpoint")
    p.add_argument("--blocks", default="0,25,49")
    args = p.parse_args()

    src = Path(args.src)
    index = json.loads((src / INDEX_NAME).read_text())["weight_map"]
    handles = {n: safe_open(str(src / n), framework="pt") for n in sorted(set(index.values()))}

    def get(k: str) -> torch.Tensor:
        return handles[index[k]].get_tensor(k).cuda().float()

    blocks = [int(b) for b in args.blocks.split(",")]

    bf = SafeTensorsFile(args.base)
    if True:
        for i in blocks:
            s = f"transformer_blocks.{i}."
            d = f"blocks.{i}."
            print(f"\n================ block {i} ================")

            # --- unambiguous control: establishes how close a fine-tune is ---
            ref_out = dequantize_reference(bf, f"{d}attn.out_proj.weight")
            print(f"  control  attn.out_proj            cos = {cos(get(s + 'attn.to_out.0.weight'), ref_out):+.5f}")
            ref_n1 = bf.get_tensor(f"{d}norm1.weight").cuda().float()
            print(f"  control  norm1                    cos = {cos(get(s + 'norm1.weight'), ref_n1):+.5f}")

            # --- qkv concat order ---
            q, k, v = (get(s + f"attn.to_{x}.weight") for x in "qkv")
            ref_qkv = dequantize_reference(bf, f"{d}attn.qkv_proj.weight")
            print("  qkv concat order:")
            for name, cand in (("[q,k,v]", (q, k, v)), ("[q,v,k]", (q, v, k)),
                               ("[k,q,v]", (k, q, v)), ("[v,k,q]", (v, k, q))):
                print(f"    {name:9s} cos = {cos(torch.cat(cand, 0), ref_qkv):+.5f}")

            # --- SwiGLU half order ---
            fc1 = get(s + "ff.net.0.proj.weight")
            first, second = fc1.chunk(2, dim=0)
            ref_fc1 = dequantize_reference(bf, f"{d}mlp.fc1.weight")
            print("  swiglu half order:")
            print(f"    as-is (value,gate)  cos = {cos(fc1, ref_fc1):+.5f}")
            print(f"    swapped (gate,value) cos = {cos(torch.cat([second, first], 0), ref_fc1):+.5f}")

            ref_fc2 = dequantize_reference(bf, f"{d}mlp.fc2.weight")
            print(f"  control  mlp.fc2                  cos = {cos(get(s + 'ff.net.2.weight'), ref_fc2):+.5f}")


if __name__ == "__main__":
    main()
