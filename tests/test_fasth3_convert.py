"""Unit tests for FastH3 Diffusers → native conversion helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "convert_fasth3",
    ROOT / "scripts" / "convert_fasth3_diffusers_to_native.py",
)
_CONV = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_CONV)


class TestFastH3ConvertHelpers(unittest.TestCase):
    def test_fuse_qkv_interleaved_per_head(self) -> None:
        # MiniMax: INNER = 56 * 128 = 7168, HIDDEN = 5376; native is per-head QKV
        heads, dim, hidden = 56, 128, 5376
        q = torch.randn(heads * dim, hidden)
        k = torch.randn(heads * dim, hidden)
        v = torch.randn(heads * dim, hidden)
        fused = _CONV._fuse_qkv(q, k, v, num_heads=heads, head_dim=dim)
        self.assertEqual(tuple(fused.shape), (heads * 3 * dim, hidden))
        # Head 0: Q then K then V (each `dim` rows)
        self.assertTrue(torch.equal(fused[:dim], q[:dim]))
        self.assertTrue(torch.equal(fused[dim : 2 * dim], k[:dim]))
        self.assertTrue(torch.equal(fused[2 * dim : 3 * dim], v[:dim]))
        # Head 1 starts at row 3*dim
        self.assertTrue(torch.equal(fused[3 * dim : 4 * dim], q[dim : 2 * dim]))
        # Round-trip against a synthetic native interleaved tensor
        native = torch.stack(
            [q.view(heads, dim, -1), k.view(heads, dim, -1), v.view(heads, dim, -1)],
            dim=1,
        ).reshape(heads * 3 * dim, hidden)
        self.assertTrue(torch.equal(fused, native))

    def test_swap_swiglu_gate_first(self) -> None:
        fc1 = torch.randn(28672, 5376)
        value, gate = fc1.chunk(2, dim=0)
        swapped = _CONV._swap_swiglu(fc1)
        self.assertTrue(torch.equal(swapped[:14336], gate))
        self.assertTrue(torch.equal(swapped[14336:], value))

    def test_top_renames_cover_native_heads(self) -> None:
        native_needed = {
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "condition_proj.weight",
            "time_embedder.proj_in.weight",
            "time_embedder.proj_out.weight",
            "final_layer.norm.weight",
            "final_layer.adaln_proj.linear.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
        }
        mapped = set(_CONV.TOP_RENAMES.values())
        self.assertTrue(native_needed.issubset(mapped))


if __name__ == "__main__":
    unittest.main()
