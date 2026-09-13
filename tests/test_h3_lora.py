"""LoRA catalog and argv wiring. No Hugging Face traffic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_backend import GenerateRequest, LoraRef, build_h3_argv
from h3_lora import BUILTIN_LORAS, lora_catalog, normalize_lora_spec, parse_lora_specs
from h3_session import build_session_argv


class LoraCatalogTests(unittest.TestCase):
    def test_builtin_includes_tutu_step100(self) -> None:
        catalog = lora_catalog(None)
        ids = [p["id"] for p in catalog]
        self.assertIn("tutu_20to8_nfe_step100", ids)
        tutu = next(p for p in catalog if p["id"] == "tutu_20to8_nfe_step100")
        self.assertEqual(tutu["scale"], 0.8)
        self.assertEqual(tutu["steps"], 8)
        self.assertIn("tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA", tutu["spec"])

    def test_normalize_blob_to_resolve(self) -> None:
        spec = normalize_lora_spec(
            "https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/"
            "blob/main/comfyui/tutu.safetensors"
        )
        self.assertIn("/resolve/main/", spec)

    def test_parse_lora_specs_by_id(self) -> None:
        parsed = parse_lora_specs(
            [{"id": "tutu_20to8_nfe_step100", "scale": 0.8}],
            BUILTIN_LORAS,
        )
        self.assertEqual(len(parsed), 1)
        self.assertIn("step000100", parsed[0][0])
        self.assertEqual(parsed[0][1], 0.8)

    def test_build_argv_emits_lora_and_rejects_ssd(self) -> None:
        lora = LoraRef(
            spec="x",
            path=Path("/tmp/tutu.safetensors"),
            scale=0.8,
        )
        req = GenerateRequest(
            prompt="fox",
            output_path=Path("/tmp/out.mp4"),
            loras=[lora],
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertIn("--lora", argv)
        self.assertTrue(any(str(a).startswith("/tmp/tutu.safetensors") for a in argv))
        session = build_session_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertIn("--lora", session)
        with self.assertRaisesRegex(ValueError, "ssd-streaming"):
            build_h3_argv(
                h3_bin=Path("/opt/h3"),
                model_dir=Path("/m"),
                req=GenerateRequest(
                    prompt="fox",
                    output_path=Path("/tmp/out.mp4"),
                    loras=[lora],
                    ssd_streaming=True,
                ),
            )


if __name__ == "__main__":
    unittest.main()
