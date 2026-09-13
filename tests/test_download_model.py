"""Pattern tests for the MiniMax-H3 downloader. No Hugging Face traffic."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import download_model as dl  # noqa: E402

# Representative paths from MiniMaxAI/MiniMax-H3 — not a live listing.
REPO_FILES = [
    "model_index.json",
    "FL2VA/model_index.json",
    "FL2VA/tokenizer/tokenizer.json",
    "FL2VA/text_encoder/model-00001-of-00014.safetensors",
    "FL2VA/transformer/config.json",
    "FL2VA/transformer/model-00001-of-00013.safetensors",
    "FL2VA/video_vae/source/model.safetensors",
    "FL2VA/audio_vae/model.safetensors",
    "Ref2VA/model_index.json",
    "Ref2VA/text_encoder/model-00001-of-00014.safetensors",
    "Ref2VA/transformer/model-00001-of-00013.safetensors",
    "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
    "transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors",
    "text_encoder/model-00001-of-00014.safetensors",
    "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "assets/demo.png",
]


class DownloadFilterTests(unittest.TestCase):
    def test_default_is_native_fl2va_only(self) -> None:
        chosen = dl._filter_names(REPO_FILES, with_ref2va=False)
        self.assertIn("FL2VA/transformer/model-00001-of-00013.safetensors", chosen)
        self.assertIn("FL2VA/text_encoder/model-00001-of-00014.safetensors", chosen)
        self.assertIn("FL2VA/video_vae/source/model.safetensors", chosen)
        self.assertNotIn("Ref2VA/transformer/model-00001-of-00013.safetensors", chosen)
        self.assertNotIn("Ref2VA/text_encoder/model-00001-of-00014.safetensors", chosen)
        for name in (
            "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
            "transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors",
            "text_encoder/model-00001-of-00014.safetensors",
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
        ):
            self.assertNotIn(name, chosen)

    def test_ref2va_adds_transformer_not_duplicate_encoder(self) -> None:
        chosen = dl._filter_names(REPO_FILES, with_ref2va=True)
        self.assertIn("Ref2VA/transformer/model-00001-of-00013.safetensors", chosen)
        self.assertIn("Ref2VA/model_index.json", chosen)
        self.assertNotIn("Ref2VA/text_encoder/model-00001-of-00014.safetensors", chosen)
        self.assertNotIn("transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors", chosen)

    def test_summarize_skips_diffusers_bytes(self) -> None:
        sizes = {n: 1_000_000_000 for n in REPO_FILES}
        sizes["FL2VA/transformer/model-00001-of-00013.safetensors"] = 5_000_000_000
        sizes["transformer/diffusion_pytorch_model-00001-of-00014.safetensors"] = 9_000_000_000
        chosen, total, skipped = dl.summarize_selection(
            REPO_FILES, sizes, with_ref2va=False
        )
        self.assertGreater(skipped, total)
        self.assertNotIn(
            "transformer/diffusion_pytorch_model-00001-of-00014.safetensors", chosen
        )


class ReuseAndLinkTests(unittest.TestCase):
    def test_reuse_never_deletes_existing_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-dl-") as tmp:
            dest = Path(tmp)
            (dest / "FL2VA" / "transformer").mkdir(parents=True)
            (dest / "FL2VA" / "transformer" / "config.json").write_text("{}")
            (dest / "transformer").mkdir()
            blob = dest / "transformer" / "weights.bin"
            blob.write_bytes(b"x" * 100)
            (dest / "text_encoder").mkdir()
            (dest / "text_encoder" / "x.bin").write_bytes(b"y" * 50)
            dl.reuse_existing_layout(dest)
            self.assertTrue(blob.is_file())
            self.assertEqual(blob.read_bytes(), b"x" * 100)
            self.assertTrue((dest / "text_encoder" / "x.bin").is_file())
            self.assertTrue((dest / "FL2VA" / "transformer" / "config.json").is_file())

    def test_reuse_links_missing_fl2va_encoder_from_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-dl-") as tmp:
            dest = Path(tmp)
            enc = dest / "text_encoder"
            enc.mkdir()
            (enc / "model.bin").write_text("already-downloaded")
            notes = dl.reuse_existing_layout(dest)
            linked = dest / "FL2VA" / "text_encoder"
            self.assertTrue(linked.is_symlink())
            self.assertEqual((linked / "model.bin").read_text(), "already-downloaded")
            self.assertTrue(any("FL2VA/text_encoder" in n for n in notes))

    def test_link_keeps_existing_ref2va_copy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-dl-") as tmp:
            dest = Path(tmp)
            enc = dest / "FL2VA" / "text_encoder"
            enc.mkdir(parents=True)
            (enc / "model.bin").write_text("shared")
            dup = dest / "Ref2VA" / "text_encoder"
            dup.mkdir(parents=True)
            (dup / "model.bin").write_text("already-downloaded")
            dl.link_shared_ref2va(dest)
            self.assertFalse((dest / "Ref2VA" / "text_encoder").is_symlink())
            self.assertEqual((dup / "model.bin").read_text(), "already-downloaded")


if __name__ == "__main__":
    unittest.main()
