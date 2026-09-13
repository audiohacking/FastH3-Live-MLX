"""Clip recipe persistence helpers. No network."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_ui import _enrich_recipe, _recipe_from_body  # noqa: E402


class RecipeFromBodyTests(unittest.TestCase):
    def test_uses_explicit_recipe(self) -> None:
        recipe = _recipe_from_body(
            {
                "prompt": "Picture 1 compiled",
                "mode": "ref2va",
                "recipe": {
                    "composer_prompt": "@img-1 hello",
                    "routing": "ref2va",
                    "refs": [{"kind": "image", "path": "/tmp/a.png", "name": "a"}],
                },
            }
        )
        self.assertEqual(recipe["composer_prompt"], "@img-1 hello")
        self.assertEqual(recipe["refs"][0]["name"], "a")

    def test_falls_back_to_generate_payload(self) -> None:
        recipe = _recipe_from_body(
            {
                "prompt": "a fox",
                "mode": "first_frame",
                "image_path": "/tmp/first.png",
                "token_reduction": False,
                "ssd_streaming": True,
                "quality": "close",
            }
        )
        self.assertEqual(recipe["composer_prompt"], "a fox")
        self.assertEqual(recipe["routing"], "fl2va")
        self.assertEqual(recipe["image_path"], "/tmp/first.png")
        self.assertFalse(recipe["token_reduction"])
        self.assertTrue(recipe["ssd_streaming"])


class EnrichRecipeTests(unittest.TestCase):
    def test_marks_missing_and_present_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upload = root / "uploads"
            upload.mkdir()
            present = upload / "face.png"
            present.write_bytes(b"x")

            class _State:
                upload_dir = upload
                output_dir = root / "out"

            _State.output_dir.mkdir()
            out = _enrich_recipe(
                _State(),  # type: ignore[arg-type]
                {
                    "refs": [
                        {"kind": "image", "path": str(present), "name": "face"},
                        {"kind": "audio", "path": str(upload / "gone.wav"), "name": "voice"},
                    ],
                    "image_path": str(upload / "missing.png"),
                },
            )
            self.assertTrue(out["refs"][0]["available"])
            self.assertTrue(str(out["refs"][0]["preview_url"]).endswith("/face.png"))
            self.assertFalse(out["refs"][1]["available"])
            self.assertFalse(out["image_path_available"])


if __name__ == "__main__":
    unittest.main()
