"""Unit tests for FastH3 Live scene pool (no network, no h3)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_live.scenes import PromptPool  # noqa: E402
from h3_live.validate import validate_scene_file  # noqa: E402
from h3_media import require_ui_canvas  # noqa: E402


class CanvasTests(unittest.TestCase):
    def test_448_is_allowed(self) -> None:
        self.assertEqual(require_ui_canvas(448, 448), (448, 448))


class PromptPoolTests(unittest.TestCase):
    def test_draw_fills_name(self) -> None:
        pool = PromptPool(curated_share=1.0)
        prompt, cast, idx = pool.draw()
        self.assertNotIn("{NAME}", prompt)
        self.assertGreaterEqual(idx, 0)
        self.assertTrue(cast)
        self.assertIn("integrated_multimodal_description", prompt)

    def test_counts(self) -> None:
        pool = PromptPool()
        total = sum(pool.counts().values())
        self.assertGreater(total, 300)


class ValidateTests(unittest.TestCase):
    def test_shipped_scenes_pass(self) -> None:
        from h3_live import DATA_DIR

        path = DATA_DIR / "prompts_scenes.txt"
        checked, failures = validate_scene_file(path)
        self.assertGreater(checked, 100)
        self.assertEqual(failures, 0)


if __name__ == "__main__":
    unittest.main()
