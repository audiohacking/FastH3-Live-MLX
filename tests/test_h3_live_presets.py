"""Unit tests for Live quality presets and ensemble draw."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_live.presets import (  # noqa: E402
    DEFAULT_PRESET,
    PRESETS,
    apply_preset_to_args,
    get_preset,
    recipe_label,
)
from h3_live.scenes import PromptPool  # noqa: E402


class PresetTests(unittest.TestCase):
    def test_default_is_live(self) -> None:
        self.assertEqual(DEFAULT_PRESET, "live")
        live = get_preset("live")
        self.assertEqual(live.render_width, 384)
        self.assertFalse(live.token_reduction)

    def test_draft_keeps_speed_path(self) -> None:
        draft = get_preset("draft")
        self.assertEqual(draft.render_width, 320)
        self.assertTrue(draft.token_reduction)

    def test_sharp_is_full_canvas(self) -> None:
        sharp = get_preset("sharp")
        self.assertEqual(sharp.render_width, 448)
        self.assertFalse(sharp.token_reduction)

    def test_apply_preset_to_args(self) -> None:
        ns = argparse.Namespace()
        apply_preset_to_args(ns, get_preset("long"))
        self.assertEqual(ns.frames, 243)
        self.assertEqual(ns.render_width, 384)
        self.assertFalse(ns.token_reduction)

    def test_recipe_label(self) -> None:
        label = recipe_label(
            preset="live",
            width=448,
            height=448,
            render_width=384,
            render_height=384,
            frames=124,
            token_reduction=False,
            steps=4,
        )
        self.assertIn("preset=live", label)
        self.assertIn("noTR", label)


class EnsembleTests(unittest.TestCase):
    def test_ensemble_only_draw(self) -> None:
        from h3_live import DATA_DIR

        # Office Live pool is 1–2 cast; exercise ensemble-only against the original pool.
        pool = PromptPool(
            [DATA_DIR / "prompts_scenes.txt"],
            DATA_DIR / "h3_characters.json",
            curated_share=1.0,
            ensemble_only=True,
            ensemble_bias=1.0,
            max_cast=5,
        )
        for _ in range(5):
            prompt, cast, _idx = pool.draw()
            self.assertNotIn("{NAME}", prompt)
            self.assertGreaterEqual(cast.count(" + ") + 1, 3)
            self.assertIn("integrated_multimodal_description", prompt)

    def test_is_ensemble(self) -> None:
        self.assertFalse(PromptPool.is_ensemble("{NAME} alone"))
        self.assertTrue(
            PromptPool.is_ensemble("{NAME} {NAME2} {NAME3} together")
        )

    def test_office_pool_has_no_trios(self) -> None:
        pool = PromptPool()
        for scene in pool.scenes():
            self.assertLessEqual(PromptPool.slot_count(scene), 2)


if __name__ == "__main__":
    unittest.main()
