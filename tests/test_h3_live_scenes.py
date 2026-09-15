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
        self.assertGreater(total, 40)

    def test_office_pool_names_show(self) -> None:
        pool = PromptPool(curated_share=1.0)
        prompt, cast, _idx = pool.draw()
        self.assertIn("from The Office", prompt)
        self.assertNotIn("{NAME}", prompt)
        # Cast is drawn from The Office names only.
        for name in cast.split(" + "):
            self.assertIn(name, pool.full)

    def test_max_cast_is_three(self) -> None:
        pool = PromptPool()
        for scene in pool.scenes():
            self.assertLessEqual(PromptPool.slot_count(scene), 3)
        for _ in range(30):
            _prompt, cast, _idx = pool.draw()
            n = 0 if cast == "(no cast)" else len(cast.split(" + "))
            self.assertLessEqual(n, 3)
            self.assertGreaterEqual(n, 1)

    def test_cast_names_are_unique(self) -> None:
        pool = PromptPool(curated_share=1.0, ensemble_bias=0.0)
        for _ in range(50):
            prompt, cast, _idx = pool.draw()
            names = cast.split(" + ")
            self.assertEqual(len(names), len(set(names)), cast)
            self.assertNotIn("{NAME}", prompt)
            self.assertNotIn("{NAME2}", prompt)
            self.assertNotIn("{NAME3}", prompt)

    def test_office_scenes_are_static_face_forward(self) -> None:
        pool = PromptPool()
        for scene in pool.scenes():
            low = scene.lower()
            self.assertTrue(
                any(
                    m in low
                    for m in (
                        "static shot",
                        "pushes in with small amplitude at slow speed",
                        "pulls out with small amplitude at slow speed",
                        "trucks left with small amplitude at slow speed",
                        "trucks right with small amplitude at slow speed",
                        "arcs around the subject at slow speed",
                    )
                ),
                scene[:120],
            )
            self.assertTrue(
                "facing the camera" in low
                or "toward the camera" in low
                or "faces the camera" in low
                or "into the camera" in low
                or "documentary camera" in low
                or "documentary lens" in low,
                scene[:120],
            )
            self.assertIn("the office", low)
            self.assertIn("mockumentary", low)
            self.assertNotRegex(low, r"from behind|over shoulder")
            self.assertLessEqual(PromptPool.slot_count(scene), 2)

    def test_wrap_idea_makes_context_ir(self) -> None:
        wrapped = PromptPool.wrap_idea_as_live_prompt(
            "happy muppets looking at the camera"
        )
        self.assertTrue(PromptPool.looks_like_context_ir(wrapped))
        self.assertIn("overall_soundscape:", wrapped)
        self.assertIn("non_diegetic_music:", wrapped)
        self.assertIn("muppets", wrapped.lower())

    def test_normalize_custom_wraps_one_liner(self) -> None:
        pool = PromptPool(curated_share=1.0)
        prompt, _cast = pool.normalize_custom("a red ball on a table")
        self.assertIn("integrated_multimodal_description:", prompt)
        self.assertIn("red ball", prompt.lower())

    def test_normalize_custom_keeps_full_ir(self) -> None:
        pool = PromptPool(curated_share=1.0)
        raw = (
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
            "{NAME} (S1) waves. The camera holds.\n"
            "overall_soundscape: Soft room tone.\n"
            "non_diegetic_music: N/A"
        )
        prompt, cast = pool.normalize_custom(raw)
        self.assertNotIn("{NAME}", prompt)
        self.assertNotEqual(cast, "(no cast)")


class ValidateTests(unittest.TestCase):
    def test_shipped_scenes_pass(self) -> None:
        from h3_live import DATA_DIR

        path = DATA_DIR / "prompts_scenes.txt"
        checked, failures = validate_scene_file(path)
        self.assertGreater(checked, 100)
        self.assertEqual(failures, 0)

    def test_office_scenes_pass(self) -> None:
        from h3_live import DATA_DIR

        path = DATA_DIR / "prompts_scenes_office.txt"
        checked, failures = validate_scene_file(path)
        self.assertGreater(checked, 40)
        self.assertEqual(failures, 0)


if __name__ == "__main__":
    unittest.main()
