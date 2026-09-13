"""Unit tests for adaptive Live play-fps pacing."""

from __future__ import annotations

import unittest

from h3_live.pace import adaptive_play_fps, ema


class TestAdaptivePlayFps(unittest.TestCase):
    def test_slower_host_gets_lower_fps(self) -> None:
        slow = adaptive_play_fps(124, 68.0, margin_ratio=1.08)
        fast = adaptive_play_fps(124, 35.0, margin_ratio=1.08)
        self.assertLess(slow, fast)
        self.assertAlmostEqual(slow, 124 / (68 * 1.08), places=3)

    def test_respects_min_max(self) -> None:
        self.assertEqual(adaptive_play_fps(124, 500.0, min_fps=0.5), 0.5)
        self.assertEqual(adaptive_play_fps(124, 1.0, max_fps=24.0), 24.0)

    def test_ema_smooths(self) -> None:
        self.assertEqual(ema(None, 10.0), 10.0)
        self.assertAlmostEqual(ema(10.0, 20.0, alpha=0.5), 15.0)


if __name__ == "__main__":
    unittest.main()
