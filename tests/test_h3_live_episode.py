"""Unit tests for episode batch helpers (no h3)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_live.episode import (  # noqa: E402
    EpisodeJob,
    clamp_scene_count,
    concat_mp4s,
)


class ClampTests(unittest.TestCase):
    def test_ok(self) -> None:
        self.assertEqual(clamp_scene_count(1), 1)
        self.assertEqual(clamp_scene_count("10"), 10)

    def test_rejects(self) -> None:
        with self.assertRaises(ValueError):
            clamp_scene_count(0)
        with self.assertRaises(ValueError):
            clamp_scene_count(11)
        with self.assertRaises(ValueError):
            clamp_scene_count("x")


class JobTests(unittest.TestCase):
    def test_lifecycle(self) -> None:
        job = EpisodeJob()
        self.assertFalse(job.is_active())
        job.reset_for_start(3)
        self.assertTrue(job.is_active())
        self.assertEqual(job.scenes_requested, 3)
        d = job.as_dict()
        self.assertEqual(d["status"], "running")
        self.assertFalse(d["download_ready"])


class ConcatTests(unittest.TestCase):
    def test_concat_two_clips(self) -> None:
        import av
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for n in range(2):
                src = Path(tmp) / f"s{n}.mp4"
                out = av.open(str(src), mode="w")
                v = out.add_stream("libx264", rate=24)
                v.width = 64
                v.height = 64
                v.pix_fmt = "yuv420p"
                a = out.add_stream("aac", rate=48000)
                a.layout = "stereo"
                for i in range(6):
                    img = np.zeros((64, 64, 3), dtype=np.uint8)
                    img[:, :, n] = 180
                    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    frame = frame.reformat(format="yuv420p")
                    for p in v.encode(frame):
                        out.mux(p)
                for p in v.encode(None):
                    out.mux(p)
                samples = np.zeros((2, 2048), dtype=np.float32)
                samples[:] = 0.05
                aframe = av.AudioFrame.from_ndarray(
                    samples, format="fltp", layout="stereo"
                )
                aframe.sample_rate = 48000
                for p in a.encode(aframe):
                    out.mux(p)
                for p in a.encode(None):
                    out.mux(p)
                out.close()
                paths.append(src)

            dest = Path(tmp) / "episode.mp4"
            concat_mp4s(paths, dest)
            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size, 500)
            probe = av.open(str(dest))
            try:
                kinds = {s.type for s in probe.streams}
                self.assertIn("video", kinds)
                self.assertIn("audio", kinds)
            finally:
                probe.close()


if __name__ == "__main__":
    unittest.main()
