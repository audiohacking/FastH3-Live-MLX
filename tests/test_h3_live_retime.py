"""Retime smoke test with a synthetic short clip (no h3)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RetimeTests(unittest.TestCase):
    def test_pyav_retime_writes_ts(self) -> None:
        import av
        import numpy as np

        from h3_live.retime import retime_to_mpegts

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mp4"
            # 8 frames @ 24 fps + short stereo tone
            out = av.open(str(src), mode="w")
            v = out.add_stream("libx264", rate=24)
            v.width = 64
            v.height = 64
            v.pix_fmt = "yuv420p"
            a = out.add_stream("aac", rate=48000)
            a.layout = "stereo"
            for i in range(8):
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                img[:, :, 0] = (i * 30) % 255
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                frame = frame.reformat(format="yuv420p")
                for p in v.encode(frame):
                    out.mux(p)
            for p in v.encode(None):
                out.mux(p)
            # ~0.3 s of audio
            samples = np.zeros((1, 2 * (48000 // 3)), dtype=np.float32)  # packed stereo
            t = np.linspace(0, 40 * np.pi, 48000 // 3)
            tone = 0.1 * np.sin(t)
            samples[0, 0::2] = tone
            samples[0, 1::2] = tone
            aframe = av.AudioFrame.from_ndarray(samples, format="flt", layout="stereo")
            aframe.sample_rate = 48000
            for p in a.encode(aframe):
                out.mux(p)
            for p in a.encode(None):
                out.mux(p)
            out.close()

            ts = retime_to_mpegts(src, Path(tmp) / "out.ts", play_fps=12.0)
            self.assertTrue(ts.is_file())
            self.assertGreater(ts.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
