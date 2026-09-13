import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_av(argv: list[str], *, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    script = _repo_root() / "h3_av.py"
    env = os.environ.copy()
    env["H3_AV_FORCE_PYAV"] = "1"
    return subprocess.run(
        [sys.executable, str(script), *argv],
        input=stdin,
        capture_output=True,
        check=False,
        env=env,
    )


class H3AvShimTests(unittest.TestCase):
    def test_ffprobe_png_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            Image.new("RGB", (96, 64), (10, 20, 30)).save(path)
            proc = _run_av(
                [
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    str(path),
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            self.assertEqual(proc.stdout.decode().strip(), "96x64")

    def test_decode_image_stretch_rgb24(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            Image.new("RGB", (40, 20), (255, 0, 0)).save(path)
            proc = _run_av(
                [
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=32:32:flags=lanczos",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            self.assertEqual(len(proc.stdout), 32 * 32 * 3)

    def test_encode_raw_rgb_mp4(self) -> None:
        width, height, frames = 32, 32, 5
        raw = bytes([i % 256 for i in range(width * height * 3)]) * frames
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.mp4"
            proc = _run_av(
                [
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    "24",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    str(out),
                ],
                stdin=raw,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 32)

    def test_av_mux_argv_keeps_pcm_on_second_input(self) -> None:
        from h3_av import _inputs_and_output

        argv = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", "1024x768", "-framerate", "24",
            "-i", "pipe:0",
            "-f", "f32le", "-ar", "32000", "-ac", "2",
            "-i", "pipe:7",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "/tmp/out.mp4",
        ]
        inputs, output, output_opts = _inputs_and_output(argv)
        self.assertEqual(output, "/tmp/out.mp4")
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0]["f"], "rawvideo")
        self.assertEqual(inputs[0]["path"], "pipe:0")
        self.assertEqual(inputs[1]["f"], "f32le")
        self.assertEqual(inputs[1]["path"], "pipe:7")
        self.assertEqual(inputs[1]["ar"], "32000")
        self.assertEqual(output_opts.get("c:v"), "libx264")
        self.assertNotEqual(output_opts.get("f"), "f32le")

    def test_encode_rgb_and_pcm_mp4(self) -> None:
        width, height, frames, fps = 32, 32, 8, 24
        raw = bytes(width * height * 3 * frames)
        samples = int(32000 * frames / fps)
        pcm = b"\x00\x00\x00\x00" * 2 * samples
        r_audio, w_audio = os.pipe()

        def _feed() -> None:
            remaining = pcm
            while remaining:
                written = os.write(w_audio, remaining)
                remaining = remaining[written:]
            os.close(w_audio)

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "av.mp4"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_repo_root() / "h3_av.py"),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-f",
                    "f32le",
                    "-ar",
                    "32000",
                    "-ac",
                    "2",
                    "-i",
                    f"pipe:{r_audio}",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(out),
                ],
                input=raw,
                capture_output=True,
                check=False,
                pass_fds=(r_audio,),
                env={**os.environ, "H3_AV_FORCE_PYAV": "1"},
            )
            os.close(r_audio)
            feeder.join(timeout=5)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 32)

    def test_real_ffmpeg_skips_shim_bindir(self) -> None:
        from h3_av import real_ffmpeg
        from h3_paths import ffmpeg_shim_bindir, h3_media_env

        bindir = ffmpeg_shim_bindir()
        found = real_ffmpeg()
        if found:
            self.assertNotEqual(Path(found).resolve(), (bindir / "ffmpeg").resolve())
            self.assertFalse(Path(found).name.startswith("h3-av"))
        os.environ["H3_AV_FORCE_PYAV"] = "1"
        try:
            self.assertIsNone(real_ffmpeg())
            env = h3_media_env({})
            self.assertTrue(str(env.get("H3_AV") or "").endswith("h3-av"))
        finally:
            os.environ.pop("H3_AV_FORCE_PYAV", None)

    def test_media_env_uses_system_ffmpeg_without_h3_av(self) -> None:
        from h3_paths import h3_media_env, real_ffmpeg

        found = real_ffmpeg()
        if not found:
            self.skipTest("no system ffmpeg")
        env = h3_media_env({})
        self.assertNotIn("H3_AV", env)
        self.assertEqual(env["H3_FFMPEG"], found)

    def test_tool_runs_rejects_dyld_crash(self) -> None:
        from h3_paths import _tool_ok, _tool_runs

        _tool_ok.pop("/usr/bin/false", None)
        self.assertFalse(_tool_runs("/usr/bin/false"))
        self.assertFalse(_tool_runs("/no/such/ffmpeg"))
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "ffmpeg"
            fake.write_text(
                "#!/bin/sh\n"
                'echo "dyld[1]: Library not loaded: '
                '/opt/homebrew/opt/x265/lib/libx265.215.dylib" >&2\n'
                "exit 1\n"
            )
            fake.chmod(0o755)
            _tool_ok.pop(str(fake), None)
            self.assertFalse(_tool_runs(str(fake)))

    def test_media_shim_ok(self) -> None:
        from h3_media import media_shim_ok

        ok, note = media_shim_ok()
        self.assertTrue(ok, note)


if __name__ == "__main__":
    unittest.main()
