import unittest
from pathlib import Path

from h3_backend import GenerateRequest, RefItem, build_h3_argv, expand_quality
from h3_media import (
    MAX_PIXELS,
    RESOLUTION_PRESETS,
    require_ui_canvas,
    seconds_to_frames,
    snap_frames,
    snap_spatial,
    validate_canvas,
)


class SnapTests(unittest.TestCase):
    def test_snap_frames_minimum(self) -> None:
        self.assertEqual(snap_frames(1), 22)
        self.assertEqual(snap_frames(22), 22)

    def test_snap_frames_up(self) -> None:
        self.assertEqual(snap_frames(23), 39)
        self.assertEqual(snap_frames(107), 107)
        self.assertEqual(snap_frames(108), 124)

    def test_seconds_to_frames(self) -> None:
        self.assertEqual(seconds_to_frames(0.9), 22)
        self.assertEqual(seconds_to_frames(1), 22)
        self.assertEqual(seconds_to_frames(2), 56)
        self.assertEqual(seconds_to_frames(5), 107)
        self.assertEqual(seconds_to_frames(10), 243)
        self.assertEqual(seconds_to_frames(15), 362)

    def test_ui_duration_presets_are_rounded(self) -> None:
        from h3_media import DURATION_PRESETS, UI_DURATION_FRAMES

        self.assertEqual([p["id"] for p in DURATION_PRESETS], ["1s", "2s", "5s", "10s", "15s"])
        for preset in DURATION_PRESETS:
            rounded = int(preset["seconds"])
            self.assertEqual(preset["seconds"], float(rounded))
            self.assertEqual(preset["num_frames"], UI_DURATION_FRAMES[rounded])

    def test_snap_spatial(self) -> None:
        self.assertEqual(snap_spatial(16), 32)
        self.assertEqual(snap_spatial(512), 512)
        self.assertEqual(snap_spatial(500), 512)

    def test_validate_canvas_cap(self) -> None:
        validate_canvas(768, 1344)
        with self.assertRaises(ValueError):
            validate_canvas(1344, 1344)

    def test_require_ui_canvas_documented_only(self) -> None:
        self.assertEqual(require_ui_canvas(512, 512), (512, 512))
        self.assertEqual(require_ui_canvas(256, 256), (256, 256))
        self.assertEqual(require_ui_canvas(1344, 768), (1344, 768))
        self.assertEqual(require_ui_canvas(1024, 576), (1024, 576))
        self.assertEqual(require_ui_canvas(576, 1024), (576, 1024))
        self.assertEqual(require_ui_canvas(768, 960), (768, 960))
        with self.assertRaises(ValueError):
            require_ui_canvas(640, 640)
        with self.assertRaises(ValueError):
            require_ui_canvas(1024, 1024)


class ResolutionPresetTests(unittest.TestCase):
    def test_every_preset_obeys_h3_spatial_rules(self) -> None:
        seen: set[str] = set()
        for preset in RESOLUTION_PRESETS:
            w, h = int(preset["width"]), int(preset["height"])
            self.assertEqual(w % 32, 0, preset["id"])
            self.assertEqual(h % 32, 0, preset["id"])
            self.assertGreaterEqual(w, 32)
            self.assertGreaterEqual(h, 32)
            self.assertLessEqual(w * h, MAX_PIXELS, preset["id"])
            self.assertEqual(require_ui_canvas(w, h), (w, h))
            self.assertNotIn(preset["id"], seen)
            seen.add(str(preset["id"]))

    def test_social_aspects_are_exact_or_tagged(self) -> None:
        from math import gcd

        def ratio(w: int, h: int) -> tuple[int, int]:
            g = gcd(w, h)
            return w // g, h // g

        by_id = {p["id"]: p for p in RESOLUTION_PRESETS}
        self.assertEqual(ratio(by_id["1024x576"]["width"], by_id["1024x576"]["height"]), (16, 9))
        self.assertEqual(ratio(by_id["576x1024"]["width"], by_id["576x1024"]["height"]), (9, 16))
        self.assertEqual(ratio(by_id["768x960"]["width"], by_id["768x960"]["height"]), (4, 5))
        self.assertEqual(ratio(by_id["960x768"]["width"], by_id["960x768"]["height"]), (5, 4))
        self.assertEqual(ratio(by_id["1024x768"]["width"], by_id["1024x768"]["height"]), (4, 3))
        self.assertEqual(ratio(by_id["768x1024"]["width"], by_id["768x1024"]["height"]), (3, 4))
        self.assertEqual(ratio(by_id["1344x768"]["width"], by_id["1344x768"]["height"]), (7, 4))
        self.assertEqual(ratio(by_id["768x1344"]["width"], by_id["768x1344"]["height"]), (4, 7))
        self.assertEqual(by_id["1024x576"]["aspect"], "16:9")
        self.assertEqual(by_id["576x1024"]["aspect"], "9:16")
        self.assertEqual(by_id["768x960"]["aspect"], "4:5")
        self.assertEqual(by_id["1344x768"]["aspect"], "7:4")
        self.assertEqual(by_id["768x1344"]["aspect"], "4:7")


class QualityTests(unittest.TestCase):
    def test_expand_quality_rejects_token_reduction_aggressive_combo(self) -> None:
        with self.assertRaisesRegex(ValueError, "token-reduction"):
            expand_quality("aggressive", token_reduction=True, width=512, height=512)

    def test_expand_quality_small_steps_forces_reuse_1(self) -> None:
        q = expand_quality("four_step")
        self.assertEqual(q["steps"], 4)
        self.assertEqual(q["reuse"], 1)

    def test_close_keeps_fifty_full_blocks(self) -> None:
        q = expand_quality("close")
        self.assertEqual(q["steps"], 50)
        self.assertEqual(q["layers"], 50)
        self.assertEqual(q["reuse"], 1)

    def test_balanced_is_h3_cli_defaults(self) -> None:
        q = expand_quality("balanced")
        self.assertEqual(q["steps"], 20)
        self.assertEqual(q["layers"], 50)
        self.assertEqual(q["reuse"], 1)

    def test_layers_and_reuse_overrides(self) -> None:
        q = expand_quality("fast", steps=20, layers=50, reuse=1)
        self.assertEqual(q["steps"], 20)
        self.assertEqual(q["layers"], 50)
        self.assertEqual(q["reuse"], 1)

    def test_build_argv_t2va(self) -> None:
        req = GenerateRequest(
            prompt="a red fox",
            output_path=Path("/tmp/out.mp4"),
            width=512,
            height=512,
            num_frames=22,
            quality="balanced",
            seed=42,
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/models/H3"), req=req)
        self.assertEqual(argv[0], "/opt/h3")
        self.assertIn("-p", argv)
        self.assertIn("a red fox", argv)
        self.assertIn("--frames", argv)
        self.assertIn("22", argv)
        self.assertEqual(argv[argv.index("--steps") + 1], "20")
        self.assertEqual(argv[argv.index("--layers") + 1], "50")
        self.assertEqual(argv[argv.index("--reuse") + 1], "1")
        self.assertIn("--seed", argv)
        self.assertIn("42", argv)
        self.assertIn("--profile", argv)

    def test_build_argv_rejects_mixed_checkpoints(self) -> None:
        req = GenerateRequest(
            prompt="x",
            output_path=Path("/tmp/out.mp4"),
            first_frame=Path("a.png"),
            refs=[RefItem(kind="image", path=Path("b.png"))],
        )
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)

    def test_build_argv_audio_needs_visual_ref(self) -> None:
        req = GenerateRequest(
            prompt="x",
            output_path=Path("/tmp/out.mp4"),
            refs=[RefItem(kind="audio", path=Path("a.wav"))],
        )
        with self.assertRaisesRegex(ValueError, "standalone audio"):
            build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)

    def test_ref_flag_order_preserved(self) -> None:
        req = GenerateRequest(
            prompt="Picture 1 then Video 1",
            output_path=Path("/tmp/out.mp4"),
            mode="ref2va",
            refs=[
                RefItem(kind="image", path=Path("a.png")),
                RefItem(kind="silent_video", path=Path("b.mp4")),
                RefItem(kind="audio", path=Path("c.wav")),
            ],
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertLess(argv.index("--ref-image"), argv.index("--ref-silent-video"))
        self.assertLess(argv.index("--ref-silent-video"), argv.index("--ref-audio"))
        self.assertEqual(argv[argv.index("--ref-image") + 1], "a.png")
        self.assertEqual(argv[argv.index("--ref-silent-video") + 1], "b.mp4")
        self.assertEqual(argv[argv.index("--ref-audio") + 1], "c.wav")

    def test_ref_video_audio_flag(self) -> None:
        req = GenerateRequest(
            prompt="Video 1",
            output_path=Path("/tmp/out.mp4"),
            mode="ref2va",
            refs=[
                RefItem(kind="image", path=Path("still.png")),
                RefItem(
                    kind="video_audio",
                    path=Path("clip.mp4"),
                    audio_path=Path("replace.wav"),
                ),
            ],
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        i = argv.index("--ref-video-audio")
        self.assertEqual(argv[i + 1], "clip.mp4")
        self.assertEqual(argv[i + 2], "replace.wav")

    def test_internal_render_flags(self) -> None:
        req = GenerateRequest(
            prompt="x",
            output_path=Path("/tmp/out.mp4"),
            width=512,
            height=512,
            render_width=384,
            render_height=384,
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertEqual(argv[argv.index("--render-width") + 1], "384")
        self.assertEqual(argv[argv.index("--render-height") + 1], "384")
        self.assertEqual(argv[argv.index("--width") + 1], "512")

    def test_fast_preset_uses_token_reduction_and_384(self) -> None:
        q = expand_quality("fast", width=512, height=512)
        self.assertTrue(q["token_reduction"])
        self.assertEqual(q["render"], (384, 384))
        req = GenerateRequest(
            prompt="x",
            output_path=Path("/tmp/out.mp4"),
            quality="fast",
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertIn("--token-reduction", argv)
        self.assertEqual(argv[argv.index("--render-width") + 1], "384")

    def test_int8_row_fc2_not_combined_with_ssd(self) -> None:
        from h3_backend import metal4_gpu, resolve_int8_row_fc2

        self.assertTrue(metal4_gpu("Apple M5 Max"))
        self.assertFalse(metal4_gpu("Apple M3 Max"))
        req = GenerateRequest(prompt="x", output_path=Path("/tmp/out.mp4"), ssd_streaming=True)
        self.assertFalse(resolve_int8_row_fc2(req, metal4=True))
        req = GenerateRequest(
            prompt="x",
            output_path=Path("/tmp/out.mp4"),
            int8_row_fc2=True,
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertIn("--use-int8-row-fc2", argv)
        req.ssd_streaming = True
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertNotIn("--use-int8-row-fc2", argv)
        self.assertIn("--ssd-streaming", argv)

    def test_256_preview_disables_token_reduction(self) -> None:
        req = GenerateRequest(
            prompt="x",
            output_path=Path("/tmp/out.mp4"),
            width=256,
            height=256,
            quality="fast",
            token_reduction=True,
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertNotIn("--token-reduction", argv)


class AudioDurationTests(unittest.TestCase):
    def test_audio_too_short(self) -> None:
        from h3_media import assert_audio_durations

        with self.assertRaisesRegex(ValueError, "at least 2"):
            assert_audio_durations([1.2])

    def test_audio_too_long(self) -> None:
        from h3_media import assert_audio_durations

        with self.assertRaisesRegex(ValueError, "at most 15"):
            assert_audio_durations([16.0])

    def test_audio_total_cap(self) -> None:
        from h3_media import assert_audio_durations

        with self.assertRaisesRegex(ValueError, "caps the sum"):
            assert_audio_durations([8.0, 8.0])

    def test_audio_ok(self) -> None:
        from h3_media import assert_audio_durations

        assert_audio_durations([2.0, 5.0, 7.0])


class SessionCommandTests(unittest.TestCase):
    def test_session_commands_clear_refs_before_anchors(self) -> None:
        from h3_session import session_commands_for_request

        req = GenerateRequest(
            prompt="fox",
            output_path=Path("/tmp/out.mp4"),
            first_frame=Path("a.png"),
            quality="balanced",
            seed=7,
        )
        cmds = session_commands_for_request(req)
        self.assertLess(cmds.index("!refs clear"), cmds.index("!first a.png"))
        self.assertIn("!last clear", cmds)
        self.assertIn("!seed 7", cmds)
        self.assertIn("!reuse 1", cmds)
        self.assertIn("!core-reuse 1", cmds)

    def test_session_argv_has_no_prompt(self) -> None:
        from h3_session import build_session_argv

        req = GenerateRequest(prompt="fox", output_path=Path("/tmp/out.mp4"))
        argv = build_session_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertNotIn("-p", argv)
        self.assertNotIn("fox", argv)
        self.assertIn("--profile", argv)

    def test_parse_done_and_progress(self) -> None:
        from h3_session import parse_cli_progress, parse_done_path, parse_outputs_dir

        text = "Outputs: /tmp/h3-abc123\nDone -> /tmp/h3-abc123/video-0001.mp4 [12.50s]\nh3> "
        self.assertEqual(parse_outputs_dir(text), Path("/tmp/h3-abc123"))
        self.assertEqual(parse_done_path(text), Path("/tmp/h3-abc123/video-0001.mp4"))
        progress = parse_cli_progress("\rdenoise                  4/20")
        self.assertIsNotNone(progress)
        assert progress is not None
        self.assertEqual(progress["step"], 4)
        self.assertEqual(progress["total"], 20)

    def test_progress_eta_matches_step_rate(self) -> None:
        from h3_backend import ProgressEta, _format_mmss, progress_console_line

        eta = ProgressEta()
        t0 = 1_000.0
        eta.enrich({"stage": "denoise", "step": 0, "total": 20}, now=t0)
        mid = eta.enrich(
            {"stage": "denoise", "step": 4, "total": 20, "label": "denoise 4/20"},
            now=t0 + 40.0,
        )
        self.assertEqual(mid["avg_step_s"], 10.0)
        self.assertEqual(mid["eta_s"], 160.0)
        self.assertIn("02:40 remaining", progress_console_line(mid))
        self.assertEqual(_format_mmss(54), "00:54")
        later = eta.enrich(
            {"stage": "video VAE load", "step": 1, "total": 36},
            now=t0 + 50.0,
        )
        self.assertIsNone(later.get("eta_s"))

    def test_repl_prompt_detects_linenoise_cr(self) -> None:
        from h3_session import has_repl_prompt

        self.assertTrue(has_repl_prompt("Size: 512x512\r\nh3> "))
        self.assertTrue(has_repl_prompt("h3> "))
        self.assertFalse(has_repl_prompt("h3> !size 512x512"))

    def test_h3_process_cwd_is_binary_dir(self) -> None:
        from h3_paths import h3_process_cwd

        self.assertEqual(h3_process_cwd(Path("/opt/h3.c/h3")), Path("/opt/h3.c"))

    def test_debug_console_defaults_on(self) -> None:
        import os

        from h3_paths import debug_console

        prev = os.environ.pop("DEBUG", None)
        try:
            self.assertTrue(debug_console())
            os.environ["DEBUG"] = "false"
            self.assertFalse(debug_console())
            os.environ["DEBUG"] = "true"
            self.assertTrue(debug_console())
        finally:
            if prev is None:
                os.environ.pop("DEBUG", None)
            else:
                os.environ["DEBUG"] = prev


class ModelLayoutTests(unittest.TestCase):
    def test_empty_fl2va_dir_is_incomplete(self) -> None:
        import tempfile

        from h3_backend import model_layout_ok

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "FL2VA").mkdir()
            ok, note = model_layout_ok(dest)
            self.assertFalse(ok)
            self.assertIn("incomplete", note)


if __name__ == "__main__":
    unittest.main()
