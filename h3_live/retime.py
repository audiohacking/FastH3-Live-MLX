"""Retime H3 24 fps A/V to a slower play fps for continuous streaming."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from h3_paths import real_ffmpeg

log = logging.getLogger("h3-live.retime")

NATIVE_FPS = 24.0


def audio_stretch_filter(rate: float, mode: str = "rubberband") -> str:
    """FFmpeg filter that slows audio to ``rate`` (play_fps / 24)."""
    if mode == "atempo":
        return f"atempo={rate:.8f}"
    if mode == "atempo2":
        half = rate**0.5
        return f"atempo={half:.8f},atempo={half:.8f}"
    if mode == "tape":
        return f"rubberband=tempo={rate:.8f}:pitch={rate:.8f}"
    return f"rubberband=tempo={rate:.8f}:transients=smooth"


def _retime_with_ffmpeg(
    src: Path,
    out: Path,
    *,
    play_fps: float,
    stretch: str,
    vbitrate: str,
    arate: int,
    lufs: float | None,
    ts_offset: float,
    ffmpeg: str,
) -> None:
    rate = float(play_fps) / NATIVE_FPS
    attempts = [stretch]
    if stretch == "rubberband":
        attempts.append("atempo2")
    last_err = ""
    for mode in attempts:
        afilt = audio_stretch_filter(rate, mode)
        if lufs is not None:
            afilt = f"{afilt},loudnorm=I={lufs}:TP=-1.5:LRA=11"
        afilt = f"{afilt},aresample={arate}"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            f"[0:v]setpts=PTS/{rate:.6f}[v];[0:a]{afilt}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-r",
            str(play_fps),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-bf",
            "0",
            "-g",
            str(max(1, int(round(play_fps)))),
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            vbitrate,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            str(arate),
            "-ac",
            "2",
            "-muxdelay",
            "0",
            "-output_ts_offset",
            f"{ts_offset:.3f}",
            "-f",
            "mpegts",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            if mode != stretch:
                log.warning("rubberband unavailable — retimed with %s", mode)
            return
        last_err = (proc.stderr or proc.stdout or "")[-800:]
        if "rubberband" in last_err.lower() or "No such filter" in last_err:
            continue
        break
    raise RuntimeError(f"ffmpeg retime failed:\n{last_err}")


def _stretch_audio_frames(frames: list, rate: float):
    """Pitch-shifting stretch by linear resample (fallback without atempo)."""
    import numpy as np
    import av

    if not frames:
        return frames
    layout = frames[0].layout.name
    rate_in = frames[0].sample_rate
    fmt = frames[0].format.name
    chunks = []
    for fr in frames:
        arr = fr.to_ndarray()
        chunks.append(arr)
    audio = np.concatenate(chunks, axis=-1)
    # Normalize to (channels, samples)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    if audio.shape[0] == 1 and "stereo" in layout:
        # packed stereo → (2, N)
        packed = audio.reshape(-1)
        if packed.size % 2 != 0:
            packed = packed[: packed.size - 1]
        audio = np.vstack([packed[0::2], packed[1::2]])
    n_in = audio.shape[1]
    n_out = max(1, int(round(n_in / rate)))
    x_old = np.linspace(0.0, 1.0, n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    stretched = np.vstack(
        [np.interp(x_new, x_old, audio[c].astype(np.float64)) for c in range(audio.shape[0])]
    ).astype(np.float32)

    out_frames = []
    hop = 1024
    for i in range(0, n_out, hop):
        sl = stretched[:, i : i + hop]
        if "flt" in fmt and sl.shape[0] > 1:
            # packed interleaved
            interleaved = np.empty((1, sl.shape[1] * sl.shape[0]), dtype=np.float32)
            for c in range(sl.shape[0]):
                interleaved[0, c :: sl.shape[0]] = sl[c]
            frame = av.AudioFrame.from_ndarray(interleaved, format="flt", layout=layout)
        else:
            frame = av.AudioFrame.from_ndarray(sl, format="fltp", layout=layout)
        frame.sample_rate = rate_in
        out_frames.append(frame)
    return out_frames


def _retime_with_pyav(
    src: Path,
    out: Path,
    *,
    play_fps: float,
    arate: int,
    ts_offset: float,
) -> None:
    """PyAV retime → MPEG-TS when system ffmpeg is unavailable.

    Video PTS are scaled for ``play_fps``. Audio is length-stretched by linear
    resample (pitch shifts slightly). Prefer a real ffmpeg + rubberband when
    available for pitch-preserving stretch.
    """
    import av

    rate = float(play_fps) / NATIVE_FPS
    # MPEG-TS likes a rational rate; keep millisecond precision for fractional fps.
    fps_num = int(round(play_fps * 1000))
    fps_den = 1000
    inp = av.open(str(src))
    outp = av.open(str(out), mode="w", format="mpegts")
    try:
        v_in = next((s for s in inp.streams if s.type == "video"), None)
        a_in = next((s for s in inp.streams if s.type == "audio"), None)
        if v_in is None:
            raise RuntimeError(f"no video in {src}")

        v_out = outp.add_stream("libx264", rate=Fraction(fps_num, fps_den))
        v_out.width = v_in.codec_context.width
        v_out.height = v_in.codec_context.height
        v_out.pix_fmt = "yuv420p"
        v_out.bit_rate = 4_000_000
        v_out.time_base = Fraction(fps_den, fps_num)
        # Baseline + frequent IDR: browsers/MSE (mpegts.js) are pickier than VLC.
        gop = max(1, int(round(play_fps)))  # ~1 keyframe / second
        v_out.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
            "profile": "baseline",
            "bf": "0",
            "g": str(gop),
            "keyint_min": str(gop),
            "sc_threshold": "0",
            "repeat-headers": "1",
        }

        # Video-only on the PyAV fallback path. Declaring AAC in the PMT without
        # early ADTS PES makes mpegts.js hang after the video init segment (VLC
        # still plays). Real ffmpeg retime keeps A/V when available.
        v_index = int(round(max(0.0, ts_offset) * play_fps))
        for frame in inp.decode(v_in):
            frame.pts = v_index
            frame.time_base = Fraction(fps_den, fps_num)
            for packet in v_out.encode(frame):
                packet.stream = v_out
                outp.mux(packet)
            v_index += 1
        for packet in v_out.encode(None):
            packet.stream = v_out
            outp.mux(packet)
        if a_in is not None:
            log.warning(
                "PyAV retime: dropping audio (no ADTS interleave) — browser MSE needs "
                "video-only or a working ffmpeg+rubberband path"
            )
    finally:
        inp.close()
        outp.close()

    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError("PyAV retime produced an empty file")
    log.warning(
        "retimed with PyAV fallback (no system ffmpeg) — video-only MPEG-TS; "
        "install a working ffmpeg+rubberband for A/V streams"
    )


def retime_to_mpegts(
    src: Path,
    dest: Path | None = None,
    *,
    play_fps: float,
    stretch: str = "rubberband",
    vbitrate: str = "4M",
    arate: int = 48000,
    lufs: float | None = None,
    ts_offset: float = 0.0,
) -> Path:
    """Rewrite one H3 MP4 to MPEG-TS at ``play_fps`` with stretched audio."""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(dest) if dest else Path(tempfile.mkstemp(suffix=".ts", prefix="h3live_")[1])
    rate = float(play_fps) / NATIVE_FPS
    if rate <= 0 or rate > 1.5:
        raise ValueError(f"play_fps {play_fps} implies bad stretch rate {rate}")

    ffmpeg = real_ffmpeg()
    if ffmpeg:
        _retime_with_ffmpeg(
            src,
            out,
            play_fps=play_fps,
            stretch=stretch,
            vbitrate=vbitrate,
            arate=arate,
            lufs=lufs,
            ts_offset=ts_offset,
            ffmpeg=ffmpeg,
        )
        return out

    _retime_with_pyav(src, out, play_fps=play_fps, arate=arate, ts_offset=ts_offset)
    return out


def feed_mpegts_to_sink(ts_path: Path, sink) -> None:
    """Copy a .ts file into an MPEG-TS muxer stdin (or any binary sink)."""
    with Path(ts_path).open("rb") as fh:
        shutil.copyfileobj(fh, sink, length=1 << 16)
