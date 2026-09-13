"""Retime H3 24 fps A/V to a slower play fps for continuous streaming.

Primary path is **PyAV** (intentional — no system ffmpeg dependency). Video PTS
are scaled for ``play_fps``; audio is length-stretched to match and muxed as
AAC ADTS into MPEG-TS with PTS interleave so mpegts.js sees early audio PES.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path

log = logging.getLogger("h3-live.retime")

NATIVE_FPS = 24.0


def _stretch_audio_frames(frames: list, rate: float, *, out_rate: int):
    """Length-stretch PCM by ``1/rate`` via linear resample → fltp stereo.

    ``rate`` is play_fps / 24 (e.g. 0.5 doubles duration). Pitch shifts slightly;
    that is acceptable for Live without a rubberband dependency.
    """
    import numpy as np
    import av

    if not frames:
        return []

    # Resample / layout-normalize first so stretch sees consistent fltp stereo.
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=out_rate)
    planar: list = []
    for fr in frames:
        for converted in resampler.resample(fr):
            arr = converted.to_ndarray()
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            planar.append(arr.astype(np.float32, copy=False))
    for converted in resampler.resample(None):
        arr = converted.to_ndarray()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        planar.append(arr.astype(np.float32, copy=False))
    if not planar:
        return []

    audio = np.concatenate(planar, axis=-1)
    if audio.shape[0] == 1:
        audio = np.vstack([audio[0], audio[0]])
    elif audio.shape[0] > 2:
        audio = audio[:2]

    n_in = int(audio.shape[1])
    if n_in < 1:
        return []
    n_out = max(1, int(round(n_in / float(rate))))
    x_old = np.linspace(0.0, 1.0, n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    stretched = np.vstack(
        [np.interp(x_new, x_old, audio[c].astype(np.float64)) for c in range(2)]
    ).astype(np.float32)

    out_frames = []
    # AAC encoders like multiples of 1024 samples.
    hop = 1024
    for i in range(0, n_out, hop):
        sl = stretched[:, i : i + hop]
        if sl.shape[1] == 0:
            continue
        if sl.shape[1] < hop:
            pad = np.zeros((2, hop - sl.shape[1]), dtype=np.float32)
            sl = np.concatenate([sl, pad], axis=1)
        frame = av.AudioFrame.from_ndarray(sl, format="fltp", layout="stereo")
        frame.sample_rate = out_rate
        out_frames.append(frame)
    return out_frames


def _packet_time_s(packet) -> float:
    pts = packet.pts
    if pts is None:
        return 0.0
    tb = packet.time_base or Fraction(1, 90000)
    return float(pts * tb)


def _retime_with_pyav(
    src: Path,
    out: Path,
    *,
    play_fps: float,
    arate: int,
    ts_offset: float,
) -> None:
    """PyAV retime → MPEG-TS with stretched AAC audio interleaved by PTS."""
    import av

    rate = float(play_fps) / NATIVE_FPS
    fps_num = int(round(play_fps * 1000))
    fps_den = 1000
    v_tb = Fraction(fps_den, fps_num)
    a_tb = Fraction(1, arate)

    inp = av.open(str(src))
    outp = av.open(str(out), mode="w", format="mpegts")
    try:
        v_in = next((s for s in inp.streams if s.type == "video"), None)
        a_in = next((s for s in inp.streams if s.type == "audio"), None)
        if v_in is None:
            raise RuntimeError(f"no video in {src}")

        # Demux both streams together — decoding video alone would skip audio
        # packets and leave a_raw empty.
        v_frames: list = []
        a_raw: list = []
        demux_streams = [v_in] + ([a_in] if a_in is not None else [])
        for packet in inp.demux(*demux_streams):
            if packet.dts is None:
                continue
            if packet.stream.type == "video":
                for frame in packet.decode():
                    v_frames.append(frame)
            elif packet.stream.type == "audio":
                for frame in packet.decode():
                    a_raw.append(frame)
        a_frames = (
            _stretch_audio_frames(a_raw, rate, out_rate=arate) if a_raw else []
        )

        v_out = outp.add_stream("libx264", rate=Fraction(fps_num, fps_den))
        v_out.width = v_in.codec_context.width
        v_out.height = v_in.codec_context.height
        v_out.pix_fmt = "yuv420p"
        v_out.bit_rate = 4_000_000
        v_out.time_base = v_tb
        gop = max(1, int(round(play_fps)))
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

        a_out = None
        if a_frames:
            a_out = outp.add_stream("aac", rate=arate)
            a_out.layout = "stereo"
            a_out.bit_rate = 128_000
            a_out.time_base = a_tb

        v_packets: list = []
        v_index = int(round(max(0.0, ts_offset) * play_fps))
        for frame in v_frames:
            frame.pts = v_index
            frame.time_base = v_tb
            for packet in v_out.encode(frame):
                packet.stream = v_out
                v_packets.append(packet)
            v_index += 1
        for packet in v_out.encode(None):
            packet.stream = v_out
            v_packets.append(packet)

        a_packets: list = []
        if a_out is not None:
            sample_pts = int(round(max(0.0, ts_offset) * arate))
            for frame in a_frames:
                frame.pts = sample_pts
                frame.time_base = a_tb
                sample_pts += frame.samples
                for packet in a_out.encode(frame):
                    packet.stream = a_out
                    a_packets.append(packet)
            for packet in a_out.encode(None):
                packet.stream = a_out
                a_packets.append(packet)

        # Interleave by presentation time; on equal PTS prefer audio so the
        # PMT's AAC stream gets an early ADTS PES (mpegts.js hangs otherwise).
        merged = sorted(
            [(_packet_time_s(p), 0, i, p) for i, p in enumerate(a_packets)]
            + [(_packet_time_s(p), 1, i, p) for i, p in enumerate(v_packets)],
            key=lambda item: (item[0], item[1], item[2]),
        )
        for _, _, _, packet in merged:
            outp.mux(packet)

        if a_frames:
            log.info(
                "PyAV retime %s → %s  play_fps=%.3f  video=%df  audio=%d frames",
                src.name,
                out.name,
                play_fps,
                len(v_frames),
                len(a_frames),
            )
        else:
            log.info(
                "PyAV retime %s → %s  play_fps=%.3f  video=%df  (no source audio)",
                src.name,
                out.name,
                play_fps,
                len(v_frames),
            )
    finally:
        inp.close()
        outp.close()

    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError("PyAV retime produced an empty file")


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
    """Rewrite one H3 MP4 to MPEG-TS at ``play_fps`` with stretched audio.

    Always uses PyAV. ``stretch`` / ``vbitrate`` / ``lufs`` are kept for call-site
    compatibility; pitch-preserving rubberband is not required on this path.
    """
    del stretch, vbitrate, lufs  # PyAV path — intentional no-ffmpeg policy
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(dest) if dest else Path(tempfile.mkstemp(suffix=".ts", prefix="h3live_")[1])
    rate = float(play_fps) / NATIVE_FPS
    if rate <= 0 or rate > 1.5:
        raise ValueError(f"play_fps {play_fps} implies bad stretch rate {rate}")

    _retime_with_pyav(src, out, play_fps=play_fps, arate=arate, ts_offset=ts_offset)
    return out


def feed_mpegts_to_sink(ts_path: Path, sink) -> None:
    """Copy a .ts file into an MPEG-TS muxer stdin (or any binary sink)."""
    with Path(ts_path).open("rb") as fh:
        shutil.copyfileobj(fh, sink, length=1 << 16)
