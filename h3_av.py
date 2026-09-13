#!/usr/bin/env python3
"""PyAV shim that impersonates the ffmpeg/ffprobe CLI calls h3.c actually makes.

h3.c posix_spawns ``ffmpeg`` / ``ffprobe`` with a small, fixed argv surface
(raw RGB24 / f32le pipes, visual-size probe). We implement that surface with
PyAV so a system ffmpeg install is not required. Point ``H3_AV``, ``H3_FFMPEG``,
and ``H3_FFPROBE`` at ``scripts/h3-av``.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
from fractions import Fraction
from pathlib import Path

from h3_paths import real_ffmpeg as _path_real_ffmpeg

SCALE_RE = re.compile(r"scale=(\d+):(\d+)")
CROP_RE = re.compile(r"crop=(\d+):(\d+)")
SIZE_RE = re.compile(r"^(\d+)x(\d+)$")
PIPE_RE = re.compile(r"^pipe:(\d+)$")
FPS_RE = re.compile(r"(?:^|,)fps=(\d+(?:\.\d+)?)")


def _die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(code)


def real_ffmpeg() -> str | None:
    return _path_real_ffmpeg()


def _inherit_extra_fds() -> None:
    for fd in range(3, 64):
        try:
            os.set_inheritable(fd, True)
        except OSError:
            continue


def _exec_system_ffmpeg(argv: list[str]) -> None:
    binary = real_ffmpeg()
    if not binary:
        return
    _inherit_extra_fds()
    print(f"h3-av: mux via {binary}", file=sys.stderr, flush=True)
    os.execv(binary, [binary, *argv[1:]])


def _require_av() -> None:
    try:
        import av  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        _die(f"h3-av requires PyAV and Pillow (pip install av Pillow): {exc}")


def _parse_size(text: str) -> tuple[int, int]:
    match = SIZE_RE.match(text.strip())
    if not match:
        _die(f"invalid size {text!r}")
    return int(match.group(1)), int(match.group(2))


def _inputs_and_output(argv: list[str]) -> tuple[list[dict[str, str]], str | None, dict[str, str]]:
    """Split ffmpeg argv the way ffmpeg does: options apply to the next ``-i``,
    then leftover options apply to the output file.

    h3.c muxes with two inputs (``pipe:0`` RGB, ``pipe:N`` PCM). A parser that
    treats everything after the first ``-i`` as output options misreads
    ``-f f32le -i pipe:N`` and tries to decode raw video as audio.
    """
    inputs: list[dict[str, str]] = []
    pending: dict[str, str] = {}
    output_opts: dict[str, str] = {}
    output: str | None = None
    i = 1
    while i < len(argv):
        token = argv[i]
        if token in ("-y",):
            i += 1
            continue
        if token.startswith("-") and token != "-":
            key = token.lstrip("-")
            value = ""
            if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
                value = argv[i + 1]
                i += 2
            else:
                i += 1
            if key == "i":
                inputs.append({**pending, "path": value})
                pending = {}
            else:
                pending[key] = value
            continue
        output = token
        output_opts = pending
        pending = {}
        i += 1
    if output is None and pending:
        output_opts = pending
    return inputs, output, output_opts


def cmd_ffprobe(argv: list[str]) -> None:
    _require_av()
    import av

    path = None
    for i, token in enumerate(argv):
        if token == "-i" and i + 1 < len(argv):
            path = argv[i + 1]
    if path is None:
        for token in reversed(argv[1:]):
            if not token.startswith("-"):
                path = token
                break
    if not path:
        _die("ffprobe: missing input path")
    try:
        container = av.open(path)
    except Exception as exc:
        _die(f"ffprobe: cannot open {path}: {exc}")
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            _die(f"ffprobe: no visual stream in {path}")
        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)
        if width < 1 or height < 1:
            _die(f"ffprobe: invalid visual size {width}x{height}")
        sys.stdout.write(f"{width}x{height}\n")
        sys.stdout.flush()
    finally:
        container.close()


def _cover_resize(image, width: int, height: int):
    from PIL import Image

    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w = max(1, int(math.ceil(src_w * scale)))
    new_h = max(1, int(math.ceil(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _stretch_resize(image, width: int, height: int):
    from PIL import Image

    return image.resize((width, height), Image.Resampling.LANCZOS)


def _parse_vf(vf: str) -> dict[str, int | bool | float]:
    spec: dict[str, int | bool | float] = {}
    fps = FPS_RE.search(vf or "")
    if fps:
        spec["fps"] = float(fps.group(1))
    scales = SCALE_RE.findall(vf or "")
    if scales:
        spec["width"] = int(scales[-1][0])
        spec["height"] = int(scales[-1][1])
    spec["cover"] = "force_original_aspect_ratio=increase" in (vf or "")
    crop = CROP_RE.search(vf or "")
    if crop:
        spec["crop_w"] = int(crop.group(1))
        spec["crop_h"] = int(crop.group(2))
        spec["cover"] = True
    return spec


def _image_to_rgb24(path: str, width: int, height: int, cover: bool) -> bytes:
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        out = _cover_resize(rgb, width, height) if cover else _stretch_resize(rgb, width, height)
        return out.tobytes()


def _frame_to_image(frame):
    from PIL import Image

    arr = frame.to_ndarray(format="rgb24")
    return Image.fromarray(arr)


def _decode_video_rgb_frames(
    path: str, width: int, height: int, *, fps: float, max_frames: int, cover: bool
) -> list[bytes]:
    import av

    container = av.open(path)
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            _die(f"no video stream in {path}")
        decoded: list[tuple[float, object]] = []
        for frame in container.decode(stream):
            t = float(frame.time) if frame.time is not None else (
                float(frame.pts * stream.time_base) if frame.pts is not None and stream.time_base else 0.0
            )
            decoded.append((t, _frame_to_image(frame)))
        if not decoded:
            _die(f"no frames in {path}")
        interval = 1.0 / float(fps)
        last_t = decoded[-1][0]
        if last_t <= 0:
            last_t = (len(decoded) - 1) * interval
        timed: list[object] = []
        idx = 0
        t = 0.0
        while len(timed) < max_frames and (t <= last_t + interval or not timed):
            while idx + 1 < len(decoded) and decoded[idx + 1][0] <= t + 1e-9:
                idx += 1
            timed.append(decoded[idx][1])
            t += interval
            if idx == len(decoded) - 1 and t > last_t + interval:
                break
        out: list[bytes] = []
        for image in timed[:max_frames]:
            resized = _cover_resize(image, width, height) if cover else _stretch_resize(image, width, height)
            out.append(resized.tobytes())
        return out
    finally:
        container.close()


def _decode_audio_f32le(path: str, *, rate: int, channels: int, max_seconds: float) -> bytes:
    """Decode + resample audio to interleaved ``rate``/``channels`` float32.

    Mirrors the ltx-ws decoder: flush the resampler tail with ``resample(None)``
    so the full waveform is produced, then pad/truncate to an exact whole number
    of frames. h3.c later checks ``received % frame_bytes == 0`` (2ch x 4B = 8),
    so the returned byte length must always be an exact multiple of the frame
    size — otherwise h3 fails with "could not decode a stereo soundtrack".
    """
    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    container = av.open(path)
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            _die(f"no audio stream in {path}")
        # Use planar "fltp" (mirrors ltx-ws). For packed "flt" stereo,
        # AVAudioFrame.to_ndarray() returns one interleaved plane (1, N)
        # where N = samples*channels; reshaping that treats each interleaved
        # element as a sample, corrupting the frame count. Planar gives a clean
        # (channels, samples) array, so we can upmix mono and interleave exactly.
        layout = "stereo" if channels == 2 else "mono"
        resampler = AudioResampler(format="fltp", layout=layout, rate=rate)
        parts: list = []
        # Emit exactly the frames whose timestamps are strictly before the
        # requested duration (real ffmpeg's `-t` semantics). h3.c sets
        # -t = seconds + 1/rate, which lands exactly on a sample boundary;
        # we must exclude that boundary frame so the byte count fits h3's
        # capacity read and the trailing EOF check passes.
        total = max_seconds * rate
        max_samples = int(total)
        if max_samples >= total:  # frame exactly at -t boundary -> excluded
            max_samples -= 1
        for frame in list(container.decode(stream)) + [None]:
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                parts.append(np.asarray(arr, dtype=np.float32))
        if not parts:
            _die(f"could not decode audio from {path}")
        # (channels, samples)
        data = np.concatenate(parts, axis=1).astype("<f4", copy=False)
        # Pad short inputs with silence up to the requested length; trim excess.
        if data.shape[1] < max_samples:
            pad = np.zeros((data.shape[0], max_samples - data.shape[1]), dtype="<f4")
            data = np.concatenate([data, pad], axis=1)
        else:
            data = data[:, :max_samples]
        # Upmix mono input to the requested number of channels.
        if data.shape[0] == 1 and channels > 1:
            data = np.repeat(data, channels, axis=0)
        elif data.shape[0] < channels:
            extra = np.zeros((channels - data.shape[0], data.shape[1]), dtype="<f4")
            data = np.concatenate([data, extra], axis=0)
        # Interleave to (samples, channels) contiguous for f32le.
        data = np.ascontiguousarray(data.T)
        return data.tobytes()
    finally:
        container.close()


def cmd_decode_raw(inputs: list[dict[str, str]], output_opts: dict[str, str]) -> None:
    _require_av()
    src = inputs[0]["path"]
    vf = output_opts.get("vf") or inputs[0].get("vf") or ""
    spec = _parse_vf(vf)
    pix = output_opts.get("pix_fmt") or output_opts.get("pixel_format") or "rgb24"
    fmt = output_opts.get("f") or ""
    frames_v = output_opts.get("frames:v") or output_opts.get("frames")
    if fmt == "f32le" or output_opts.get("vn") == "":
        rate = int(output_opts.get("ar") or 32000)
        channels = int(output_opts.get("ac") or 2)
        duration = float(output_opts.get("t") or 15.0)
        blob = _decode_audio_f32le(src, rate=rate, channels=channels, max_seconds=duration)
        sys.stdout.buffer.write(blob)
        sys.stdout.buffer.flush()
        return
    if pix != "rgb24":
        _die(f"unsupported pix_fmt {pix}")
    width = int(spec.get("width") or 0)
    height = int(spec.get("height") or 0)
    if width < 1 or height < 1:
        _die("decode requires scale=W:H in -vf")
    cover = bool(spec.get("cover"))
    max_frames = int(frames_v) if frames_v else 1
    fps = float(spec["fps"]) if spec.get("fps") else 0.0
    if max_frames <= 1 and not fps:
        blob = _image_to_rgb24(src, width, height, cover)
        sys.stdout.buffer.write(blob)
        sys.stdout.buffer.flush()
        return
    fps = fps or 24.0
    frames = _decode_video_rgb_frames(src, width, height, fps=fps, max_frames=max_frames, cover=cover)
    for blob in frames:
        sys.stdout.buffer.write(blob)
    sys.stdout.buffer.flush()


def _open_pipe(spec: str, mode: str):
    if spec in ("pipe:0", "pipe:", "-"):
        return sys.stdin.buffer if "r" in mode else sys.stdout.buffer
    if spec in ("pipe:1",):
        return sys.stdout.buffer
    match = PIPE_RE.match(spec)
    if match:
        fd = int(match.group(1))
        return os.fdopen(fd, mode, buffering=0)
    _die(f"unsupported pipe {spec}")
    raise AssertionError


def _drain_fd(handle, sink: bytearray) -> None:
    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            break
        sink.extend(chunk)


def _add_video_stream(container, fps_frac: Fraction, width: int, height: int):
    last: Exception | None = None
    for codec in ("libx264", "h264_videotoolbox", "mpeg4"):
        try:
            stream = container.add_stream(codec, rate=fps_frac, width=width, height=height)
            stream.pix_fmt = "yuv420p"
            if codec == "libx264":
                stream.options = {"crf": "18", "preset": "fast"}
            return stream
        except Exception as exc:
            last = exc
    _die(f"cannot open an H.264 encoder: {last}")
    raise AssertionError


def cmd_encode_mp4(inputs: list[dict[str, str]], output: str) -> None:
    _require_av()
    import av
    import numpy as np

    if not output or output.startswith("pipe"):
        _die("encode requires an output path")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    video_in = next((item for item in inputs if item.get("f") == "rawvideo" or "video_size" in item), inputs[0])
    audio_in = next((item for item in inputs[1:] if item.get("f") == "f32le"), None)
    size = video_in.get("video_size") or video_in.get("s")
    if not size:
        _die("rawvideo encode needs -video_size WxH")
    width, height = _parse_size(size)
    fps = int(float(video_in.get("framerate") or video_in.get("r") or 24))
    frame_bytes = width * height * 3
    video_handle = _open_pipe(video_in["path"], "rb")

    audio_blob = bytearray()
    audio_thread = None
    audio_rate = 32000
    audio_channels = 2
    if audio_in is not None:
        audio_rate = int(audio_in.get("ar") or 32000)
        audio_channels = int(audio_in.get("ac") or 2)
        audio_handle = _open_pipe(audio_in["path"], "rb")
        audio_thread = threading.Thread(target=_drain_fd, args=(audio_handle, audio_blob), daemon=True)
        audio_thread.start()

    fps_frac = Fraction(fps, 1)
    print(f"h3-av: mux {width}x{height} @{fps}fps → {output}", file=sys.stderr, flush=True)
    with av.open(output, "w") as container:
        vstream = _add_video_stream(container, fps_frac, width, height)
        vstream.time_base = Fraction(1, fps)
        astream = None
        layout = "stereo" if audio_channels == 2 else "mono"
        if audio_in is not None:
            astream = container.add_stream("aac", rate=audio_rate)
            astream.bit_rate = 192000
        pts = 0
        while True:
            buf = video_handle.read(frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                _die(f"short RGB frame ({len(buf)} of {frame_bytes} bytes)")
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = pts
            for packet in vstream.encode(frame):
                container.mux(packet)
            pts += 1
        for packet in vstream.encode(None):
            container.mux(packet)

        if audio_thread is not None and astream is not None:
            audio_thread.join()
            pcm = np.frombuffer(bytes(audio_blob), dtype="<f4")
            if pcm.size == 0:
                print("h3-av: audio pipe was empty; writing video-only MP4", file=sys.stderr, flush=True)
            else:
                if audio_channels > 1:
                    usable = (pcm.size // audio_channels) * audio_channels
                    pcm = pcm[:usable].reshape((-1, audio_channels))
                sample_count = int(pcm.shape[0]) if pcm.ndim == 2 else int(pcm.size)
                hop = 1024
                sample_pts = 0
                for start in range(0, sample_count, hop):
                    block = pcm[start : start + hop]
                    if block.ndim == 1:
                        planar = np.ascontiguousarray(block.reshape(1, -1))
                    else:
                        planar = np.ascontiguousarray(block.T)
                    aframe = av.AudioFrame.from_ndarray(planar, format="fltp", layout=layout)
                    aframe.sample_rate = audio_rate
                    aframe.pts = sample_pts
                    sample_pts += aframe.samples
                    for packet in astream.encode(aframe):
                        container.mux(packet)
                for packet in astream.encode(None):
                    container.mux(packet)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if not argv:
        argv = ["h3-av"]
    aliases = {"ffmpeg", "ffprobe", "h3-av", "h3-ffmpeg", "h3-ffprobe"}
    cleaned = [argv[0]]
    for token in argv[1:]:
        if not token.startswith("-") and Path(token).name.lower() in aliases:
            continue
        cleaned.append(token)
    argv = cleaned
    if "rawvideo" in argv and any(str(t).endswith(".mp4") for t in argv):
        _exec_system_ffmpeg(argv)
    name = Path(argv[0]).name.lower()
    if name in {"h3-ffprobe", "ffprobe"} or "-show_entries" in argv or "-select_streams" in argv:
        cmd_ffprobe(argv)
        return 0
    inputs, output, output_opts = _inputs_and_output(argv)
    fmt = output_opts.get("f") or ""
    has_raw_in = any(item.get("f") == "rawvideo" or "video_size" in item for item in inputs)
    writing_file = bool(output) and output not in {"pipe:1", "pipe:", "-"} and not str(output).startswith("pipe")
    if has_raw_in and writing_file:
        _exec_system_ffmpeg(argv)
        cmd_encode_mp4(inputs, output)
        return 0
    if output in {"pipe:1", "pipe:", "-"} or fmt in {"rawvideo", "f32le"}:
        if not inputs:
            _die("decode requires -i PATH")
        cmd_decode_raw(inputs, output_opts)
        return 0
    if has_raw_in:
        if not output:
            _die("encode requires an output path")
        _exec_system_ffmpeg(argv)
        cmd_encode_mp4(inputs, output)
        return 0
    binary = real_ffmpeg()
    if binary:
        print(f"h3-av: unrecognized argv, handing off to {binary}", file=sys.stderr, flush=True)
        os.execv(binary, [binary, *argv[1:]])
    _die("h3-av: unrecognized ffmpeg argv: " + " ".join(argv[1:40]))
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
