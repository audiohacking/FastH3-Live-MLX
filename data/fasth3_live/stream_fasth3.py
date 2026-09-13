# SPDX-License-Identifier: Apache-2.0
"""Continuous FastH3 stream: generate -> retime -> push, with no growing file pile.

Generation is slower than playback at 24 fps, so the stream is retimed on the
way out: the frames are authored at 24 fps and played at a lower rate, with the
audio slowed by the same factor, which preserves pitch. The break-even playback
rate is ``length / generation seconds``, so it must be derived from a
back-to-back measurement, never from a single best run -- planning against a
best case is exactly how this script ended up defaulting to an ``--fps`` it
could not sustain. Use ``profile_h3_nodes.py`` to re-measure after any change.

Retiming happens on the way to the muxer, never in the generation loop, so it
costs the generator nothing (a whole clip re-encodes in ~0.7 s).

Architecture:

    producer thread   POST /prompt -> poll /history -> clip mp4  -> queue
    feeder  thread    queue -> ffmpeg (retime, mpegts) -> stdin of ->
    output  ffmpeg    -re paced -> http/udp

``-re`` on the output paces the stream at real time, so a full pipe blocks the
feeder: that backpressure is what bounds memory, and a queue that drains to zero
is exactly the "generation fell behind" signal this script is meant to expose.
Each clip is deleted the moment it has been fed.

The producer keeps ``--pipeline`` prompts in ComfyUI's own queue rather than
submitting one and waiting for it, so the GPU is never idle across the gap
between a job finishing and the next one being accepted.

Open the printed URL in VLC.
"""

from __future__ import annotations

import argparse
import collections
import os
import queue
import random
import shutil
import subprocess
import sys
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
import urllib.error

from submit_h3 import get, post, to_api  # noqa: E402

import json

FFMPEG = r"E:\soft\ffmpeg\bin\ffmpeg.exe"
COMFY_OUT = r"F:\Comfy-Desktop\ComfyUI-Shared\output"
SCRATCH_PREFIX = "stream_tmp/clip"

# VHS_VideoCombine writes a metadata PNG of the first frame and, when audio is
# attached, a silent intermediate mp4 alongside the muxed one. Both are dead
# weight for a clip that is fed once and deleted, and the PNG is a full libpng
# encode per clip. These two flags are read out of the workflow "extra" block,
# which reaches the node through /prompt's extra_data.
VHS_EXTRA_DATA = {
    "extra_pnginfo": {"workflow": {"extra": {"VHS_MetadataImage": False,
                                             "VHS_KeepIntermediate": False}}}
}

# How ComfyUI writes the clip. Both stock nodes spend their time in Python
# rather than in the encoder -- ffmpeg alone does this payload in 0.21 s -- so
# the ranking is about how few Python passes each makes over the frames, not
# about the codec. h3fast is the custom node in
# custom_nodes/h3_fast_writer, which converts in chunks and muxes the audio in
# the same pass. The file is re-encoded by feed_clip a second later anyway, so
# the intermediate only has to be cheap and not lossy enough to compound.
# h3fast additionally hands the encode to a background thread and returns, so
# ComfyUI can start the next prompt instead of holding an idle GPU for the
# write. Its reported filename therefore names a file that does not exist yet;
# the node renames the clip into place when it is complete, which is why
# Producer.collect waits for the name rather than trusting /history alone.
WRITERS = {
    "h3fast": ("H3FastWriteVideo",
               {"crf": 16, "preset": "veryfast", "chunk_frames": 32,
                "async_write": True}),
    "vhs-nvenc": ("VHS_VideoCombine", {"format": "video/nvenc_h264-mp4",
                                       "pix_fmt": "yuv420p", "bitrate": 10,
                                       "megabit": True, "save_metadata": False}),
    "vhs-x264": ("VHS_VideoCombine", {"format": "video/h264-mp4",
                                      "pix_fmt": "yuv420p", "crf": 16,
                                      "save_metadata": False,
                                      "trim_to_audio": False}),
}

DEFAULT_PROMPTS = [
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a night market alley in the rain, red lanterns strung overhead and reflections breaking on the wet stone. The camera pushes in with small amplitude at slow speed as a vendor in a canvas apron turns skewers over a charcoal grill, sending sparks upward. [Shot 2] At 00:07.000, the shot cuts to a close-up of the grill, fat dripping onto the coals and flaring.

overall_soundscape: Rain patters on canvas awnings while charcoal hisses and crackles under dripping fat. Distant conversation and the clatter of tongs carry down the alley.

non_diegetic_music: N/A""",
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot frames an empty subway platform at night, fluorescent tubes flickering along the curved ceiling. The camera trucks right with small amplitude at slow speed past tiled columns as warm air pushes litter along the platform edge. [Shot 2] At 00:08.000, the camera cuts to a low shot of the tunnel mouth as headlights grow and a train sweeps through.

overall_soundscape: A low ventilation hum fills the station, broken by the rising roar of an approaching train and the squeal of brakes on steel.

non_diegetic_music: A slow synthesizer drone with a single repeating bass note, rising in volume as the train arrives.""",
    """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot frames rain running down a workshop window at dusk, tools hanging in silhouette behind the glass. The camera pulls out with small amplitude at slow speed to reveal a wooden bench covered in brass parts and open notebooks. [Shot 2] At 00:09.000, the shot transitions to an overhead close-up of hands sorting small gears into a shallow tin.

overall_soundscape: Steady rain runs down glass over a quiet room tone. Small brass parts click against tin, and a chair creaks as weight shifts.

non_diegetic_music: Sparse piano notes at a slow tempo with long gaps between phrases.""",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class PromptPool:
    """Random scene x random character.

    The scene files carry a ``{NAME}`` placeholder rather than a baked-in
    character, so 60 scenes and 503 characters give tens of thousands of
    distinct clips instead of 60 repeats. ``--curated-share`` of draws come
    from the small pool the community reports as the most reliable text-only
    recognitions; the rest come from the full verified-usable index.
    """

    def __init__(self, scenes_path, characters_path: str, curated_share: float,
                 explicit: bool = False):
        # One path or several. Several are drawn from as a single flat pool, so
        # a file's share of the draws is its share of the blocks -- adding 100
        # scenes to 321 makes them 24% of the stream, not 50%.
        self.scenes_paths = [scenes_path] if isinstance(scenes_path, str) else list(scenes_path)
        self.explicit = explicit
        self.curated_share = curated_share
        pools = json.load(open(characters_path, encoding="utf-8"))
        self.curated = pools["curated"]
        self.full = pools["full"]
        counts = self.counts()
        log("prompt pool: " + " + ".join(f"{n} from {os.path.basename(p)}"
                                         for p, n in counts.items())
            + f" = {sum(counts.values())} scenes x {len(self.full)} characters "
              f"({len(self.curated)} curated, {curated_share:.0%} of draws)")

    def counts(self) -> dict:
        return {p: len(self._blocks(p)) for p in self.scenes_paths}

    def _blocks(self, path: str) -> list[str]:
        """One file's scenes, re-read every call so it can be edited live.

        A missing file is fatal only when it was asked for by name. The shipped
        default may list a file this download does not have, and losing the
        stream over an optional extra would be the wrong trade.
        """
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            if self.explicit:
                raise SystemExit(f"--scenes {path}: {exc}")
            if path not in getattr(self, "_missing_warned", ()):
                self._missing_warned = set(getattr(self, "_missing_warned", ())) | {path}
                log(f"scenes: {os.path.basename(path)} not found, skipping it")
            return []
        return [b.strip() for b in text.split("\n---\n") if b.strip()]

    def scenes(self) -> list[str]:
        out: list[str] = []
        for p in self.scenes_paths:
            out.extend(self._blocks(p))
        if not out:
            raise SystemExit("no scenes to draw from; check --scenes")
        return out

    def _pick(self) -> str:
        pool = self.curated if (self.curated and random.random() < self.curated_share) else self.full
        return random.choice(pool)

    def draw(self) -> tuple[str, str, int]:
        """Fill every character placeholder a scene declares.

        A scene carries ``{NAME}`` and, for an ensemble, ``{NAME2}``..``{NAME5}``.
        However many it declares, that many are drawn under the same
        curated/full split and forced to be distinct, so a five-hander is five
        different faces rather than the same one five times.

        Also returns the scene's index in the file, so a clip on screen can be
        traced back to the block that produced it -- which is the only way to
        act on "that one looked wrong" without regenerating to find it.
        """
        scenes = self.scenes()
        idx = random.randrange(len(scenes))
        scene = scenes[idx]

        # replace longest-first: "{NAME}" is a prefix of nothing here, but doing
        # it in declared order would let a {NAME} substitution corrupt {NAME2}
        slots = ["{NAME}"] + [f"{{NAME{i}}}" for i in range(2, 10)]
        slots = [s for s in slots if s in scene]

        picked: list[str] = []
        for _ in slots:
            name = self._pick()
            for _ in range(20):
                if name not in picked:
                    break
                name = self._pick()
            picked.append(name)

        for slot, name in sorted(zip(slots, picked), key=lambda p: -len(p[0])):
            scene = scene.replace(slot, name)
        return scene, " + ".join(picked), idx


def set_video_vae(prompt: dict, name: str | None) -> dict:
    """Point the VIDEO VAELoader at ``name``, leaving the audio one alone.

    Both VAEs load through the same node type, so a class-level override would
    silently retarget the audio VAE too; pick the loader whose current filename
    identifies it.
    """
    if not name:
        return prompt
    hit = [k for k, v in prompt.items() if v["class_type"] == "VAELoader"
           and "video" in v["inputs"].get("vae_name", "").lower()]
    if len(hit) != 1:
        raise SystemExit(f"expected exactly one video VAELoader, found {len(hit)}")
    out = dict(prompt)
    out[hit[0]] = {**out[hit[0]],
                   "inputs": {**out[hit[0]]["inputs"], "vae_name": name}}
    return out


def set_vae_tiling(prompt: dict, tile_size: int) -> dict:
    """Insert H3VideoVaeTiling between the video VAELoader and VAEDecode.

    The H3 video VAE hardcodes 256 px spatial tiles, and decode time tracks the
    tile count rather than the pixels. Enlarging the tile is therefore tempting
    and wrong -- the decoder is a ViT that attends within a tile, so a bigger
    one is out of distribution and the picture degrades badly. This exists to
    pin 256 explicitly, because the node mutates the VAE in place and the
    setting outlives the prompt: ``tile_size <= 0`` inherits whatever was set
    last, which is not the same as restoring the default.
    """
    if tile_size <= 0:
        return prompt
    dec = next((k for k, v in prompt.items() if v["class_type"] == "VAEDecode"), None)
    if dec is None:
        raise SystemExit("no VAEDecode node to retile")
    out = dict(prompt)
    out["vae_tiling"] = {"class_type": "H3VideoVaeTiling",
                         "inputs": {"vae": out[dec]["inputs"]["vae"],
                                    "tiling": True, "tile_size": tile_size}}
    out[dec] = {**out[dec], "inputs": {**out[dec]["inputs"], "vae": ["vae_tiling", 0]}}
    return out


FFPLAY = os.path.join(os.path.dirname(FFMPEG), "ffplay.exe")

ACCEL_PACKS = {
    "SpectrumApplyMiniMaxH3": "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3",
    "MiniMaxH3MemoryEfficientSolAttentionPatch": "https://github.com/Saganaki22/ComfyUI-sol-attn",
    "H3SLAAttention": "https://github.com/ethanfel/ComfyUI-PlagueKind-Nodes-only-sparse",
}


class MissingNode(RuntimeError):
    """An --accel mode needs a node pack that is not installed.

    Raised rather than exiting so the caller can decide: an explicit --accel is
    a request and should fail loudly, while the default is a preference and
    should fall back, because a fresh download must run without installing
    anything (see ``default_workflow``).
    """

    def __init__(self, class_type: str):
        super().__init__(class_type)
        self.class_type = class_type
        self.url = ACCEL_PACKS.get(class_type, "")


def play_when_up(args, url: str, stop: threading.Event) -> subprocess.Popen | None:
    """Open a borderless ffplay on the stream, once it is actually serving.

    Borderless so the window carries no player chrome: "Go Live -> Application"
    in Discord then captures the picture and nothing else, and takes the audio
    with it. A virtual camera would be the other way to do this and is worse --
    ffmpeg has no output device on Windows, and a camera device carries no
    sound, so audio would need its own path and its own sync.

    The wait is a TCP connect rather than a sleep: ffmpeg does not bind the port
    until it has probed its input, which is the whole prefill plus the first
    clip, and a player started before that dies on connection refused.

    Lifted from ``stream_chat.py``, which had it first and still runs its own --
    it passes ``--no-player`` down so the two do not both open a window.
    """
    host, _, port = url.split("//", 1)[1].split("/")[0].partition(":")
    host, port = host or "127.0.0.1", int(port or 80)
    while not stop.is_set():
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            stop.wait(1.0)
    if stop.is_set():
        return None
    cmd = [FFPLAY, "-noborder",
           "-left", str(args.player_left), "-top", str(args.player_top),
           "-window_title", args.player_title, "-loglevel", "error"]
    if args.player_ontop:
        cmd.append("-alwaysontop")

    env = dict(os.environ)
    if args.player_software:
        # SDL puts the window on a D3D context by default. 448x448 at 18 fps is
        # nothing for a CPU blit, and on this box the card is the scarce
        # resource: the sampler goes bimodal (19 s vs 29 s) the moment the DiT
        # no longer fits with room to spare.
        env["SDL_RENDER_DRIVER"] = "software"
        env["SDL_FRAMEBUFFER_ACCELERATION"] = "0"

    log(f"player: {args.player_title} at {args.player_left},{args.player_top}"
        f"{' (software render)' if args.player_software else ''}")
    return subprocess.Popen(cmd + [url], env=env)


ACCEL_MODES = ("sage", "sol", "sla", "spectrum", "solspec", "slaspec")
SAGE_FPS, SPECTRUM_FPS = 17.5, 20.2       # measured ceilings, see set_accel()
DEFAULT_SCENES = ["prompts_scenes.txt", "prompts_scenes_2.txt"]


def _node_defaults(object_info: dict, class_type: str, **override) -> dict:
    """Build a node's inputs from its published schema, then apply overrides.

    Hand-written input dicts drop fields: a missing ``sink_conditioning`` cost a
    whole benchmark run to a bare ``HTTP 400 required_input_missing``, whose
    reason lives only in the response body. Generating from the schema cannot
    make that mistake.
    """
    if class_type not in object_info:
        raise MissingNode(class_type)
    out = {}
    for k, v in object_info[class_type]["input"]["required"].items():
        if len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]:
            out[k] = v[1]["default"]
        elif isinstance(v[0], list) and v[0]:
            out[k] = v[0][0]
    out.update(override)
    return out


def set_accel(prompt: dict, object_info: dict, mode: str) -> dict:
    """Swap the attention backend and/or stack a step-skipping patch on it.

    Measured on t2va 448x448 x362 at 4 steps, A-B-B-A with three runs per arm
    (``bench_attn.py``), sampler seconds:

        sage 12.65 | Sol 11.73 | SLA 10.36 | Spectrum 10.36 | Sol+Spectrum 9.64

    Sol and Spectrum compose because they remove different work -- Sol makes the
    same maths faster, Spectrum evaluates fewer steps. SLA and Spectrum do not:
    both cut redundancy, and stacked they came out at 10.54, worse than either
    alone. EasyCache did nothing at all, and structurally cannot: it skips steps
    whose neighbours agree, and four steps have no such redundancy.

    ★ The speed order is NOT the quality order. Watched back at a matched
    ``--fps``, the ranking inverts almost exactly:

        spectrum  as good as sage -- usable
        sla       faces and hands still readable
        sol       faces and hands smear into a blob -- the worst of the four
        solspec   worse still

    So ``spectrum`` is the one to ship: sampler 10.36 s (-18% on sage) for no
    visible cost, while the two fastest arms are unusable. Nothing in the timing
    predicted that, and an earlier version of this docstring called Sol
    "lossless-ish" purely because it was fast. Never infer quality from seconds;
    watch a clip.
    """
    if mode == "sage":
        return prompt
    if mode not in ACCEL_MODES:
        raise SystemExit(f"--accel must be one of {', '.join(ACCEL_MODES)}")
    out = {k: {**v, "inputs": dict(v.get("inputs", {}))} for k, v in prompt.items()}

    hit = [k for k, v in out.items() if v["class_type"] == "MiniMaxH3BlockAttentionSplit"]
    if len(hit) > 1:
        raise SystemExit(f"expected at most one MiniMaxH3BlockAttentionSplit, found {len(hit)}")
    attn = hit[0] if hit else None

    if mode in ("sol", "solspec", "sla", "slaspec"):
        # Replacing the attention backend needs a backend node to replace. The
        # reference graph can have had it spliced out (--r2v-patches plain), and
        # inventing one there would silently change a deliberately plain run.
        if attn is None:
            raise SystemExit(
                f"--accel {mode} replaces the attention backend, but this graph has no "
                "MiniMaxH3BlockAttentionSplit to replace (--r2v-patches plain removes it). "
                "Use --accel spectrum, which stacks instead of replacing.")
        cls, over = (("MiniMaxH3MemoryEfficientSolAttentionPatch",
                      dict(tau=1.3, min_tokens=4096))
                     if mode in ("sol", "solspec") else ("H3SLAAttention", {}))
        out[attn] = {"class_type": cls,
                     "inputs": _node_defaults(object_info, cls,
                                              model=list(out[attn]["inputs"]["model"]), **over)}

    if mode in ("spectrum", "solspec", "slaspec"):
        # Anchored on whatever currently drives the guider rather than on the
        # attention node, so this also works on a graph whose patches were
        # spliced out. Spectrum only needs to sit somewhere on the MODEL edge.
        guider = next((k for k, v in out.items() if v["class_type"] == "BasicGuider"), None)
        if guider is None:
            raise SystemExit("no BasicGuider to hang Spectrum off")
        upstream = out[guider]["inputs"]["model"]
        for consumer, node in list(out.items()):
            if consumer == "accel_spectrum":
                continue
            for field, val in node.get("inputs", {}).items():
                if isinstance(val, list) and val == list(upstream) and field == "model":
                    out[consumer]["inputs"][field] = ["accel_spectrum", 0]
        out["accel_spectrum"] = {"class_type": "SpectrumApplyMiniMaxH3",
                                 "inputs": _node_defaults(object_info,
                                                          "SpectrumApplyMiniMaxH3",
                                                          model=list(upstream))}
    return out


def swap_writer(prompt: dict, writer: str, prefix: str) -> dict:
    """Rewrite CreateVideo -> SaveVideo into a single faster writer node.

    Done on the API prompt rather than in the workflow file so the workflow
    stays the one that is shipped and opened in the UI; the substitution is a
    property of how the stream writes clips, not of the graph.
    """
    if writer not in WRITERS:
        return prompt
    node_type, params = WRITERS[writer]
    save = next((k for k, v in prompt.items() if v["class_type"] == "SaveVideo"), None)
    if save is None:
        raise SystemExit("no SaveVideo node to replace; --writer needs the stock workflow")
    create_id = prompt[save]["inputs"]["video"][0]
    create = prompt[create_id]
    if create["class_type"] != "CreateVideo":
        raise SystemExit(f"SaveVideo is fed by {create['class_type']}, not CreateVideo")

    fps = float(create["inputs"].get("fps", 24.0))
    if node_type == "H3FastWriteVideo":
        inputs = {"images": create["inputs"]["images"], "fps": fps,
                  "filename_prefix": prefix}
    else:
        inputs = {"images": create["inputs"]["images"], "frame_rate": fps,
                  "loop_count": 0, "filename_prefix": prefix,
                  "pingpong": False, "save_output": True}
    if "audio" in create["inputs"]:
        inputs["audio"] = create["inputs"]["audio"]
    inputs.update(params)

    out = {k: v for k, v in prompt.items() if k not in (save, create_id)}
    out["clip_writer"] = {"class_type": node_type, "inputs": inputs}
    return out


class Producer(threading.Thread):
    """Generate clips forever and hand their paths to the feeder."""

    def __init__(self, args, out_q: queue.Queue, stop: threading.Event, pool: PromptPool):
        super().__init__(daemon=True)
        self.args, self.q, self.stop, self.pool = args, out_q, stop, pool
        self.workflow = json.load(open(args.workflow, encoding="utf-8"))
        self.object_info = get("/object_info")
        self.check_models()
        self.n = 0
        self.submitted = 0
        self.last_done = time.time()

    def check_models(self) -> None:
        """Fail at startup, not mid-prompt, on a model ComfyUI cannot see.

        The video VAE in particular defaults to a file this repository tells you
        to build rather than one you download, so a fresh checkout will not have
        it. ComfyUI's own error for that arrives as a rejected prompt, several
        seconds in and wrapped in validation JSON.
        """
        combos = {
            "--video-vae": (self.args.video_vae, "VAELoader", "vae_name"),
            "--dit": (self.args.dit, "UNETLoader", "unet_name"),
            "--clip": (self.args.clip, "CLIPLoader", "clip_name"),
        }
        for flag, (want, node, field) in combos.items():
            if not want:
                continue
            spec = self.object_info.get(node, {}).get("input", {}).get("required", {})
            have = spec.get(field, [None])[0]
            if isinstance(have, list) and want not in have:
                lines = [f"{flag}: ComfyUI does not list {want!r} for {node}.",
                         f"  available: {', '.join(map(str, have[:8]))}"
                         + (" ..." if len(have) > 8 else "")]
                if flag == "--video-vae":
                    lines.append("  the default video VAE is built rather than downloaded"
                                 " -- see the README, or pass one you already have.")
                raise SystemExit("\n".join(lines))

    def apply_accel(self, prompt: dict) -> dict:
        """Apply --accel, falling back to sage when the default's pack is absent.

        A fresh download of this repository must run without installing
        anything, and the shipped default (``spectrum``) needs a pack that is
        not redistributed here. So an unmet default degrades to sage with a
        line saying what it cost; an unmet *explicit* ``--accel`` is a direct
        request and stops instead.

        Warned once rather than per clip: this runs for every prompt.
        """
        try:
            return set_accel(prompt, self.object_info, self.args.accel)
        except MissingNode as miss:
            if self.args.accel_explicit:
                raise SystemExit(
                    f"--accel {self.args.accel} needs {miss.class_type}, which is not "
                    f"loaded.\n  install it and restart ComfyUI:  {miss.url}")
            if not getattr(self, "_accel_warned", False):
                self._accel_warned = True
                log(f"accel: {miss.class_type} not installed -- falling back to sage "
                    f"({SAGE_FPS} fps instead of {SPECTRUM_FPS}). To get it back:")
                log(f"  cd ComfyUI/custom_nodes && git clone {miss.url}")
            return set_accel(prompt, self.object_info, "sage")

    def build(self, prompt_text: str, seed: int) -> dict:
        ov = {
            "MiniMaxH3ImageToVideo": {"prompt": prompt_text, "width": self.args.width,
                                      "height": self.args.height, "length": self.args.length},
            "RandomNoise": {"noise_seed": seed},
            "SaveVideo": {"filename_prefix": SCRATCH_PREFIX},
        }
        if self.args.dit:
            ov["UNETLoader"] = {"unet_name": self.args.dit}
        if self.args.clip:
            ov["CLIPLoader"] = {"clip_name": self.args.clip}
        prompt = to_api(self.workflow, self.object_info, ov)
        prompt = set_video_vae(prompt, self.args.video_vae)
        prompt = set_vae_tiling(prompt, self.args.vae_tile)
        prompt = self.apply_accel(prompt)
        return swap_writer(prompt, self.args.writer, SCRATCH_PREFIX)

    def next_seed(self) -> int:
        """One seed per clip, walking up from --seed.

        Off the SUBMIT counter and not the completion counter: with several
        prompts in flight the two differ, and a reused seed would silently ship
        a duplicate clip. A walk rather than a random draw because it never
        collides, and the scene and character are drawn randomly anyway -- the
        clip a seed lands on is different every run regardless.
        """
        return (self.args.seed + self.submitted) & 0x7FFFFFFF

    def submit(self) -> tuple[str, str, int, int] | None:
        text, who, idx = self.pool.draw()
        seed = self.next_seed()
        try:
            pid = post("/prompt", {"prompt": self.build(text, seed),
                                   "client_id": "fasth3-stream",
                                   "extra_data": VHS_EXTRA_DATA})["prompt_id"]
        except urllib.error.HTTPError as e:
            log(f"submit rejected: {e.read().decode('utf-8', 'replace')[:300]}")
            return None
        self.submitted += 1
        return pid, who, idx, seed

    def collect(self, pid: str) -> str | None:
        """Block until ``pid`` leaves the queue, then return its clip path."""
        path = None
        while not self.stop.is_set():
            hist = get(f"/history/{pid}")
            if pid in hist:
                for node_out in hist[pid].get("outputs", {}).values():
                    for items in node_out.values():
                        if not isinstance(items, list):
                            continue
                        for it in items:
                            if isinstance(it, dict) and it.get("filename", "").endswith(".mp4"):
                                path = os.path.join(COMFY_OUT, it.get("subfolder", ""),
                                                    it["filename"])
                return self.await_file(path) if path else None
            time.sleep(0.25)
        return None

    def await_file(self, path: str) -> str | None:
        """Wait for an asynchronously written clip to be renamed into place.

        With ``--writer h3fast`` the node returns as soon as it has handed the
        frames to its encoder thread, so /history reports the clip a moment
        before it exists. The node writes to ``<name>.part`` and renames, so
        the name appearing is the completion signal -- and because the rename
        is atomic, seeing it means the whole clip is there.
        """
        deadline = time.time() + self.args.write_timeout
        while not self.stop.is_set():
            if os.path.exists(path):
                return path
            if time.time() > deadline:
                log(f"clip never appeared within {self.args.write_timeout:g}s: "
                    f"{os.path.basename(path)}")
                return None
            time.sleep(0.02)
        return None

    def run(self) -> None:
        inflight: list[tuple[str, str, int, int]] = []
        while not self.stop.is_set():
            # Top up ComfyUI's queue first. Blocking on q.put below is the
            # backpressure that bounds this: it stalls the top-up, and the
            # overshoot is at most --pipeline clips.
            while len(inflight) < self.args.pipeline and not self.stop.is_set():
                job = self.submit()
                if job is None:
                    time.sleep(3)
                    break
                inflight.append(job)
            if not inflight:
                continue

            pid, who, idx, seed = inflight.pop(0)
            path = self.collect(pid)
            if path and os.path.exists(path):
                self.n += 1
                now = time.time()
                interval, self.last_done = now - self.last_done, now
                # names last and untruncated: the cast is drawn from an unseeded
                # random, so scene + seed alone do not reproduce a clip -- the
                # names are the rest of the recipe, and a five-hander needs all
                # of them
                log(f"clip {self.n:04d}  scene {idx:03d}  seed {seed:<10d} "
                    f"{interval:5.1f}s  queue={self.q.qsize() + 1}  {who}")
                # the interval between completions, not the job's own wall time:
                # with a pipeline the latter includes waiting behind its
                # predecessor and would overstate the cost per clip roughly
                # --pipeline-fold
                self.q.put((path, interval, idx, seed))
            elif not self.stop.is_set():
                log("generation produced no file; retrying")
                time.sleep(2)


class Broadcast:
    """Fan one paced MPEG-TS stream out to any number of HTTP clients.

    ffmpeg's own ``-listen 1`` HTTP output serves exactly one client and then
    DIES: the moment that client disconnects the muxer aborts (-10053) and the
    port is gone for good, so a single premature connection attempt kills the
    stream permanently. Pacing still belongs to ffmpeg (``-re``), but the socket
    does not -- this owns it so viewers can come and go.
    """

    def __init__(self, host: str, port: int):
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        broadcast = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp2t")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q: queue.Queue = queue.Queue(maxsize=256)
                with broadcast.lock:
                    broadcast.clients.add(q)
                log(f"viewer connected ({len(broadcast.clients)} now)")
                try:
                    while True:
                        self.wfile.write(q.get())
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with broadcast.lock:
                        broadcast.clients.discard(q)
                    log(f"viewer left ({len(broadcast.clients)} now)")

            def log_message(self, *a):  # silence per-request logging
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def push(self, chunk: bytes) -> None:
        with self.lock:
            targets = list(self.clients)
        for q in targets:
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass  # slow viewer: drop rather than stall the whole stream


class Output:
    """Paced MPEG-TS sink; ``stdin`` is what the feeder writes clips into."""

    def __init__(self, args, broadcast: "Broadcast | None"):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-re",
               "-fflags", "+genpts", "-f", "mpegts", "-i", "pipe:0", "-c", "copy",
               "-mpegts_flags", "+resend_headers", "-pat_period", "0.1",
               "-f", "mpegts"]
        self.broadcast = broadcast
        if broadcast is not None:
            self.proc = subprocess.Popen(cmd + ["pipe:1"], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE)
            threading.Thread(target=self._pump, daemon=True).start()
        else:
            self.proc = subprocess.Popen(cmd + [args.url], stdin=subprocess.PIPE)
        self.stdin = self.proc.stdin

    def _pump(self) -> None:
        while True:
            chunk = self.proc.stdout.read(1 << 15)
            if not chunk:
                break
            self.broadcast.push(chunk)

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def audio_stretch(rate: float, mode: str) -> str:
    """Filter chain that slows audio to ``rate`` without wrecking it.

    ``atempo`` is documented as valid from 0.5, so the 12 fps case runs it at
    exactly its limit -- its WSOLA overlap-add then leaves a periodic metallic
    ring on broadband material like rain and wind. rubberband is a real phase
    vocoder, has no such floor, and is the default for that reason.
    """
    if mode == "atempo":
        return f"atempo={rate:.8f}"
    if mode == "atempo2":                       # two gentler stages, product == rate
        half = rate ** 0.5
        return f"atempo={half:.8f},atempo={half:.8f}"
    if mode == "tape":                          # pitch falls with speed, as on tape
        return f"rubberband=tempo={rate:.8f}:pitch={rate:.8f}"
    return f"rubberband=tempo={rate:.8f}:transients=smooth"


def default_workflow() -> str:
    """The authored workflow if this is the machine it was authored on, else the
    copy shipped next to this script.

    The absolute path is kept first because editing the graph in ComfyUI writes
    back to that file, and a stream started afterwards should pick the edit up.
    On any other machine that path does not exist and the bundled copy is the
    right answer, so a fresh download runs without editing anything.
    """
    authored = (r"F:\Comfy-Desktop\ComfyUI-Installs\MiniMax H3\ComfyUI"
                r"\user\default\workflows\FastH3_4step_T2VA.json")
    if os.path.exists(authored):
        return authored
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "FastH3_4step_T2VA.json")


def drop_clip(path: str) -> None:
    """Delete a fed clip and any siblings the writer left next to it.

    VHS names the muxed file ``clip_00001-audio.mp4`` and the silent pass it
    was built from ``clip_00001.mp4``. VHS_KeepIntermediate=False should have
    removed the latter already; sweeping it here as well means a stale flag or
    an older VHS cannot quietly grow a file pile over a long run.
    """
    for p in ({path, path.replace("-audio.mp4", ".mp4")}
              if path.endswith("-audio.mp4") else {path}):
        try:
            os.remove(p)
        except OSError:
            pass


def feed_clip(path: str, offset: float, args, sink) -> None:
    """Retime one clip and write it to the output muxer as MPEG-TS."""
    rate = args.fps / 24.0
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-filter_complex",
           f"[0:v]setpts=PTS/{rate:.6f}[v];"
           # loudnorm resamples internally and emits 96 kHz, which is out of spec for
           # AAC-LC in MPEG-TS -- VLC then fails to open the audio decoder and plays
           # silence. Pin the rate back down right after it.
           f"[0:a]{audio_stretch(rate, args.stretch)}"
           + (f",loudnorm=I={args.lufs}:TP=-1.5:LRA=11" if args.lufs else "")
           + f",aresample={args.arate}[a]",
           "-map", "[v]", "-map", "[a]", "-r", str(args.fps),
           "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
           "-g", "24", "-keyint_min", "24", "-sc_threshold", "0",
           "-pix_fmt", "yuv420p", "-b:v", args.vbitrate,
           "-c:a", "aac", "-b:a", "128k", "-ar", str(args.arate), "-ac", "2",
           "-muxdelay", "0", "-output_ts_offset", f"{offset:.3f}",
           # identical PIDs on every clip, so a boundary is not a PMT change,
           # and frequent tables so a viewer joining mid-clip finds the audio ES
           "-streamid", "0:256", "-streamid", "1:257",
           "-mpegts_flags", "+resend_headers", "-pat_period", "0.1", "-sdt_period", "0.5",
           "-f", "mpegts", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        shutil.copyfileobj(proc.stdout, sink, length=1 << 16)
    finally:
        proc.stdout.close()
        proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workflow", default=default_workflow(),
                   help="UI workflow to convert and submit. Defaults to the copy in "
                        "ComfyUI's workflow folder if it is there, else the one shipped "
                        "beside this script.")
    p.add_argument("--scenes", action="append", metavar="FILE",
                   help="scene file; blocks separated by a line containing ---, each with "
                        "{NAME}. Repeat the flag to pool several. Passing it at all "
                        "REPLACES the default set, so --scenes prompts_scenes_2.txt draws "
                        "from that file alone. Default: "
                        + " + ".join(DEFAULT_SCENES))
    p.add_argument("--characters", default="h3_characters.json",
                   help="written by build_h3_characters.py")
    p.add_argument("--curated-share", type=float, default=0.30,
                   help="fraction of draws taken from the high-recognition pool")
    p.add_argument("--url", default="http://127.0.0.1:9000",
                   help="http://HOST:PORT (VLC opens the same URL) or udp://HOST:PORT")
    p.add_argument("--fps", type=float, default=18.0,  # measured sustainable at 448x448
                   help="playback rate; frames are authored at 24, so this slows motion by fps/24")
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--length", type=int, default=362)
    p.add_argument("--seed", type=int, default=90000,
                   help="first noise seed; each clip takes the next one up. Logged per "
                        "clip, so a clip worth keeping can be reproduced from its log "
                        "line together with the scene index printed beside it.")
    p.add_argument("--prefill", type=int, default=3, help="clips to bank before opening the stream")
    p.add_argument("--queue-max", type=int, default=8)
    p.add_argument("--vbitrate", default="2M")
    p.add_argument("--stretch", default="rubberband",
                   choices=["rubberband", "atempo", "atempo2", "tape"],
                   help="how the audio is slowed to match --fps; atempo bottoms out at 0.5 "
                        "(= 12 fps) and rings on broadband sound, rubberband does not")
    p.add_argument("--arate", type=int, default=48000,
                   help="output audio sample rate; must be a normal AAC rate")
    p.add_argument("--lufs", type=float, default=0.0,
                   help="EBU R128 loudness target, e.g. -16. Off by default: H3's own level "
                        "is fine on a correctly configured output, and loudnorm adds a "
                        "resampling stage and dynamic gain the stream does not need.")
    p.add_argument("--dit", default="minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors")
#    p.add_argument("--dit", default="minimax_h3_fl2va_fasth3_dense_pruned_asym_w4a8_int8.safetensors")
#    p.add_argument("--clip", default="qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors")
    # Deliberately the 25.9 GB int8 encoder and NOT the 15.0 GB nvfp4 one. The
    # nvfp4 encode really is faster (0.80 s vs 1.95 s -- the cost is almost all
    # weight movement, not compute), but it is small enough to STAY RESIDENT in
    # VRAM afterwards, leaving the 20 GB DiT without a card it can fully hold.
    # Six clean runs then split bimodally: sampler 12.0/12.5 s on the good ones
    # and 19.3/35.1/37.7 s on the bad, with no paging involved. The int8 encoder
    # cannot coexist with the DiT at all, so it is evicted wholesale and the
    # sampler stays flat at 11.3-11.6 s. Bigger is faster here; measure before
    # "upgrading" this.
    p.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                   help="text encoder. nvfp4_awq since v1.2.0: it encodes in 0.70 s "
                        "against int8_convrot's 1.67 s and the sampler does not slow down "
                        "to pay for it (10.05 vs 10.40), so the clip is 8%% faster "
                        "end to end. Prompt adherence was unchanged on inspection.")
    # x264 over nvenc: both land at ~1.6 s, because what is left after moving
    # the encode out of Python is the per-frame tobytes() and the pipe, not the
    # encoder. Given a tie, libx264 avoids standing up a second CUDA context on
    # a card with ~3 GB free, and is kinder to the re-encode that follows.
    p.add_argument("--writer", default="h3fast",
                   choices=["h3fast", "vhs-x264", "vhs-nvenc", "savevideo"],
                   help="how ComfyUI writes the clip. savevideo is the stock node and "
                        "the slowest by far; the vhs- writers pipe raw frames to ffmpeg")
    p.add_argument("--pipeline", type=int, default=2,
                   help="prompts kept in ComfyUI's queue, so the GPU does not idle "
                        "between jobs; 1 restores the old submit-and-wait behaviour")
    # w4a8 rather than the int8 build, quantized straight from the official fp16
    # weights by requantize_comfy_h3.py. It is not faster -- int8 is already the
    # fastest matmul path comfy_kitchen offers, and 4-bit weights are expanded
    # back to int8 for it -- but it stages 1,657 MB against 2,677 MB at PSNR
    # ~38 dB, and this box has no host RAM to spare.
    p.add_argument("--video-vae", default="minimax_h3_video_vae_w4a8_from_fp16.safetensors",
                   help="video VAE to load; the audio VAE is left alone")
    # Pinned at the stock 256, and pinned EXPLICITLY. Larger tiles are much
    # faster (384 -> 3.58 s, 640 -> 3.07 s against 6.37 s) but the decoder is a
    # ViT whose attention spans one tile, so enlarging it is out of
    # distribution: 384 visibly softens hands and faces, 640 smears the image
    # into strokes. The node also mutates the VAE in place, so passing 0 here
    # would inherit whatever a previous run left set rather than restore 256.
    p.add_argument("--no-player", action="store_true",
                   help="do not open ffplay; the stream is served either way")
    p.add_argument("--player-left", type=int, default=100)
    p.add_argument("--player-top", type=int, default=100)
    p.add_argument("--player-title", default="FastH3",
                   help="window title, which is what Discord's Application picker lists")
    p.add_argument("--player-ontop", action="store_true")
    p.add_argument("--player-software", action="store_true",
                   help="render the player on the CPU, leaving the card to the sampler")
    p.add_argument("--accel", default="spectrum", choices=list(ACCEL_MODES),
                   help="attention/step-skip stack. spectrum (default) = sage + Spectrum: "
                        "sampler 10.36 s vs sage 12.65, and watched back it is as good as "
                        "sage. sol and solspec are FASTER but their faces and hands smear "
                        "-- do not ship them. sage = the old behaviour.")
    p.add_argument("--vae-tile", type=int, default=256,
                   help="H3 video VAE spatial tile edge in pixels. 256 is the only value "
                        "that decodes correctly; larger is faster and visibly worse. "
                        "0 leaves whatever is currently set on the loaded VAE.")
    p.add_argument("--window", type=int, default=8,
                   help="clips averaged for the sustainable-fps readout; a trailing "
                        "window rather than the all-time mean, which a slow cold start "
                        "would poison for hundreds of clips")
    p.add_argument("--write-timeout", type=float, default=60.0,
                   help="how long to wait for an asynchronously written clip to appear "
                        "before giving up on it and moving to the next")
    p.add_argument("--keep-clips", action="store_true", help="do not delete clips after streaming")
    p.add_argument("--max-clips", type=int, default=0, help="stop after N clips (0 = forever); for smoke tests")
    args = p.parse_args()
    # append with a default cannot express "replace the default", so the
    # default is applied here and the flag being present means exactly it.
    args.scenes_explicit = args.scenes is not None
    if args.scenes is None:
        args.scenes = list(DEFAULT_SCENES)
    # Whether --accel was asked for or merely defaulted decides what an absent
    # node pack means: a hard stop or a fallback. argparse does not record it.
    args.accel_explicit = any(a == "--accel" or a.startswith("--accel=") for a in sys.argv[1:])

    # sweep anything a previous run left behind, so the scratch dir never grows
    scratch = os.path.join(COMFY_OUT, os.path.dirname(SCRATCH_PREFIX))
    if os.path.isdir(scratch):
        # .part too: the async writer's encoder thread is a daemon, so a clip
        # in flight when ComfyUI or this script is killed leaves one behind
        # with no chance to run its own cleanup.
        stale = [f for f in os.listdir(scratch) if f.endswith((".mp4", ".png", ".part"))]
        for f in stale:
            try:
                os.remove(os.path.join(scratch, f))
            except OSError:
                pass
        if stale:
            log(f"swept {len(stale)} stale clip(s) from {scratch}")

    clip_play = args.length / args.fps
    log(f"{args.width}x{args.height} x {args.length} frames  "
        f"writer={args.writer}  pipeline={args.pipeline}")
    log(f"playback {clip_play:.2f}s per clip at {args.fps:g} fps "
        f"(motion at {args.fps / 24 * 100:.0f}% speed); to keep up, generation "
        f"must average under {clip_play:.2f}s")

    q: queue.Queue = queue.Queue(maxsize=args.queue_max)
    stop = threading.Event()
    pool = PromptPool(args.scenes, args.characters, args.curated_share,
                      explicit=args.scenes_explicit)
    producer = Producer(args, q, stop, pool)
    producer.start()

    broadcast = None
    if args.url.startswith("http"):
        hostport = args.url.split("//", 1)[1].split("/")[0]
        host, _, port = hostport.partition(":")
        broadcast = Broadcast(host or "127.0.0.1", int(port or 8080))
        log(f"STREAM OPEN -- open this in VLC now:  {args.url}")
        log("  you may connect during prefill; playback starts when clips arrive")
    else:
        log(f"stream target:  {args.url.replace('udp://', 'udp://@')}  (open after prefill)")

    player: list = []
    if args.no_player:
        pass
    elif not args.url.startswith("http"):
        log(f"player: not opening one for {args.url} -- ffplay is wired up for the "
            "http stream; open the udp URL yourself")
    elif not os.path.exists(FFPLAY):
        log(f"player: {FFPLAY} not found; open the stream yourself")
    else:
        threading.Thread(target=lambda: player.append(play_when_up(args, args.url, stop)),
                         daemon=True).start()

    log(f"prefilling {args.prefill} clips")
    seen = -1
    while q.qsize() < args.prefill and not stop.is_set():
        if q.qsize() != seen:
            seen = q.qsize()
            log(f"  prefill {seen}/{args.prefill}")
        time.sleep(0.5)

    out = Output(args, broadcast)
    # ffmpeg does not bind the HTTP listener until it has probed its input, so
    # the port is dead for the whole prefill plus the first clip's first bytes.
    # Announce the URL only once a TCP connect actually succeeds.
    if broadcast is None:
        log(f"STREAM OPEN -- in VLC use:  {args.url.replace('udp://', 'udp://@')}")

    offset = 0.0
    played = 0
    # A trailing window, not the all-time mean: the first clip after a cold
    # ComfyUI is always several seconds slow, and an all-time mean carries that
    # for hundreds of clips, which is exactly long enough to mislead someone
    # deciding whether the current --fps is sustainable.
    recent: collections.deque[float] = collections.deque(maxlen=args.window)
    t_start = time.time()
    try:
        while True:
            path, gen_s, scene_idx, seed = q.get()
            depth = q.qsize()
            feed_clip(path, offset, args, out.stdin)
            out.stdin.flush()
            offset += clip_play
            played += 1
            # The first clip's "interval" is measured from before ComfyUI had
            # loaded anything, so on a cold server it is a model load plus a
            # generation -- 40 s or more against a steady state near 20 s. Left
            # in an 8-wide window it drags the mean for the window's whole
            # length, which is long enough to report BEHIND for the entire
            # opening of every run. It is a load time, not a throughput sample.
            if played > 1:
                recent.append(gen_s)
            if not args.keep_clips:
                drop_clip(path)

            head = (f"streamed {played:04d}  scene {scene_idx:03d} seed {seed:<10d} "
                    f"buf={depth}  gen {gen_s:5.1f}s ")
            if not recent:
                log(head + "(cold start, excluded from the average)")
            else:
                # The whole question this script exists to answer: is generation
                # keeping up, and if not, what --fps would it keep up with?
                mean = sum(recent) / len(recent)
                margin = clip_play - mean        # seconds of slack won per clip
                can_do = args.length / mean      # the fps this rate sustains
                verdict = "OK" if margin >= 0 else "BEHIND"
                if margin < 0 and depth:
                    # the buffer holds depth*clip_play seconds of playback, and
                    # each clip cycle eats `-margin` of it
                    verdict += f" ~{int(depth * clip_play / -margin)} clips of runway"
                elif margin < 0:
                    verdict += " buffer empty"
                log(head + f"(avg{len(recent)} {mean:5.1f})  play {clip_play:5.2f}s  "
                           f"margin {margin:+5.2f}s/clip  sustains {can_do:4.1f}fps  "
                           f"{verdict}")
            if args.max_clips and played >= args.max_clips:
                log("max-clips reached")
                break
    except (KeyboardInterrupt, BrokenPipeError):
        log("stopping")
    finally:
        stop.set()
        for proc in player:
            if proc is not None and proc.poll() is None:
                proc.terminate()
        if not args.keep_clips:
            while not q.empty():
                leftover, _, _, _ = q.get()
                drop_clip(leftover)
        out.close()


if __name__ == "__main__":
    sys.exit(main())
