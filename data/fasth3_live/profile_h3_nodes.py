# SPDX-License-Identifier: Apache-2.0
"""Per-node wall-clock breakdown of a FastH3 job, straight from ComfyUI's websocket.

``/history`` only reports a single "Prompt executed in N seconds", which is
useless for deciding what to optimise: it cannot say whether the cost sits in
the text encoder, the sampler, or the VAE. The websocket ``executing`` event
fires as each node *starts*, so the gap between consecutive events is that
node's duration. That is the only per-node timing ComfyUI exposes without
patching it.

Runs each config twice by default and reports the second run, because the first
run after a config change is always polluted by model re-staging (see the
h3-cost-structure notes).

    .venv\\Scripts\\python.exe profile_h3_nodes.py --size 576x320 --size 512x288
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

import websockets.sync.client as wsclient

from stream_fasth3 import (VHS_EXTRA_DATA, set_vae_tiling, set_video_vae,
                           swap_writer)
from submit_h3 import get, post, to_api

WS = "ws://127.0.0.1:8188/ws?clientId="

PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a night market alley in the rain, red lanterns strung overhead and reflections breaking on the wet stone. The camera pushes in with small amplitude at slow speed as a vendor in a canvas apron turns skewers over a charcoal grill, sending sparks upward. [Shot 2] At 00:07.000, the shot cuts to a close-up of the grill, fat dripping onto the coals and flaring.

overall_soundscape: Rain patters on canvas awnings while charcoal hisses and crackles under dripping fat. Distant conversation and the clatter of tongs carry down the alley.

non_diegetic_music: N/A"""


def run_once(wf, oi, ov, client_id, writer="savevideo", vae_tile=0, video_vae=None,
             timeout=600.0):
    """Submit one job and return [(class_type, seconds), ...] plus the total."""
    prompt = set_video_vae(to_api(wf, oi, ov), video_vae)
    prompt = set_vae_tiling(prompt, vae_tile)
    prompt = swap_writer(prompt, writer, "profile_tmp/probe")
    names = {nid: n["class_type"] for nid, n in prompt.items()}
    with wsclient.connect(WS + client_id, open_timeout=20) as ws:
        pid = post("/prompt", {"prompt": prompt, "client_id": client_id,
                               "extra_data": VHS_EXTRA_DATA})["prompt_id"]
        spans, cur, t_prev, t0 = [], None, None, None
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = ws.recv(timeout=deadline - time.time())
            except TimeoutError:
                break
            if isinstance(msg, bytes):
                continue
            ev = json.loads(msg)
            d = ev.get("data", {})
            if d.get("prompt_id") not in (None, pid):
                continue
            if ev["type"] == "execution_start":
                t0 = t_prev = time.time()
            elif ev["type"] == "executing":
                now = time.time()
                if cur is not None and t_prev is not None:
                    spans.append((names.get(cur, cur), cur, now - t_prev))
                cur, t_prev = d.get("node"), now
                if d.get("node") is None:
                    return spans, now - (t0 or now), pid
    raise SystemExit("websocket timed out waiting for the job")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workflow",
                   default=r"F:\Comfy-Desktop\ComfyUI-Installs\MiniMax H3\ComfyUI"
                           r"\user\default\workflows\FastH3_4step_T2VA.json")
    p.add_argument("--size", action="append", default=None,
                   help="WxH, repeatable; default 576x320 and 512x288")
    p.add_argument("--length", type=int, default=362)
    p.add_argument("--dit", default="minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors")
    p.add_argument("--clip", default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    p.add_argument("--runs", type=int, default=2, help="runs per config; the last is reported")
    p.add_argument("--seed", type=int, default=777000)
    p.add_argument("--writer", default="savevideo",
                   choices=["savevideo", "h3fast", "vhs-x264", "vhs-nvenc"])
    p.add_argument("--video-vae", default=None, help="override the video VAELoader")
    p.add_argument("--prompt-file", help="use this scene instead of the built-in one, so "
                                         "several resolutions can be compared on identical "
                                         "content")
    p.add_argument("--keep", action="store_true", help="do not vary the prompt per run")
    p.add_argument("--vae-tile", type=int, default=0,
                   help="H3 video VAE spatial tile edge in px; 0 = stock 256")
    args = p.parse_args()

    prompt_text = (open(args.prompt_file, encoding="utf-8").read().strip()
                   if args.prompt_file else PROMPT)
    sizes = args.size or ["576x320", "512x288"]
    wf = json.load(open(args.workflow, encoding="utf-8"))
    oi = get("/object_info")
    client_id = f"h3-profile-{uuid.uuid4().hex[:8]}"

    results = {}
    for i, size in enumerate(sizes):
        w, h = (int(x) for x in size.lower().split("x"))
        for r in range(args.runs):
            ov = {
                # the run marker keeps ComfyUI from serving a cached sampler
                # result when the same prompt is measured twice; --keep drops it
                # so several resolutions can be compared on identical text
                "MiniMaxH3ImageToVideo": {"prompt": prompt_text if args.keep else
                                          prompt_text + f"\n\n<!-- {size} run {r} -->",
                                          "width": w, "height": h, "length": args.length},
                "RandomNoise": {"noise_seed": args.seed + i * 100 + r},
                "SaveVideo": {"filename_prefix": "profile_tmp/probe"},
                "UNETLoader": {"unet_name": args.dit},
                "CLIPLoader": {"clip_name": args.clip},
            }
            spans, total, pid = run_once(wf, oi, ov, client_id, args.writer, args.vae_tile, args.video_vae)
            tag = "reported" if r == args.runs - 1 else "warm-up"
            print(f"\n=== {size} x {args.length} [{args.writer}]  "
                  f"run {r + 1}/{args.runs} ({tag})   total {total:.2f}s")
            for name, nid, secs in spans:
                if secs >= 0.02:
                    print(f"    {secs:7.2f}s  {secs / total * 100:5.1f}%  {name} (#{nid})")
            if r == args.runs - 1:
                results[size] = (spans, total)

    if len(results) > 1:
        print("\n=== comparison (reported runs) ===")
        keys = list(results)
        allnodes = {n for s, _ in results.values() for n, _, _ in s}
        wid = max(len(n) for n in allnodes)
        print(f"{'node':<{wid}} " + " ".join(f"{k:>10}" for k in keys))
        for node in sorted(allnodes):
            row = []
            for k in keys:
                secs = sum(sc for n, _, sc in results[k][0] if n == node)
                row.append(f"{secs:9.2f}s")
            print(f"{node:<{wid}} " + " ".join(row))
        print(f"{'TOTAL':<{wid}} " + " ".join(f"{results[k][1]:9.2f}s" for k in keys))


if __name__ == "__main__":
    main()
