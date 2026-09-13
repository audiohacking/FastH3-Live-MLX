# SPDX-License-Identifier: Apache-2.0
"""Submit a FastH3 workflow straight to ComfyUI's HTTP API and verify the result.

comfy-cli is not installed, so the MCP server cannot drive this box; POSTing to
/prompt does the same job. Converting the UI workflow here also bypasses the
frontend's ``step: 17`` widget snapping, which is the suspected cause of the
"asked for 736 frames, got 515" reports -- whatever this script sends is
literally what the node receives.

Prints the requested vs delivered frame count so a truncation cannot go
unnoticed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:8188"
FFPROBE = r"E:\soft\ffmpeg\bin\ffprobe.exe"


def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.loads(r.read())


def to_api(workflow: dict, object_info: dict, overrides: dict) -> dict:
    """UI workflow -> API prompt, mapping widgets_values by the live schema."""
    by_id = {n["id"]: n for n in workflow["nodes"]}
    links = {l[0]: l for l in workflow["links"]}

    def resolve(node_id: int, slot: int) -> list | None:
        """Follow bypassed (mode 4) nodes to the real upstream producer.

        A bypassed node passes an input straight through to the output of the
        same type, so a reference to it must be rewritten, not dropped.
        """
        node = by_id[node_id]
        if node.get("mode") != 4:
            return [str(node_id), slot]
        out_type = node["outputs"][slot]["type"]
        for inp in node["inputs"]:
            if inp["type"] == out_type and inp.get("link") is not None:
                lk = links[inp["link"]]
                return resolve(lk[1], lk[2])
        return None  # nothing of that type flows through; drop the connection

    prompt = {}
    for n in workflow["nodes"]:
        if n["type"] == "MarkdownNote" or n.get("mode") in (2, 4):
            continue
        info = object_info.get(n["type"])
        if info is None:
            raise SystemExit(f"unknown node type {n['type']}")
        it = info["input"]
        # widgets_values comes in three shapes depending on node and frontend
        # version: a positional list, a name->value dict, or a list plus a
        # parallel widgets_values_named. Positional zipping against the wrong
        # one silently mis-assigns every widget.
        widget_names = [i["name"] for i in n["inputs"] if "widget" in i]
        raw = n.get("widgets_values")
        named = n.get("widgets_values_named")
        if isinstance(named, dict) and named:
            inputs = {k: v for k, v in named.items() if k in widget_names}
        elif isinstance(raw, dict):
            inputs = {k: v for k, v in raw.items() if k in widget_names}
        else:
            vals = list(raw or [])
            inputs = {name: vals[i] for i, name in enumerate(widget_names) if i < len(vals)}
        for i in n["inputs"]:
            if i.get("link") is not None:
                lk = links[i["link"]]
                ref = resolve(lk[1], lk[2])
                if ref is not None:
                    inputs[i["name"]] = ref
        for key, value in overrides.get(n["type"], {}).items():
            if key in it.get("required", {}) or key in it.get("optional", {}):
                inputs[key] = value
        prompt[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return prompt


def probe(path: str) -> str:
    if not os.path.exists(FFPROBE):
        return "(ffprobe not found)"
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height",
         "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.split()
    return " ".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workflow", required=True)
    p.add_argument("--length", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--dit", help="override UNETLoader unet_name")
    p.add_argument("--clip", help="override CLIPLoader clip_name")
    p.add_argument("--video-vae", help="override the video VAELoader vae_name")
    p.add_argument("--timeout", type=float, default=900.0)
    args = p.parse_args()

    wf = json.load(open(args.workflow, encoding="utf-8"))
    oi = get("/object_info")

    ov: dict[str, dict] = {}
    i2v = {k: v for k, v in (("length", args.length), ("width", args.width),
                             ("height", args.height)) if v is not None}
    if i2v:
        ov["MiniMaxH3ImageToVideo"] = dict(i2v)
        ov["MiniMaxH3ReferenceToVideo"] = dict(i2v)
    if args.seed is not None:
        ov["RandomNoise"] = {"noise_seed": args.seed}
    if args.dit:
        ov["UNETLoader"] = {"unet_name": args.dit}
    if args.clip:
        ov["CLIPLoader"] = {"clip_name": args.clip}


    prompt = to_api(wf, oi, ov)
    GEN = ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo")
    if args.video_vae:
        # both VAEs load through the same node type, so target the one whose
        # current filename identifies it as the video VAE
        hit = [v for v in prompt.values() if v["class_type"] == "VAELoader"
               and "video" in v["inputs"].get("vae_name", "").lower()]
        if len(hit) != 1:
            raise SystemExit(f"expected exactly one video VAELoader, found {len(hit)}")
        hit[0]["inputs"]["vae_name"] = args.video_vae
    req = prompt[next(k for k, v in prompt.items() if v["class_type"] in GEN)]["inputs"]
    dit = next((v["inputs"]["unet_name"] for v in prompt.values()
                if v["class_type"] == "UNETLoader"), "?")
    steps = next((v["inputs"]["steps"] for v in prompt.values()
                  if v["class_type"] == "BasicScheduler"), "?")
    print(f"submitting: {req.get('width')}x{req.get('height')} length={req.get('length')} "
          f"steps={steps}  dit={dit[:52]}")

    try:
        res = post("/prompt", {"prompt": prompt, "client_id": "fasth3-probe"})
    except urllib.error.HTTPError as e:
        raise SystemExit("rejected:\n" + e.read().decode("utf-8", "replace")[:2000])
    pid = res["prompt_id"]
    print(f"prompt_id {pid}")

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        hist = get(f"/history/{pid}")
        if pid in hist:
            h = hist[pid]
            wall = time.time() - t0
            status = h.get("status", {})
            print(f"finished in {wall:.1f}s  status={status.get('status_str')}")
            for node_out in h.get("outputs", {}).values():
                for vids in node_out.values():
                    if not isinstance(vids, list):
                        continue
                    for v in vids:
                        if not isinstance(v, dict) or "filename" not in v:
                            continue
                        fp = os.path.join(r"F:\Comfy-Desktop\ComfyUI-Shared\output",
                                          v.get("subfolder", ""), v["filename"])
                        print(f"  {v['filename']}")
                        print(f"  probe (w,h,nb_frames,duration): {probe(fp)}")
                        print(f"  requested length={req.get('length')}")
            if status.get("status_str") != "success":
                print(json.dumps(status, indent=2)[:1500])
            return
        time.sleep(2.0)
    print("timed out waiting for history")


if __name__ == "__main__":
    main()
