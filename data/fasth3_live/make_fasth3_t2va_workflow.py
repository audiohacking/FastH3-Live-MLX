# SPDX-License-Identifier: Apache-2.0
"""Generate a ComfyUI t2va workflow for the converted FastH3 4-step checkpoint.

Built to match the shape ComfyUI's own frontend emits (verified against the
shipped ``video_minimax_h3_r2v`` template), and validated against the live
node schemas before it is written: every node type must exist, and every link
must connect an output type to an input of the same type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIT = "minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"
# INT8 ConvRot: 12.1 s of decode against fp16's 14.0 s, 2677 MB staged against
# 4965 MB, and PSNR 44.4 dB / SSIM 0.984 against the fp16 decode of the same latent.
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

PROMPT = """integrated_multimodal_description:
[Shot 1] A woman in a dark green raincoat stands at a harbour railing at dusk, \
facing the camera from the centre of the frame. She lifts her right hand from the \
railing and pushes wet hair back from her forehead, then turns her head to look off \
to the left edge of the frame. The camera slowly pushes in from a medium shot to a \
medium close-up. Rain falls steadily; harbour lights blur into orange circles behind her.

overall_soundscape:
Steady rain on water and on the fabric of the coat. Low harbour swell slapping against \
a stone wall. A distant foghorn sounds once, far off to the right.

non_diegetic_music:
N/A"""

NOTE = """## FastH3 4-step T2VA

DiT: `minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors`
(converted from `FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree`)

### Sampler settings are not free parameters

**4 steps / euler / simple / BasicGuider** reproduces the ladder the student was
distilled on, `[999, 749, 500, 250]`, under the model's shift 12 (video) / 3
(audio). Other step counts are off-distribution.

`BasicGuider` applies no CFG, which is what a guidance-distilled model needs --
do not swap in `CFGGuider`.

`MiniMaxH3SigmaShift` here just states the defaults explicitly. If 4 steps look
soft or over-cooked, shift is the first knob to try, not the step count.

### Prompt format

T2VA wants three fields, all present, in this order:

```
integrated_multimodal_description:  shots, camera motion, what moves where
overall_soundscape:                 diegetic ambience and action sound
non_diegetic_music:                 score only, or N/A
```

Geometry beats vocabulary: write "enters from the left edge of the frame",
not "on her left".

### Where a 3.2 s step actually goes (measured at 864x480x124, seq 15448)

| part | time | share |
|---|---|---|
| attention, `pytorch attention` | 2.05 s | 64% |
| the four INT8 linears x50 blocks | 1.01 s | 32% |
| norms / adaln / rope / residuals | ~0.2 s | 4% |

So attention is the target, not the weights. `attention_sage` measures
14.41 ms/block against pytorch's 40.97 -> **0.72 s instead of 2.05 s**.

**MiniMax H3 Block Attention Split** is what switches it. It is a normal node
and does nothing unless it is in the graph -- there is no global setting.
`head_pct`/`tail_pct` keep that share of the first/last blocks on the slower,
higher-fidelity backend; they ship at 0/0 here (every block on sage, maximum
speed). If the 4-step output degrades, raise both to 20 before giving up on
sage -- the ends of the stack are the quantisation-sensitive part.

NVFP4 is deliberately not used for the DiT: comfy_kitchen's NVFP4 matmul takes
its fast path only when *both* operands are NVFP4, and the activations here are
BF16, so it dequantizes and runs at exactly BF16 speed (measured 67.88 ms/block
vs BF16's 67.85). It would cost 9x the weight error for nothing.

### torch.compile -- low ceiling, read this before spending time on it

With sage on, a 1.92 s step is linears 1.01 + attention 0.72 + everything else
~0.20. Compile can only attack that last 0.20 s, i.e. **~10% of the step and
~4% of a 17.5 s run**. It is shipped bypassed on purpose.

Backend is `cudagraphs`, not `inductor`. `inductor` fails here: the H3 forward
does `float(1.0 - sigma_v)` (model.py:577), which breaks the graph and makes
inductor emit a tiny **CPU** kernel; compiling it needs `omp.h`, which
`cpp_prefix.h` includes unconditionally, and ComfyUI is not launched from a
Visual Studio developer environment so `INCLUDE` is unset:

    fatal error C1083: Cannot open include file: 'omp.h'

To use `inductor` anyway, launch ComfyUI with the MSVC include/lib dirs
exported, e.g. from "x64 Native Tools Command Prompt for VS 2022", or set
INCLUDE/LIB to the toolset that matches the `cl` on PATH (here
`...\VC\Tools\MSVC.41.34120\include`). `cudagraphs` needs none of that --
it captures CUDA graphs instead of generating C++, which is the right tool for
launch overhead anyway.

### Video VAE

The workflow uses `minimax_h3_video_vae_int8_convrot`, not the fp16 VAE. Measured here by
re-running with the whole video-decode path removed:

| VAE | full run | audio only | video decode | staged |
|---|---:|---:|---:|---:|
| fp16 | 24.2 s | 10.2 s | 14.0 s | 4965 MB |
| int8_convrot | 22.2 s | 10.1 s | **12.1 s** | 2677 MB |

1.16x on decode and 2.3 GB less resident, for PSNR 44.4 dB / SSIM 0.984 against the fp16
decode of an identical latent -- no visible cost. It needs ComfyUI 0.31.0 or newer; older
builds decode it to black frames.

Decode is still the largest single cost in a run, ahead of the four denoising steps.

### Resolution

Set on the **MiniMax H3 Image to Video** node. Short edge must be >= 480.
`length` snaps to the 17k+5 grid (124 = ~5.2 s at 24 fps; trained range 124-362).

### This is fl2va / t2va only

There is no ref2va FastH3 student. Wire `first_frame` / `last_frame` on the
same node for fl2va; character replacement still needs the base ref2va model.
"""

# name, type, is_widget, optional
SCHEMA: dict[str, list[tuple[str, str, bool, bool]]] = {
    "UNETLoader": [("unet_name", "COMBO", True, False), ("weight_dtype", "COMBO", True, False)],
    "CLIPLoader": [("clip_name", "COMBO", True, False), ("type", "COMBO", True, False),
                   ("device", "COMBO", True, True)],
    "VAELoader": [("vae_name", "COMBO", True, False)],
    "MiniMaxH3ImageToVideo": [
        ("clip", "CLIP", False, False), ("vae", "VAE", False, False),
        ("prompt", "STRING", True, False), ("width", "INT", True, False),
        ("height", "INT", True, False), ("length", "INT", True, False),
        ("first_frame", "IMAGE", False, True), ("last_frame", "IMAGE", False, True)],
    "MiniMaxH3SigmaShift": [("model", "MODEL", False, False), ("shift_video", "FLOAT", True, False),
                            ("shift_audio", "FLOAT", True, False)],
    "MiniMaxH3BlockAttentionSplit": [
        ("model", "MODEL", False, False), ("edge_backend", "COMBO", True, False),
        ("middle_backend", "COMBO", True, False), ("head_pct", "FLOAT", True, False),
        ("tail_pct", "FLOAT", True, False)],
    "TorchCompileModel": [("model", "MODEL", False, False), ("backend", "COMBO", True, False)],
    "BasicGuider": [("model", "MODEL", False, False), ("conditioning", "CONDITIONING", False, False)],
    "BasicScheduler": [("model", "MODEL", False, False), ("scheduler", "COMBO", True, False),
                       ("steps", "INT", True, False), ("denoise", "FLOAT", True, False)],
    "KSamplerSelect": [("sampler_name", "COMBO", True, False)],
    "RandomNoise": [("noise_seed", "INT", True, False)],
    "SamplerCustomAdvanced": [("noise", "NOISE", False, False), ("guider", "GUIDER", False, False),
                              ("sampler", "SAMPLER", False, False), ("sigmas", "SIGMAS", False, False),
                              ("latent_image", "LATENT", False, False)],
    "VAEDecode": [("samples", "LATENT", False, False), ("vae", "VAE", False, False)],
    "VAEDecodeAudio": [("samples", "LATENT", False, False), ("vae", "VAE", False, False)],
    # frontend puts the optional AUDIO link right after images, as in the shipped template
    "CreateVideo": [("images", "IMAGE", False, False), ("audio", "AUDIO", False, True),
                    ("fps", "FLOAT", True, False), ("bit_depth", "INT", True, True)],
    "SaveVideo": [("video", "VIDEO", False, False), ("filename_prefix", "STRING", True, False),
                  ("format", "COMBO", True, False), ("codec", "COMFY_DYNAMICCOMBO_V3", True, True)],
}

OUTPUTS: dict[str, list[tuple[str, str]]] = {
    "UNETLoader": [("MODEL", "MODEL")],
    "CLIPLoader": [("CLIP", "CLIP")],
    "VAELoader": [("VAE", "VAE")],
    "MiniMaxH3ImageToVideo": [("positive", "CONDITIONING"), ("LATENT", "LATENT")],
    "MiniMaxH3SigmaShift": [("MODEL", "MODEL")],
    "MiniMaxH3BlockAttentionSplit": [("model", "MODEL"), ("split_report", "STRING")],
    "TorchCompileModel": [("MODEL", "MODEL")],
    "BasicGuider": [("GUIDER", "GUIDER")],
    "BasicScheduler": [("SIGMAS", "SIGMAS")],
    "KSamplerSelect": [("SAMPLER", "SAMPLER")],
    "RandomNoise": [("NOISE", "NOISE")],
    "SamplerCustomAdvanced": [("output", "LATENT"), ("denoised_output", "LATENT")],
    "VAEDecode": [("IMAGE", "IMAGE")],
    "VAEDecodeAudio": [("AUDIO", "AUDIO")],
    "CreateVideo": [("VIDEO", "VIDEO")],
    "SaveVideo": [("video", "VIDEO")],
}


class Graph:
    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self.by_id: dict[int, dict] = {}
        self._nid = 0
        self._lid = 0

    def add(self, node_type: str, pos, size, widgets=None, models=None, title=None,
            mode: int = 0) -> int:
        """mode: 0 = active, 2 = muted, 4 = bypassed (passes its input through)."""
        self._nid += 1
        nid = self._nid
        node = {
            "id": nid, "type": node_type, "pos": list(pos), "size": list(size),
            "flags": {}, "order": 0, "mode": mode,
            "inputs": [
                {"localized_name": n, "name": n, "type": t,
                 **({"shape": 7} if opt else {}),
                 **({"widget": {"name": n}} if is_w else {}),
                 "link": None}
                for n, t, is_w, opt in SCHEMA[node_type]],
            "outputs": [
                {"localized_name": n, "name": n, "type": t, "links": []}
                for n, t in OUTPUTS[node_type]],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets if widgets is not None else [],
        }
        if models:
            node["properties"]["models"] = models
        if title:
            node["title"] = title
        self.nodes.append(node)
        self.by_id[nid] = node
        return nid

    def note(self, text: str, pos, size) -> int:
        self._nid += 1
        nid = self._nid
        self.nodes.append({"id": nid, "type": "MarkdownNote", "pos": list(pos), "size": list(size),
                           "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [],
                           "title": "Read me", "properties": {}, "widgets_values": [text],
                           "color": "#432", "bgcolor": "#653"})
        return nid

    def link(self, src: int, src_slot: int, dst: int, dst_name: str) -> None:
        s, d = self.by_id[src], self.by_id[dst]
        out = s["outputs"][src_slot]
        slot = next(i for i, inp in enumerate(d["inputs"]) if inp["name"] == dst_name)
        inp = d["inputs"][slot]
        if inp["type"] != out["type"]:
            raise SystemExit(f"type mismatch: {s['type']}.{out['name']}({out['type']}) -> "
                             f"{d['type']}.{dst_name}({inp['type']})")
        self._lid += 1
        out["links"].append(self._lid)
        inp["link"] = self._lid
        self.links.append([self._lid, src, src_slot, dst, slot, out["type"]])

    def finalize(self) -> dict:
        # topological order for the frontend's execution hint
        deps = {n["id"]: {self.links[lk - 1][1] for i in n["inputs"] if (lk := i["link"])}
                for n in self.nodes}
        done: list[int] = []
        while len(done) < len(self.nodes):
            progressed = False
            for n in self.nodes:
                if n["id"] in done:
                    continue
                if deps[n["id"]] <= set(done):
                    n["order"] = len(done)
                    done.append(n["id"])
                    progressed = True
            if not progressed:
                raise SystemExit("cycle in graph")
        for n in self.nodes:
            for o in n["outputs"]:
                if not o["links"]:
                    o["links"] = None
        return {"id": "fasth3-t2va-4step", "revision": 0,
                "last_node_id": self._nid, "last_link_id": self._lid,
                "nodes": self.nodes, "links": self.links, "groups": [],
                "config": {}, "extra": {}, "version": 0.4}


def build(width: int, height: int, length: int, steps: int, seed: int, sampler: str,
          head_pct: float = 0.0, tail_pct: float = 0.0, compile_on: bool = False,
          compile_backend: str = "cudagraphs") -> dict:
    g = Graph()
    g.note(NOTE, (-1560, 4560), (620, 900))

    unet = g.add("UNETLoader", (-1560, 5540), (640, 82), [DIT, "default"], title="FastH3 4-step DiT")
    clip = g.add("CLIPLoader", (-1560, 5680), (640, 110), [CLIP, "minimax", "default"])
    vvae = g.add("VAELoader", (-1560, 5840), (640, 70), [VIDEO_VAE])
    avae = g.add("VAELoader", (-1560, 5950), (640, 70), [AUDIO_VAE])

    attn = g.add("MiniMaxH3BlockAttentionSplit", (-840, 5400), (360, 160),
                 ["pytorch attention", "sage attention", head_pct, tail_pct],
                 title="Attention backend (the 64% of each step)")
    shift = g.add("MiniMaxH3SigmaShift", (-840, 5610), (330, 106), [12.0, 3.0],
                  title="Sigma shift (defaults, shown for tuning)")
    compile_node = g.add("TorchCompileModel", (-840, 5760), (330, 82), [compile_backend],
                         title=f"torch.compile / {compile_backend}"
                               + ("" if compile_on else " (BYPASSED - ctrl+B to enable)"),
                         mode=0 if compile_on else 4)
    i2v = g.add("MiniMaxH3ImageToVideo", (-840, 5900), (560, 420),
                [PROMPT, width, height, length], title="T2VA prompt + empty AV latent")

    guider = g.add("BasicGuider", (-200, 5540), (280, 66))
    sched = g.add("BasicScheduler", (-200, 5660), (280, 130), ["simple", steps, 1.0])
    samp = g.add("KSamplerSelect", (-200, 5830), (280, 58), [sampler])
    noise = g.add("RandomNoise", (-200, 5930), (280, 82), [seed, "fixed"])

    adv = g.add("SamplerCustomAdvanced", (140, 5620), (300, 130))
    dec_v = g.add("VAEDecode", (500, 5620), (240, 66))
    dec_a = g.add("VAEDecodeAudio", (500, 5740), (240, 66))
    mkvid = g.add("CreateVideo", (800, 5620), (270, 110), [24, 8])
    save = g.add("SaveVideo", (1120, 5540), (760, 620), ["video/FastH3_t2va", "auto", "auto"])

    g.link(unet, 0, attn, "model")
    g.link(attn, 0, shift, "model")
    g.link(shift, 0, compile_node, "model")
    g.link(compile_node, 0, guider, "model")
    g.link(compile_node, 0, sched, "model")
    g.link(clip, 0, i2v, "clip")
    g.link(vvae, 0, i2v, "vae")
    g.link(i2v, 0, guider, "conditioning")
    g.link(noise, 0, adv, "noise")
    g.link(guider, 0, adv, "guider")
    g.link(samp, 0, adv, "sampler")
    g.link(sched, 0, adv, "sigmas")
    g.link(i2v, 1, adv, "latent_image")
    g.link(adv, 0, dec_v, "samples")
    g.link(vvae, 0, dec_v, "vae")
    g.link(adv, 0, dec_a, "samples")
    g.link(avae, 0, dec_a, "vae")
    g.link(dec_v, 0, mkvid, "images")
    g.link(dec_a, 0, mkvid, "audio")
    g.link(mkvid, 0, save, "video")
    return g.finalize()


def validate(wf: dict, comfy_root: str) -> None:
    """Check node types and input names against the live ComfyUI registry."""
    import asyncio
    import os
    import sys
    cwd = os.getcwd()
    os.chdir(comfy_root)
    sys.path.insert(0, comfy_root)
    try:
        import nodes
        asyncio.run(nodes.init_extra_nodes(init_api_nodes=False))
        problems = []
        for n in wf["nodes"]:
            if n["type"] == "MarkdownNote":
                continue
            cls = nodes.NODE_CLASS_MAPPINGS.get(n["type"])
            if cls is None:
                problems.append(f"unknown node type {n['type']}")
                continue
            it = cls.INPUT_TYPES()
            valid = set(it.get("required", {})) | set(it.get("optional", {}))
            for inp in n["inputs"]:
                if inp["name"] not in valid:
                    problems.append(f"{n['type']}: no input named {inp['name']}")
            n_out = len(getattr(cls, "RETURN_TYPES", ()))
            if len(n["outputs"]) != n_out:
                problems.append(f"{n['type']}: {len(n['outputs'])} outputs declared, class has {n_out}")
        if problems:
            raise SystemExit("validation failed:\n  " + "\n  ".join(problems))
        print(f"validated {len(wf['nodes'])} nodes / {len(wf['links'])} links against live node registry")
    finally:
        os.chdir(cwd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--length", type=int, default=124)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--head-pct", type=float, default=0.0,
                   help="%% of leading blocks kept on the slower/high-fidelity attention backend")
    p.add_argument("--tail-pct", type=float, default=0.0)
    p.add_argument("--compile", action="store_true",
                   help="ship the TorchCompileModel node active instead of bypassed")
    p.add_argument("--compile-backend", default="cudagraphs", choices=["cudagraphs", "inductor"],
                   help="inductor needs a C++ toolchain on PATH *and* INCLUDE set; cudagraphs does not")
    p.add_argument("--comfy-root", default=r"F:\Comfy-Desktop\ComfyUI-Installs\MiniMax H3\ComfyUI")
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()

    wf = build(args.width, args.height, args.length, args.steps, args.seed, args.sampler,
               args.head_pct, args.tail_pct, args.compile, args.compile_backend)
    if not args.no_validate:
        validate(wf, args.comfy_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
