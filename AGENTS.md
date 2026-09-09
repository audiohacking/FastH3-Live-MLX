# AGENTS.md

Canonical guide for AI agents using **h3-ws** to generate video on Apple Silicon.

**Read [`ROADMAP.md`](ROADMAP.md)** for architecture and phase status. Prompting for H3 should follow a Context-IR skeleton (scene, action, camera, look, audio) — a full DIRECTOR.md lands in P7.

## Stack

| Piece | Role |
|-------|------|
| `server.py` | Local MiniMax-H3 inference via native **h3.c**. One job at a time. Embeds Web UI. |
| `h3_backend.py` | Spawns `./h3 -p … -o`. FL2VA keeps a resident interactive session; Ref2VA is one-shot. |
| `web_ui.py` + `web/` | Browser library, quality presets, SSE progress. |
| Weights | `models/MiniMax-H3/{FL2VA,Ref2VA}` from `MiniMaxAI/MiniMax-H3` |

**Default Web UI:** http://127.0.0.1:8765/
**WebSocket:** ws://127.0.0.1:8765/ws

The app opens in **Ref2VA** mode by default (add an image/video/audio reference to enable Generate; switch modes as needed).

This stack does **not** run cloud prompt expansion. What you send is what H3 sees. h3.c media I/O is the PyAV shim (`scripts/h3-av`); no system ffmpeg. The `third_party/h3.c` submodule is a **local fork** — we manage it inside this repo (commit locally, never push upstream); it carries the PyAV shim wiring, LoRA fold, and INT8 support. See [`DEV.md`](DEV.md) for the current handoff state.

## Weights (mandatory)

**Never download MiniMax-H3 weights on the development workstation.** Fetch them only on the Apple Silicon test host.

Only the official MiniMax-H3 **native** checkpoint trees. Not LTX, not Diffusers root shards, not community quants.

The Hugging Face repo is ~464 GB. Unfiltered `hf download MiniMaxAI/MiniMax-H3` pulls everything: native `FL2VA/` (~134 GB), native `Ref2VA/` (~134 GB, of which ~72 GB duplicates FL2VA encoder/VAE), and a Diffusers copy at repo root (~196 GB) that h3.c never opens.

On the test host:

```
python scripts/download_model.py                 # FL2VA only, ~134 GB; resumes, never deletes
python scripts/download_model.py --with-ref2va   # + Ref2VA transformer ~62 GB
python scripts/download_model.py --status        # what is already on disk
```

Build the engine: `./scripts/build_h3.sh` (needs the `third_party/h3.c` submodule).

## Frame math

H3 snaps **up** to `5 + 17n` at 24 fps (22, 39, 56, 107, 243, 362, …). Width and height must each be multiples of 32, at least 32, and their product must not exceed 768×1344. The UI offers tagged canvases: 1:1 (256 through 768), exact 16:9 `1024×576` / 9:16 `576×1024`, largest near-16:9 `1248×704` / `704×1248`, 4:5 `768×960` and 5:4 `960×768`, 4:3 / 3:4, and the 7:4 / 4:7 pixel-cap extremes `1344×768` / `768×1344`. H3-Base is a 768p model; 512×512 is the default development size.

## Modes (v1)

- `t2va` — text to video+audio (FL2VA)
- `first_frame` / `last_frame` / `fl2va` — image anchors
- `ref2va` — ordered image / silent video / video / video+audio / audio references (Ref2VA)

Do not mix first/last-frame anchors with Ref2VA references. Prompt Ref2VA with `Picture N` / `Video N` / `Audio N` in list order. Standalone audio must accompany an image or video. Limits: ≤9 images, ≤3 videos, ≤3 audio, mixed files ≤12.

## Quality presets

`four_step` · `aggressive` · `fast` (default) · `balanced` · `close`

h3.c defaults are `--steps 20 --layers 50 --reuse 1`. The UI always lets you edit steps, layers, and reuse (a preset fills them). Close keeps `--steps 50` explicit: 50 complete 50-block denoiser forwards — the oracle when a fast mode changes subject, anatomy, motion, or composition.

`--reuse` and `--core-reuse` are mutually exclusive. Do not combine token-reduction with `--layers 40 --reuse 3`. `--ssd-streaming` saves RAM and makes denoise much slower — leave it off unless the process is killed for memory. On M5, `--use-int8-row-fc2` is on automatically.

## LoRA

The Web UI LoRA menu can download and enable FL2VA DiT adapters. First builtin: [Tutu MiniMax-H3 Audio-Video 20→8 NFE](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA) (step 100 at strength 0.8, 8 steps). h3.c fuses `W += scale * B @ A` at DiT load (`--lora PATH:SCALE`). Not compatible with `--ssd-streaming`. Rebuild `./h3` after pull so the fuse patch is in the binary.
