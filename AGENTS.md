# AGENTS.md

Canonical guide for AI agents on **FastH3-Live-MLX** (Apple Silicon Live stream).

This repo is a **fork of h3-ws** focused on continuous FastH3 Live. Prefer
[`LIVE.md`](LIVE.md) and [`README.md`](README.md) for Live. The library Web UI
(`server.py` :8765) may still be present as inherited tooling — Live’s product
surface is **`liveserver.py` :9000**.

**Engine is antirez h3.c Metal** (`third_party/h3.c` local fork). Do **not**
replace it with MLX or ComfyUI as runtime despite the GitHub slug containing
`MLX`.

Prompting for H3 should follow a Context-IR skeleton (scene, action, camera,
look, audio). Live dashboard: random scene pool or custom prompt applied to the
**next** clip without stopping the stream.

## Live stack

| Piece | Role |
|-------|------|
| `liveserver.py` | Continuous producer + dashboard + MPEG-TS (`:9000`) |
| `h3_live/` | Scenes, cast, retime, adaptive play-fps, metrics |
| `h3_backend.py` | Spawns `./h3 -p … -o` (oneshot jobs for Live) |
| Weights | `models/MiniMax-H3-FastH3` / `-INT8` (+ MiniMax-H3 encoder/VAE) |

**Live dashboard:** http://127.0.0.1:9000/  
**VLC:** http://127.0.0.1:9000/stream.ts

## Inherited library stack (optional)

| Piece | Role |
|-------|------|
| `server.py` | h3-ws Web UI on `:8765` (Ref2VA/FL2VA library) |
| `web_ui.py` + `web/` | Browser library UI |

What you send is what H3 sees (no cloud prompt expansion). Media I/O is the
PyAV shim (`scripts/h3-av`). `third_party/h3.c` is a **local fork** — commit
here; never push upstream to antirez.

## Weights (mandatory)

**Never download MiniMax-H3 / FastH3 weights on a thin development workstation.**
Fetch them only on the Apple Silicon test host. See [`LIVE.md`](LIVE.md) for the
FastH3 student convert path; `scripts/download_model.py` for base FL2VA/Ref2VA.

Build the engine: `./scripts/build_h3.sh`.

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
