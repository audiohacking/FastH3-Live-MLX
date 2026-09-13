# FastH3 Live (Apple Metal / h3.c)

Standalone **FastH3 Live** streamer for Apple Silicon. Forked from
[h3-ws](https://github.com/lmangani/h3-ws) (`feat/fasth3-live`) after the Live
path diverged into its own product: continuous MPEG-TS + embedded dashboard,
adaptive play-fps, FastH3 student DiT on Metal.

> **Repo name note:** `FastH3-Live-MLX` is the GitHub slug. The **runtime is
> antirez [h3.c](https://github.com/antirez/h3.c) (Metal TensorOps)** — not MLX.
> MLX ports remain idea sources only.

Canonical community map: [awesome-minimax-H3](https://github.com/wildminder/awesome-minimax-H3).

## What you get

| Piece | Role |
|-------|------|
| `liveserver.py` | Producer + HTTP dashboard + MPEG-TS fan-out (`:9000`) |
| `h3_live/` | Scenes, cast, retime, adaptive pace, dashboard |
| `data/fasth3_live/` | Apache scene/character assets |
| `third_party/h3.c/` | Local Metal fork (INT8, LoRA fold, PyAV shim, TensorOps) |
| `scripts/` | Diffusers→native convert, INT8 quant, `h3-av` mux |

**Not** the full h3-ws library Web UI workflow — Live is demand-driven: generation
starts when a viewer hits Play (or VLC opens `/stream.ts`).

## Quick start

```bash
git clone https://github.com/audiohacking/FastH3-Live-MLX.git
cd FastH3-Live-MLX

# Build Metal engine
./scripts/build_h3.sh

# Weights: prepare FastH3 student tree on the Apple Silicon host only
# (never download MiniMax-H3 on a thin/dev laptop — see LIVE.md)
./scripts/prepare_fasth3_native_tree.sh   # after Diffusers download
# optional INT8: scripts/quantize_fasth3_native_int8.py

export H3_FORCE_TENSOROPS=1 H3_AV=$PWD/scripts/h3-av
python3 liveserver.py --host 0.0.0.0 --port 9000 -v
```

- Dashboard: http://127.0.0.1:9000/
- VLC / ffplay: http://127.0.0.1:9000/stream.ts
- Prompt: Random pool (~721 scenes) or **Custom → Apply to next clip** (no stream stop)

## Recipe (defaults)

- Canvas **448²**, DiT/VAE render **320²**, **124** frames, **4** steps
- Play fps **adaptive** (`--fps 0`): `frames / (gen_s × 1.08)`
- 0 viewers → cancel in-flight generate immediately

Details, measured sustain (~4.3 fps warm on M3 Ultra @ render-320), and overnight
notes: [`LIVE.md`](LIVE.md).

## Licence

- Scene/code under `data/fasth3_live/`: Apache-2.0
- FastH3 / MiniMax weights: MiniMax H3 Community License (see `NOTICE` in assets)
- Character names are real people — label published streams as AI-generated

## Relation to h3-ws

This repo is the **Live fork**. Day-to-day Live work happens here. Upstream
[h3-ws](https://github.com/lmangani/h3-ws) remains the broader library/UI stack;
do not expect this tree to track every h3-ws Web UI change.
