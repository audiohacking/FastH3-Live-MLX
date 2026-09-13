# FastH3 Live on Apple Metal

**Can we run a continuous FastH3 Live stream on Apple Silicon — same idea as the
CUDA/Comfy “Live” card (~19–22 fps on a 5090) — without leaving Metal?**

That is the experiment in this repo.

We generate short 4-step FastH3 clips back-to-back, retime them to a playable
MPEG-TS wire, and fan them out to a browser dashboard and/or VLC. Generation is
**demand-driven** (idle until someone watches). Play rate is **adaptive** so a
faster Mac (e.g. M5 Pro) automatically raises fps; a slower one drops instead of
falling behind.

> **Slug vs runtime:** the GitHub name is `FastH3-Live-MLX`. The **engine is
> antirez [h3.c](https://github.com/antirez/h3.c) on Metal (TensorOps / INT8)** —
> not MLX. MLX and Comfy ports are idea sources only.
> Community map: [awesome-minimax-H3](https://github.com/wildminder/awesome-minimax-H3).

Forked from [h3-ws](https://github.com/lmangani/h3-ws) after the Live path
diverged into its own stack. Day-to-day Live work lives **here**.

---

## What “Live” means here

| Layer | What we run |
|-------|-------------|
| Model | **FastH3 Dense-DataFree student DiT** (4-step DMD2) — not base H3 + a speed LoRA |
| Engine | Local `third_party/h3.c` Metal fork (`./h3`) |
| Wire | Continuous **MPEG-TS** over HTTP |
| Clients | Embedded **mpegts.js** dashboard **and** VLC / ffplay (multi-viewer) |
| Pace | Adaptive play-fps from measured gen time (`frames / (gen × 1.08)`) |
| Demand | 0 viewers → cancel generate immediately; Play / `/stream.ts` starts work |
| Prompts | ~721 scene × cast pool, or **custom prompt applied to the next clip** (no stream stop) |

CUDA Live cheats with pruned INT8 ConvRot + a custom Comfy graph. On Mac we have
to **convert** the Diffusers student into the native fused layout h3.c loads,
fix QKV interleaving, enable Metal TensorOps on Metal 4 chips (including M3 Ultra),
and build the whole live mux ourselves.

---

## What’s involved (the real work)

1. **Weights** — Download Diffusers FastH3 → convert to native fused QKV /
   gate-first SwiGLU (`scripts/convert_fasth3_diffusers_to_native.py`) → optional
   offline INT8 (`scripts/quantize_fasth3_native_int8.py`) → symlink encoder/VAE
   from MiniMax-H3 (`scripts/prepare_fasth3_native_tree.sh`).
2. **Correctness** — Diffusers `cat([Q,K,V])` ≠ native per-head interleaved QKV.
   Wrong layout → muddy frames (std≈6). Fixed mapping → real picture (std≈50+).
3. **Speed knobs on Metal** — TensorOps / runtime INT8, token-reduction,
   `--render-width/height 320` into a 448 output canvas (~2× sustain vs full 448
   denoise on M3 Ultra). Not: MLX swap, Sage-Attn, dual-clock samplers.
4. **Live plumbing** — `liveserver.py` producer + HTTP fan-out; PyAV retime to
   MPEG-TS when Homebrew ffmpeg is broken; 188-aligned TS chunks; video-only
   browser path when AAC PES is empty (VLC still fine).
5. **UX** — Dashboard with CPU/GPU meters, session watch + heartbeats, custom
   next-clip prompts, adaptive fps across machines.

Operator notes and overnight archaeology: [`LIVE.md`](LIVE.md), [`OVERNIGHT.md`](OVERNIGHT.md).

---

## Where we are (measured)

Host: **Apple M3 Ultra**, INT8 FastH3 student, 448² out / **320** render, 124f, 4 steps.

| | Wall | Notes |
|--|------|--------|
| Warm denoise + VAE | ~29 s / clip | ~**4.3 fps** sustain |
| Same without render-320 | ~68 s / clip | ~1.8 fps |
| CUDA Live reference | ~19–22 fps | Different stack (Comfy INT8 ConvRot) |

Gap to 5090 Live is still large (attention / denoise wait on Metal). The stream
is already **continuous and watchable** with adaptive play-fps and a working
picture after the QKV fix.

---

## Layout

| Path | Role |
|------|------|
| [`liveserver.py`](liveserver.py) | Live producer + dashboard + MPEG-TS |
| [`h3_live/`](h3_live/) | Scenes, cast, retime, pace, dashboard HTML |
| [`data/fasth3_live/`](data/fasth3_live/) | Apache scenes / characters |
| [`third_party/h3.c/`](third_party/h3.c/) | Vendored Metal fork (not upstream) |
| [`scripts/`](scripts/) | Convert, quantize, `h3-av`, build |

Weights stay under `models/` (gitignored). Never bulk-download MiniMax-H3 on a
thin laptop — prepare trees on the Apple Silicon host only.

---

## Run

```bash
git clone https://github.com/audiohacking/FastH3-Live-MLX.git
cd FastH3-Live-MLX

# Python deps + Metal binary
# uv venv --python 3.12 --seed && source .venv/bin/activate
# uv pip install -r requirements.txt
./scripts/build_h3.sh

# One-time weight prep (see LIVE.md for Diffusers download)
./scripts/prepare_fasth3_native_tree.sh
# optional: python scripts/quantize_fasth3_native_int8.py …

export H3_FORCE_TENSOROPS=1 H3_AV=$PWD/scripts/h3-av
python3 liveserver.py --host 0.0.0.0 --port 9000 -v
```

| Client | URL |
|--------|-----|
| Dashboard (Play starts gen) | http://127.0.0.1:9000/ |
| VLC / ffplay | http://127.0.0.1:9000/stream.ts |

**Defaults:** 448² · render 320 · 124 frames · 4 steps · adaptive play-fps ·
random scene pool (or Custom → Apply for the next clip).

```bash
# Pin play rate on a known machine
python3 liveserver.py --fps 3.0 -v
```

---

## Licence

- Scene/code under `data/fasth3_live/`: **Apache-2.0**
- FastH3 / MiniMax weights: **MiniMax H3 Community License** (territorial limits —
  read `NOTICE` in the assets)
- Character names are real people — label published streams as AI-generated

---

## Status

Working continuous Live on Metal with real picture, multi-client (browser + VLC),
adaptive fps, and next-clip prompt control. Still an experiment: closing the
fps gap vs CUDA Live is the open frontier (denoise wait / attention on h3.c),
not swapping the runtime for MLX.
