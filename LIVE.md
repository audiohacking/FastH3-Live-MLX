# FastH3 Live on h3.c (Apple Metal)

Standalone continuous stream for Apple Silicon — **this repo**
([audiohacking/FastH3-Live-MLX](https://github.com/audiohacking/FastH3-Live-MLX)).
Forked from h3-ws; runtime is **h3.c Metal**, not MLX (slug is historical).

The speed asset is the **FastH3 Dense-DataFree student DiT** (4-step DMD2), not
base MiniMax-H3 + a LoRA. On CUDA/Comfy that student ships as a pruned INT8
ConvRot single file; on Metal we convert the upstream Diffusers BF16 student
into the native fused layout `./h3` already loads, then keep h3.c’s own Metal
INT8 activation path.

```bash
# 1) Apache/prompt assets (optional refresh)
./scripts/sync_fasth3_live_bucket.sh

# 2) Distilled student transformer (~66 GB Diffusers shards)
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree \
  --include 'transformer/*' \
  --include 'LICENSE' --include 'README.md' --include 'fastvideo_inference.json' \
  --local-dir models/FastH3-Dense-DataFree

# 3) Convert Diffusers → native + symlink text encoder / VAEs from MiniMax-H3
./scripts/prepare_fasth3_native_tree.sh

# 4) Stream (dashboard + MPEG-TS; Play starts generation)
export H3_FORCE_TENSOROPS=1 H3_AV=$PWD/scripts/h3-av
python3 liveserver.py --host 0.0.0.0 --port 9000 -v
# → http://127.0.0.1:9000/   browser player, or VLC → …/stream.ts
# Play fps adapts to measured gen time (faster Mac → higher fps).
# Pin: python liveserver.py --fps 2.0
# Custom prompt: dashboard → Custom → Apply to next clip (no stream stop)
```

## Recipe

| Knob | Default |
|------|---------|
| DiT | `models/MiniMax-H3-FastH3` (converted FastH3 student) |
| Canvas | 448×448 (DiT/VAE render 320×320, scaled up) |
| Frames | 124 (~5.2 s at 24 fps authored; 362 once sustain rises) |
| Steps | 4 (DMD ladder ≈ stock `--steps 4` under shift 12/3) |
| Layers / reuse | 50 / 1 (do not thin a 4-step student) |
| Token reduction | on (Metal; `--no-token-reduction` to disable) |
| INT8 row FC2 | on when TensorOps available (`--no-int8-row-fc2` to disable) |
| LoRA | none (student replaces the base DiT) |
| Play fps | **adaptive** (`--fps 0`, default): `frames / (gen_s × --margin)`; clamp `--min-fps`/`--max-fps`. Fixed: `--fps N` |

Engine: Metal **`./h3`** (antirez fork) — **not** MLX. Comfy INT8 ConvRot / Sage /
Spectrum stay idea sources; we invent Metal equivalents (TensorOps on Metal 4
chips including M3 Ultra, runtime int8, token reduction, warm FL2VA session).

### Smoke (M3 Ultra, 448×448×22, 4 steps, token-reduction + int8-row-fc2)

| Phase | Wall |
|-------|------|
| Text encoder (cold) | ~9.3 s |
| DiT load | ~8.7 s |
| DiT denoise (hot) | **~8.5 s** |
| Video VAE | ~3.8 s |
| Audio VAE | ~0.3 s |

Warm live clips skip encoder/DiT load after the first job. Measured M3 Ultra
INT8 Live with **`--render-width/height 320`** (output still 448²×124, 4 steps):
denoise **~21.2 s** + VAE **~7.0 s** → warm-ish **~29 s/clip**, sustain **~4.3 fps**
(picture std≈59). Without render-320 the same host was ~68 s / ~1.8 fps.
Env knobs (`H3_GPU_SAMPLER*`, single-tile VAE at 448) did not help; keep 2×2@256
VAE tiles. Play fps defaults to adaptive so faster Macs raise the rate automatically.

## Why not base + LoRA

The Live card’s measured ~19–22 fps on a 5090 is the **full student** (then
quantized/pruned for Comfy). A ~1 GB LoRA on stock FL2VA is a different, slower
object. Metal frontier work here:

1. **Weights** — Diffusers FastH3 → native fused QKV / gate-first SwiGLU / F32 heads
   (`scripts/convert_fasth3_diffusers_to_native.py`)
2. **Schedule** — `--steps 4` ≈ trained ladder under shift 12/3
3. **Kernels** — enable TensorOps/int8 on Metal 4 (not only chips named “M5”);
   token-reduction; no SSD streaming on 512 GiB unified memory
4. **INT8 DiT** — offline `blocks.*` quant (`scripts/quantize_fasth3_native_int8.py`);
   keep `token_refiner` BF16
5. **Retime** — PyAV→mpegts fallback when Homebrew ffmpeg is dyld-broken
6. **Demand** — producer idles until an HTTP viewer is connected
7. **Next** — INT8 re-quant from interleaved QKV; Live picture check; VAE speed
8. **QKV layout (fixed)** — Diffusers `cat([Q,K,V])` was wrong; native needs
   per-head interleaved rows (muddy std≈6 → picture std≈50)

## Overnight findings (2026-09-13)

**Symptom:** continuous Live MPEG-TS worked (~1.7 play fps) but picture is **muddy / no structure** (std≈6), not high-variance noise. Same in source MP4 and retimed `.ts` → **not retime**.

**Evidence**

| Check | Result |
|-------|--------|
| Prior BF16-tree smoke (`clip.mp4`) vs offline INT8 (`clip-int8.mp4`) | Nearly identical muddy stats |
| Offline INT8 scales | Present `[rows,1]` F32; dequant vs BF16 max_err≈0.014 — OK |
| Diffusers→native QKV / SwiGLU | Cosine vs base FL2VA confirms **QKV + swap** (`scripts/_overnight_validate_mapping.py`) |
| Metal4 INT8 | **Auto-on** at DiT load even without `--use-int8-row-fc2`; need `--use-slower-bf16-*` for true BF16 |
| True BF16 22f attempt | Denoise ~10 s + VAE OK; **encode failed** (Homebrew ffmpeg `libx265` dyld) → no `noise-bf16-22.mp4` |

**Hypothesis:** collapse is in the FastH3 Metal generate path (student + auto runtime INT8 and/or schedule), **not** missing offline scales. Prefer diagnosing on `models/MiniMax-H3-FastH3` with slower BF16 flags; do not trust INT8 Live picture yet.

Full commands + interpret: [`OVERNIGHT.md`](OVERNIGHT.md).

## Layout

| Path | Role |
|------|------|
| [`liveserver.py`](liveserver.py) | Producer + HTTP MPEG-TS live wire |
| [`h3_live/`](h3_live/) | Scenes, cast, retime |
| [`data/fasth3_live/`](data/fasth3_live/) | Synced scenes/characters/validators (Apache) |
| [`scripts/convert_fasth3_diffusers_to_native.py`](scripts/convert_fasth3_diffusers_to_native.py) | Student layout bridge |
| [`scripts/prepare_fasth3_native_tree.sh`](scripts/prepare_fasth3_native_tree.sh) | Hybrid `-d` tree |
| [`scripts/sync_fasth3_live_bucket.sh`](scripts/sync_fasth3_live_bucket.sh) | Prompt/code (+ optional Comfy refs) |

## Measure

```bash
# Adaptive (default): after each clip, play_fps = frames / (ema_gen × 1.08)
python liveserver.py --prefill 1 --max-clips 3 -v

# Fixed play fps (A/B or pin for a known host)
python liveserver.py --fps 1.7 --margin 1.08 --prefill 1 --max-clips 3 -v
```

Each clip logs gen seconds, EMA, chosen play_fps, margin, and sustained fps.
Dashboard `/api/status` exposes the current `play_fps`.

## Licence note

FastH3 / bucket weights are MiniMax H3 Community License (territorial
restrictions — read `NOTICE`). Scene/code under `data/fasth3_live/` is
Apache-2.0. Character names are real people — label streams as AI-generated if
you publish.
