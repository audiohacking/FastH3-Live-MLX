# FastH3 Live on h3.c (Apple Metal)

Standalone continuous stream — **not** the Web UI.

The speed asset is the **FastH3 Dense-DataFree student DiT** (4-step DMD2), not
base MiniMax-H3 + a LoRA. On CUDA/Comfy that student ships as a pruned INT8
ConvRot single file; on Metal we convert the upstream Diffusers BF16 student
into the native fused layout `./h3` already loads, then keep h3.c’s own Metal
INT8 activation path.

```bash
git checkout feat/fasth3-live

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
python liveserver.py
# → http://127.0.0.1:9000/   browser player, or VLC → …/stream.ts
# Play fps adapts to measured gen time (faster Mac → higher fps).
# Pin: python liveserver.py --fps 2.0
```

## Recipe

| Knob | Default (`live` preset) |
|------|---------|
| DiT | `models/MiniMax-H3-FusedTurbo*` if present, else `MiniMax-H3-FastH3(-INT8)` |
| Canvas | 448×448 out · **render 384×384** · token-reduction **off** |
| Frames | 124 (~5.2 s authored; `long`=243 is optional / not a continuous Live path) |
| Steps | 4 (DMD ladder ≈ stock `--steps 4` under shift 12/3) |
| Layers / reuse | 50 / 1 (do not thin a 4-step student) |
| Token reduction | **off** in `live`/`sharp`/`long` (on only for `draft`) |
| Curated cast | 50% curated / 50% full; **ensemble-bias 0** (solo/duo face locks); **max 1–3** |
| INT8 row FC2 | on when TensorOps available (`--no-int8-row-fc2` to disable) |
| LoRA | none (student replaces the base DiT) |
| Play fps | **adaptive** (`--fps 0`, default): `frames / (gen_s × --margin)`; clamp `--min-fps`/`--max-fps`. Fixed: `--fps N` |

**Prompt pool (default):** The Office only — `prompts_scenes_office.txt` × `h3_characters_office.json` (Michael, Dwight, Jim, Pam, Andy) + `h3_office_lines.json`. **On-set mockumentary look.** Faces already toward camera. Mostly solos + sparse two-shots with subtle micro-motion. Speak beats use `{QUOTE}` filled from that character’s **short** famous lines (≤~6 words for <5s clips). Fair cast weighting. Originals: `prompts_scenes.txt`, `prompts_scenes_2.txt`, `h3_characters.json`.

```bash
python liveserver.py \
  --scenes data/fasth3_live/prompts_scenes.txt \
          data/fasth3_live/prompts_scenes_2.txt \
  --characters data/fasth3_live/h3_characters.json \
  --max-cast 5
```

Quality presets (CLI `--quality-preset` / dashboard **Apply quality** → next clip):

| Preset | render | TR | frames | Role |
|--------|--------|----|--------|------|
| `draft` | 320 | on | 124 | Legacy speed path — soft picture |
| **`live`** | **384** | **off** | 124 | **Default** — continuous Live target |
| `sharp` | 448 | off | 124 | jacokon-matched full canvas (slower) |
| `long` | 384 | off | 243 | Optional longer clip (not continuous Live) |

Engine: Metal **`./h3`** (antirez fork) — **not** MLX. Comfy INT8 ConvRot / Sage /
Spectrum stay idea sources; we invent Metal equivalents (TensorOps on Metal 4
chips including M3 Ultra, runtime int8, token reduction, warm FL2VA session).

### jacokon ↔ Metal gap matrix

| jacokon technique | Metal Live |
|-------------------|------------|
| Full 448² DiT+VAE (no render-down) | Preset `sharp`; default is `live` @ 384 |
| Dense-DataFree FastH3 student | ✅ converted native tree |
| Fused turbo DiT (MATLOWAI) | Prefer `models/MiniMax-H3-FusedTurbo*` when present; Comfy INT8 ConvRot has **no** convert yet (`scripts/inspect_fused_turbo_for_native.py` — verified on jacokon’s FastH3 ConvRot file: native key names + `weight_scale` / `comfy_quant`). Stub: `scripts/prepare_fused_turbo_native_tree.sh` |
| Sage / Spectrum / Sol / SLA | ❌ CUDA-only; Spectrum/Sol hurt quality — do not port |
| nvfp4 TE / W4A8 VAE | ❌ speed/RAM only; skip for quality |
| H3FastWriteVideo | N/A (PyAV retime) |
| Ensemble scenes + curated cast | ✅ bias + curated-share + ensemble-only toggle |
| Context-IR prompts | ✅ pool + custom wrap |

**Draft cause (fixed):** earlier Live stacked **render 320 + token-reduction**. Internal ~102k px sits below jacokon’s ~200k-px flare floor, then upscales to 448. Default raised render to **384** and keeps **TR off**.

**Token-reduction @384 rejected (2026-09-13):** warm A/B was **+22% wall** (29.1→22.6 s @56f) with healthy luma std, but Live stream showed **horizontal doubling / striped subjects (characters appear twice)**. FastH3 4-step + TR is not the README’s validated 20-step / layers-45 path — do not ship TR for `live`. Keep TR on `draft` only.

Ladder smoke (M3 Ultra, INT8 student, **22f** oneshot wall incl. cold load):

| Arm | render | TR | wall |
|-----|--------|----|------|
| draft | 320 | on | **16.4 s** |
| live | 384 | off | **22.7 s** |
| sharp | 448 | off | **24.8 s** |

`live` stays the default. `sharp` is only ~9% slower than `live` at 22f — use it when margin allows. Re-measure warm 124f after changing presets. `./scripts/live_quality_ladder.sh` reproduces the arms.

### Warm Live profile (M3 Ultra, INT8 student, `live` 448 out / 384 render / 4 steps)

**Rule:** only ship changes that improve measured sustain fps. No speculative knobs.

| Phase (124f warm) | Wall | Share |
|-------------------|------|-------|
| Text encoder | ~2.9 s | ~4% |
| DiT load | **0** if cache hit | — |
| DiT denoise (GPU wait) | **~48 s** | **~71%** |
| Video VAE decode | ~15 s | ~22% |
| Audio + mux | ~1 s | ~1% |
| **Sustain** | **~67 s/clip → ~1.8 fps** | |

56f A/B (same host): cold ~37 s; equal-token DiT+VAE hit ~31 s (**−16%**). Without fixed text width, sequential Live prompts share a token count only **~1.5%** of the time → DiT miss every clip. Idle `!refs clear` / `!first clear` / `!last clear` used to call `h3_cache_clear` even when empty (fixed). Prepared DiT cache now rebinds text via `h3_dit_reset_run` (refine + maps); AdaLN stays resident.

Optional `H3_PAD_TEXT_TOKENS=N` pads/truncates T2VA ids so every Live clip shares one DiT width. **Measured (56f, M3 Ultra):** cold pad-160 overhead **~0**; warm sustain **31.1 s vs 36.8 s nopad (−5.7 s/clip)** with DiT+VAE hit. `liveserver.py` defaults **`H3_PAD_TEXT_TOKENS=192`** (covers pool max ~185).

**Warm sustain A/Bs (same host, INT8, pad-192):**

| Arm | Warm wall | fps | Notes |
|-----|-----------|-----|-------|
| 56f / blocks=30 / **384 noTR** | 30.6 s | 1.83 | prior baseline |
| 56f / blocks=30 / **384 + TR** | **22.6 s** | **2.48** | **+22% wall vs noTR but Live picture doubled/striped — rejected** |
| 124f / blocks=30 / 384 noTR | 68.6 s | 1.81 | fps ≈ flat vs 56 — little amortisation |
| 243f cold | 161 s | 1.51 | product-dead for continuous Live; skip |
| `H3_DIT_COMMAND_BLOCKS` 20/30/40/0 @56f | 33.0–33.5 s | 1.67–1.70 | noise; keep default **30** |
| `--use-int8-row-fc2` on vs off @56f warm | on slightly faster | — | **keep on** |
| INT8 MLP knobs (`FC2_LOCAL128`, `FC1_LOCAL`, `GROUP_QUANT_128`) @56f | 29.1–29.6 s | ~0% | no ship |
| Metal toggles (no Morton/Morton4, MPS GQA, no coop QKV) @56f | 29.1–29.2 s | ~0% | no ship |
| MLP stage sample (`H3_PROFILE_MLP_STAGES`, 22f / 384) | FC1≈20 ms · FC2≈12 ms · quants≈0 (prequantized) | — | **FC1+SwiGLU is the MLP hotspot**; matmul already `tensor_inline` |
| FC1 register-local gate on full_k5376 (tried, discarded) | 30.6 vs 29.1 s | **+5% wall** | register pressure across 2nd full-K matmul — **not shipped** |

`H3_PROFILE_DIT_PHASES=1` (step 0, first blocks): **MLP ~59%** / attn ~41% of sampled block time. Denoise GPU wait remains the Metal frontier; TE (~4%) cannot move sustain ≥5% alone. Env knobs around existing INT8/NAX MLP paths are exhausted at Live geometry — next gain needs a new kernel or VAE decode work. `H3_PROFILE_MLP_STAGES=N` (rebuild `./h3`) samples the first N MLP calls with submit barriers to name quant / FC1 / FC2.

### Metal literature check (standing practice)

At each new bottleneck, search recent Apple Metal / Apple Silicon inference work **before** committing to a kernel rewrite. Use findings mostly to **avoid known failing approaches**, not to cap experimentation. Host is **M3 Ultra** (Apple9-class) — treat M5 Neural Accelerator papers as idea sources, not drop-in wins.

| Source (2025–2026) | Claim | Live relevance |
|--------------------|-------|----------------|
| [Cider INT8 GEMM tutorial](https://github.com/Mininglamp-AI/cider/blob/main/tutorial/how_to_write_efficient_int_gemm_m5_en.md) | Dropping **threadgroup staging** ≈2.3× on unified memory; deep K-loop | Audit our NAX INT8 MLP for TG copies; M5 TensorOps path is conditional |
| [Rigel / M4 Max matmul2d](https://arxiv.org/html/2606.12765v1) | On pre–Neural-Accel GPUs, TensorOps is shader-emulated; **epilogue fusion** +6–13% | Matches M3 better than M5 blogs — favor fuse-in-register over chasing matmul2d alone |
| [Draw Things Metal Quantized Attention](https://releases.drawthings.ai/p/metal-quantized-attention-pulling) | Online INT8 QK/V attention + fused dequant ~1.6–1.9× vs FP16 attn | Attn is ~41% of our block; M5-first but fusion pattern is portable |
| [mlx-teacache](https://github.com/IonDen/mlx-teacache) | TeaCache **does not engage** on 4–8 step distilled schedules (zero skips + overhead) | **Skip** algorithmic step-cache on FastH3 4-step |
| PyTorch MPS GLU→Metal | Standalone SwiGLU custom ≈1.03× fwd | Don’t rewrite SwiGLU alone; only fuse with GEMM |
| WWDC26 Metal tensors / BaseRT / mtlgemm | M5 cooperative_tensor INT8 | Idea bank when/if we run M5; not the current host’s free lunch |

### Smoke (M3 Ultra, 448×448×22, 4 steps, token-reduction + int8-row-fc2)

| Phase | Wall |
|-------|------|
| Text encoder (cold) | ~9.3 s |
| DiT load | ~8.7 s |
| DiT denoise (hot) | **~8.5 s** |
| Video VAE | ~3.8 s |
| Audio VAE | ~0.3 s |

Historical M3 Ultra INT8 Live with **`--render-width/height 320`** (output still 448²×124, 4 steps):
denoise **~21.2 s** + VAE **~7.0 s** → warm-ish **~29 s/clip**, sustain **~4.3 fps**
(picture std≈59). Without render-320 the same host was ~68 s / ~1.8 fps.
Default is now **384 / noTR** (`live`); TR @384 rejected for picture.
Env knobs (`H3_GPU_SAMPLER*`, single-tile VAE at 448) did not help; keep 2×2@256
VAE tiles. Play fps defaults to adaptive so faster Macs raise the rate automatically.

## Why not base + LoRA

The Live card’s measured ~19–22 fps on a 5090 is the **full student** (then
quantized/pruned for Comfy). A multi-GB LoRA on stock FL2VA is a different, slower
object — default Live stays on FastH3-INT8. **Exception:** explicit A/B with
[`--lora taomate_h3_3step`](#taomate-3-step-trial-base-fl2va--lora). Metal frontier work here:

1. **Weights** — Diffusers FastH3 → native fused QKV / gate-first SwiGLU / F32 heads
   (`scripts/convert_fasth3_diffusers_to_native.py`)
2. **Schedule** — `--steps 4` ≈ trained ladder under shift 12/3
3. **Kernels** — enable TensorOps/int8 on Metal 4 (not only chips named “M5”);
   token-reduction; no SSD streaming on 512 GiB unified memory
4. **INT8 DiT** — offline `blocks.*` quant (`scripts/quantize_fasth3_native_int8.py`);
   keep `token_refiner` BF16
5. **Retime** — PyAV→mpegts with stretched AAC (no system ffmpeg dependency)
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

## TaoMate 3-step trial (base FL2VA + LoRA)

[TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) is Alibaba’s streaming
3-step EMA adapter for **stock MiniMax-H3 FL2VA** — not the FastH3 student DiT.
ComfyUI conversion we wire: [Robert1212star/TaoMate-H3-3Step-ComfyUI](https://huggingface.co/Robert1212star/TaoMate-H3-3Step-ComfyUI)
(`taomate_h3_3step_comfy.safetensors`, ~2.5 GB). Catalog id: **`taomate_h3_3step`**.

```bash
# Downloads LoRA on first run; uses models/MiniMax-H3 (not FastH3-INT8).
python liveserver.py --lora taomate_h3_3step -v
# → steps=3, scale=0.8, stock FL2VA automatically
# Optional: --lora-scale 0.65 --steps 3 --model-dir models/MiniMax-H3
# Dashboard LoRA slider Apply changes strength for the next clip (DiT reload).
# Speed A/B (keeps quality TR off): --render-width 320 --render-height 320
# Metal phase dump: --profile
```

Warm baseline @384² / 124f / 3 steps (scale 1.0, M3 Ultra): ~55 s/clip —
**denoise ~36 s (66%)**, **video VAE ~15 s (27%)**, text ~2 s. Shorter frames
barely help sustain fps (fixed overhead). Prefer smaller render or fewer DiT
layers for speed; do not enable token-reduction (doubled subjects on Live).
Dashboard `/api/status` exposes live `progress` (stage/step/ETA) and `last_phases`.

Do **not** stack TaoMate on `MiniMax-H3-FastH3*`. Compare warm sustain vs the default
FastH3 student (`python liveserver.py`) before adopting. h3.c has no ManualSigmas —
community 3-step ladder `1.0, 0.961165, 0.853333, 0.0` is Comfy-only guidance.

## Episode batch (download)

Exclusive with Live: **Generate episode** stops the stream, clears watchers, and
owns ``./h3`` until the job finishes. Dashboard knobs (no Live sustain limits):
**1–10** scenes, per-clip duration **5 / 10 / 15 s**, render **320 / 384 / 448**,
steps / layers / reuse / token-reduction. Native **24 fps** MP4 concat (no
play-fps retime). Uses the active LoRA if loaded. Download ``/api/episode.mp4``
when status is ready. Live Play works again after the batch ends or is cancelled.


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
