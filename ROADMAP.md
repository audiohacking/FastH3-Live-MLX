# h3-ws roadmap

Local **MiniMax-H3** video+audio generation on Apple Silicon — Web UI, CLI, and MCP over a single WebSocket server — with **[h3.c](https://github.com/antirez/h3.c)** as the Metal inference engine.

This is a parallel of [ltx-ws](https://github.com/audiohacking/ltx-ws): **same UI/UX shell**, **new model surface**. Do not port LTX LoRAs, pipeline profiles, or frame math. Rebuild options from the h3.c CLI and MiniMax-H3 checkpoints.

---

## Product intent

| Keep from ltx-ws | Rebuild for H3 |
|------------------|----------------|
| Browser generate + library + progress | Modes, resolution, duration, quality controls |
| One-job-at-a-time WS server on `:8765` | Native `h3` process (not in-process Python MLX) |
| CLI + MCP clients | H3 flags, checkpoints, prompting |
| Multi-clip chains + merge | Last-frame → `--first-frame`; optional Ref2VA continue |
| Apple Silicon only | Metal / MPSGraph / TensorOps via h3.c |

**Not in v1:** face swap, IC-LoRA, LipDub, ID-LoRA, LTX retake/extend pipelines, CFG/STG, distilled vs HQ profiles, H3-Regenerate-2K (unreleased), hosted H3-Context-IR. **DiT LoRA picker is in:** catalog + HF download + `--lora` fuse at load (Tutu 20→8 NFE first).

---

## Engine facts (source of truth)

Upstream: [antirez/h3.c](https://github.com/antirez/h3.c) (`h3-metal`). Weights: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (`FL2VA/` and `Ref2VA/` trees).

| Constraint | Rule |
|------------|------|
| Output | Joint **video + 32 kHz stereo audio**, 24 fps |
| Canvas | Width and height multiples of **32**; product ≤ **768×1344** (so exact 768p 16:9 / 9:16 do not fit) |
| Temporal shape | Frames snap **up** to `5 + 17n` (22, 39, 56, 107, 243, 362, …) |
| Checkpoints | **FL2VA** (text / first / last frame) and **Ref2VA** (ordered image/video/audio refs) are **distinct** and **must not mix** in one run |
| Memory | ~33 GB transformer; ~40 GB peak physical on M5 Max full-residency; `--ssd-streaming` drops tracked DiT storage to ~2 GiB |
| Validated sizes | `512×512` (dev), `768×768`, `1344×768` / `768×1344`, `1024×768` / `768×1024`; `256×256` is a fast preview only. UI also offers snapped 16:9 / 9:16 / 4:5 / 5:4 under the same cap. |
| Hardware | Built for M3 Max / M5 Max; int8 MLP + TensorOps are M5 defaults |

`--steps` is always the number of denoising **passes**. `--reuse` and `--core-reuse` are mutually exclusive. `--seconds` and `--frames` are mutually exclusive.

---

## Architecture

```
  Web UI (React)     CLI          MCP
        │             │            │
        └─────────────┼────────────┘
                      │  HTTP + WS  (ltx-ws protocol shapes)
                      ▼
              server.py / web_ui.py
              job queue · library · snap/validate
                      │
                      ▼
              h3_backend.py  (process manager)
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    resident `./h3`         one-shot `./h3 -p … -o`
    (warm FL2VA session)    (Ref2VA / cold / isolation)
           │
           ▼
    third_party/h3.c  →  Metal / MPSGraph
    models/MiniMax-H3/{FL2VA,Ref2VA}
```

**Why a subprocess, not Python MLX:** h3.c is the production Metal engine. The Python layer only orchestrates jobs, the library, and the UI — same split as ltx-ws vs `ltx_pipelines_mlx`, except the model never lives in the Python process.

**Warm session (P5):** h3.c’s interactive mode keeps BF16 prompt conditioning, prepared DiT, and the video decoder resident. First generate pays load cost; repeats with a new seed do not. Target this for FL2VA (t2va + first/last + autocontinue). Ref2VA loads a different transformer — switch or spawn, never mix anchors with refs.

**Optional later:** [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx) as a Python MLX fallback. Not v1.

---

## Mode mapping (ltx-ws → h3-ws)

| ltx-ws mode | h3-ws v1 | h3.c path |
|-------------|----------|-----------|
| Text to video | **t2va** | FL2VA, prompt only |
| Image to video | **first-frame** | `--first-frame` / `!first` |
| Keyframe interpolation | **first+last** | `--first-frame` + `--last-frame` |
| V2V / extend (approx.) | **ref-video** | `--ref-silent-video` or `--ref-video` (Ref2VA) |
| Audio to video | **ref-av** | `--ref-audio` **plus** an image or video ref (audio cannot stand alone) |
| Autocontinue chain | **autocontinue** | last decoded frame → next `--first-frame` |
| Retake | **drop** | no native region-edit |
| IC-LoRA / LipDub / face swap / ID-LoRA | **drop** | no LoRA; identity-ish work uses Ref2VA refs |
| Native LTX extend | **later** | closest is Ref2VA continue, not in-place latent extend |

UI generation modes for v1:

1. `t2va` — Text to video+audio
2. `first_frame` — First frame → video
3. `last_frame` — Last frame → video
4. `fl2va` — First and last frame
5. `ref2va` — Ordered references (images / video / audio)

---

## Settings surface (replace LoRA + pipeline profiles)

### Quality presets (replace `pipeline_profiles`)

| id | Label | `--steps` | `--layers` | `--reuse` | Extra |
|----|-------|-----------|------------|-----------|--------|
| `four_step` | Four-step | 4 | 50 | 1 | Keep reuse 1 at small budgets |
| `aggressive` | Aggressive preview | 20 | 40 | 3 | Internal `320` for `512` output; **no** token-reduction with this combo |
| `fast` | Fast (UI default) | 20 | 45 | 2 | `--token-reduction`; internal `384` for `512` output |
| `balanced` | Balanced | 20 | 50 | 1 | h3.c CLI defaults |
| `close` | Close / reference | **50** | 50 | 1 | 50 complete 50-block denoiser forwards. Oracle when a fast mode changes subject, anatomy, motion, or composition. |

The UI always exposes `--steps`, `--layers`, and `--reuse` (preset fills them; testers can then edit). `--steps` 50 on close is explicit: it is much more expensive than the default 20×50 path.

Advanced (collapsed, not on the first row): `--core-reuse` (exclusive with reuse), `--token-reduction`, `--render-width` / `--render-height`, `--ssd-streaming`, `--use-int8-row-fc2` (M5), slower BF16 diagnostic flags only in CLI/env.

### LoRA picker

The UI catalog downloads FL2VA DiT adapters into `models/loras/` and passes `--lora PATH:SCALE` so h3.c can fuse `W += scale * (alpha/rank) * B @ A` at BF16 load. First builtin: [Tutu 20→8 NFE](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA) (step 100, strength 0.8, 8 steps). Not compatible with `--ssd-streaming`. h3.c still uses its own shifted Euler `--steps`; it does not reproduce ComfyUI ManualSigmas or video/audio shift 12/3.

### Resolution presets

| Canvas | Role |
|--------|------|
| 256×256 | 1:1 fast preview (auto RoPE adapt; no token-reduction) |
| 512×512 | 1:1 default / safest |
| 768×768 | 1:1 square 768p |
| 1024×576 / 576×1024 | Exact 16:9 / 9:16 (largest exact under the cap) |
| 1248×704 / 704×1248 | Largest near-16:9 / 9:16 under the cap |
| 1024×768 / 768×1024 | Exact 4:3 / 3:4 |
| 960×768 / 768×960 | Exact 5:4 / 4:5 social (Instagram feed) |
| 1344×768 / 768×1344 | 7:4 / 4:7 pixel-cap extremes (not 16:9 / 9:16) |

### Duration presets (`5 + 17n` @ 24 fps)

| Request | Frames | Actual |
|---------|--------|--------|
| 1s (dev) | 22 | 0.917 s |
| 2s | 56 | 2.333 s |
| 5s | 107 | 4.458 s |
| 10s | 243 | 10.125 s |
| 15s | 362 | 15.083 s |

Released workflow is **~4–15 s**. Short clips are for iteration.

### Server flags (v1)

| Flag | Default | Notes |
|------|---------|-------|
| `--host` / `--port` | `0.0.0.0` / `8765` | Web UI + `/ws` |
| `--h3-bin` | `third_party/h3.c/h3` | Built binary |
| `--model-dir` | `./models/MiniMax-H3` | HF snapshot with `FL2VA/` + `Ref2VA/` |
| `--quality` | `fast` | Preset above |
| `--width` / `--height` | `512` / `512` | Snap to ×32; enforce pixel cap |
| `--frames` or `--seconds` | 22 frames | Snap to `5+17n` |
| `--ssd-streaming` | off | Memory/speed tradeoff; ~2× slower denoise. Do not auto-enable. |
| `--web-output-dir` | `./web_outputs` | Library |
| `--no-web-ui` | off | WS only |

---

## Implementation phases

**P0–P6 UI + P5 warm FL2VA in tree (this checkout).** P7 CLI/MCP still ahead.

### P0 — Scaffold

Done: repo layout, gitignore, `third_party/h3.c` submodule, React chrome without LoRA, `requirements.txt`.

### P1 — Engine bootstrap

Done: `scripts/build_h3.sh`, `scripts/download_model.py`, `h3 --info` via `H3Engine.info()`, FFmpeg check, RAM probe → SSD-streaming hint.

### P2 — First generate (one-shot)

Done: `h3_backend.py` one-shot spawn, WS framing, `/api/generate` + library + SSE, default 512×512 / 22 frames / balanced.

### P3 — H3 settings in the UI

Done: quality presets, seed, resolution, duration, steps, layers, reuse, token-reduction, SSD streaming. Core-reuse / internal render size remain CLI/body advanced (P8 polish).

### P4 — FL2VA image modes + autocontinue

Done: first/last/fl2va uploads, library-frame capture, clip ×N last-frame chain + autoconcat.

### P5 — Resident `h3` session

**In tree:** FL2VA jobs reuse an interactive `./h3` process (PTY, linenoise). Settings are applied with `!size` / `!frames` / `!first` / …; the MP4 is copied out of the session directory. Ref2VA still one-shots (interactive CLI only has `!ref-image`, and mixing checkpoints in one process is unsafe). Session crash or start failure falls back to one-shot. Cancel tears the process down.

### P6 — Ref2VA

**UI (one-shot) is in tree:** ordered ref list under the prompt (image / silent video / video / video+audio / audio), Picture N / Video N tokens, mutual exclusion with first/last-frame, documented canvas dropdown, audio duration checks (2–15 s each, total ≤15 s) at ingest. **Ref2VA is the default UI mode on open** (add a reference to enable Generate).

**Audio import (v1) is fixed:** mono/stereo audio references are decoded through the PyAV shim with planar `fltp` resampling so the byte stream satisfies h3.c's frame-alignment check — see `DEV.md`.

**INT8 default for NEW installs (pending):** the Comfy-Org INT8 convrot model
(`minimax_h3_fl2va_pruned_int8_convrot`) worked well for Ref2VA; default new
`download_model.py` setups to that INT8 FL2VA instead of the full BF16 FL2VA. Existing
setups are left untouched.

Still remaining for the warm session:

- Interactive `!ref-video` / `!ref-audio` (upstream CLI only has `!ref-image` today)
- Switching FL2VA ↔ Ref2VA without dropping the FL2VA process (today Ref2VA stops the warm session so two transformers are not resident at once)

### P7 — CLI + MCP + agent docs

- CLI client (protocol-compatible with the WS server): prompt, image, refs, quality, frames, count, autocontinue, autoconcat
- MCP: `h3_server_healthcheck`, `h3_generate_video`, `h3_generate_sequence`
- `AGENTS.md` / `DIRECTOR.md` / `CLAUDE.md`: Context-IR-style prompts (subject, action, camera, look, **audio**), H3 frame math, no LTX LoRA advice
- `Start H3-WS.command` double-click launcher

### P8 — Hardening

- Tests: snap_frames `5+17n`, canvas limits, preset expansion, FL2VA vs Ref2VA mutual exclusion, queue fairness
- Frontend hot reload (`web` Vite :5299 → :8765) like ltx-ws
- Benchmark script wrapping `h3 --profile`
- Document M3 vs M5 paths (int8 default, TensorOps, ssd-streaming)

### P9 — Generation preview (ComfyUI-Continuity style)

- Research https://github.com/roadmaus/ComfyUI-Continuity and replicate a live
  frame-by-frame preview of the running generation plus a cancel control, streamed over
  the existing WS/SSE progress channel.

---

## Prompting (DIRECTOR.md seed)

H3 expects a Context-IR-like description, not a three-word caption. Gold prompt skeleton:

```
Scene: …
Action: …
Camera: …
Look: …
Audio: …
```

Keep identity and object counts explicit. `--seed` default in h3.c is 42; UI should still offer random. There is **no** on-device prompt rewriter in v1 (same as ltx-ws `enhancement_enabled=false`).

---

## File plan (target)

```
server.py              WS + embedded UI
web_ui.py              HTTP API, library, jobs
h3_backend.py          h3.c process manager
h3_media.py            frame extract, concat, snap helpers
h3_paths.py            model / output / binary paths
h3cli.py               WS CLI client
mcp_server.py          MCP adapter
web/                   React UI (ltx-ws visual language)
third_party/h3.c       submodule
scripts/build_h3.sh
scripts/download_model.py
models/                gitignored MiniMax-H3 snapshot
web_outputs/
```

Copy UI chrome from `../ltx-ws/web` (layout, library, progress, clip multiplier). Replace the generate-form controls and `types.ts` config shape.

---

## Decisions already made

1. **Engine = h3.c**, not MiniMax PyTorch/SGLang/vLLM, not minimax-h3-mlx (v1).
2. **UI/UX clone** of ltx-ws; **options are H3-native**.
3. **One generation at a time**, fair queue.
4. **FL2VA warm, Ref2VA on demand.**
5. **Apple Silicon only** (Metal). No CUDA story.
6. **h3.c media is a PyAV shim.** `H3_AV`/`H3_FFMPEG`/`H3_FFPROBE` point at `scripts/h3-av` (`h3_av.py`). No system ffmpeg install. Patch: `patches/h3-prefer-H3_AV.patch`.
7. **`third_party/h3.c` is a local fork** that we manage inside this repo. We commit it
   locally (the parent tracks the gitlink); we do **not** push to upstream `antirez/h3.c`,
   and we sync upstream manually only if ever needed. No upstream relations or PRs.

## Open questions (resolve during P2–P5, not before)

- Exact stdin protocol robustness of interactive `h3` (parse `!help` once we have a local binary).
- Whether to pin a h3.c git SHA in the submodule (yes, once the first green generate lands).
- Branding in the header: **H3-WS** (drop “Videofentanyl” unless we decide to keep the in-joke).
- Default duration: 22-frame dev vs 107-frame (~4.5 s) “real clip”. Lean **22 for first boot**, **107 as the UI default after P3**.
