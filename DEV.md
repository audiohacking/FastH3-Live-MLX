# DEV.md — h3-ws development state & notes

Handoff doc for the next agent. Captures the current working state, what was just
implemented, known constraints, and the open roadmap items. Read
[`ROADMAP.md`](ROADMAP.md) for architecture/phases and [`AGENTS.md`](AGENTS.md) for
operating guidance.

**Read this first, then ROADMAP.md.** The docs in this repo are kept close to the
walking state of the tree.

---

## Current running state (commit `2b46c46` + uncommitted work below)

- **Engine**: local MiniMax-H3 via native `third_party/h3.c` (Metal) on Apple Silicon.
  M3 Ultra, 512 GiB unified, Metal (not Metal 4).
- **Server**: `python server.py --host 0.0.0.0 --port 8765`. Web UI + `/ws` protocol.
- **Models** (`models/MiniMax-H3`): FL2VA ✅, Ref2VA ✅ (both fully present, verified),
  video VAE ✅, audio VAE ✅. Health `engine_ok:true`.
- **Muxer**: `h3-av` PyAV shim (system Homebrew ffmpeg is broken/dyld-crashed and is
  skipped; the shim is the only media path).

### Committed so far (`2b46c46`)
- Model download progress via SSE + task-aware resume + non-blocking Models modal UX.
- Web UI auto-build on startup (`server.py` → `ensure_web_dist_built`).

---

## Uncommitted work in the tree (this branch)

> These are the changes NOT yet committed as of this handoff. The submodule is a
> **local fork** — we do NOT push to upstream `antirez/h3.c`. We commit the submodule
> locally and the parent tracks our gitlink. Sync upstream only manually if ever needed.

### Parent repo (Python + web)
- **`h3_av.py` — PyAV audio decode fix (IMPORTANT).**
  - Bug: audio import failed with `h3: FFmpeg could not decode a stereo soundtrack`.
  - Root cause: packed `format="flt"` for stereo. PyAV `to_ndarray()` on packed stereo
    returns a single interleaved plane `(1, N)` where N = samples × channels; the code's
    `.T`/reshape treated each interleaved element as a separate sample. For a mono→stereo
    upmix that yielded an odd frame count → byte length not a multiple of 8 →
    h3's `received % frame_bytes` check failed.
  - Fix: use planar `format="fltp"` (mirrors `../ltx-ws/ltx_media.py`), giving clean
    `(channels, samples)`; upmix mono, interleave exactly, and exclude the frame exactly
    at the `-t` boundary so output fits h3.c's capacity read + trailing EOF check.
  - Verified: 57 tests pass; exact h3 spawn produces frame-aligned stereo for both
    non-truncate (ref-audio) and truncate (video soundtrack) paths; live gen wrote MP4.
- **`web/src/App.tsx` — Ref2VA is now the DEFAULT mode** (`useState("ref2va")`).
  App opens in Ref2VA with the reference picker visible; Generate is disabled until a
  ref is added. Loading a clip/preset still restores that item's own stored mode.
- **`web/src/App.tsx` — UI/UX cleanups**: "Model Reads" (`WhatTheModelReads`) moved below
  the prompt and made compact; removed the `TurboInfo` overlay; Models modal no longer
  locks on close/backdrop (`openModels` always opens; `closeModels` guards on download).
- **`web/src/components/media/ModelsManager.tsx`**: releases the download-active flag on
  unmount so the Models modal can re-open.
- **`web/web_ui.py`**: simplified `/api/info` note (dropped RAM/SSD/Metal-4 prose).
- **`web/h3_media.py`**: relabeled 512×512 default ("Balanced default resolution.").
- **`.gitignore`**: added `.claude/`, `.serena/`, `web/.claude/` (local tooling caches).

### `third_party/h3.c` submodule (local fork — commit inside, do NOT push upstream)
- **`H3_AV` shim wiring** in `h3_ffmpeg.c` (`ffmpeg_program()`/`ffprobe_program()` honor
  `H3_AV` env first) — this makes the PyAV audio/video shim actually get used.
- **LoRA support**: `h3_lora.c` / `h3_lora.h` (Accelerate fold of `W += scale*B@A`) and
  `tools/fold_turbo_lora.py`.
- **INT8 quantization support** across `h3_weights.c`, `h3_dit.c`, `h3_gpu.m`,
  `h3_shaders.metal`, `h3_tokenizer.m`, `h3_safetensors.c`, `h3_video_vae.c`.
- Makefile, main.c tweaks for the above. (317 insertions / 35 deletions across 12 files.)
- The `h3` binary at `third_party/h3.c/h3` is a local build already containing these;
  it is gitignored.

---

## Working / verified

- FL2VA generation (t2va) ✅
- Ref2VA generation with **image + audio references** ✅ (produces video+audio with the
  subject matching the image and voice from the audio). This is the primary daily flow.
- Audio decode via PyAV shim ✅ (fixed, see above).
- Model download + resume + live progress ✅.
- Models modal open/close/reopen ✅.

---

## Known constraints / gotchas

- **No system ffmpeg.** Homebrew `/opt/homebrew/bin/ffmpeg` & `ffprobe` crash on missing
  dylibs and are skipped by `_tool_runs()`. All media I/O goes through the PyAV shim
  (`scripts/h3-av` → `h3_av.py`), wired via `H3_AV`/`H3_FFMPEG`/`H3_FFPROBE`.
- **h3.c audio decode byte rule**: decoded f32le byte count must satisfy
  `received % (channels*4) == 0` and `trailing == 0`. Mono→stereo upmix and
  `-t` boundary frames are the two places this silently breaks.
- **Standalone audio ref is invalid**: audio must accompany an image or video reference.
- **INT8 convrot model worked well for Ref2VA** → default NEW installs to
  `minimax_h3_fl2va_pruned_int8_convrot` (same Comfy-Org source as the Ref2VA int8).
  Only for new installs/downloads; do not change existing setups. **Not yet implemented.**
- No generation **preview** yet (see roadmap below).

---

## Open roadmap items (next)

1. **Default new installs to INT8 convrot FL2VA** (`minimax_h3_fl2va_pruned_int8_convrot`)
   in `scripts/download_model.py`, only for new setups. Keep existing installs on BF16 FL2VA.
2. **Live generation preview + cancel** (ComfyUI-Continuity style). Research
   https://github.com/roadmaus/ComfyUI-Continuity and replicate a frame-by-frame preview
   of the running generation plus a cancel control.
3. P7: CLI client + MCP server (`h3cli.py`, `mcp_server.py`).
4. P8: more tests (frame snap, canvas limits, FL2VA/Ref2VA exclusion, queue fairness);
   frontend hot reload (Vite `:5299` → `:8765`); benchmark wrapper.

See `ROADMAP.md` P5–P8 for the fuller phase context.
