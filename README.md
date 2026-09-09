# h3-ws

Local **MiniMax-H3** video+audio generation on Apple Silicon. Native [h3.c](https://github.com/antirez/h3.c) (Metal) with a browser UI.

Open **http://127.0.0.1:8765/** after starting the server.

The UI starts in **Ref2VA** mode (add an image/video/audio reference to generate). See
[`DEV.md`](DEV.md) for the current development state and [`ROADMAP.md`](ROADMAP.md) for the plan.

## Requirements

- Apple Silicon Mac (Metal)
- Python 3.12+ with packages from `requirements.txt` (PyAV `av` for all media, including the h3.c mux/decode shim)
- Node.js 18+ (to build the UI)
- Xcode command-line tools (`make`)
- Enough unified memory for the model (~40 GB peak). Use `--ssd-streaming` if you have under ~64 GB.

## Install

```bash
git clone --recurse-submodules https://github.com/audiohacking/h3-ws.git
cd h3-ws

# If you already cloned without submodules:
git submodule update --init --recursive

uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -r requirements.txt

./scripts/build_h3.sh
python scripts/download_model.py          # ~134 GB FL2VA only (enough to generate)
# python scripts/download_model.py --with-ref2va   # +~62 GB Ref2VA transformer

cd web && npm install && npm run build && cd ..
```

Do **not** `hf download MiniMaxAI/MiniMax-H3` without filters. The Hugging Face repo is ~464 GB: native `FL2VA/` + `Ref2VA/` plus a second Diffusers copy at the repo root (`transformer/`, `transformer_ref/`, `text_encoder/`, `vae/`). There are no extra quantizations in that repo. h3.c only loads the native trees.

Default `python scripts/download_model.py` fetches **FL2VA only** (~134 GB: Qwen encoder + DiT + VAEs). Pass `--with-ref2va` for the extra ~62 GB Ref2VA transformer. Existing files are never deleted; a second run resumes and skips what is already on disk. If a previous unfiltered download already saved root `text_encoder/`, that copy is reused for `FL2VA/text_encoder` instead of downloading it again.

## Run

```bash
source .venv/bin/activate
python server.py
```

Or double-click `Start H3-WS.command`.

Default generate is 512×512, **1s** (22 frames), **fast** quality (token-reduction + 384 internal render). Lower RAM if a run is killed for memory (this makes denoise slower):

```bash
python server.py --ssd-streaming
```

## Generate

Prompts should describe scene, action, camera, look, and audio. What you type is what H3 sees — there is no cloud rewriter.

| Mode | Use |
|------|-----|
| Text to video+audio | Prompt only |
| First / last / first+last frame | Image anchors |
| Ordered references (Ref2VA) | Image, silent video, video, video+audio, and/or audio |

First/last-frame anchors cannot be mixed with Ref2VA references. For references, prompt with `Picture 1`, `Video 1`, `Audio 1` in list order. Standalone audio must accompany an image or video (≤9 images, ≤3 videos, ≤3 audio).

**Quality:** Four-step · Aggressive · Fast (default) · Balanced · Close.

**Canvas** (dropdown, grouped): **1:1** 256×256 preview, 512×512 (safest), 512×512 with 384 or 320 internal, 768×768. **16:9** 1024×576 (exact) and 1248×704 (largest under the cap). **7:4** 1344×768 max landscape. **4:3** 1024×768. **5:4** 960×768. **4:5** 768×960 Instagram/social. **3:4** 768×1024. **9:16** 576×1024 (exact Stories/Reels/TikTok) and 704×1248 (largest). **4:7** 768×1344 max portrait. Each side is a multiple of 32; pixel count cannot exceed 768×1344. True 768p 16:9 (1365×768) does not fit that cap. H3-Base is a 768p model; 512×512 is the usual working size.

Output is 24 fps video with 32 kHz stereo audio. Duration picks are **1s, 2s, 5s, 10s, 15s** (each snapped to a legal H3 frame count).
