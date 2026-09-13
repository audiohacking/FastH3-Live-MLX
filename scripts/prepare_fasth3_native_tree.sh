#!/usr/bin/env bash
# Build a MiniMax-H3-shaped tree whose FL2VA/transformer is the FastH3 student
# (Diffusers → native via convert_fasth3_diffusers_to_native.py). Everything else
# is symlinked from the official FL2VA tree already on disk.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${FASTH3_DIFFUSERS:-$ROOT/models/FastH3-Dense-DataFree/transformer}"
BASE="${H3_BASE:-$ROOT/models/MiniMax-H3}"
OUT="${FASTH3_NATIVE:-$ROOT/models/MiniMax-H3-FastH3}"

if [[ ! -f "$SRC/diffusion_pytorch_model.safetensors.index.json" ]]; then
  echo "missing FastH3 Diffusers transformer at $SRC" >&2
  echo "pull it with:" >&2
  echo "  hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree \\" >&2
  echo "    --include 'transformer/*' --local-dir models/FastH3-Dense-DataFree" >&2
  exit 1
fi
if [[ ! -d "$BASE/FL2VA/transformer" ]]; then
  echo "missing base FL2VA at $BASE/FL2VA" >&2
  exit 1
fi

mkdir -p "$OUT/FL2VA"
echo "→ converting Diffusers FastH3 → native transformer"
python3 "$ROOT/scripts/convert_fasth3_diffusers_to_native.py" \
  --src "$SRC" \
  --ref "$BASE/FL2VA/transformer" \
  --dst "$OUT/FL2VA/transformer"

link() {
  local name="$1"
  local target="$BASE/FL2VA/$name"
  local dest="$OUT/FL2VA/$name"
  rm -rf "$dest"
  ln -s "$target" "$dest"
  echo "  link $name → $target"
}

echo "→ linking shared FL2VA components from $BASE"
for name in text_encoder tokenizer video_vae audio_vae; do
  link "$name"
done

# Optional: copy NOTICE from Live assets if present
if [[ -f "$ROOT/data/fasth3_live/NOTICE" ]]; then
  cp "$ROOT/data/fasth3_live/NOTICE" "$OUT/NOTICE"
fi

echo
echo "ready: $OUT"
echo "  python liveserver.py --model-dir $OUT"
echo "  ./h3 -d $OUT -p '…' --steps 4 --width 448 --height 448 --frames 22 -o /tmp/out.mp4"
