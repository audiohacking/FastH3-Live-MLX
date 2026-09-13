#!/usr/bin/env bash
# Prepare models/MiniMax-H3-FusedTurbo once a *native* (or Diffusers) DiT exists.
#
# The MATLOWAI Hugging Face bake is Comfy INT8 ConvRot — inspect first:
#   python scripts/inspect_fused_turbo_for_native.py --src PATH/TO.safetensors
#
# If Diffusers BF16 shards become available:
#   python scripts/convert_fasth3_diffusers_to_native.py \
#     --src models/FusedTurbo-Diffusers/transformer \
#     --ref models/MiniMax-H3/FL2VA/transformer \
#     --dst models/MiniMax-H3-FusedTurbo/FL2VA/transformer
#   then symlink TE/VAE/tokenizer like prepare_fasth3_native_tree.sh.
#
# If the file is already native BF16 shards, copy them into
#   models/MiniMax-H3-FusedTurbo/FL2VA/transformer/
# and run the symlink block below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${BASE:-$ROOT/models/MiniMax-H3}"
DST="${DST:-$ROOT/models/MiniMax-H3-FusedTurbo}"
TR="$DST/FL2VA/transformer"
if [[ ! -d "$TR" ]] || ! ls "$TR"/*.safetensors >/dev/null 2>&1; then
  echo "missing native transformer at $TR" >&2
  echo "Comfy INT8 ConvRot cannot be converted by this script yet." >&2
  echo "Run: python scripts/inspect_fused_turbo_for_native.py --src <file>" >&2
  exit 1
fi
mkdir -p "$DST/FL2VA"
for link in text_encoder tokenizer video_vae audio_vae; do
  src="$BASE/FL2VA/$link"
  dst="$DST/FL2VA/$link"
  if [[ -e "$src" && ! -e "$dst" ]]; then
    ln -s "$(realpath "$src")" "$dst"
    echo "linked $dst -> $src"
  fi
done
echo "ready: $DST (liveserver prefers this over FastH3 student when present)"
