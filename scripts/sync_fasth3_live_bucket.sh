#!/usr/bin/env bash
# Sync FastH3 Live reference assets from the private HF Storage Bucket.
# Code/prompts → data/fasth3_live/ (git-friendly).
# Optional Comfy weights → models/fasth3-live-ref/ (gitignored).
#
# The Metal live path does NOT load those Comfy files. It uses the FastVideo
# Dense-DataFree student converted to native layout:
#   hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree \
#     --include 'transformer/*' --local-dir models/FastH3-Dense-DataFree
#   ./scripts/prepare_fasth3_native_tree.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET="hf://buckets/audiohacking/fasth3-live-bucket"
CODE_DIR="$ROOT/data/fasth3_live"
WEIGHT_DIR="$ROOT/models/fasth3-live-ref"

mkdir -p "$CODE_DIR" "$WEIGHT_DIR"

echo "→ syncing code/prompts → $CODE_DIR"
hf buckets sync "$BUCKET" "$CODE_DIR" \
  --exclude '*.safetensors' \
  --exclude 'custom_nodes/**' \
  --exclude '.gitattributes'

if [[ "${1:-}" == "--with-weights" ]]; then
  echo "→ syncing Comfy reference weights → $WEIGHT_DIR"
  # Prefer explicit cp for large Xet objects (sync has left 0-byte stubs).
  hf buckets cp \
    "$BUCKET/minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors" \
    "$WEIGHT_DIR/minimax_h3_fl2va_fasth3_dense_pruned_int8_convrot.safetensors"
  hf buckets cp \
    "$BUCKET/minimax_h3_video_vae_w4a8_from_fp16.safetensors" \
    "$WEIGHT_DIR/minimax_h3_video_vae_w4a8_from_fp16.safetensors"
  hf buckets cp "$BUCKET/NOTICE" "$WEIGHT_DIR/NOTICE"
  hf buckets cp "$BUCKET/LICENSE-MiniMax-H3.txt" "$WEIGHT_DIR/LICENSE-MiniMax-H3.txt"
  echo "Comfy weights are reference/quant ideas only. Live Metal path:"
  echo "  models/MiniMax-H3-FastH3 via scripts/prepare_fasth3_native_tree.sh"
else
  echo "(skip weights; pass --with-weights to pull Comfy DiT/VAE for inspection)"
fi

echo "done."
