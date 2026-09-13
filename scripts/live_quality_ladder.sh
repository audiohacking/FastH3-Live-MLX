#!/usr/bin/env bash
# Quality-ladder smoke for FastH3 Live (Metal).
# Runs short 22-frame oneshots per preset arm; prints denoise-ish wall times.
# Does not start liveserver — use for locking the default preset.
#
#   ./scripts/live_quality_ladder.sh
#   ./scripts/live_quality_ladder.sh --frames 22 --max-arms 3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export H3_FORCE_TENSOROPS="${H3_FORCE_TENSOROPS:-1}"
export H3_AV="${H3_AV:-$ROOT/scripts/h3-av}"
export PYTHONPATH="$ROOT"

FRAMES="${FRAMES:-22}"
STEPS=4
WIDTH=448
HEIGHT=448
OUT="${OUT:-/tmp/h3-ws/quality_ladder}"
mkdir -p "$OUT"

MODEL=""
for cand in \
  models/MiniMax-H3-FusedTurbo-INT8 \
  models/MiniMax-H3-FusedTurbo \
  models/MiniMax-H3-FastH3-INT8 \
  models/MiniMax-H3-FastH3
do
  if [[ -d "$cand/FL2VA/transformer" ]] && ls "$cand/FL2VA/transformer"/*.safetensors >/dev/null 2>&1; then
    MODEL="$cand"
    break
  fi
done
if [[ -z "$MODEL" ]]; then
  echo "no FastH3 / FusedTurbo tree under models/" >&2
  exit 1
fi
H3="${H3:-$ROOT/third_party/h3.c/h3}"
if [[ ! -x "$H3" ]]; then
  echo "missing $H3 — build with ./scripts/build_h3.sh" >&2
  exit 1
fi

PROMPT='integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium group shot frames three people in a small front room. The camera pushes in with small amplitude at slow speed. [Shot 2] At 00:02.000, the shot cuts to a close-up continuing the same beat.
overall_soundscape: Soft room tone.
non_diegetic_music: N/A'

echo "model=$MODEL  frames=$FRAMES  out=$OUT"
echo "arms: draft(320+TR) live(384 noTR) sharp(448 noTR)"

run_arm() {
  local name="$1" rw="$2" rh="$3" tr="$4"
  local mp4="$OUT/${name}.mp4"
  local -a cmd=(
    "$H3"
    -d "$MODEL"
    -p "$PROMPT"
    --width "$WIDTH" --height "$HEIGHT"
    --render-width "$rw" --render-height "$rh"
    --frames "$FRAMES" --steps "$STEPS" --layers 50 --reuse 1
    --use-int8-row-fc2
    -o "$mp4"
  )
  if [[ "$tr" == "1" ]]; then cmd+=(--token-reduction); fi
  echo "=== $name render=${rw}x${rh} tr=$tr ==="
  /usr/bin/time -p "${cmd[@]}" 2>"$OUT/${name}.log" || true
  if [[ -f "$mp4" ]]; then
    echo "wrote $mp4 ($(wc -c <"$mp4") bytes)"
  else
    echo "FAILED $name — see $OUT/${name}.log" >&2
  fi
  grep -E 'denoise|VAE|total|profile|step|real |user |sys ' "$OUT/${name}.log" | tail -n 30 || true
}

run_arm draft 320 320 1
run_arm live 384 384 0
run_arm sharp 448 448 0

echo "Done. Compare wall times in $OUT/*.log and pick the sharpest arm that sustains Live margin≥1.0 at 124f."
