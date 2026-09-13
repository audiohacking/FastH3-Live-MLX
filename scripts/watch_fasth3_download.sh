#!/usr/bin/env bash
# Byte-level progress for FastH3 weight pulls (tqdm only counts finished files).
#   watch -n 5 ./scripts/watch_fasth3_download.sh
#   ./scripts/watch_fasth3_download.sh   # one-shot
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 - <<'PY'
import os
from pathlib import Path

def nbytes(path: str) -> int:
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total

student = nbytes("models/FastH3-Dense-DataFree")
comfy = nbytes("models/fasth3-live-ref")
print(f"student  {student/1e9:5.1f} / 66.3 GB  {100 * student / 66.3e9:5.1f}%")
print(f"comfy    {comfy/1e9:5.1f} / 21.0 GB  {100 * comfy / 21.0e9:5.1f}%")

inc = sorted(
    (
        p.stat().st_size,
        str(p.relative_to("models/FastH3-Dense-DataFree"))[:72],
    )
    for p in Path("models/FastH3-Dense-DataFree").rglob("*.incomplete")
    if p.is_file()
)
if inc:
    print(f"\nincompletes ({len(inc)}):")
    for sz, name in sorted(inc, reverse=True)[:8]:
        print(f"  {sz/1e9:5.2f} GB  {name}")
PY

echo
if pgrep -fq 'hf download FastVideo'; then
  echo "student: RUNNING (screen fasth3-dl)"
else
  echo "student: not running"
fi
if pgrep -fq 'hf buckets cp.*fasth3'; then
  echo "comfy:   RUNNING (screen fasth3-comfy)"
else
  echo "comfy:   not running"
fi
echo
echo "attach live tqdm:  screen -r fasth3-dl"
echo "logs:              tail -f /tmp/fasth3-student-dl.log"
