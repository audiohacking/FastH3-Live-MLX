#!/bin/bash
# Start H3-WS.command — double-click on Mac to run the Web UI.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
PORT="${H3_WS_PORT:-8765}"
UI_URL="http://127.0.0.1:${PORT}/"
export PYTHONUNBUFFERED=1

say() { printf '%s\n' "$*"; }

die() { say "ERROR: $*"; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "This launcher is for macOS. Run: python server.py"
[[ -f "$ROOT/server.py" ]] || die "server.py not found"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  say "Creating .venv…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python python3.12 --seed "$ROOT/.venv" || uv venv --seed "$ROOT/.venv"
  else
    python3 -m venv "$ROOT/.venv" || die "python3 -m venv failed"
  fi
fi
PY="$ROOT/.venv/bin/python"

say "Checking Python packages…"
"$PY" -c "from h3_bootstrap import ensure_python_requirements; ensure_python_requirements()" \
  || die "Python packages failed (see requirements.txt)"

if [[ ! -x "$ROOT/third_party/h3.c/h3" ]]; then
  say "h3 binary missing — building (needs submodule + Xcode CLT)…"
  git submodule update --init --recursive || true
  bash "$ROOT/scripts/build_h3.sh" || say "WARNING: h3 build failed. UI will start but generate will error."
fi

if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
  command -v npm >/dev/null 2>&1 || die "npm missing; run: cd web && npm install && npm run build"
  (cd "$ROOT/web" && npm install && npm run build) || die "Web UI build failed"
fi

say "Starting H3-WS on $UI_URL"
open "$UI_URL" 2>/dev/null || true
exec "$PY" -u "$ROOT/server.py" --host 127.0.0.1 --port "$PORT"
