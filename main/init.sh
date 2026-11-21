#!/usr/bin/env bash
# Sets up and runs the detector API + UI natively (no Docker, no network once installed).
#
# Usage:
#   ./init.sh                    set up (if needed) and run the API (reload) + UI (dev) in the foreground
#   ./init.sh --api-only         run only the API (no Node / npm needed)
#   ./init.sh --check            only report whether this machine is ready (touches no network)
#   ./init.sh --offline          never touch the network; fail with a clear message if something is missing
#   ./init.sh --prepare-offline  on a machine WITH internet, download everything an offline machine needs
#
# Installing is skipped when the dependencies are already present, so a machine
# without internet works as long as .venv (and web/node_modules) are installed.
# If a ./wheelhouse directory exists (see --prepare-offline), Python packages are
# installed from it with no network access.
#
# Ctrl-C stops both the API and the UI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

API_ONLY=0
CHECK_ONLY=0
OFFLINE=0
PREPARE=0
for arg in "$@"; do
  case "$arg" in
    --api-only) API_ONLY=1 ;;
    --check) CHECK_ONLY=1 ;;
    --offline) OFFLINE=1 ;;
    --prepare-offline) PREPARE=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see ./init.sh --help)" >&2; exit 2 ;;
  esac
done

# Next.js phones home for anonymous telemetry; keep it off so nothing leaves the machine.
export NEXT_TELEMETRY_DISABLED=1

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 isn't installed or isn't on PATH. Install Python 3.12+ first: https://www.python.org/downloads/" >&2
  exit 1
fi

if [ "$API_ONLY" -eq 0 ] && { ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; }; then
  echo "node/npm aren't installed or aren't on PATH. Install Node 22+ (https://nodejs.org/), or run ./init.sh --api-only." >&2
  exit 1
fi

CHECK_ARGS=()
[ "$API_ONLY" -eq 1 ] && CHECK_ARGS+=(--backend)

if [ "$CHECK_ONLY" -eq 1 ]; then
  # Read-only: no venv creation, no installs.
  CHECK_PY=.venv/bin/python
  [ -x "$CHECK_PY" ] || CHECK_PY=python3
  exec "$CHECK_PY" check_env.py "${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}"
fi

if [ ! -d .venv ]; then
  echo "Creating Python virtualenv at .venv ..."
  python3 -m venv .venv
fi
PY=.venv/bin/python

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "pip isn't available inside .venv. Re-create the venv or install pip manually." >&2
  exit 1
fi

if [ "$PREPARE" -eq 1 ]; then
  echo "Downloading Python wheels into ./wheelhouse (needs internet) ..."
  "$PY" -m pip download -q -r requirements.txt -d wheelhouse
  if [ "$API_ONLY" -eq 0 ]; then
    echo "Installing UI dependencies into web/node_modules (needs internet) ..."
    (cd web && npm ci)
  fi
  echo "Done. Copy this directory, including ./wheelhouse and web/node_modules, to the offline machine"
  echo "and run ./init.sh --offline there. Wheels are platform-specific: use the same OS, CPU and Python."
  exit 0
fi

backend_ready() { "$PY" check_env.py --backend >/dev/null 2>&1; }

if ! backend_ready; then
  if [ -d wheelhouse ]; then
    echo "Installing Python dependencies from ./wheelhouse (no network) ..."
    "$PY" -m pip install -q --no-index --find-links wheelhouse -r requirements.txt
  elif [ "$OFFLINE" -eq 1 ]; then
    echo "Python dependencies are missing and --offline was given with no ./wheelhouse to install from." >&2
    "$PY" check_env.py --backend >&2 || true
    exit 1
  else
    # opencv-python-headless needs libgl1/libglib2.0-0 at import time; install them
    # on Debian/Ubuntu-based systems if they're missing (mirrors main/Dockerfile).
    if command -v apt-get >/dev/null 2>&1 \
        && ! "$PY" -c "import ctypes; ctypes.CDLL('libGL.so.1')" >/dev/null 2>&1; then
      echo "Installing system dependencies (libgl1, libglib2.0-0) ..."
      SUDO=""
      if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
          SUDO="sudo"
        else
          echo "libgl1/libglib2.0-0 appear to be missing and 'sudo' isn't available to install them as non-root. Install them manually: apt-get install -y libgl1 libglib2.0-0" >&2
          exit 1
        fi
      fi
      $SUDO apt-get update
      $SUDO apt-get install -y --no-install-recommends libgl1 libglib2.0-0
    fi
    echo "Installing Python dependencies (needs internet) ..."
    "$PY" -m pip install -q -r requirements.txt
  fi
else
  echo "Python dependencies already installed; skipping install."
fi

if [ "$API_ONLY" -eq 0 ] && [ ! -d web/node_modules ]; then
  if [ "$OFFLINE" -eq 1 ]; then
    echo "web/node_modules is missing and --offline was given. Copy a prepared web/node_modules here (see --prepare-offline), or use ./init.sh --api-only." >&2
    exit 1
  fi
  echo "Installing UI dependencies (needs internet) ..."
  (cd web && npm install)
fi

# Final gate: stops here with a readable list if anything (e.g. an OpenCV build without
# the face detector) is still wrong, instead of failing or silently degrading later.
"$PY" check_env.py "${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}"

echo "Starting the detector API (port 8008) ..."
"$PY" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8008 &
API_PID=$!

cleanup() {
  if kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "API:  http://localhost:8008"
if [ "$API_ONLY" -eq 1 ]; then
  echo "Press Ctrl-C to stop."
  wait "$API_PID"
else
  echo "Starting the UI (port 3000) ..."
  echo "UI:   http://localhost:3000"
  echo "Press Ctrl-C to stop both."
  (cd web && npm run dev)
fi
