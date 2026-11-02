#!/usr/bin/env bash
# Sets up and runs the detector API + UI natively (no Docker) for local development.
#
# Usage:
#   ./init.sh          set up (if needed) and run the API (reload) + UI (dev) in the foreground
#
# Ctrl-C stops both the API and the UI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 isn't installed or isn't on PATH. Install Python 3.12+ first: https://www.python.org/downloads/" >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "node/npm aren't installed or aren't on PATH. Install Node 22+ first: https://nodejs.org/" >&2
  exit 1
fi

# opencv-python-headless needs libgl1/libglib2.0-0 at import time; install them
# on Debian/Ubuntu-based systems if they're missing (mirrors main/Dockerfile).
if command -v apt-get >/dev/null 2>&1; then
  if ! python3 -c "import ctypes; ctypes.CDLL('libGL.so.1')" >/dev/null 2>&1; then
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
fi

if [ ! -d .venv ]; then
  echo "Creating Python virtualenv at .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "pip isn't available inside .venv. Re-create the venv or install pip manually." >&2
  exit 1
fi

echo "Installing Python dependencies ..."
python3 -m pip install -q -r requirements.txt

if [ ! -d web/node_modules ]; then
  echo "Installing UI dependencies ..."
  (cd web && npm install)
fi

echo "Starting the detector API (port 8008) ..."
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8008 &
API_PID=$!

cleanup() {
  if kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting the UI (port 3000) ..."
echo "API:  http://localhost:8008"
echo "UI:   http://localhost:3000"
echo "Press Ctrl-C to stop both."
(cd web && npm run dev)
