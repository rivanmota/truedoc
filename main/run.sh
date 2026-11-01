#!/usr/bin/env bash
# Builds and runs the detector API + UI on Docker.
#
# Usage:
#   ./run.sh          build + run in the foreground (Ctrl-C to stop)
#   ./run.sh -d        build + run in the background
#   ./run.sh down      stop and remove the containers
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker isn't installed or isn't on PATH. Install Docker Desktop first: https://docs.docker.com/get-docker/" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose (v2) isn't available. Update Docker Desktop, or install the compose plugin." >&2
  exit 1
fi

if [ "${1:-}" = "down" ]; then
  docker compose down
  exit 0
fi

echo "Building and starting the detector API (port 8008) and UI (port 3000) ..."
docker compose up --build "$@"
