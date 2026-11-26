#!/usr/bin/env bash
# Wrapper around `blenderproc run` that uses the project venv and, if present,
# an unbundled Blender copy (macOS blocks BlenderProc from installing its pip
# packages inside a signed Blender.app bundle).
#   ./run_blenderproc.sh src/deform.py --seed 0
set -e
cd "$(dirname "$0")"
BLENDER_DIR="${BLENDER_DIR:-$HOME/blender/blender-4.2.1-unbundled}"
if [ -d "$BLENDER_DIR" ]; then
  exec .venv/bin/blenderproc run --custom-blender-path "$BLENDER_DIR" "$@"
else
  exec .venv/bin/blenderproc run "$@"
fi
