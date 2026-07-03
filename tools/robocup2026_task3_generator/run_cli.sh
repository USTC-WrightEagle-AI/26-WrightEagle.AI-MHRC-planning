#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/CompetitionTemplate}"

if [[ -x "$SCRIPT_DIR/.venv-uv/bin/athome-generator" ]]; then
  "$SCRIPT_DIR/.venv-uv/bin/athome-generator" -d "$DATA_DIR" "$@"
else
  conda run -n task3 athome-generator -d "$DATA_DIR" "$@"
fi
