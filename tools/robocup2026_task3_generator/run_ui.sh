#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/CompetitionTemplate}"
export GPSR_UI_LOG="${GPSR_UI_LOG:-$SCRIPT_DIR/gpsr-ui.log}"

conda run -n task3 athome-generator-gpsr-ui -d "$DATA_DIR" "$@"

