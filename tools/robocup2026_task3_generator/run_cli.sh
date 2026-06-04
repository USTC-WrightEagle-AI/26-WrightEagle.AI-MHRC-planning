#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/CompetitionTemplate}"

conda run -n task3 athome-generator -d "$DATA_DIR" "$@"

