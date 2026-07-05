#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-task1}"
REALSENSE_CAMERA_MODEL="${REALSENSE_CAMERA_MODEL:-D455}"
REALSENSE_SERIAL="${REALSENSE455_SERIAL:-${REALSENSE_SERIAL:-${REALSENSE515_SERIAL:-${CADE_REALSENSE_SERIAL:-}}}}"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# start_all.sh sets PYTHONNOUSERSITE=1 for catkin node stability, but the
# Jetson OpenCV module used by ultralytics is installed in ~/.local.
unset PYTHONNOUSERSITE
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cd "$SCRIPT_DIR"
exec python vision_cache_daemon.py \
  --camera-model "$REALSENSE_CAMERA_MODEL" \
  --serial-number "$REALSENSE_SERIAL" \
  "$@"
