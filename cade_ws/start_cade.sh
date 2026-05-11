#!/bin/bash
# =============================================================
# CADE 一键启动脚本
# =============================================================
# 用法:
#   ./start_cade.sh              # 启动全部节点（需要 roscore）
#   ./start_cade.sh vision       # 只启动视觉节点
#   ./start_cade.sh brain        # 只启动大脑节点
#   ./start_cade.sh voice        # 只启动语音节点
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR"

# 检查 ROS 环境
if [ -z "$ROS_DISTRIBUTION" ]; then
    echo "Warning: ROS environment not sourced. Trying to source..."
    if [ -f /opt/ros/noetic/setup.bash ]; then
        source /opt/ros/noetic/setup.bash
    elif [ -f /opt/ros/melodic/setup.bash ]; then
        source /opt/ros/melodic/setup.bash
    else
        echo "Error: ROS not found. Please source setup.bash manually."
        exit 1
    fi
fi

# Source workspace
if [ -f "$WS_DIR/devel/setup.bash" ]; then
    source "$WS_DIR/devel/setup.bash"
else
    echo "Workspace not built. Run: cd $WS_DIR && catkin_make"
    echo "Attempting to run without workspace sourcing..."
fi

# 加载 .env
if [ -f "$WS_DIR/.env" ]; then
    export $(grep -v '^#' "$WS_DIR/.env" | xargs)
    echo "Loaded environment from .env"
elif [ -f "$SCRIPT_DIR/../.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
    echo "Loaded environment from parent .env"
fi

MODE=${1:-full}

case $MODE in
    voice)
        echo "Starting Voice Layer (ASR + TTS)..."
        roslaunch cade_voice voice.launch
        ;;
    vision)
        echo "Starting Vision Node..."
        rosrun cade_vision open_vision_node.py \
            --model "${CADE_YOLO_MODEL:-yolo11x-seg.pt}" \
            --device "${CADE_YOLO_DEVICE:-cuda}" \
            --conf "${CADE_YOLO_CONF:-0.25}" \
            --serial-number "${CADE_REALSENSE_SERIAL:-333422301212}"
        ;;
    brain)
        echo "Starting Brain Node..."
        rosrun cade_brain brain_node.py
        ;;
    full|*)
        echo "Starting CADE Full System..."
        roslaunch cade_ws cade_full.launch
        ;;
esac
