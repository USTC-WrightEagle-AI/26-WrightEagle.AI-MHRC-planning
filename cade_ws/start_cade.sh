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

load_env_defaults() {
    local env_file="$1"
    while IFS= read -r line || [ -n "$line" ]; do
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" == *"="* ]] || continue

        local key="${line%%=*}"
        local value="${line#*=}"
        key="$(printf '%s' "$key" | xargs)"

        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        if [ -z "${!key:-}" ] && [ -n "$value" ]; then
            export "$key=$value"
        fi
    done < "$env_file"
}

# 加载 .env 默认值；已存在的环境变量优先，例如外部注入的 CADE_CLOUD_API_KEY。
if [ -f "$WS_DIR/.env" ]; then
    load_env_defaults "$WS_DIR/.env"
    echo "Loaded environment from .env"
elif [ -f "$SCRIPT_DIR/../.env" ]; then
    load_env_defaults "$SCRIPT_DIR/../.env"
    echo "Loaded environment from parent .env"
fi

# 当前任务要求只走 Cloud API key 通道，暂不使用本地 Ollama。
export CADE_MODE=CLOUD

MODE=${1:-full}

case $MODE in
    voice)
        echo "Starting Voice Layer (ASR + TTS)..."
        roslaunch cade_voice voice.launch
        ;;
    vision)
        echo "Starting Vision Node..."
        VISION_ARGS=(
            --device "${CADE_YOLO_DEVICE:-cuda}"
            --conf "${CADE_YOLO_CONF:-0.25}"
            --serial-number "${CADE_REALSENSE_SERIAL:-333422301212}"
        )
        if [ -n "${CADE_YOLO_MODEL:-}" ]; then
            VISION_ARGS+=(--model "$CADE_YOLO_MODEL")
        fi
        rosrun cade_vision open_vision_node.py "${VISION_ARGS[@]}"
        ;;
    brain)
        echo "Starting Brain Node..."
        rosrun cade_brain brain_node.py
        ;;
    full|*)
        echo "Starting CADE Full System..."
        roslaunch "$WS_DIR/launch/cade_full.launch"
        ;;
esac
