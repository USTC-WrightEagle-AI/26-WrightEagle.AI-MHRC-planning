#!/bin/bash
# CADE 静默测试分发脚本
# 交互式选择测试指令，通过虚拟音频线缆静默注入

set -e

# 脚本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="${SCRIPT_DIR}/test_audio"

# 测试指令列表
declare -A TEST_INSTRUCTIONS=(
    [1]="你好"
    [2]="你叫什么名字"
    [3]="你能做什么"
    [4]="去厨房"
    [5]="回到起点"
    [6]="帮我找苹果"
    [7]="找到水杯"
    [8]="把苹果拿到桌子上"
    [9]="我渴了，帮我拿瓶水"
    [10]="今天天气怎么样"
    [11]="给我讲个笑话"
)

# 检查音频文件目录
if [ ! -d "$AUDIO_DIR" ]; then
    echo "❌ 音频目录不存在: $AUDIO_DIR"
    echo "   请先运行: python scripts/gen_test_audio.py"
    exit 1
fi

# 检查 paplay 命令
if ! command -v paplay >/dev/null 2>&1; then
    echo "❌ paplay 命令未找到，请安装 pulseaudio-utils:"
    echo "   sudo apt install pulseaudio-utils"
    exit 1
fi

# 检查虚拟 Sink 是否存在
if ! pactl list short sinks | grep -q "CADE_Speaker"; then
    echo "⚠️  未找到 CADE_Speaker 虚拟 Sink"
    echo "   请先运行: bash scripts/setup_virtual_audio.sh"
    echo ""
    echo "是否要现在设置虚拟音频接口？[y/N]"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        bash scripts/setup_virtual_audio.sh
        echo ""
        echo "请按照上面的指南配置 pavucontrol，然后重新运行此脚本"
        exit 0
    else
        echo "❌ 需要虚拟音频接口才能继续测试"
        exit 1
    fi
fi

# 显示菜单
show_menu() {
    echo "============================================================"
    echo "🎤 CADE 静默测试系统"
    echo "============================================================"
    echo ""
    echo "选择要模拟的语音指令:"
    echo ""

    for i in {1..11}; do
        printf "  %2d. %s\n" "$i" "${TEST_INSTRUCTIONS[$i]}"
    done

    echo ""
    echo "  0. 退出"
    echo ""
    echo "============================================================"
}

# 播放音频到虚拟 Sink
play_to_virtual_sink() {
    local idx="$1"
    local text="$2"
    local audio_file="${AUDIO_DIR}/${idx}.wav"

    if [ ! -f "$audio_file" ]; then
        echo "❌ 音频文件不存在: $audio_file"
        echo "   请先运行: python scripts/gen_test_audio.py"
        return 1
    fi

    echo ""
    echo "🎧 正在模拟: \"$text\""
    echo "   📁 文件: $audio_file"
    echo "   🎯 目标: CADE_Speaker (虚拟扬声器)"
    echo "   🔗 链路: 音频正在注入 CADE_Speaker -> 转发至 CADE_Mic"
    echo ""

    # 播放到虚拟 Sink
    if paplay -d CADE_Speaker "$audio_file" 2>/dev/null; then
        echo "✅ 音频已静默注入虚拟线缆"
        echo ""
        echo "💡 ASR 节点应该能接收到此音频并转译文字"
        echo "   检查 ROS 终端查看 ASR 输出"
        return 0
    else
        echo "❌ 播放失败，检查音频设备设置"
        echo "   请确保:"
        echo "   1. PulseAudio 正在运行"
        echo "   2. CADE_Speaker 虚拟 Sink 存在"
        echo "   3. 有权限访问音频设备"
        return 1
    fi
}

# 主循环
main() {
    while true; do
        show_menu

        echo -n "请输入编号 [0-11]: "
        read -r choice

        case "$choice" in
            0)
                echo ""
                echo "👋 退出测试系统"
                echo ""
                exit 0
                ;;
            [1-9]|1[0-1])
                # 验证选择有效
                if [[ -n "${TEST_INSTRUCTIONS[$choice]}" ]]; then
                    text="${TEST_INSTRUCTIONS[$choice]}"
                    play_to_virtual_sink "$choice" "$text"

                    echo -n "按 Enter 继续，或输入 q 退出: "
                    read -r continue_choice
                    if [[ "$continue_choice" == "q" ]]; then
                        echo "👋 退出测试系统"
                        exit 0
                    fi
                else
                    echo "❌ 无效的选择"
                fi
                ;;
            *)
                echo "❌ 无效输入，请输入 0-11 之间的数字"
                ;;
        esac

        echo ""
    done
}

# 运行主函数
main