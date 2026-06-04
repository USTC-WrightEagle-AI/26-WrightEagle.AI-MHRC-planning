#!/bin/bash
# CADE 虚拟音频回环系统设置脚本
# 创建虚拟扬声器 CADE_Speaker 和虚拟麦克风 CADE_Mic，形成对录链路

set -e

echo "============================================================"
echo "🔧 CADE 虚拟音频回环系统设置"
echo "============================================================"

# 检查 pulseaudio/pulsewire 是否运行
if ! pactl info > /dev/null 2>&1; then
    echo "❌ PulseAudio/PipeWire 未运行，请先启动音频服务"
    echo "   尝试运行: systemctl --user start pipewire pulseaudio"
    exit 1
fi

echo "✅ 音频服务正在运行"

# 清理旧的虚拟设备模块
echo "清理旧的虚拟设备模块..."

# 卸载可能已存在的 CADE_Speaker 相关模块
CADE_SPEAKER_MODULE=$(pactl list short modules | grep "sink_name=CADE_Speaker" | cut -f1)
if [ -n "$CADE_SPEAKER_MODULE" ]; then
    echo "卸载现有 CADE_Speaker 模块 (ID: $CADE_SPEAKER_MODULE)..."
    pactl unload-module "$CADE_SPEAKER_MODULE" 2>/dev/null || true
fi

# 卸载可能已存在的 CADE_Mic 相关模块
CADE_MIC_MODULE=$(pactl list short modules | grep "source_name=CADE_Mic" | cut -f1)
if [ -n "$CADE_MIC_MODULE" ]; then
    echo "卸载现有 CADE_Mic 模块 (ID: $CADE_MIC_MODULE)..."
    pactl unload-module "$CADE_MIC_MODULE" 2>/dev/null || true
fi

# 也清理旧的 CADE_Test 模块（兼容旧版本）
CADE_TEST_MODULE=$(pactl list short modules | grep "sink_name=CADE_Test" | cut -f1)
if [ -n "$CADE_TEST_MODULE" ]; then
    echo "卸载旧版 CADE_Test 模块 (ID: $CADE_TEST_MODULE)..."
    pactl unload-module "$CADE_TEST_MODULE" 2>/dev/null || true
fi

sleep 1

# 创建虚拟扬声器 (Sink) - 播放端
echo "创建虚拟扬声器 (Sink): CADE_Speaker..."
SPEAKER_MODULE_ID=$(pactl load-module module-null-sink \
    sink_name=CADE_Speaker \
    sink_properties=device.description="CADE_Speaker")

if [ -z "$SPEAKER_MODULE_ID" ]; then
    echo "❌ 创建虚拟扬声器 CADE_Speaker 失败"
    exit 1
fi

echo "✅ 虚拟扬声器创建成功 (模块 ID: $SPEAKER_MODULE_ID)"
echo "   名称: CADE_Speaker"
echo "   描述: CADE_Speaker"

# 创建虚拟麦克风 (Source) - 录音端
# 使用 module-remap-source 将 CADE_Speaker.monitor 映射为独立输入设备
echo "创建虚拟麦克风 (Source): CADE_Mic..."
MIC_MODULE_ID=$(pactl load-module module-remap-source \
    source_name=CADE_Mic \
    source_properties=device.description="CADE_Mic" \
    master=CADE_Speaker.monitor)

if [ -z "$MIC_MODULE_ID" ]; then
    echo "❌ 创建虚拟麦克风 CADE_Mic 失败"
    echo "   尝试卸载已创建的扬声器模块..."
    pactl unload-module "$SPEAKER_MODULE_ID" 2>/dev/null || true
    exit 1
fi

echo "✅ 虚拟麦克风创建成功 (模块 ID: $MIC_MODULE_ID)"
echo "   名称: CADE_Mic"
echo "   描述: CADE_Mic"
echo "   Master: CADE_Speaker.monitor"

sleep 1

# 验证输出
echo ""
echo "📋 当前音频 Sink 列表:"
pactl list short sinks

echo ""
echo "📋 当前音频 Source 列表:"
pactl list short sources

echo ""
echo "============================================================"
echo "🔌 虚拟音频回环链路已建立"
echo "============================================================"
echo ""
echo "链路结构:"
echo "  CADE_Speaker (虚拟扬声器) ← 接收测试脚本的音频流"
echo "  CADE_Mic (虚拟麦克风)     ← 监听 CADE_Speaker.monitor"
echo ""
echo "ASR 节点将搜索设备名包含 'CADE' 且具有输入通道的设备。"
echo "CADE_Mic 应被识别为独立输入设备 (max_input_channels > 0)。"
echo ""
echo "============================================================"
echo "🎛️  配置指南"
echo "============================================================"
echo ""
echo "使用 pavucontrol 配置:"
echo ""
echo "1. 打开 pavucontrol (如果没有安装: sudo dnf install pavucontrol)"
echo ""
echo "2. 切换到 'Recording' 选项卡"
echo ""
echo "3. 找到 ASR 节点 (asr_node.py) 的进程"
echo ""
echo "4. 将其输入源改为 'CADE_Mic'"
echo ""
echo "5. 切换到 'Playback' 选项卡"
echo ""
echo "6. 确保 TTS 节点 (tts_node.py) 的输出设备保持为物理扬声器"
echo ""
echo "7. 切换到 'Output Devices' 选项卡，确认能看到 'CADE_Speaker'"
echo "8. 切换到 'Input Devices' 选项卡，确认能看到 'CADE_Mic'"
echo ""
echo "============================================================"
echo "💾 持久化设置"
echo "============================================================"
echo ""
echo "虚拟设备在系统重启后会失效，如需持久化:"
echo ""
echo "1. 将以下行添加到 ~/.config/pulse/default.pa:"
echo ""
echo "   # CADE 虚拟音频回环系统"
echo "   load-module module-null-sink sink_name=CADE_Speaker sink_properties=device.description=CADE_Speaker"
echo "   load-module module-remap-source source_name=CADE_Mic source_properties=device.description=CADE_Mic master=CADE_Speaker.monitor"
echo ""
echo "2. 重启音频服务:"
echo "   systemctl --user restart pipewire pipewire-pulse"
echo "   或"
echo "   pulseaudio -k && pulseaudio --start"
echo ""
echo "============================================================"
echo "✅ 设置完成！现在可以运行 test_me.sh 进行测试"
echo "============================================================"