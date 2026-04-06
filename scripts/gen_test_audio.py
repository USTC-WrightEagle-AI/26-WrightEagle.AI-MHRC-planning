#!/usr/bin/env python3
"""
批量生成标准测试音频集

功能：
1. 使用 TTS 引擎生成11个标准指令的语音
2. 在每个音频首尾拼接0.5秒静音以优化VAD识别
3. 保存为 test_audio/{idx}.wav 格式
"""

import os
import sys
import numpy as np
import soundfile as sf
import time

# 添加项目路径以便导入 tts_node 中的函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/asr_tts/scripts'))

try:
    import sherpa_onnx
    print("✅ sherpa_onnx 导入成功")
except ImportError:
    print("❌ 请安装 sherpa_onnx: pip install sherpa-onnx")
    sys.exit(1)

# 测试指令清单
TEST_INSTRUCTIONS = [
    "你好",
    "你叫什么名字",
    "你能做什么",
    "去厨房",
    "回到起点",
    "帮我找苹果",
    "找到水杯",
    "把苹果拿到桌子上",
    "我渴了，帮我拿瓶水",
    "今天天气怎么样",
    "给我讲个笑话"
]

def create_tts():
    """创建 TTS 引擎（复用 tts_node.py 中的配置）"""
    # 模型文件路径（相对于项目根目录）
    project_root = os.path.join(os.path.dirname(__file__), '..')
    model_dir = os.path.join(project_root, "src/asr_tts/models/vits-zh-aishell3")

    model_path = os.path.join(model_dir, "vits-aishell3.int8.onnx")
    tokens_path = os.path.join(model_dir, "tokens.txt")
    lexicon_path = os.path.join(model_dir, "lexicon.txt")

    # 检查文件是否存在
    for path in [model_path, tokens_path, lexicon_path]:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            print(f"   请确保已下载 VITS 模型文件到 {model_dir}")
            sys.exit(1)

    # 基础配置：推理引擎设置
    model_config = sherpa_onnx.OfflineTtsModelConfig(
        provider="cpu",
        num_threads=1,
        debug=False
    )

    # 配置 VITS 模型
    model_config.vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=model_path,
        tokens=tokens_path,
        lexicon=lexicon_path,
        noise_scale=0.667,
        noise_scale_w=0.8,
    )

    tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)

    if not tts_config.validate():
        raise ValueError("TTS 配置校验失败，请检查模型文件路径是否正确。")

    return sherpa_onnx.OfflineTts(tts_config)

def generate_speech(tts, text, sid=0, speed=1.0):
    """生成语音

    Args:
        tts: TTS引擎实例
        text: 要转换的文本
        sid: 说话人ID (0=清晰女声)
        speed: 语速

    Returns:
        audio: 生成的音频数据
        sample_rate: 采样率
    """
    start = time.time()
    audio = tts.generate(text, sid=sid, speed=speed)
    end = time.time()

    if len(audio.samples) == 0:
        raise ValueError("Error in generating audio")

    elapsed_seconds = end - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    print(f"  生成时间: {elapsed_seconds:.3f}s")
    print(f"  音频时长: {audio_duration:.3f}s")
    print(f"  RTF: {real_time_factor:.3f}")

    # 转换为numpy数组并返回
    return np.array(audio.samples, dtype=np.float32), audio.sample_rate

def add_silence(audio, sample_rate, silence_duration=0.5):
    """在音频首尾添加静音

    Args:
        audio: 原始音频数据（列表或numpy数组）
        sample_rate: 采样率
        silence_duration: 静音时长（秒）

    Returns:
        添加静音后的音频数据（numpy数组）
    """
    # 确保audio是numpy数组
    if isinstance(audio, list):
        audio = np.array(audio, dtype=np.float32)

    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros(silence_samples, dtype=audio.dtype)

    # 在首尾添加静音
    result = np.concatenate([silence, audio, silence])
    return result

def main():
    """主函数：生成所有测试音频"""
    print("=" * 60)
    print("🤖 CADE 静默测试音频生成器")
    print("=" * 60)

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '../test_audio')
    os.makedirs(output_dir, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print(f"生成 {len(TEST_INSTRUCTIONS)} 个测试音频...")

    # 创建 TTS 引擎
    print("\n初始化 TTS 引擎...")
    try:
        tts = create_tts()
        print("✅ TTS 引擎初始化成功")
    except Exception as e:
        print(f"❌ TTS 引擎初始化失败: {e}")
        sys.exit(1)

    # 生成每个音频
    for i, text in enumerate(TEST_INSTRUCTIONS, 1):
        print(f"\n[{i}/{len(TEST_INSTRUCTIONS)}] 生成: \"{text}\"")

        try:
            # 生成语音
            audio, sample_rate = generate_speech(tts, text, sid=0, speed=1.0)

            # 添加首尾静音
            audio_with_silence = add_silence(audio, sample_rate, silence_duration=0.5)

            # 保存文件
            output_path = os.path.join(output_dir, f"{i}.wav")
            sf.write(
                output_path,
                audio_with_silence,
                samplerate=sample_rate,
                subtype="PCM_16",
            )

            # 计算统计信息
            original_duration = len(audio) / sample_rate
            final_duration = len(audio_with_silence) / sample_rate
            print(f"✅ 保存到: {output_path}")
            print(f"   原始时长: {original_duration:.2f}s")
            print(f"   最终时长: {final_duration:.2f}s (+1.0s 静音)")

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            continue

    # 生成完成
    print("\n" + "=" * 60)
    print("🎉 所有测试音频生成完成！")
    print(f"总计: {len(TEST_INSTRUCTIONS)} 个音频文件")
    print(f"位置: {output_dir}")
    print("\n文件名对应关系:")
    for i, text in enumerate(TEST_INSTRUCTIONS, 1):
        print(f"  {i}.wav -> \"{text}\"")
    print("=" * 60)

if __name__ == "__main__":
    main()