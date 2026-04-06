# ！/home/tiger/miniforge3/envs/ros/bin/python
import soundfile as sf
import sherpa_onnx
import sounddevice as sd
import time
import argparse
import rospy
from std_msgs.msg import String
import rospkg
import os
import sys
pkg_path = rospkg.RosPack().get_path('asr_tts')


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # TTS 相关参数
# TTS 核心模型文件
    parser.add_argument(
        "--tts-model",
        type=str,
        default=os.path.join(pkg_path, "models/vits-zh-aishell3/vits-aishell3.int8.onnx"),  # 默认指向新模型
        help="Path to the TTS model file (.onnx)",
    )

    # 词表文件（VITS/Kitten 通用）
    parser.add_argument(
        "--tts-tokens",
        type=str,
        default=os.path.join(pkg_path, "models/vits-zh-aishell3/tokens.txt"),
        help="Path to the tokens.txt file",
    )

    # 发音词典（VITS 需要，Kitten 不需要但可留空）
    parser.add_argument(
        "--tts-lexicon",
        type=str,
        default=os.path.join(pkg_path, "models/vits-zh-aishell3/lexicon.txt"),
        help="Path to the lexicon.txt file (optional for some models)",
    )

    # 专用数据目录（Kitten 用这个，VITS 通常用 lexicon）
    parser.add_argument(
        "--tts-data-dir",
        type=str,
        default="",
        help="Path to the model data directory (e.g., espeak-ng-data)",
    )

    # 声音定义文件（Kitten 专用）
    parser.add_argument(
        "--tts-voices",
        type=str,
        default="",
        help="Path to the voices.bin file (specific to Kitten models)",
    )

    parser.add_argument(
        "--sid",
        type=int,
        default=0,
        help="Speaker ID for TTS",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="TTS speech speed",
    )

    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Text to convert to speech",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(pkg_path, "output.wav"),
        help="Path to save generated audio",
    )

    parser.add_argument(
        "--play",
        action="store_false",
        default=True,
        help="Play the generated audio (default: True)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Inference provider: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads",
    )

    return parser.parse_known_args()


def create_tts(args):
    """通用的 TTS 引擎创建函数"""
    model_path = args.tts_model.lower()

    # 基础配置：推理引擎设置
    model_config = sherpa_onnx.OfflineTtsModelConfig(
        provider=args.provider,
        num_threads=args.num_threads,
        debug=False
    )

    # 根据路径或参数判定模型架构
    if "vits" in model_path:
        model_config.vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=args.tts_model,
            tokens=args.tts_tokens,
            lexicon=args.tts_lexicon,
            noise_scale=0.667,
            noise_scale_w=0.8,
        )
    elif "kitten" in model_path:
        model_config.kitten = sherpa_onnx.OfflineTtsKittenModelConfig(
            model=args.tts_model,
            voices=args.tts_voices,
            tokens=args.tts_tokens,
            data_dir=args.tts_data_dir,
        )
    else:
        # 如果是其他模型（如 Matcha），可以继续扩展 elif
        raise ValueError(f"Unsupported model architecture in path: {args.tts_model}")

    tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)

    if not tts_config.validate():
        raise ValueError("TTS 配置校验失败，请检查模型文件路径是否正确。")

    return sherpa_onnx.OfflineTts(tts_config)


def play_audio(audio, sample_rate, device=None, wav_path=None):
    """播放音频，优先使用系统播放器 (paplay)"""
    import subprocess

    # 如果有保存的 wav 文件，直接用系统播放器
    if wav_path and os.path.exists(wav_path):
        try:
            print(f"使用系统播放器: paplay {wav_path}")
            subprocess.run(['paplay', wav_path], check=True)
            print("播放完成")
            return
        except Exception as e:
            print(f"paplay 失败: {e}，尝试用 sounddevice")

    # 回退到 sounddevice
    import numpy as np
    try:
        if device is not None:
            device_info = sd.query_devices(device)
            device_sample_rate = int(device_info['default_samplerate'])

            if sample_rate != device_sample_rate:
                print(f"重采样: {sample_rate} Hz -> {device_sample_rate} Hz")
                ratio = device_sample_rate / sample_rate
                new_length = int(len(audio) * ratio)
                old_indices = np.arange(len(audio))
                new_indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(new_indices, old_indices, audio)
                sample_rate = device_sample_rate

            sd.play(audio, sample_rate, device=device)
        else:
            sd.play(audio, sample_rate)
        sd.wait()
        print("播放完成")
    except Exception as e:
        print(f"Error playing audio: {e}")


def generate_speech(tts, text, sid=0, speed=1.0):
    """生成语音

    Args:
        tts: TTS引擎实例
        text: 要转换的文本
        sid: 说话人ID
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

    print(f"TTS generation time: {elapsed_seconds:.3f}s")
    print(f"Audio duration: {audio_duration:.3f}s")
    print(f"RTF: {real_time_factor:.3f}")

    return audio.samples, audio.sample_rate


class TTSNode:
    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.args, _ = get_args()

        # 1. 从参数服务器获取“人定的”设备真名
        # 默认值可以设为 "default"，但建议在 launch 中明确指定
        target_name = rospy.get_param("~device_name", "default")

        import sounddevice as sd
        self.output_device = None

        devices = sd.query_devices()
        rospy.loginfo(f"🔍 [TTS] 正在尝试锁定输出设备: \"{target_name}\"")

        # 2. 确定性匹配逻辑
        for i, d in enumerate(devices):
            # 注意：我们要找的是有输出通道（max_output_channels > 0）的设备
            if target_name in d['name'] and d['max_output_channels'] > 0:
                self.output_device = i
                rospy.loginfo(f"✅ [TTS] 成功锁定输出设备: [{i}] {d['name']}")
                break

        # 3. 强制校验：找不到就报错，不准乱猜
        if self.output_device is None:
            rospy.logerr(f"❌ [TTS] 无法找到指定的输出设备: \"{target_name}\"")
            rospy.logerr("请运行 'python3 -m sounddevice' 确认设备名，并在 launch 中修改。")
            # 根据你的容错需求，可以选择 sys.exit(1) 或者设为默认值
            sys.exit(1)

        rospy.loginfo("Initializing TTS engine...")
        self.tts = create_tts(self.args)
        self.tts_subscription_ = rospy.Subscriber('tts', String, self.TTS, queue_size=10)
        rospy.loginfo("TTS Node is READY!")

    def TTS(self, msg: String):
        print(f"Generating speech for: '{msg.data}'")
        audio, sample_rate = generate_speech(self.tts, msg.data, self.args.sid, self.args.speed)
        # 保存音频
        sf.write(
            self.args.output,
            audio,
            samplerate=sample_rate,
            subtype="PCM_16",
        )
        print(f"Saved to {self.args.output}")

        # 播放音频
        if self.args.play:
            print("Playing audio...")
            play_audio(audio, sample_rate, device=self.output_device, wav_path=self.args.output)


def main():
    node = TTSNode("tts_node")
    rospy.spin()


if __name__ == "__main__":
    main()
