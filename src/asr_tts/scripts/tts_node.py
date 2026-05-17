#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Node - Text-to-Speech (VITS / Kokoro)
Subscribes to /tts topic, generates speech, and plays audio.
Supports both VITS and Kokoro architectures (auto-detected from model path).
"""

import subprocess
import soundfile as sf
import sherpa_onnx
import sounddevice as sd
import time
import argparse
import numpy as np
import os
import sys

import rospy
from std_msgs.msg import String
import rospkg


pkg_path = rospkg.RosPack().get_path('asr_tts')


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core TTS model
    parser.add_argument(
        "--tts-model",
        type=str,
        default=os.path.join(pkg_path, "models/kokoro-int8-en-v0_19/model.int8.onnx"),
        help="Path to the TTS model file (.onnx)",
    )

    # Token file (shared by VITS/Kokoro)
    parser.add_argument(
        "--tts-tokens",
        type=str,
        default=os.path.join(pkg_path, "models/kokoro-int8-en-v0_19/tokens.txt"),
        help="Path to the tokens.txt file",
    )

    # Lexicon (required by VITS, not needed for Kokoro)
    parser.add_argument(
        "--tts-lexicon",
        type=str,
        default="",
        help="Path to the lexicon.txt file (optional for some models)",
    )

    # Data directory (used by Kokoro for espeak-ng-data)
    parser.add_argument(
        "--tts-data-dir",
        type=str,
        default=os.path.join(pkg_path, "models/kokoro-int8-en-v0_19/espeak-ng-data"),
        help="Path to the model data directory (e.g., espeak-ng-data)",
    )

    # Voice definition file (Kokoro-specific)
    parser.add_argument(
        "--tts-voices",
        type=str,
        default=os.path.join(pkg_path, "models/kokoro-int8-en-v0_19/voices.bin"),
        help="Path to the voices.bin file (specific to Kokoro models)",
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
        default="cuda",
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
    """Create a TTS engine instance (supports VITS, Kitten, Kokoro)."""

    model_path = args.tts_model.lower()

    model_config = sherpa_onnx.OfflineTtsModelConfig(
        provider=args.provider,
        num_threads=args.num_threads,
        debug=False,
    )

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
    elif "kokoro" in model_path or "model.int8.onnx" in model_path:
        import os as _os
        _dict_dir = _os.path.join(_os.path.dirname(args.tts_data_dir), "") \
                   if args.tts_data_dir else ""
        model_config.kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=args.tts_model,
            voices=args.tts_voices,
            tokens=args.tts_tokens,
            data_dir=args.tts_data_dir,
            dict_dir=_dict_dir,
            length_scale=1.0,
        )
    else:
        pass
    tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)

    if not tts_config.validate():
        raise ValueError("Invalid TTS configuration. Check model file paths.")

    return sherpa_onnx.OfflineTts(tts_config)


def play_audio(audio, sample_rate, device=None, wav_path=None):
    """Play audio using sounddevice with specified device."""

    try:
        if device is not None:
            device_info = sd.query_devices(device)
            device_sample_rate = int(device_info['default_samplerate'])

            if sample_rate != device_sample_rate:
                rospy.loginfo(f"[TTS] 重采样: {sample_rate} Hz -> {device_sample_rate} Hz")
                ratio = device_sample_rate / sample_rate
                new_length = int(len(audio) * ratio)
                old_indices = np.arange(len(audio))
                new_indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(new_indices, old_indices, audio)
                sample_rate = device_sample_rate

            rospy.loginfo(f"[TTS] 播放设备: [{device}] {device_info['name']}")

            sd.play(audio, sample_rate, device=device)
        else:
            rospy.loginfo("[TTS] 使用默认播放设备")
            sd.play(audio, sample_rate)

        audio_duration = len(audio) / sample_rate
        rospy.loginfo(f"[TTS] 开始播放, 时长约 {audio_duration:.2f}s")
        sd.wait()
        rospy.loginfo("[TTS] 播放完成")
    except Exception as e:
        rospy.logerr(f"[TTS] 播放音频时出错: {e}")


def generate_speech(tts, text, sid=0, speed=1.0):
    """
    Generate speech from text.

    Returns:
        tuple: (audio_samples, sample_rate)
    """

    rospy.loginfo(f"[TTS] 开始生成语音: '{text[:50]}...'")
    start = time.time()
    audio = tts.generate(text, sid=sid, speed=speed)
    end = time.time()

    if len(audio.samples) == 0:
        raise ValueError("TTS 生成音频失败")

    elapsed_seconds = end - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    rospy.loginfo(f"[TTS] 生成耗时: {elapsed_seconds:.3f}s, 音频时长: {audio_duration:.3f}s, RTF: {real_time_factor:.3f}")

    return audio.samples, audio.sample_rate


class TTSNode:
    """ROS node for Text-to-Speech synthesis."""

    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.args, _ = get_args()

        rospy.loginfo("[TTS] 初始化 TTS 节点")
        rospy.loginfo(f"[TTS] 模型路径: {self.args.tts_model}")
        rospy.loginfo(f"[TTS] 参数: speed={self.args.speed}, sid={self.args.sid}, provider={self.args.provider}")

        target_name = rospy.get_param("~device_name", "default")
        self.output_device = None

        devices = sd.query_devices()
        rospy.loginfo(f"[TTS] 系统可用播放设备数: {len(devices)}")
        rospy.loginfo(f"[TTS] 正在查找输出设备: \"{target_name}\"")

        for i, d in enumerate(devices):
            if target_name in d['name'] and d['max_output_channels'] > 0:
                self.output_device = i
                rospy.loginfo(f"[TTS] 输出设备已锁定: [{i}] {d['name']}")
                break

        if self.output_device is None:
            rospy.logwarn(f"[TTS] 找不到 \"{target_name}\"，尝试 fallback 到 pulse")
            for i, d in enumerate(devices):
                if "pulse" in d['name'].lower() and d['max_output_channels'] > 0:
                    self.output_device = i
                    rospy.loginfo(f"[TTS] Fallback 成功: 使用 [{i}] {d['name']}")
                    break

        if self.output_device is None:
            rospy.logwarn(f"[TTS] pulse 不可用，尝试 fallback 到 default")
            default_idx = sd.default.device[1]
            if default_idx is not None and default_idx < len(devices):
                d = devices[int(default_idx)]
                if d['max_output_channels'] > 0:
                    self.output_device = int(default_idx)
                    rospy.loginfo(f"[TTS] Fallback 成功: 使用 default [{self.output_device}] {d['name']}")

        if self.output_device is None:
            rospy.logerr(f"[TTS] 找不到任何可用的输出设备")
            rospy.loginfo("[TTS] 可用播放设备列表:")
            for i, d in enumerate(devices):
                if d['max_output_channels'] > 0:
                    rospy.loginfo(f"[TTS]   [{i}] {d['name']}")
            sys.exit(1)

        rospy.loginfo("[TTS] 正在初始化 TTS 引擎...")
        self.tts = create_tts(self.args)
        rospy.loginfo("[TTS] TTS 引擎初始化完成")

        # 发布 TTS 播放状态, 供 ASR 节点用于回声消除
        self._pub_tts_status = rospy.Publisher("/tts/playing", String, queue_size=10, latch=True)

        self.tts_subscription_ = rospy.Subscriber('tts', String, self.TTS, queue_size=10)
        rospy.loginfo("[TTS] 已订阅 /tts 话题")
        rospy.loginfo("[TTS] 节点就绪, 等待文本输入!")

    def TTS(self, msg: String):
        """Callback: generate and play speech from received text."""
        rospy.loginfo(f"[TTS] 收到文本: '{msg.data}'")
        audio, sample_rate = generate_speech(self.tts, msg.data, self.args.sid, self.args.speed)

        sf.write(
            self.args.output,
            audio,
            samplerate=sample_rate,
            subtype="PCM_16",
        )
        rospy.loginfo(f"[TTS] 音频已保存到: {self.args.output}")

        if self.args.play:
            rospy.loginfo("[TTS] 开始播放音频")
            # 通知 ASR 节点暂停识别 (回声消除)
            self._pub_tts_status.publish(String(data="playing"))
            play_audio(
                audio, sample_rate,
                device=self.output_device,
                wav_path=self.args.output,
            )
            # 播放完成, 通知 ASR 恢复识别
            self._pub_tts_status.publish(String(data="idle"))
            rospy.loginfo("[TTS] 播放完成, ASR 已恢复")


def main():
    node = TTSNode("tts_node")
    rospy.spin()


if __name__ == "__main__":
    main()
