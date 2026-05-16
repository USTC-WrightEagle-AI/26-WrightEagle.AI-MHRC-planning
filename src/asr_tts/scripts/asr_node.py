#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR Node - VAD + Offline (Non-streaming) ASR

订阅 /audio/raw (Float32MultiArray), 使用 Silero VAD +
SenseVoice/Whisper/Paraformer/Transducer 模型进行离线语音识别。
识别结果发布到 /asr topic (std_msgs/String)。

回声消除: 订阅 /tts/playing 话题, TTS 播放期间及冷却期内丢弃识别结果。
"""

import re
import wave
import time
import argparse
import os
import queue
from pathlib import Path
from typing import Optional, List

import numpy as np
import sherpa_onnx

import rospy
from std_msgs.msg import Float32MultiArray, String
import rospkg

pkg_path = rospkg.RosPack().get_path('asr_tts')

WHISPER_MODEL_DIR = os.path.join(pkg_path, "models/sherpa-onnx-whisper-small.en")

TTS_COOLDOWN = 3.0

NON_SPEECH_PATTERN = re.compile(r'^\([^)]*\)$|^\[.*\]$')


# ============================================================
# WAV 保存工具
# ============================================================

def save_wav(filepath, audio_data, sample_rate: int = 16000):
    if isinstance(audio_data, list):
        audio_data = np.array(audio_data)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    with wave.open(str(filepath), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_int16.tobytes())


def save_wav_concat(filepath, samples_list, sample_rate: int = 16000):
    all_audio = np.concatenate([
        np.array(s) if isinstance(s, list) else s for s in samples_list
    ])
    save_wav(filepath, all_audio, sample_rate)


# ============================================================
# 命令行参数
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- VAD ---
    parser.add_argument("--silero-vad-model", type=str,
                        default=os.path.join(pkg_path, "models/silero_vad.onnx"),
                        help="Path to silero_vad.onnx")

    # --- 通用 ---
    parser.add_argument("--tokens", type=str,
                        default=os.path.join(WHISPER_MODEL_DIR, "small.en-tokens.txt"),
                        help="Path to tokens.txt")
    parser.add_argument("--num-threads", type=int, default=2,
                        help="Number of threads for neural network computation")
    parser.add_argument("--provider", type=str, default="cuda",
                        choices=["cpu", "cuda", "coreml"],
                        help="Inference provider")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate of the feature extractor")
    parser.add_argument("--feature-dim", type=int, default=80,
                        help="Feature dimension. Must match the model expectation")
    parser.add_argument("--blank-penalty", type=float, default=0.0,
                        help="Penalty applied on blank symbol during decoding")
    parser.add_argument("--decoding-method", type=str, default="greedy_search",
                        help="Valid values: greedy_search, modified_beam_search")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Show debug messages when loading models")
    parser.add_argument("--hr-lexicon", type=str, default="",
                        help="Lexicon.txt for homophone replacer (optional)")
    parser.add_argument("--hr-rule-fsts", type=str, default="",
                        help="Replace.fst for homophone replacer (optional)")
    parser.add_argument("--save-audio", type=str,
                        default=os.path.join(pkg_path, "recordings"),
                        help="Folder path to save recorded audio segments")

    # --- Transducer ---
    parser.add_argument("--encoder", default="", type=str,
                        help="Path to the transducer encoder model")
    parser.add_argument("--decoder", default="", type=str,
                        help="Path to the transducer decoder model")
    parser.add_argument("--joiner", default="", type=str,
                        help="Path to the transducer joiner model")

    # --- Paraformer ---
    parser.add_argument("--paraformer", default="", type=str,
                        help="Path to the model.onnx from Paraformer")

    # --- SenseVoice ---
    parser.add_argument("--sense-voice", default="", type=str,
                        help="Path to the model.onnx from SenseVoice")

    # --- Whisper ---
    parser.add_argument("--whisper-encoder", type=str,
                        default=os.path.join(WHISPER_MODEL_DIR, "small.en-encoder.int8.onnx"),
                        help="Path to whisper encoder model")
    parser.add_argument("--whisper-decoder", type=str,
                        default=os.path.join(WHISPER_MODEL_DIR, "small.en-decoder.int8.onnx"),
                        help="Path to whisper decoder model")
    parser.add_argument("--whisper-language", type=str, default="en",
                        help="Spoken language (en, fr, de, zh, jp, etc.)")
    parser.add_argument("--whisper-task", type=str, default="transcribe",
                        choices=["transcribe", "translate"],
                        help="For multilingual models, translate outputs in English")
    parser.add_argument("--whisper-tail-paddings", type=int, default=-1,
                        help="Number of tail padding frames. Use -1 for default")

    # --- Moonshine ---
    parser.add_argument("--moonshine-preprocessor", default="", type=str,
                        help="Path to moonshine preprocessor model")
    parser.add_argument("--moonshine-encoder", default="", type=str,
                        help="Path to moonshine encoder model")
    parser.add_argument("--moonshine-uncached-decoder", default="", type=str,
                        help="Path to moonshine uncached decoder model")
    parser.add_argument("--moonshine-cached-decoder", default="", type=str,
                        help="Path to moonshine cached decoder model")

    return parser.parse_known_args()


# ============================================================
# 模型创建
# ============================================================

def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def _assert_exclusive(args, *exclude_fields):
    """确保只有一个模型类型被指定"""
    model_fields = [
        ("encoder", args.encoder),
        ("paraformer", args.paraformer),
        ("sense_voice", args.sense_voice),
        ("whisper_encoder", args.whisper_encoder),
        ("moonshine_preprocessor", args.moonshine_preprocessor),
    ]
    exclude_set = set(exclude_fields)
    for name, val in model_fields:
        if name not in exclude_set and val:
            raise ValueError(f"不能同时指定 {name} 和 {', '.join(exclude_set)}")


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    common_kwargs = dict(
        tokens=args.tokens,
        num_threads=args.num_threads,
        decoding_method=args.decoding_method,
        debug=args.debug,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
        provider=args.provider,
    )

    if args.encoder:
        _assert_exclusive(args, "encoder")
        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=args.encoder, decoder=args.decoder, joiner=args.joiner,
            sample_rate=args.sample_rate, feature_dim=args.feature_dim,
            blank_penalty=args.blank_penalty, **common_kwargs,
        )

    if args.paraformer:
        _assert_exclusive(args, "paraformer")
        assert_file_exists(args.paraformer)
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            sample_rate=args.sample_rate, feature_dim=args.feature_dim,
            **common_kwargs,
        )

    if args.sense_voice:
        _assert_exclusive(args, "sense_voice")
        assert_file_exists(args.sense_voice)
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice, use_itn=True, **common_kwargs,
        )

    if args.whisper_encoder:
        _assert_exclusive(args, "whisper_encoder")
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder, decoder=args.whisper_decoder,
            language=args.whisper_language, task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings, **common_kwargs,
        )

    if args.moonshine_preprocessor:
        _assert_exclusive(args, "moonshine_preprocessor")
        assert_file_exists(args.moonshine_preprocessor)
        assert_file_exists(args.moonshine_encoder)
        assert_file_exists(args.moonshine_uncached_decoder)
        assert_file_exists(args.moonshine_cached_decoder)
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=args.moonshine_preprocessor,
            encoder=args.moonshine_encoder,
            uncached_decoder=args.moonshine_uncached_decoder,
            cached_decoder=args.moonshine_cached_decoder,
            **common_kwargs,
        )

    raise ValueError("Please specify at least one model")


# ============================================================
# ASR ROS 节点
# ============================================================

class ASRNode:
    """ROS node for VAD-based offline ASR."""

    def __init__(self, node_name: str = "asr_node"):
        rospy.init_node(node_name)
        self._pub_asr = rospy.Publisher("asr", String, queue_size=10)

        self._recognizer = None
        self._vad = None
        self._sample_rate = 16000
        self._window_size = 512

        self._audio_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._buffer = np.array([])
        self._speech_segments: List[np.ndarray] = []
        self._texts: List[str] = []

        self._tts_playing = False
        self._tts_stop_time = 0.0

        self._save_folder: Optional[Path] = None
        self._all_samples: Optional[List[np.ndarray]] = None
        self._segment_count = 0
        self._base_name = "test"

    # ---- 初始化 ----

    def _init_models(self, args):
        assert_file_exists(args.tokens)
        assert_file_exists(args.silero_vad_model)
        assert args.num_threads > 0, args.num_threads

        self._sample_rate = args.sample_rate

        rospy.loginfo(f"[ASR] 采样率: {self._sample_rate} Hz")
        rospy.loginfo(f"[ASR] 模型参数: threads={args.num_threads}, provider={args.provider}")
        rospy.loginfo(f"[ASR] Whisper 语言: {args.whisper_language}, 任务: {args.whisper_task}")

        rospy.loginfo("[ASR] 正在加载识别模型, 请稍候...")
        self._recognizer = create_recognizer(args)
        rospy.loginfo("[ASR] 模型加载完成")

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = args.silero_vad_model
        config.silero_vad.min_silence_duration = 0.25
        config.sample_rate = self._sample_rate

        self._window_size = config.silero_vad.window_size
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
        rospy.loginfo("[ASR] VAD (语音活动检测) 初始化完成")

    def _init_save_folder(self, args):
        if args.save_audio:
            self._save_folder = Path(args.save_audio)
            self._all_samples = []
            self._save_folder.mkdir(parents=True, exist_ok=True)
            rospy.loginfo(f"[ASR] 音频保存目录: {self._save_folder}")

    def _init_subscribers(self):
        rospy.Subscriber("/tts/playing", String, self._on_tts_status, queue_size=10)
        rospy.loginfo(f"[ASR] 已订阅 /tts/playing 话题 (回声消除, 冷却期={TTS_COOLDOWN}s)")

        rospy.Subscriber("/audio/raw", Float32MultiArray, self._on_audio, queue_size=20)
        rospy.loginfo("[ASR] 已订阅 /audio/raw 话题")

    # ---- ROS 回调 ----

    def _on_tts_status(self, msg):
        if msg.data == "playing":
            self._tts_playing = True
        elif msg.data == "idle":
            self._tts_playing = False
            self._tts_stop_time = time.time()

    def _on_audio(self, msg):
        try:
            self._audio_queue.put_nowait(np.array(msg.data, dtype=np.float32))
        except queue.Full:
            pass

    # ---- 回声消除 ----

    def _is_echo(self) -> bool:
        if self._tts_playing:
            return True
        if self._tts_stop_time > 0 and (time.time() - self._tts_stop_time) < TTS_COOLDOWN:
            return True
        return False

    # ---- VAD 处理 ----

    def _feed_vad(self, samples: np.ndarray):
        self._buffer = np.concatenate([self._buffer, samples])
        while len(self._buffer) > self._window_size:
            self._vad.accept_waveform(self._buffer[:self._window_size])
            self._buffer = self._buffer[self._window_size:]

        while not self._vad.empty():
            self._speech_segments.append(self._vad.front.samples)
            self._vad.pop()

    # ---- 语音识别 ----

    def _recognize(self, speech_samples: np.ndarray) -> Optional[str]:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._sample_rate, speech_samples)
        self._recognizer.decode_stream(stream)

        text = stream.result.text.strip().lower()
        if not text:
            return None

        if NON_SPEECH_PATTERN.match(text):
            rospy.loginfo(f"[ASR] 过滤非语音标记: '{text}'")
            return None

        return text

    def _publish_result(self, text: str, speech_samples: np.ndarray):
        self._texts.append(text)
        rospy.loginfo(f"[ASR] 识别结果: '{text}'")

        msg = String()
        msg.data = text
        self._pub_asr.publish(msg)

        if self._save_folder:
            self._segment_count += 1
            path = self._save_folder / f"{self._base_name}-{self._segment_count}.wav"
            save_wav(path, speech_samples, self._sample_rate)
            rospy.loginfo(f"[ASR] 音频段已保存: {path}")

    # ---- 主循环 ----

    def run(self):
        args, _ = get_args()
        self._init_models(args)
        self._init_save_folder(args)
        self._init_subscribers()

        rospy.loginfo("[ASR] 节点就绪, 等待语音输入!")

        try:
            while not rospy.is_shutdown():
                try:
                    samples = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if self._all_samples is not None:
                    self._all_samples.append(samples)

                self._feed_vad(samples)

                if not self._speech_segments:
                    continue

                if self._is_echo():
                    self._speech_segments.pop(0)
                    continue

                speech_samples = self._speech_segments.pop(0)
                text = self._recognize(speech_samples)
                if text:
                    self._publish_result(text, speech_samples)

        except KeyboardInterrupt:
            rospy.loginfo("[ASR] Ctrl+C, 退出")
            if self._save_folder and self._all_samples:
                path = self._save_folder / f"{self._base_name}.wav"
                save_wav_concat(path, self._all_samples, self._sample_rate)
                rospy.loginfo(f"[ASR] 完整录音已保存: {path}")


# ============================================================
# 入口
# ============================================================

def main():
    node = ASRNode("asr_node")
    node.run()


if __name__ == "__main__":
    main()
