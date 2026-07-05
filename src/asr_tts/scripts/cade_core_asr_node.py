#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS ASR node backed by the vendored cade-voice-core ASR engine.

The node keeps the Task1 ROS contract unchanged:
  - subscribes to /audio/raw so audio_capture_node remains the only mic owner
  - publishes recognized text to /asr
  - publishes JSON evidence to /asr/segment
  - observes /tts/playing for echo suppression
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import time
import wave
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray, String


PKG_PATH = Path(rospkg.RosPack().get_path("asr_tts"))
VOICE_CORE_PATH = PKG_PATH / "third_party" / "cade_voice_core"

DEFAULT_ASR_MODEL_DIR = (
    "/home/nvidia/audio/models/asr/"
    "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
)
DEFAULT_ASR_FALLBACK_DIR = (
    "/home/nvidia/audio/models/asr/"
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-mobile"
)
DEFAULT_VAD_MODEL = "/home/nvidia/audio/models/asr/silero_vad.onnx"

NON_SPEECH_PATTERN = re.compile(r"^(?:\([^)]*\)?|\[[^\]]*\]?|<[^>]*>?)$")


def _install_voice_core_path() -> None:
    if not VOICE_CORE_PATH.is_dir():
        raise RuntimeError(f"cade-voice-core vendor path not found: {VOICE_CORE_PATH}")
    sys.path.insert(0, str(VOICE_CORE_PATH))


def _stripped_or_none(value: str) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _save_wav(filepath: Path, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio_data, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    audio_int16 = (samples * 32767).astype(np.int16)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_int16.tobytes())


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default=DEFAULT_ASR_MODEL_DIR)
    parser.add_argument("--model-type", default="streaming_nemotron")
    parser.add_argument("--fallback-model-dir", default=DEFAULT_ASR_FALLBACK_DIR)
    parser.add_argument("--fallback-model-type", default="streaming_zipformer")
    parser.add_argument("--vad-model", default=DEFAULT_VAD_MODEL)
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "coreml"])
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-sample-rate", type=int, default=16000)
    parser.add_argument("--language", default="en")
    parser.add_argument("--tts-cooldown", type=float, default=0.8)
    parser.add_argument("--vad-min-silence-duration", type=float, default=0.8)
    parser.add_argument(
        "--save-audio",
        default=str(PKG_PATH / "recordings" / "asr_segments"),
    )
    parser.add_argument("--streaming-save-tail-sec", type=float, default=8.0)
    return parser.parse_known_args()


class CadeCoreASRNode:
    def __init__(self):
        rospy.init_node("cade_core_asr_node")
        self._pub_asr = rospy.Publisher("/asr", String, queue_size=10)
        self._pub_asr_segment = rospy.Publisher("/asr/segment", String, queue_size=10)

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1000)
        self._history: Deque[np.ndarray] = deque()
        self._history_samples = 0
        self._history_limit_samples = 0
        self._segment_count = 0
        self._buffer = np.array([], dtype=np.float32)

        self._engine = None
        self._recognizer = None
        self._vad = None
        self._stream = None
        self._is_streaming = False
        self._sample_rate = 16000
        self._input_sample_rate = 16000
        self._window_size = 512
        self._save_folder = Path("/tmp/asr_segments")

        self._tts_playing = False
        self._tts_stop_time = 0.0
        self._tts_cooldown_sec = 0.8

    def _init_engine(self, args) -> None:
        _install_voice_core_path()
        from cade.asr.engine import ASREngine
        import sherpa_onnx

        self._sample_rate = int(args.sample_rate)
        self._input_sample_rate = int(
            rospy.get_param("/audio/raw/sample_rate", args.input_sample_rate)
        )
        self._tts_cooldown_sec = max(0.0, float(args.tts_cooldown))
        self._save_folder = Path(args.save_audio)
        self._save_folder.mkdir(parents=True, exist_ok=True)
        self._history_limit_samples = int(
            max(0.5, float(args.streaming_save_tail_sec)) * self._input_sample_rate
        )

        fallback_type = _stripped_or_none(args.fallback_model_type)
        fallback_dir = _stripped_or_none(args.fallback_model_dir)

        rospy.loginfo(
            "[CADE-ASR] loading model: type=%s dir=%s provider=%s threads=%s",
            args.model_type,
            args.model_dir,
            args.provider,
            args.num_threads,
        )
        self._engine = ASREngine(
            model_dir=args.model_dir,
            vad_model=args.vad_model,
            model_type=args.model_type,
            provider=args.provider,
            sample_rate=self._sample_rate,
            num_threads=args.num_threads,
            language=args.language,
            fallback_model_type=fallback_type,
            fallback_model_dir=fallback_dir,
        )
        self._recognizer = self._engine._recognizer
        self._is_streaming = bool(self._engine._is_streaming)

        if not self._is_streaming:
            self._engine._vad_config.silero_vad.min_silence_duration = max(
                0.1, float(args.vad_min_silence_duration)
            )
            self._window_size = self._engine._vad_window_size
            self._vad = sherpa_onnx.VoiceActivityDetector(
                self._engine._vad_config,
                buffer_size_in_seconds=100,
            )
        else:
            self._stream = self._recognizer.create_stream()

        rospy.loginfo(
            "[CADE-ASR] ready: active_model=%s streaming=%s input_rate=%s model_rate=%s",
            self._engine.active_model_name,
            self._is_streaming,
            self._input_sample_rate,
            self._sample_rate,
        )

    def _init_subscribers(self) -> None:
        rospy.Subscriber("/tts/playing", String, self._on_tts_status, queue_size=10)
        rospy.Subscriber("/audio/raw", Float32MultiArray, self._on_audio, queue_size=20)
        rospy.loginfo(
            "[CADE-ASR] subscribed: /audio/raw and /tts/playing "
            "(cooldown=%.2fs)",
            self._tts_cooldown_sec,
        )

    def _on_tts_status(self, msg: String) -> None:
        state = str(msg.data or "").strip().lower()
        if state == "playing":
            self._tts_playing = True
            self._reset_decoder_state()
        elif state == "idle":
            self._tts_playing = False
            self._tts_stop_time = time.time()
            self._reset_decoder_state()

    def _on_audio(self, msg: Float32MultiArray) -> None:
        try:
            samples = np.asarray(msg.data, dtype=np.float32)
            self._audio_queue.put_nowait(samples)
        except queue.Full:
            pass

    def _is_echo_window(self) -> bool:
        if self._tts_playing:
            return True
        return (
            self._tts_stop_time > 0
            and (time.time() - self._tts_stop_time) < self._tts_cooldown_sec
        )

    def _reset_decoder_state(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        if self._is_streaming and self._stream is not None:
            try:
                self._recognizer.reset(self._stream)
            except Exception:
                rospy.logdebug("[CADE-ASR] recognizer reset failed", exc_info=True)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                return

    def _remember_audio(self, samples: np.ndarray) -> None:
        if self._history_limit_samples <= 0 or samples.size == 0:
            return
        copied = np.asarray(samples, dtype=np.float32).copy()
        self._history.append(copied)
        self._history_samples += copied.size
        while self._history and self._history_samples > self._history_limit_samples:
            removed = self._history.popleft()
            self._history_samples -= removed.size

    def _history_audio(self) -> np.ndarray:
        if not self._history:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self._history)).astype(np.float32)

    def _resample_if_needed(self, samples: np.ndarray) -> np.ndarray:
        if self._input_sample_rate == self._sample_rate:
            return np.asarray(samples, dtype=np.float32)
        return self._engine._resample(
            np.asarray(samples, dtype=np.float32),
            self._input_sample_rate,
            self._sample_rate,
        )

    def _clean_text(self, text: str) -> Optional[str]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        cleaned = self._engine._normalize_text(cleaned).strip().lower()
        cleaned = " ".join(cleaned.split())
        if not cleaned or NON_SPEECH_PATTERN.match(cleaned):
            return None
        return cleaned

    def _publish_result(self, text: str, audio_samples: Optional[np.ndarray]) -> None:
        cleaned = self._clean_text(text)
        if not cleaned:
            return

        self._segment_count += 1
        wav_path = ""
        if audio_samples is not None and audio_samples.size > 0:
            path = self._save_folder / f"cade-core-{self._segment_count}.wav"
            _save_wav(path, audio_samples, self._input_sample_rate)
            wav_path = str(path.resolve())

        rospy.loginfo("[CADE-ASR] recognized: %r wav=%s", cleaned, wav_path or "none")
        segment = {
            "text": cleaned,
            "wav_path": wav_path,
            "sample_rate": self._input_sample_rate,
            "backend": "cade_core",
            "model": getattr(self._engine, "active_model_name", ""),
        }
        self._pub_asr_segment.publish(String(data=json.dumps(segment, ensure_ascii=False)))
        self._pub_asr.publish(String(data=cleaned))

    def _process_offline(self, samples: np.ndarray) -> None:
        model_samples = self._resample_if_needed(samples)
        self._buffer = np.concatenate([self._buffer, model_samples])
        while len(self._buffer) > self._window_size:
            self._vad.accept_waveform(self._buffer[: self._window_size])
            self._buffer = self._buffer[self._window_size :]

        while not self._vad.empty():
            speech = self._vad.front.samples
            self._vad.pop()

            stream = self._recognizer.create_stream()
            stream.accept_waveform(self._sample_rate, speech)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                self._publish_result(text, speech)

    def _process_streaming(self, samples: np.ndarray) -> None:
        model_samples = self._resample_if_needed(samples)
        self._stream.accept_waveform(self._sample_rate, model_samples.astype(np.float32))

        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        if not self._recognizer.is_endpoint(self._stream):
            return

        text = self._recognizer.get_result(self._stream).strip()
        self._recognizer.reset(self._stream)
        if text:
            self._publish_result(text, self._history_audio())

    def run(self) -> None:
        args, _ = get_args()
        self._init_engine(args)
        self._init_subscribers()

        rospy.loginfo("[CADE-ASR] node ready")
        while not rospy.is_shutdown():
            try:
                samples = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._is_echo_window():
                self._reset_decoder_state()
                self._drain_queue()
                continue

            self._remember_audio(samples)
            if self._is_streaming:
                self._process_streaming(samples)
            else:
                self._process_offline(samples)


def main() -> None:
    node = CadeCoreASRNode()
    node.run()


if __name__ == "__main__":
    main()
