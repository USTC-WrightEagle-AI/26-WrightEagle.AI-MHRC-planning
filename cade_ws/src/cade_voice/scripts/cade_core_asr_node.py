#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task3 ROS ASR node backed by the vendored cade-voice-core ASR engine.

This node preserves the Task3 contract: recognized commands are published as
plain std_msgs/String on /asr for cade_brain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import rospy
import rospkg
from std_msgs.msg import String


PKG_PATH = Path(rospkg.RosPack().get_path("cade_voice"))
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
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
NOISE_ONLY_PHRASES = {
    "a",
    "ah",
    "an",
    "and",
    "eh",
    "er",
    "hmm",
    "hm",
    "huh",
    "mm",
    "mmm",
    "oh",
    "or",
    "so",
    "the",
    "uh",
    "um",
}


def _install_voice_core_path() -> None:
    if not VOICE_CORE_PATH.is_dir():
        raise RuntimeError(f"cade-voice-core vendor path not found: {VOICE_CORE_PATH}")
    sys.path.insert(0, str(VOICE_CORE_PATH))


def _stripped_or_none(value: str) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


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
    parser.add_argument("--language", default="en")
    parser.add_argument("--tts-cooldown", type=float, default=1.2)
    parser.add_argument("--pre-roll-sec", type=float, default=0.5)
    parser.add_argument("--post-roll-sec", type=float, default=0.5)
    parser.add_argument(
        "--save-audio",
        default=str(PKG_PATH / "recordings" / "cade_core_asr"),
    )
    return parser.parse_known_args()


class CadeCoreASRNode:
    def __init__(self):
        rospy.init_node("cade_core_asr_node")
        self._pub_asr = rospy.Publisher("/asr", String, queue_size=10)
        self._pub_asr_segment = rospy.Publisher("/asr/segment", String, queue_size=10)
        self._tts_playing = False
        self._tts_stop_time = 0.0
        self._tts_busy_until = 0.0
        self._tts_cooldown_sec = 1.2
        self._segment_count = 0
        self._engine = None
        self._active_model_name = ""

    def _on_tts_status(self, msg: String) -> None:
        state = str(msg.data or "").strip().lower()
        if state == "playing":
            self._tts_playing = True
            self._tts_busy_until = max(self._tts_busy_until, time.time() + 2.0)
        elif state == "idle":
            self._tts_playing = False
            self._tts_stop_time = time.time()

    def _on_tts_request(self, msg: String) -> None:
        text = str(msg.data or "").strip()
        if not text:
            return
        estimated_duration = max(1.5, min(35.0, len(text) * 0.12))
        self._tts_playing = True
        self._tts_busy_until = max(
            self._tts_busy_until,
            time.time() + estimated_duration + self._tts_cooldown_sec,
        )

    def _can_decode(self) -> bool:
        now = time.time()
        if self._tts_playing and now >= self._tts_busy_until:
            self._tts_playing = False
            self._tts_stop_time = now
        if self._tts_playing or now < self._tts_busy_until:
            return False
        return not (
            self._tts_stop_time > 0
            and (now - self._tts_stop_time) < self._tts_cooldown_sec
        )

    def _clean_text(self, text: str) -> Optional[str]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        cleaned = " ".join(cleaned.split()).lower()
        if not cleaned or NON_SPEECH_PATTERN.match(cleaned):
            return None
        tokens = TOKEN_PATTERN.findall(cleaned)
        if len(tokens) == 1 and tokens[0] in NOISE_ONLY_PHRASES:
            rospy.loginfo("[CADE-ASR] ignored noise-like one-word transcript: %r", cleaned)
            return None
        return cleaned

    def _publish_transcript(self, text: str) -> None:
        cleaned = self._clean_text(text)
        if not cleaned:
            return

        self._segment_count += 1
        rospy.loginfo("[CADE-ASR] recognized: %r", cleaned)
        self._pub_asr.publish(String(data=cleaned))
        segment = {
            "text": cleaned,
            "wav_path": "",
            "sample_rate": 16000,
            "backend": "cade_core",
            "model": self._active_model_name,
            "segment_index": self._segment_count,
        }
        self._pub_asr_segment.publish(String(data=json.dumps(segment, ensure_ascii=False)))

    def run(self) -> None:
        args, _ = get_args()
        _install_voice_core_path()
        from cade.asr.engine import ASREngine

        self._tts_cooldown_sec = max(0.0, float(args.tts_cooldown))
        device_name = rospy.get_param("~device_name", "default")
        save_audio = rospy.get_param("~save_audio", args.save_audio)
        save_audio = str(save_audio or "").strip() or None

        fallback_type = _stripped_or_none(args.fallback_model_type)
        fallback_dir = _stripped_or_none(args.fallback_model_dir)

        rospy.Subscriber("/tts", String, self._on_tts_request, queue_size=10)
        rospy.Subscriber("/tts/playing", String, self._on_tts_status, queue_size=10)
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
            sample_rate=int(args.sample_rate),
            num_threads=int(args.num_threads),
            language=args.language,
            fallback_model_type=fallback_type,
            fallback_model_dir=fallback_dir,
            streaming_pre_roll_sec=float(args.pre_roll_sec),
            streaming_post_roll_sec=float(args.post_roll_sec),
        )
        self._active_model_name = getattr(self._engine, "active_model_name", args.model_type)
        rospy.loginfo(
            "[CADE-ASR] ready: active_model=%s device=%s cooldown=%.2fs",
            self._active_model_name,
            device_name,
            self._tts_cooldown_sec,
        )

        self._engine.start_listening(
            callback=self._publish_transcript,
            device_name=device_name,
            save_dir=save_audio,
            should_decode=self._can_decode,
        )


def main() -> None:
    node = CadeCoreASRNode()
    node.run()


if __name__ == "__main__":
    main()
