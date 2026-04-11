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
    """Play audio, preferring system player (paplay) over sounddevice."""

    if wav_path and os.path.exists(wav_path):
        try:
            print(f"Using system player: paplay {wav_path}")
            subprocess.run(['paplay', wav_path], check=True)
            print("Playback finished")
            return
        except Exception as e:
            print(f"paplay failed: {e}, falling back to sounddevice")

    try:
        if device is not None:
            device_info = sd.query_devices(device)
            device_sample_rate = int(device_info['default_samplerate'])

            if sample_rate != device_sample_rate:
                print(f"Resampling: {sample_rate} Hz -> {device_sample_rate} Hz")
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
        print("Playback finished")
    except Exception as e:
        print(f"Error playing audio: {e}")


def generate_speech(tts, text, sid=0, speed=1.0):
    """
    Generate speech from text.

    Returns:
        tuple: (audio_samples, sample_rate)
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
    """ROS node for Text-to-Speech synthesis."""

    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.args, _ = get_args()

        target_name = rospy.get_param("~device_name", "default")
        self.output_device = None

        devices = sd.query_devices()
        rospy.loginfo(f"[TTS] Trying to lock output device: \"{target_name}\"")

        for i, d in enumerate(devices):
            if target_name in d['name'] and d['max_output_channels'] > 0:
                self.output_device = i
                rospy.loginfo(f"[TTS] Output device locked: [{i}] {d['name']}")
                break

        if self.output_device is None:
            rospy.logerr(f"[TTS] Cannot find output device: \"{target_name}\"")
            rospy.logerr("Run 'python3 -m sounddevice' to check device names.")
            sys.exit(1)

        rospy.loginfo("Initializing TTS engine...")
        self.tts = create_tts(self.args)
        self.tts_subscription_ = rospy.Subscriber('tts', String, self.TTS, queue_size=10)
        rospy.loginfo("TTS Node is READY!")

    def TTS(self, msg: String):
        """Callback: generate and play speech from received text."""
        print(f"Generating speech for: '{msg.data}'")
        audio, sample_rate = generate_speech(self.tts, msg.data, self.args.sid, self.args.speed)

        sf.write(
            self.args.output,
            audio,
            samplerate=sample_rate,
            subtype="PCM_16",
        )
        print(f"Saved to {self.args.output}")

        if self.args.play:
            print("Playing audio...")
            play_audio(
                audio, sample_rate,
                device=self.output_device,
                wav_path=self.args.output,
            )


def main():
    node = TTSNode("tts_node")
    rospy.spin()


if __name__ == "__main__":
    main()
