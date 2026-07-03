#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Node - Text-to-Speech (VITS / Kokoro)
Subscribes to /tts topic, generates speech, and plays audio.
Supports both VITS and Kokoro architectures (auto-detected from model path).
"""

import subprocess
import time
import argparse
import os
import sys
import wave

import rospy
from std_msgs.msg import String
import rospkg
import numpy as np


def _package_path() -> str:
    return rospkg.RosPack().get_path("cade_voice")


def _add_local_sherpa_runtime() -> None:
    runtime_path = os.path.join(_package_path(), "src")
    if os.path.isdir(os.path.join(runtime_path, "sherpa_onnx")):
        sys.path.insert(0, runtime_path)


_add_local_sherpa_runtime()
import sherpa_onnx  # noqa: E402


pkg_path = _package_path()


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


def import_sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415
    except ImportError:
        print("Please install sounddevice first. You can use")
        print()
        print("  pip install sounddevice")
        print()
        raise

    return sd


def is_default_device_name(target_name: str) -> bool:
    return str(target_name or "").strip().lower() in ("", "default", "auto")


def log_available_output_devices(devices) -> None:
    rospy.logerr("Available output devices:")
    for i, device in enumerate(devices):
        if device["max_output_channels"] > 0:
            rospy.logerr("  [%d] %s", i, device["name"])


def resolve_output_device(devices, target_name):
    """Resolve index/name-fragment speaker selection; default returns None."""
    if is_default_device_name(target_name):
        return None, "system default output"

    target = str(target_name).strip()
    if target.isdigit():
        index = int(target)
        if 0 <= index < len(devices) and devices[index]["max_output_channels"] > 0:
            return index, f"[{index}] {devices[index]['name']}"
        return None, ""

    target_lower = target.lower()
    for i, device in enumerate(devices):
        if target_lower in device["name"].lower() and device["max_output_channels"] > 0:
            return i, f"[{i}] {device['name']}"

    return None, ""


def play_audio(audio, sample_rate, device=None, wav_path=None):
    """Play audio, preferring system player (paplay) over sounddevice."""

    if device is None and wav_path and os.path.exists(wav_path):
        try:
            print(f"Using system player: paplay {wav_path}")
            subprocess.run(['paplay', wav_path], check=True)
            print("Playback finished")
            return
        except Exception as e:
            print(f"paplay failed: {e}, falling back to sounddevice")

    try:
        sd = import_sounddevice()
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


def write_wav(path, audio, sample_rate):
    """Write mono PCM16 WAV without requiring the soundfile package."""
    samples = np.asarray(audio, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16.tobytes())


class TTSNode:
    """ROS node for Text-to-Speech synthesis."""

    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.args, _ = get_args()

        target_name = rospy.get_param("~device_name", "default")
        self.output_device = None

        rospy.loginfo(f"[TTS] Trying to lock output device: \"{target_name}\"")

        if is_default_device_name(target_name):
            rospy.loginfo("[TTS] Output device locked: system default output")
        else:
            sd = import_sounddevice()
            devices = sd.query_devices()
            self.output_device, output_device_label = resolve_output_device(
                devices, target_name
            )
            if self.output_device is not None:
                rospy.loginfo("[TTS] Output device locked: %s", output_device_label)

        if not is_default_device_name(target_name) and self.output_device is None:
            rospy.logerr(f"[TTS] Cannot find output device: \"{target_name}\"")
            rospy.logerr("Run 'python3 -m sounddevice' to check device names.")
            log_available_output_devices(devices)
            sys.exit(1)

        rospy.loginfo("Initializing TTS engine...")
        self.tts = create_tts(self.args)
        self.playing_pub = rospy.Publisher("/tts/playing", String, queue_size=10, latch=True)
        self.tts_subscription_ = rospy.Subscriber('tts', String, self.TTS, queue_size=10)
        self.playing_pub.publish(String(data="idle"))
        rospy.loginfo("TTS Node is READY!")

    def TTS(self, msg: String):
        """Callback: generate and play speech from received text."""
        print(f"Generating speech for: '{msg.data}'")
        audio, sample_rate = generate_speech(self.tts, msg.data, self.args.sid, self.args.speed)

        write_wav(
            self.args.output,
            audio,
            sample_rate,
        )
        print(f"Saved to {self.args.output}")

        if self.args.play:
            print("Playing audio...")
            self.playing_pub.publish(String(data="playing"))
            try:
                play_audio(
                    audio, sample_rate,
                    device=self.output_device,
                    wav_path=self.args.output,
                )
            finally:
                self.playing_pub.publish(String(data="idle"))


def main():
    node = TTSNode("tts_node")
    rospy.spin()


if __name__ == "__main__":
    main()
