#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming ASR Node with Endpoint Detection (Zipformer Transducer)
Uses OnlineRecognizer with built-in rule-based endpoint detection.
No VAD required - continuously streams audio to the recognizer.
Publishes recognized text to /asr topic (std_msgs/String).
"""

import rospy
from std_msgs.msg import String
import rospkg
import os
import argparse
import sys
from pathlib import Path
import queue

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first")
    sys.exit(-1)

import numpy as np
import sherpa_onnx


pkg_path = rospkg.RosPack().get_path('asr_tts')
TARGET_SAMPLE_RATE = 16000


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt",
        ),
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-streaming-zipformer-en-2023-06-26/"
            "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        ),
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-streaming-zipformer-en-2023-06-26/"
            "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        ),
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-streaming-zipformer-en-2023-06-26/"
            "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        ),
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values: greedy_search, modified_beam_search",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Inference provider: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="File containing hotwords (one per line)",
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="Hotword score boost for biasing",
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="Penalty applied on blank symbol during decoding",
    )

    parser.add_argument(
        "--hr-lexicon",
        type=str,
        default="",
        help="Lexicon.txt for homophone replacer (optional)",
    )

    parser.add_argument(
        "--hr-rule-fsts",
        type=str,
        default="",
        help="Replace.fst for homophone replacer (optional)",
    )

    return parser.parse_known_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args):
    """Create an online (streaming) recognizer with endpoint detection."""

    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=TARGET_SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method=args.decoding_method,
        provider=args.provider,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
    )
    return recognizer


class StreamingASREndpointNode:
    """ROS node for streaming ASR with endpoint detection."""

    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.publisher = rospy.Publisher("asr", String, queue_size=10)

    def run(self):
        """Main loop - PortAudio callback + queue pattern."""

        devices = sd.query_devices()
        if len(devices) == 0:
            rospy.logerr("No audio devices found!")
            sys.exit(1)

        target_name = rospy.get_param("~device_name", "default")
        rospy.loginfo(f"Trying to lock audio device: \"{target_name}\"")

        default_input_device_idx = None
        for i, d in enumerate(devices):
            if target_name in d['name'] and d['max_input_channels'] > 0:
                default_input_device_idx = i
                break

        if default_input_device_idx is None:
            rospy.logerr(f"Cannot find device: \"{target_name}\"")
            rospy.logerr("Run 'python3 -m sounddevice' to check device names.")
            sys.exit(1)

        print(
            f"Microphone locked: [{default_input_device_idx}] "
            f"{devices[default_input_device_idx]['name']}"
        )

        args, _ = get_args()
        print("Creating streaming ASR recognizer. Please wait...")
        recognizer = create_recognizer(args)
        print("Started! Please speak")

        device_sample_rate = int(devices[default_input_device_idx]['default_samplerate'])

        if device_sample_rate != TARGET_SAMPLE_RATE:
            print(
                f"Device sample rate: {device_sample_rate} Hz, "
                f"target: {TARGET_SAMPLE_RATE} Hz"
            )

        def resample_audio(audio, from_rate, to_rate):
            """Resample audio using linear interpolation."""
            if from_rate == to_rate:
                return audio
            ratio = to_rate / from_rate
            new_length = int(len(audio) * ratio)
            old_indices = np.arange(len(audio))
            new_indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(new_indices, old_indices, audio)

        stream = recognizer.create_stream()

        # --- Audio capture via PortAudio callback ---
        audio_queue = queue.Queue(maxsize=1000)

        def audio_callback(indata, frames, time_info=None, status=None):
            if status:
                print(f"Audio stream status: {status}", file=sys.stderr)
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

        # --- Main processing loop ---
        try:
            with sd.InputStream(
                device=default_input_device_idx,
                channels=1,
                dtype="float32",
                samplerate=max(device_sample_rate, TARGET_SAMPLE_RATE),
                callback=audio_callback,
            ):
                print("Audio stream started, listening...")

                while True:
                    indata = audio_queue.get()
                    samples = indata.flatten()

                    if device_sample_rate != TARGET_SAMPLE_RATE:
                        samples = resample_audio(
                            samples, device_sample_rate, TARGET_SAMPLE_RATE
                        )

                    stream.accept_waveform(TARGET_SAMPLE_RATE, samples)

                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                    is_endpoint = recognizer.is_endpoint(stream)
                    result = recognizer.get_result(stream)

                    if result:
                        print(f"[speaker0]: {result}")
                        asr_rst = String()
                        asr_rst.data = result
                        self.publisher.publish(asr_rst)

                    if is_endpoint:
                        if result:
                            print(f"[FINAL] {result}")
                        recognizer.reset(stream)

        except KeyboardInterrupt:
            print("\nCaught Ctrl+C. Exiting")


def main():
    node = StreamingASREndpointNode("streaming_asr_endpoint_node")
    node.run()


if __name__ == "__main__":
    main()
