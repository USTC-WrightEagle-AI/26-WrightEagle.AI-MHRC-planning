#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAD + Streaming ASR Node (Zipformer Transducer)
Combines Silero VAD for speech boundary detection with streaming ASR
for real-time transcription within detected speech segments.
Publishes recognized text to /asr topic (std_msgs/String).
"""

import rospy
from std_msgs.msg import String
import rospkg
import os
import argparse
import sys
import time
from pathlib import Path
import queue

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first")
    sys.exit(-1)

import sherpa_onnx
import wave


pkg_path = rospkg.RosPack().get_path('asr_tts')
TARGET_SAMPLE_RATE = 16000


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        default=os.path.join(pkg_path, "models/silero_vad.onnx"),
        help="Path to silero_vad.onnx",
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
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Inference provider: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the model expectation",
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

    parser.add_argument(
        "--save-audio",
        type=str,
        default="",
        help="Folder path to save recorded audio",
    )

    return parser.parse_known_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_streaming_recognizer(args) -> sherpa_onnx.OnlineRecognizer:
    """Create a streaming recognizer with endpoint detection enabled."""

    assert_file_exists(args.tokens)
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        sample_rate=TARGET_SAMPLE_RATE,
        feature_dim=args.feature_dim,
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


class VADStreamingASRNode:
    """ROS node for VAD-gated streaming ASR."""

    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.publisher = rospy.Publisher("asr", String, queue_size=10)

    def run(self):
        """Main loop - VAD gates streaming ASR processing."""

        args, _ = get_args()

        print("Creating streaming ASR recognizer. Please wait...")
        recognizer = create_streaming_recognizer(args)

        print("Loading VAD model. Please wait...")
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = args.silero_vad_model
        vad_config.silero_vad.min_silence_duration = 0.25
        vad_config.silero_vad.min_speech_duration = 0.25
        vad_config.sample_rate = TARGET_SAMPLE_RATE
        if not vad_config.validate():
            raise ValueError("Errors in vad config")

        window_size = vad_config.silero_vad.window_size
        vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

        # --- Device lookup ---
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
        print("VAD + Streaming ASR (Zipformer)")
        print("Speak normally and see real-time transcription")
        print("Press Ctrl+C to exit")

        device_sample_rate = int(devices[default_input_device_idx]['default_samplerate'])
        read_sample_rate = max(device_sample_rate, TARGET_SAMPLE_RATE)

        save_folder = Path(args.save_audio) if args.save_audio else None
        all_samples = [] if save_folder else None
        segment_count = 0
        base_name = "test"

        if save_folder:
            save_folder.mkdir(parents=True, exist_ok=True)

        # --- Helper functions ---

        def resample_audio(audio, from_rate, to_rate):
            """Resample audio using linear interpolation."""
            if from_rate == to_rate:
                return audio
            ratio = to_rate / from_rate
            new_length = int(len(audio) * ratio)
            old_indices = np.arange(len(audio))
            new_indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(new_indices, old_indices, audio)

        def save_wav(filepath, audio_data):
            """Save audio data to WAV file."""
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(str(filepath), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(TARGET_SAMPLE_RATE)
                f.writeframes(audio_int16.tobytes())

        def save_wav_from_samples_list(filepath, samples_list):
            """Concatenate and save multiple sample chunks to WAV file."""
            all_audio = np.concatenate([
                np.array(s) if isinstance(s, list) else s for s in samples_list
            ])
            audio_int16 = (all_audio * 32767).astype(np.int16)
            with wave.open(str(filepath), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(TARGET_SAMPLE_RATE)
                f.writeframes(audio_int16.tobytes())

        # --- State variables ---
        idx = 0
        buffer = []
        started = False
        current_stream = None
        speech_buffer = []
        speech_start_time = None
        last_speech_time = None

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
                samplerate=read_sample_rate,
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

                    if all_samples is not None:
                        all_samples.append(samples)

                    buffer = np.concatenate([buffer, samples])

                    while len(buffer) > window_size:
                        vad.accept_waveform(buffer[:window_size])
                        buffer = buffer[window_size:]

                    is_speech = vad.is_speech_detected()
                    current_time = time.time()

                    # Speech onset detection
                    if is_speech and not started:
                        if last_speech_time is None or (current_time - last_speech_time) > 2.0:
                            started = True
                            current_stream = recognizer.create_stream()
                            speech_buffer = []
                            speech_start_time = current_time
                            print("\n[Speech detected]")

                    if is_speech:
                        last_speech_time = current_time

                    # Feed audio to streaming recognizer during speech
                    if started:
                        speech_buffer.extend(samples.tolist())
                        current_stream.accept_waveform(TARGET_SAMPLE_RATE, samples)

                        while recognizer.is_ready(current_stream):
                            recognizer.decode_stream(current_stream)

                        # Speech end detection (silence timeout)
                        if not is_speech and started:
                            silence_duration = current_time - last_speech_time
                            if silence_duration > 1.0:
                                if len(speech_buffer) > 0.5 * TARGET_SAMPLE_RATE:
                                    result = recognizer.get_result(current_stream)
                                    if result:
                                        print(f"\n[{idx}] [speaker0]: {result}")
                                        asr_rst = String()
                                        asr_rst.data = result
                                        self.publisher.publish(asr_rst)

                                        if save_folder:
                                            segment_count += 1
                                            segment_path = save_folder / f"{base_name}-{segment_count}.wav"
                                            save_wav(segment_path, np.array(speech_buffer))

                                recognizer.reset(current_stream)
                                current_stream = None
                                started = False
                                speech_buffer = []
                                idx += 1

        except KeyboardInterrupt:
            print("\nCaught Ctrl+C. Exiting")
            if save_folder and all_samples:
                save_wav_from_samples_list(save_folder / f"{base_name}.wav", all_samples)


def main():
    node = VADStreamingASRNode("vad_streaming_asr_node")
    node.run()


if __name__ == "__main__":
    main()
