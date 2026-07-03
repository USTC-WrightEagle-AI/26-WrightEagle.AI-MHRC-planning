#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR Node - VAD + Offline (Non-streaming) ASR
Uses SenseVoice/Whisper/Paraformer/Transducer models with Silero VAD.
Publishes recognized text to /asr topic (std_msgs/String).
"""

import os
import sys
import wave
import argparse
import inspect
import queue
from pathlib import Path

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
            "models/sherpa-onnx-whisper-small.en/small.en-tokens.txt",
        ),
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to the model.onnx from SenseVoice",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="Inference provider: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--whisper-encoder",
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-whisper-small.en/small.en-encoder.int8.onnx",
        ),
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-whisper-small.en/small.en-decoder.int8.onnx",
        ),
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="en",
        type=str,
        help="""Spoken language in the input file.
        Example values: en, fr, de, zh, jp.
        If not specified, language is inferred from audio.""",
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="For multilingual models, translate outputs in English.",
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="Number of tail padding frames. Use -1 for default.",
    )

    parser.add_argument(
        "--moonshine-preprocessor",
        default="",
        type=str,
        help="Path to moonshine preprocessor model",
    )

    parser.add_argument(
        "--moonshine-encoder",
        default="",
        type=str,
        help="Path to moonshine encoder model",
    )

    parser.add_argument(
        "--moonshine-uncached-decoder",
        default="",
        type=str,
        help="Path to moonshine uncached decoder model",
    )

    parser.add_argument(
        "--moonshine-cached-decoder",
        default="",
        type=str,
        help="Path to moonshine cached decoder model",
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="Penalty applied on blank symbol during decoding.",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values: greedy_search, modified_beam_search.",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Show debug messages when loading models.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the feature extractor.",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the model expectation.",
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
        default=os.path.join(pkg_path, "recordings"),
        help="Folder path to save recorded audio segments.",
    )

    parser.add_argument(
        "--input-gain-db",
        type=float,
        default=0.0,
        help="Pre-VAD microphone gain in dB. Use a small value for far-field speech.",
    )

    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD speech probability threshold.",
    )

    parser.add_argument(
        "--normalize-speech",
        action="store_true",
        help="Normalize each VAD speech segment before ASR decoding.",
    )

    parser.add_argument(
        "--target-rms-dbfs",
        type=float,
        default=-22.0,
        help="Target RMS for --normalize-speech.",
    )

    return parser.parse_known_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def call_with_supported_kwargs(func, **kwargs):
    """Call sherpa_onnx helpers across versions with slightly different signatures."""
    supported = inspect.signature(func).parameters
    return func(**{key: value for key, value in kwargs.items() if key in supported})


def import_sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415
    except ImportError:
        print("Please install sounddevice first. You can use")
        print()
        print("  pip install sounddevice")
        print()
        sys.exit(-1)

    return sd


def is_default_device_name(target_name: str) -> bool:
    return str(target_name or "").strip().lower() in ("", "default", "auto")


def log_available_audio_devices(devices, direction: str) -> None:
    channel_key = "max_input_channels" if direction == "input" else "max_output_channels"
    rospy.logerr("Available %s devices:", direction)
    for i, device in enumerate(devices):
        if device[channel_key] > 0:
            rospy.logerr("  [%d] %s", i, device["name"])


def resolve_input_device(sd, devices, target_name):
    """Resolve default/index/name-fragment microphone selection."""
    if is_default_device_name(target_name):
        try:
            device_info = sd.query_devices(kind="input")
            return None, device_info, "system default input"
        except Exception:
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    return i, device, f"[{i}] {device['name']}"
            return None, None, ""

    target = str(target_name).strip()
    if target.isdigit():
        index = int(target)
        if 0 <= index < len(devices) and devices[index]["max_input_channels"] > 0:
            return index, devices[index], f"[{index}] {devices[index]['name']}"
        return None, None, ""

    target_lower = target.lower()
    for i, device in enumerate(devices):
        if target_lower in device["name"].lower() and device["max_input_channels"] > 0:
            return i, device, f"[{i}] {device['name']}"

    return None, None, ""


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.sense_voice) == 0, args.sense_voice
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert len(args.moonshine_uncached_decoder) == 0, args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            blank_penalty=args.blank_penalty,
            debug=args.debug,
            hr_rule_fsts=args.hr_rule_fsts,
            hr_lexicon=args.hr_lexicon,
            provider=args.provider,
        )
    elif args.paraformer:
        assert len(args.sense_voice) == 0, args.sense_voice
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert len(args.moonshine_uncached_decoder) == 0, args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.paraformer)

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
            hr_rule_fsts=args.hr_rule_fsts,
            hr_lexicon=args.hr_lexicon,
            provider=args.provider,
        )
    elif args.sense_voice:
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert len(args.moonshine_uncached_decoder) == 0, args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        assert_file_exists(args.sense_voice)
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice,
            tokens=args.tokens,
            num_threads=args.num_threads,
            use_itn=True,
            debug=args.debug,
            hr_rule_fsts=args.hr_rule_fsts,
            hr_lexicon=args.hr_lexicon,
            provider=args.provider,
        )
    elif args.whisper_encoder:
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert len(args.moonshine_uncached_decoder) == 0, args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        recognizer = call_with_supported_kwargs(
            sherpa_onnx.OfflineRecognizer.from_whisper,
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
            hr_rule_fsts=args.hr_rule_fsts,
            hr_lexicon=args.hr_lexicon,
            provider=args.provider,
        )
    elif args.moonshine_preprocessor:
        assert_file_exists(args.moonshine_preprocessor)
        assert_file_exists(args.moonshine_encoder)
        assert_file_exists(args.moonshine_uncached_decoder)
        assert_file_exists(args.moonshine_cached_decoder)

        recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=args.moonshine_preprocessor,
            encoder=args.moonshine_encoder,
            uncached_decoder=args.moonshine_uncached_decoder,
            cached_decoder=args.moonshine_cached_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            debug=args.debug,
            hr_rule_fsts=args.hr_rule_fsts,
            hr_lexicon=args.hr_lexicon,
            provider=args.provider,
        )
    else:
        raise ValueError("Please specify at least one model")

    return recognizer


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


def apply_gain(samples, gain_db: float):
    samples = np.asarray(samples, dtype=np.float32)
    if abs(gain_db) < 1e-6:
        return samples
    return np.clip(samples * db_to_gain(gain_db), -1.0, 1.0)


def normalize_rms(samples, target_dbfs: float = -22.0, max_gain_db: float = 18.0):
    """Normalize one speech segment with a limiter to avoid clipping."""
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) == 0:
        return samples

    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms <= 1e-6:
        return samples

    target_rms = db_to_gain(target_dbfs)
    gain = min(target_rms / rms, db_to_gain(max_gain_db))
    peak_after_gain = float(np.max(np.abs(samples * gain)))

    if peak_after_gain > 0.98:
        gain *= 0.98 / peak_after_gain

    return np.clip(samples * gain, -1.0, 1.0)


class ASRNode:
    """ROS node for VAD-based offline ASR."""

    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.publisher = rospy.Publisher("asr", String, queue_size=10)

    def run(self):
        """Main loop - runs on main thread using PortAudio callback + queue."""
        sd = import_sounddevice()
        devices = sd.query_devices()
        if len(devices) == 0:
            rospy.logerr("No audio devices found!")
            sys.exit(1)

        target_name = rospy.get_param("~device_name", "default")
        rospy.loginfo(f"Trying to lock audio input device: \"{target_name}\"")

        input_device_idx, input_device_info, input_device_label = resolve_input_device(
            sd, devices, target_name
        )

        if input_device_info is None:
            rospy.logerr(f"Cannot find device: \"{target_name}\"")
            rospy.logerr("Run 'python3 -m sounddevice' to check device names.")
            log_available_audio_devices(devices, "input")
            sys.exit(1)

        print(f"Microphone locked: {input_device_label}")

        args, _ = get_args()
        assert_file_exists(args.tokens)
        assert_file_exists(args.silero_vad_model)

        assert args.num_threads > 0, args.num_threads

        device_sample_rate = int(input_device_info['default_samplerate'])
        target_sample_rate = args.sample_rate

        if device_sample_rate != target_sample_rate:
            print(f"Device sample rate: {device_sample_rate} Hz, resampling to: {target_sample_rate} Hz")

        print("Creating recognizer. Please wait...")
        recognizer = create_recognizer(args)

        input_gain_db = float(rospy.get_param("~input_gain_db", args.input_gain_db))
        vad_threshold = float(rospy.get_param("~vad_threshold", args.vad_threshold))
        normalize_speech = bool(rospy.get_param("~normalize_speech", args.normalize_speech))
        target_rms_dbfs = float(rospy.get_param("~target_rms_dbfs", args.target_rms_dbfs))

        print(
            "ASR audio preprocessing: "
            f"input_gain_db={input_gain_db:.1f}, "
            f"vad_threshold={vad_threshold:.2f}, "
            f"normalize_speech={normalize_speech}, "
            f"target_rms_dbfs={target_rms_dbfs:.1f}"
        )

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = args.silero_vad_model
        config.silero_vad.min_silence_duration = 0.25
        config.silero_vad.threshold = vad_threshold
        config.sample_rate = target_sample_rate

        window_size = config.silero_vad.window_size

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)

        print("Started! Please speak")

        save_folder = Path(args.save_audio) if args.save_audio else None
        all_samples = [] if save_folder else None
        segment_count = 0
        base_name = "test"

        if save_folder:
            save_folder.mkdir(parents=True, exist_ok=True)

        buffer = np.array([])
        texts = []
        speech_segments = []

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
            audio_data = np.clip(np.asarray(audio_data, dtype=np.float32), -1.0, 1.0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(str(filepath), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_int16.tobytes())

        def save_wav_from_samples_list(filepath, samples_list):
            """Concatenate and save multiple sample chunks to WAV file."""
            all_audio = np.concatenate([
                np.array(s) if isinstance(s, list) else s for s in samples_list
            ])
            all_audio = np.clip(np.asarray(all_audio, dtype=np.float32), -1.0, 1.0)
            audio_int16 = (all_audio * 32767).astype(np.int16)
            with wave.open(str(filepath), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_int16.tobytes())

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
                device=input_device_idx,
                channels=1,
                callback=audio_callback,
                samplerate=device_sample_rate,
                dtype='float32',
            ):
                print("Audio stream started, listening...")

                while True:
                    indata = audio_queue.get()
                    samples = indata.flatten()

                    rms = np.sqrt(np.mean(samples ** 2))
                    max_amp = np.max(np.abs(samples))

                    if device_sample_rate != target_sample_rate:
                        samples = resample_audio(samples, device_sample_rate, target_sample_rate)

                    samples = apply_gain(samples, input_gain_db)

                    if all_samples is not None:
                        all_samples.append(samples)

                    buffer = np.concatenate([buffer, samples])
                    while len(buffer) > window_size:
                        vad.accept_waveform(buffer[:window_size])
                        buffer = buffer[window_size:]

                    while not vad.empty():
                        speech_segments.append(vad.front.samples)
                        vad.pop()

                    if speech_segments:
                        speech_samples = speech_segments.pop(0)
                        if normalize_speech:
                            speech_samples = normalize_rms(speech_samples, target_rms_dbfs)

                        stream = recognizer.create_stream()
                        stream.accept_waveform(target_sample_rate, speech_samples)
                        recognizer.decode_stream(stream)

                        text = stream.result.text.strip().lower()
                        if len(text):
                            idx = len(texts)
                            texts.append(text)
                            print(f"\n[speaker0]: {text}")

                            asr_rst = String()
                            asr_rst.data = text
                            self.publisher.publish(asr_rst)

                            if save_folder:
                                segment_count += 1
                                segment_path = save_folder / f"{base_name}-{segment_count}.wav"
                                save_wav(segment_path, speech_samples)
                                print(f"VAD segment saved to {segment_path}")

        except KeyboardInterrupt:
            print("\nCaught Ctrl+C. Exiting")
            if save_folder and all_samples:
                save_wav_from_samples_list(save_folder / f"{base_name}.wav", all_samples)
                print(f"Full recording saved to {save_folder / f'{base_name}.wav'}")


def main():
    node = ASRNode("asr_node")
    node.run()


if __name__ == "__main__":
    main()
