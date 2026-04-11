#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic VAD ASR Node - Speaker Registration + VAD + Offline ASR
Automatically registers unknown speakers via Wespeaker embeddings,
then performs offline ASR (Whisper/SenseVoice/Paraformer/Transducer).
Publishes recognized text with speaker label to /asr topic.
"""

import json
import wave

import numpy as np
import sherpa_onnx
from pathlib import Path
from typing import Dict, List, Optional
import queue

import rospy
from std_msgs.msg import String
import rospkg
import os
import argparse
import sys

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first")
    sys.exit(-1)


pkg_path = rospkg.RosPack().get_path('asr_tts')
TARGET_SAMPLE_RATE = 16000


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # VAD model
    parser.add_argument(
        "--silero-vad-model",
        type=str,
        default=os.path.join(pkg_path, "models/silero_vad.onnx"),
        help="Path to silero_vad.onnx",
    )

    # Speaker embedding model
    parser.add_argument(
        "--speaker-model",
        type=str,
        default=os.path.join(pkg_path, "models/wespeaker_en_voxceleb_CAM++_LM.onnx"),
        help="Path to speaker embedding model",
    )

    parser.add_argument(
        "--speaker-db",
        type=str,
        default=os.path.join(pkg_path, "speakers/speaker_db.json"),
        help="Path to speaker database JSON file",
    )

    # ASR models
    parser.add_argument(
        "--tokens",
        type=str,
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-whisper-medium.en/medium.en-tokens.txt",
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
        "--whisper-encoder",
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-whisper-medium.en/medium.en-encoder.int8.onnx",
        ),
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default=os.path.join(
            pkg_path,
            "models/sherpa-onnx-whisper-medium.en/medium.en-decoder.int8.onnx",
        ),
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="Spoken language in the input file",
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type_str=str,
        help="For multilingual models: transcribe or translate",
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="Number of tail padding frames",
    )

    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to sense voice model",
    )

    # Common parameters
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Show debug messages when loading models",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Inference provider: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for speaker identification",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=192,
        help="Feature dimension. Must match the model expectation",
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


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    """Create an offline recognizer from available model arguments."""

    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=TARGET_SAMPLE_RATE,
            feature_dim=args.feature_dim,
            decoding_method="greedy_search",
            debug=args.debug,
            provider=args.provider,
        )
    elif args.paraformer:
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.paraformer)

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=TARGET_SAMPLE_RATE,
            feature_dim=args.feature_dim,
            decoding_method="greedy_search",
            debug=args.debug,
            provider=args.provider,
        )
    elif args.whisper_encoder:
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)

        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method="greedy_search",
            debug=args.debug,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
            provider=args.provider,
        )
    elif args.sense_voice:
        assert_file_exists(args.sense_voice)
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice,
            tokens=args.tokens,
            num_threads=args.num_threads,
            use_itn=True,
            debug=args.debug,
            provider=args.provider,
        )
    else:
        raise ValueError("Please specify at least one ASR model")

    return recognizer


def load_speaker_embedding_model(args):
    """Load the speaker embedding extractor."""

    assert_file_exists(args.speaker_model)
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.speaker_model,
        num_threads=args.num_threads,
        debug=args.debug,
        provider=args.provider,
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


# --- Speaker database management ---

def load_speaker_db(db_path: str) -> Dict[str, List[float]]:
    """Load speaker database from JSON file."""
    if not Path(db_path).exists():
        print(f"Speaker database not found. Creating new one at {db_path}")
        return {}

    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} speakers from database")
            for name in data.keys():
                print(f"  - {name}")
            return data
    except Exception as e:
        print(f"Error loading speaker database: {e}")
        return {}


def save_speaker_db(db_path: str, db: Dict[str, List[float]]):
    """Save speaker database to JSON file."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)


# --- Speaker embedding and identification ---

def compute_speaker_embedding(
    samples: np.ndarray,
    sample_rate: int,
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
) -> Optional[np.ndarray]:
    """Compute speaker embedding vector from audio samples."""
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    stream.input_finished()

    if extractor.is_ready(stream):
        embedding = extractor.compute(stream)
        return np.array(embedding)
    return None


def get_next_speaker_name(speaker_db: Dict[str, List[float]]) -> str:
    """Generate a unique speaker ID (speaker0, speaker1, ...)."""
    existing_ids = []
    for name in speaker_db.keys():
        if name.startswith("speaker"):
            try:
                speaker_id = int(name[7:])
                existing_ids.append(speaker_id)
            except ValueError:
                pass

    next_id = max(existing_ids) + 1 if existing_ids else 0
    return f"speaker{next_id}"


def register_speaker(
    name: str,
    embedding: np.ndarray,
    manager: sherpa_onnx.SpeakerEmbeddingManager,
    speaker_db: Dict[str, List[float]],
) -> bool:
    """Register a new speaker in the manager and database."""
    status = manager.add(name, embedding.tolist())
    if status:
        speaker_db[name] = embedding.tolist()
        print(f"\n[NEW SPEAKER] Registered: {name}")
        return True
    else:
        print(f"\n[ERROR] Failed to register speaker '{name}'")
        return False


def identify_speaker(
    embedding: np.ndarray,
    manager: sherpa_onnx.SpeakerEmbeddingManager,
    threshold: float,
) -> Optional[str]:
    """Identify a speaker by searching against registered speakers."""
    name = manager.search(embedding.tolist(), threshold=threshold)
    return name


class DynamicVADASRNode:
    """ROS node for dynamic speaker registration + VAD + offline ASR."""

    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.publisher = rospy.Publisher("asr", String, queue_size=10)

    def run(self):
        """Main loop - process speech segments with speaker ID and ASR."""

        args, _ = get_args()

        print("\n[SYSTEM] Creating ASR recognizer. Please wait...")
        recognizer = create_recognizer(args)

        print("[SYSTEM] Loading speaker embedding model. Please wait...")
        extractor = load_speaker_embedding_model(args)

        speaker_db = load_speaker_db(args.speaker_db)

        manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
        for name, embedding in speaker_db.items():
            status = manager.add(name, embedding)
            if not status:
                print(f"[ERROR] Failed to load speaker {name}")

        print("[SYSTEM] Initializing VAD...")
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

        device_sample_rate = int(devices[default_input_device_idx]['default_samplerate'])
        read_sample_rate = max(device_sample_rate, TARGET_SAMPLE_RATE)

        print("\n" + "=" * 60)
        print("Dynamic Speaker Registration + VAD + ASR")
        print("Unknown speakers will be auto-registered as speaker0, speaker1, etc.")
        print("Speak normally for recognition")
        print("Press Ctrl+C to exit")
        print("=" * 60 + "\n")

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

                    buffer = np.concatenate([buffer, samples])

                    if all_samples is not None:
                        all_samples.append(samples)

                    while len(buffer) > window_size:
                        vad.accept_waveform(buffer[:window_size])
                        buffer = buffer[window_size:]

                    while not vad.empty():
                        speech_samples = vad.front.samples.copy()
                        vad.pop()

                        # Skip short segments (< 0.5s)
                        if len(speech_samples) < 0.5 * TARGET_SAMPLE_RATE:
                            continue

                        # Compute speaker embedding
                        embedding = compute_speaker_embedding(
                            speech_samples, TARGET_SAMPLE_RATE, extractor
                        )
                        if embedding is None:
                            continue

                        # Identify or register speaker
                        speaker_name = identify_speaker(
                            embedding, manager, args.threshold
                        )
                        if speaker_name is None or speaker_name == "":
                            speaker_name = get_next_speaker_name(speaker_db)
                            register_speaker(
                                speaker_name, embedding, manager, speaker_db
                            )
                            save_speaker_db(args.speaker_db, speaker_db)

                        # Run offline ASR on the speech segment
                        asr_stream = recognizer.create_stream()
                        asr_stream.accept_waveform(
                            sample_rate=TARGET_SAMPLE_RATE,
                            waveform=speech_samples,
                        )
                        recognizer.decode_stream(asr_stream)
                        text = asr_stream.result.text.strip()

                        if text:
                            print(f"[{speaker_name}]: {text}")
                            asr_rst = String()
                            asr_rst.data = f"[{speaker_name}]: {text}"
                            self.publisher.publish(asr_rst)

                            if save_folder:
                                segment_count += 1
                                segment_path = save_folder / f"{base_name}-{segment_count}.wav"
                                save_wav(segment_path, speech_samples)

                        idx += 1

        except KeyboardInterrupt:
            save_speaker_db(args.speaker_db, speaker_db)
            print("\n\n[SYSTEM] Caught Ctrl+C. Exiting and saving speaker database")
            if save_folder and all_samples:
                save_wav_from_samples_list(save_folder / f"{base_name}.wav", all_samples)


def main():
    node = DynamicVADASRNode("dynamic_vad_asr_node")
    node.run()


if __name__ == "__main__":
    main()
