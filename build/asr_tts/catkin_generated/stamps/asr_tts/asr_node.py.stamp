# ！/home/tiger/miniforge3/envs/ros/bin/python

import wave
import sherpa_onnx
import numpy as np
import threading
from pathlib import Path
import sys
import argparse
import rospy
from std_msgs.msg import String
import rospkg
import os
pkg_path = rospkg.RosPack().get_path('asr_tts')


try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        # required=True,
        default=os.path.join(pkg_path, "models/silero_vad.onnx"),
        help="Path to silero_vad.onnx",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default=os.path.join(pkg_path, "models/sherpa-onnx-whisper-medium.en/medium.en-tokens.txt"),
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
        default=os.path.join(pkg_path, "models/sherpa-onnx-whisper-medium.en/medium.en-encoder.int8.onnx"),
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default=os.path.join(pkg_path, "models/sherpa-onnx-whisper-medium.en/medium.en-decoder.int8.onnx"),
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="""It specifies the spoken language in the input file.
        Example values: en, fr, de, zh, jp.
        Available languages for multilingual models can be found at
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        If not specified, we infer the language from the input audio file.
        """,
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="""For multilingual models, if you specify translate, the output
        will be in English.
        """,
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="""Number of tail padding frames.
        We have removed the 30-second constraint from whisper, so you need to
        choose the amount of tail padding frames by yourself.
        Use -1 to use a default value for tail padding.
        """,
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
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Valid values are greedy_search and modified_beam_search.
        modified_beam_search is valid only for transducer models.
        """,
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages when loading modes.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="""Sample rate of the feature extractor. Must match the one
        expected by the model.""",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
    )

    parser.add_argument(
        "--hr-lexicon",
        type=str,
        default="",
        help="If not empty, it is the lexicon.txt for homophone replacer",
    )

    parser.add_argument(
        "--hr-rule-fsts",
        type=str,
        default="",
        help="If not empty, it is the replace.fst for homophone replacer",
    )

    parser.add_argument(
        "--save-audio",
        type=str,
        default=os.path.join(pkg_path, "recordings"),
        help="Folder path to save recorded audio (e.g., asr_tts/recordings). "
             "VAD segments saved as test-1.wav, test-2.wav, etc. "
             "Full recording saved as test.wav on interrupt.",
    )

    return parser.parse_known_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.sense_voice) == 0, args.sense_voice
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder
        assert len(args.moonshine_preprocessor) == 0, args.moonshine_preprocessor
        assert len(args.moonshine_encoder) == 0, args.moonshine_encoder
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
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
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
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
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
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
        assert (
            len(args.moonshine_uncached_decoder) == 0
        ), args.moonshine_uncached_decoder
        assert len(args.moonshine_cached_decoder) == 0, args.moonshine_cached_decoder

        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
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


class ASRNode:
    def __init__(self, node_name: str):
        rospy.init_node(node_name)
        self.publisher = rospy.Publisher("asr", String, queue_size=10)
        self.audio_thread = threading.Thread(
            target=self.ASR,
            daemon=True
        )
        self.audio_thread.start()

    def ASR(self):
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            sys.exit(0)

        print(devices)

        # If you want to select a different input device, please use
        # default_input_device_idx =xxx
        # where xxx is the device number

        default_input_device_idx = sd.default.device[0]
        print(f'Use default device: {devices[default_input_device_idx]["name"]}')

        args, _ = get_args()
        assert_file_exists(args.tokens)
        assert_file_exists(args.silero_vad_model)

        assert args.num_threads > 0, args.num_threads

        assert (
            args.sample_rate == 16000
        ), f"Only sample rate 16000 is supported.Given: {args.sample_rate}"

        print("Creating recognizer. Please wait...")
        recognizer = create_recognizer(args)

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = args.silero_vad_model
        config.silero_vad.min_silence_duration = 0.25
        config.sample_rate = args.sample_rate

        window_size = config.silero_vad.window_size

        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)

        samples_per_read = int(0.1 * args.sample_rate)  # 0.1 second = 100 ms

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

        def save_wav(filepath, audio_data):
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(str(filepath), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_int16.tobytes())

        def save_wav_from_samples_list(filepath, samples_list):
            all_audio = np.concatenate([np.array(s) if isinstance(s, list) else s for s in samples_list])
            audio_int16 = (all_audio * 32767).astype(np.int16)
            with wave.open(str(filepath), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_int16.tobytes())

        with sd.InputStream(channels=1, dtype="float32", samplerate=args.sample_rate, device=default_input_device_idx) as s:
            try:
                while True:
                    # 1. 读取麦克风数据（阻塞操作，但时间很短）
                    samples, _ = s.read(samples_per_read)  # 0.1秒的数据
                    samples = samples.reshape(-1)

                    # 2. 保存所有样本（如果需要）
                    if all_samples is not None:
                        all_samples.append(samples)

                    # 3. 处理 VAD
                    buffer = np.concatenate([buffer, samples])
                    while len(buffer) > window_size:
                        vad.accept_waveform(buffer[:window_size])
                        buffer = buffer[window_size:]

                    # 4. 收集 VAD 检测到的语音片段
                    while not vad.empty():
                        speech_segments.append(vad.front.samples)
                        vad.pop()

                    # 5. 处理语音片段（如果有）
                    if speech_segments:
                        # 只处理一个语音片段，避免阻塞主循环
                        samples = speech_segments.pop(0)
                        stream = recognizer.create_stream()
                        stream.accept_waveform(args.sample_rate, samples)

                        # 6. 执行 ASR 识别
                        recognizer.decode_stream(stream)

                        # 7. 处理识别结果
                        text = stream.result.text.strip().lower()
                        if len(text):
                            idx = len(texts)
                            texts.append(text)
                            print(f"[speaker0]: {text}")
                            asr_rst = String()
                            asr_rst.data = text
                            self.publisher.publish(asr_rst)

                            if save_folder:
                                segment_count += 1
                                segment_path = save_folder / f"{base_name}-{segment_count}.wav"
                                save_wav(segment_path, samples)
                                print(f"VAD segment saved to {segment_path}")
            except KeyboardInterrupt:
                print("\nCaught Ctrl + C. Exiting")
                if save_folder and all_samples:
                    save_wav_from_samples_list(save_folder / f"{base_name}.wav", all_samples)
                    print(f"Full recording saved to {save_folder / f'{base_name}.wav'}")


def main():
    node = ASRNode("asr_node")
    rospy.spin()


if __name__ == "__main__":
    main()
