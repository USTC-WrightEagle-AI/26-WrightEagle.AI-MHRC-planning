#!/usr/bin/env python3
"""
Streaming audio tagging from microphone with dpdfnet8 speech denoising preprocessing.

Uses dpdfnet8 for speech enhancement + CED-mini for audio tagging.
- 每2秒累积去噪后音频进行一次检测
- 保留1秒重叠, 避免漏检声音边界
"""

import logging
import sys
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

MODEL_NAME = "dpdfnet8"


def create_speech_denoiser():
    model_file = "../models/dpdfnet8.onnx"
    if not Path(model_file).is_file():
        raise ValueError(f"Please download {model_file}")

    config = sherpa_onnx.OnlineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(
                model=model_file
            ),
            debug=False,
            num_threads=1,
            provider="cpu",
        )
    )
    if not config.validate():
        raise ValueError(f"Please check the config: {config}")

    return sherpa_onnx.OnlineSpeechDenoiser(config)


def create_audio_tagger():
    model_file = "../models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx"
    label_file = "../models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19/class_labels_indices.csv"

    if not Path(model_file).is_file():
        raise ValueError(f"Please download {model_file}")
    if not Path(label_file).is_file():
        raise ValueError(f"Please download {label_file}")

    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            ced=model_file,
            num_threads=1,
            debug=False,
            provider="cpu",
        ),
        labels=label_file,
        top_k=5,
    )
    if not config.validate():
        raise ValueError(f"Please check the config: {config}")

    return sherpa_onnx.AudioTagging(config)


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    try:
        from scipy.signal import resample_poly

        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return resample_poly(audio, up, down)
    except ImportError:
        logging.warning("scipy not available, using simple decimation for resampling")
        ratio = orig_sr / target_sr
        if ratio == int(ratio):
            return audio[::int(ratio)]
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio)


def main():
    logging.info(f"Creating speech denoiser ({MODEL_NAME}) and audio tagger")
    denoiser = create_speech_denoiser()
    audio_tagger = create_audio_tagger()

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    denoiser_sr = denoiser.sample_rate
    tagger_sr = 16000
    frame_shift = denoiser.frame_shift_in_samples

    logging.info(f"Denoiser sample rate: {denoiser_sr}, frame_shift: {frame_shift}")
    logging.info("Now, listening for audio events with denoising...")

    samples_per_read = int(0.2 * denoiser_sr)
    buffer_duration = 2.0
    buffer_read_count = int(buffer_duration / 0.2)
    overlap_read_count = 5

    denoised_buffer = []

    with sd.InputStream(channels=1, dtype="float32", samplerate=denoiser_sr) as stream:
        while True:
            samples, _ = stream.read(samples_per_read)
            samples = samples.reshape(-1)

            denoised_chunks = []
            for start in range(0, len(samples), frame_shift):
                chunk = samples[start : start + frame_shift]
                if len(chunk) < frame_shift:
                    chunk = np.pad(chunk, (0, frame_shift - len(chunk)))
                denoised = denoiser(chunk, denoiser_sr)
                denoised_chunks.append(np.asarray(denoised.samples, dtype=np.float32))

            if denoised_chunks:
                denoised_audio = np.concatenate(denoised_chunks)
                denoised_audio = resample_audio(denoised_audio, denoiser_sr, tagger_sr)
                denoised_buffer.append(denoised_audio)

            if len(denoised_buffer) >= buffer_read_count:
                all_audio = np.concatenate(denoised_buffer)

                s = audio_tagger.create_stream()
                s.accept_waveform(tagger_sr, all_audio)
                result = audio_tagger.compute(s)

                if result:
                    best = max(result, key=lambda e: e.prob)
                    logging.info(f"[{MODEL_NAME}] 检测到: {best.name}, 概率: {best.prob:.4f}")

                denoised_buffer = denoised_buffer[-overlap_read_count:]


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    try:
        main()
    except KeyboardInterrupt:
        logging.info("Ctrl+C caught. Exiting.")
