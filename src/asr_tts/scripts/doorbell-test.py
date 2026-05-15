#!/usr/bin/env python3
"""
Streaming audio tagging from microphone using sherpa-onnx.

用 AudioTagging 实现流式麦克风输入
- 每2秒累积音频后进行一次检测
- 保留1秒重叠, 避免漏检声音边界
"""

import logging
import sys
from pathlib import Path
import numpy as np
import sherpa_onnx
import sounddevice as sd

def create_audio_tagger():
    model_file = "./models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19/model.int8.onnx"
    label_file = "./models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19/class_labels_indices.csv"

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

def main():
    logging.info("Create audio tagger")
    audio_tagger = create_audio_tagger()

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')
    logging.info("Now, listening for audio events...")

    sample_rate = 16000
    samples_per_read = int(0.2 * sample_rate)  # 200ms per read
    buffer_duration = 2.0                       # 检测间隔 2秒
    buffer_read_count = int(buffer_duration / 0.2)  # 10 reads
    overlap_read_count = 10                      # 保留10个reads重叠 (2.0秒)

    buffer = []  # 音频缓冲

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as stream:
        while True:
            samples, _ = stream.read(samples_per_read)
            samples = samples.reshape(-1)
            buffer.append(samples)

            if len(buffer) >= buffer_read_count:
                # 累积满2秒，合并音频
                all_audio = np.concatenate(buffer)

                # 创建新流并检测
                s = audio_tagger.create_stream()
                s.accept_waveform(sample_rate, all_audio)
                result = audio_tagger.compute(s)

                # 门铃检测逻辑
                flag = False
                prob = 0.0
                for event in result:
                    if event.name == "Doorbell" or event.name == "Ding" or event.name == "Ding-dong":
                        prob += event.prob
                    if prob > 0.5:
                        flag = True

                if flag:
                    logging.info(f"检测到门铃声，概率为 {prob:.2f}")

                # 保留重叠部分，避免漏检
                buffer = buffer[-overlap_read_count:]

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    try:
        main()
    except KeyboardInterrupt:
        logging.info("Ctrl+C caught. Exiting.")
