#!/usr/bin/env python3
"""
Streaming audio tagging from microphone using sherpa-onnx.

用 AudioTagging 实现流式麦克风输入
- 每2秒累积音频后进行一次检测
- 保留1秒重叠, 避免漏检声音边界
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import sherpa_onnx
import sounddevice as sd

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


def _device_arg_type(value):
    """argparse 类型: 把 '4' 解析成 int 4, 其他原样保留为 str."""
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return s


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=_device_arg_type,
        default="pulse",
        help="音频输入设备. 传数字 = 设备索引 (int), 传字符串 = 名称片段(忽略大小写). "
             "不传则使用系统默认输入设备. default 可以是 int 或 str.",
    )
    return parser.parse_known_args()[0]


def select_input_device(device_arg):
    """
    选择音频输入设备。
    优先级: --device 索引 > --device 名称片段 > sd.default.device[0] > 第一个输入设备
    """
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    # 1) 先打印所有设备 (含输出设备, 方便对照)
    print("All devices:")
    for i, d in enumerate(devices):
        marker = ""
        if d["max_input_channels"] > 0 and d["max_output_channels"] > 0:
            marker = " (in/out)"
        elif d["max_input_channels"] > 0:
            marker = " (in)"
        elif d["max_output_channels"] > 0:
            marker = " (out)"
        else:
            marker = " (none)"
        print(f"  [{i}] {d['name']}  in_ch={d['max_input_channels']}  "
              f"out_ch={d['max_output_channels']}{marker}")

    # 2) 再过滤出可用的输入设备
    input_devices = [
        (i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0
    ]
    if not input_devices:
        print("No input device with input channels found")
        sys.exit(0)

    print("Input-capable devices:")
    for i, d in input_devices:
        print(f"  [{i}] {d['name']}  (in_ch={d['max_input_channels']})")

    # 1. 用户没指定, 用系统默认
    if device_arg is None:
        idx = sd.default.device[0]
        if idx is None or idx < 0 or idx >= len(devices):
            idx = input_devices[0][0]
        print(f"--device 未指定, 使用系统默认输入设备 [{idx}]")
        return idx

    # 兼容 int 输入 (比如某些调用方传 int)
    if isinstance(device_arg, int):
        device_arg = str(device_arg)
    elif not isinstance(device_arg, str):
        device_arg = str(device_arg)

    # 2. 纯数字: 当作设备索引
    if device_arg.isdigit():
        idx = int(device_arg)
        for i, d in input_devices:
            if i == idx:
                print(f"按索引选中: [{idx}] {d['name']}")
                return idx
        print(f"索引 {idx} 不是有效的输入设备")
        sys.exit(1)

    # 3. 字符串: 按名称片段匹配 (忽略大小写)
    wanted = device_arg.strip().lower()
    matches = [(i, d) for i, d in input_devices if wanted in d["name"].lower()]
    if len(matches) == 1:
        i, d = matches[0]
        print(f"按名称片段 '{device_arg}' 选中: [{i}] {d['name']}")
        return i
    if len(matches) > 1:
        print(f"名称 '{device_arg}' 匹配到多个设备, 请精确指定:")
        for i, d in matches:
            print(f"  [{i}] {d['name']}")
        sys.exit(1)
    print(f"找不到名称包含 '{device_arg}' 的输入设备")
    sys.exit(1)


def main():
    logging.info("Create audio tagger")
    audio_tagger = create_audio_tagger()

    args = get_args()
    default_input_device_idx = select_input_device(args.device)
    print(f'Use device: [{default_input_device_idx}] {sd.query_devices(default_input_device_idx)["name"]}')
    logging.info("Now, listening for audio events...")

    sample_rate = 16000
    samples_per_read = int(0.2 * sample_rate)
    buffer_duration = 2.0
    buffer_read_count = int(buffer_duration / 0.2)
    overlap_read_count = 5

    buffer = []

    with sd.InputStream(device=default_input_device_idx, channels=1, dtype="float32", samplerate=sample_rate) as stream:
        while True:
            samples, _ = stream.read(samples_per_read)
            samples = samples.reshape(-1)
            buffer.append(samples)

            if len(buffer) >= buffer_read_count:
                all_audio = np.concatenate(buffer)

                s = audio_tagger.create_stream()
                s.accept_waveform(sample_rate, all_audio)
                result = audio_tagger.compute(s)

                if result:
                    top = result[0]
                    logging.info(f"检测结果: {top.name}, 概率为 {top.prob:.2f}")

                buffer = buffer[-overlap_read_count:]

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    try:
        main()
    except KeyboardInterrupt:
        logging.info("Ctrl+C caught. Exiting.")
