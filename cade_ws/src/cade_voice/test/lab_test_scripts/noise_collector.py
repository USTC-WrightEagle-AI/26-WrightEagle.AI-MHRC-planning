#!/usr/bin/env python3
"""
噪声/语音采集脚本 - noise_collector.py

在机器人上采集环境噪声或带标注的语音数据。
支持两种模式：
  1. pure_noise   — 纯背景噪声录制（风扇、电机等），用于后续SNR合成测试
  2. speech       — 带标注语音采集（念预设文本），自动记录ground truth标签

用法:
    # 录制30秒纯噪声（场景标签: idle）
    python noise_collector.py --mode pure_noise --scene idle --duration 30

    # 录制机器人移动时的噪声
    python noise_collector.py --mode pure_noise --scene moving --duration 30

    # 带标注语音采集（逐条念预设文本）
    python noise_collector.py --mode speech --speaker test_user_01

    # 指定麦克风设备ID（先用 robot_env_probe.py 查看设备列表）
    python noise_collector.py --mode pure_noise --duration 20 --device-id 2

    # 自定义输出目录
    python noise_collector.py --mode pure_noise --duration 30 --output-dir ./my_noises
"""

import json
import os
import sys
import time
import wave
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

# 尝试导入 sounddevice
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    print("[WARNING] sounddevice not installed. Run: pip install sounddevice")

SAMPLE_RATE = 16000  # 目标采样率，与ASR模型一致


def list_audio_devices():
"""列出所有可用的音频输入设备"""
    if not SD_AVAILABLE:
        return []
    devices = sd.query_devices()
    input_devices = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            input_devices.append({
                "id": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "samplerate": int(d["default_samplerate"]),
            })
    return input_devices


def record_audio(duration_seconds, sample_rate=SAMPLE_RATE, device_id=None, channels=1):
    """
    录制音频
    返回 numpy 数组 (mono, float64)
    """
    print(f"    Recording {duration_seconds}s @ {sample_rate}Hz ...")
    if device_id is not None:
        print(f"    Using device ID: {device_id}")

    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float64",
        device=device_id,
        blocking=True,
    )
    return audio.flatten()


def save_wav(filepath, audio_data, sample_rate=SAMPLE_RATE):
    """保存为 WAV 文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sd.write(str(filepath), audio_data, samplerate=sample_rate)
    return str(filepath.resolve())


def record_pure_noise(args):
    """纯噪声录制模式"""
    output_dir = Path(args.output_dir) / "pure_noises"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = args.scene or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    duration = args.duration or 30

    filename = f"{scene}_{timestamp}_{duration}s.wav"
    filepath = output_dir / filename

    print(f"\n{'='*50}")
    print(f"  Pure Noise Recording Mode")
    print(f"  Scene:     {scene}")
    print(f"  Duration:  {duration}s")
    print(f"  Output:    {filepath.name}")
    print(f"{'='*50}")

    # 倒计时
    print(f"\n    Starting in 3 seconds... (keep quiet!)")
    for i in range(3, 0, -1):
        print(f"      {i}...")
        time.sleep(1)

    audio = record_audio(duration, device_id=args.device_id)
    saved_path = save_wav(filepath, audio)

    # 记录元信息
    meta = {
        "filename": filename,
        "filepath": str(saved_path),
        "mode": "pure_noise",
        "scene_label": scene,
        "duration_sec": duration,
        "sample_rate": SAMPLE_RATE,
        "device_id": args.device_id,
        "timestamp": datetime.now().isoformat(),
        "samples": len(audio),
        "file_size_bytes": os.path.getsize(saved_path) if os.path.exists(saved_path) else 0,
    }
    meta_path = output_dir / f"{filename}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 简单统计
    rms = float(__import__("numpy").sqrt((audio ** 2).mean()))
    peak = float(abs(audio).max())
    print(f"\n  [DONE] Saved: {saved_path}")
    print(f"         RMS level: {rms:.6f}, Peak: {peak:.6f}")
    print(f"         Meta:     {meta_path.name}")

    meta["rms_level"] = rms
    meta["peak_level"] = peak
    return meta


def record_speech(args):
    """带标注语音采集模式 — 逐条念预设文本"""
    # 导入预设文本集
    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))
    from test_texts import ALL_TEST_TEXTS

    speaker = args.speaker or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / "speech_collection" / f"{speaker}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings_meta = []

    print(f"\n{'='*50}")
    print(f"  Speech Collection Mode")
    print(f"  Speaker:   {speaker}")
    print(f"  Texts:     {len(ALL_TEST_TEXTS)} items")
    print(f"  Output:    {output_dir.name}/")
    print(f"{'='*50}")

    for idx, text in enumerate(ALL_TEST_TEXTS):
        text_short = text[:60] + ("..." if len(text) > 60 else "")
        filename = f"speech_{idx:03d}_{timestamp}.wav"
        filepath = output_dir / filename

        print(f"\n  --- [{idx+1}/{len(ALL_TEST_TEXTS)}] ---")
        print(f'  Text: "{text_short}"')
        print(f"    Press Enter when ready to speak, or 'q' to skip, 'x' to quit...")

        user_input = input("    > ").strip().lower()
        if user_input == "x":
            print("    Stopped by user.")
            break
        if user_input == "q":
            print(f"    Skipped #{idx+1}")
            continue

        print(f"    Recording now... (press Ctrl+C to stop early)")
        t0 = time.time()

        try:
            audio = record_audio(args.duration or 15, device_id=args.device_id)
            elapsed = time.time() - t0
        except KeyboardInterrupt:
            elapsed = time.time() - t0
            print(f"\n    Stopped after {elapsed:.1f}s")
            # 获取已录制的部分
            audio = sd.rec(int(elapsed * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                          channels=1, dtype="float64", device=args.device_id, blocking=False)
            # 这里简化处理：如果被中断就跳过
            continue

        saved_path = save_wav(filepath, audio)

        item_meta = {
            "idx": idx,
            "filename": filename,
            "filepath": str(saved_path),
            "ground_truth": text,
            "speaker": speaker,
            "sample_rate": SAMPLE_RATE,
            "recorded_duration_sec": round(len(audio) / SAMPLE_RATE, 2),
            "timestamp": datetime.now().isoformat(),
        }
        recordings_meta.append(item_meta)

        print(f"    Saved: {filename} ({len(audio)/SAMPLE_RATE:.1f}s)")

        time.sleep(0.5)  # 短暂间隔

    # 保存总标注文件
    manifest = {
        "collection_mode": "speech",
        "speaker": speaker,
        "timestamp": timestamp,
        "num_recordings": len(recordings_meta),
        "items": recordings_meta,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Recorded {len(recordings_meta)}/{len(ALL_TEST_TEXTS)} utterances")
    print(f"       Manifest: {manifest_path.name}")
    return manifest


def main():
    parser = ArgumentParser(description="Noise/Speech Collector for Robot Audio Testing")
    parser.add_argument("--mode", choices=["pure_noise", "speech"], required=True,
                       help="采集模式: pure_noise(纯噪声) 或 speech(带标注语音)")
    parser.add_argument("--scene", default=None,
                       help="场景标签 (仅pure_noise模式): idle, moving, fan_high, lab_busy 等")
    parser.add_argument("--duration", type=int, default=30,
                       help="每段录制时长秒数 (默认: 30)")
    parser.add_argument("--speaker", default=None,
                       help="说话人标识 (仅speech模式)")
    parser.add_argument("--device-id", type=int, default=None,
                       help="麦克风设备ID (用 robot_env_probe.py 查看)")
    parser.add_argument("--output-dir", default="./noise_data",
                       help="根输出目录 (默认: ./noise_data)")
    parser.add_argument("--list-devices", action="store_true",
                       help="列出可用音频设备后退出")
    args = parser.parse_args()

    if not SD_AVAILABLE:
        print("[ERROR] sounddevice is required but not installed.")
        print("       Install it with: pip install sounddevice")
        return 1

    if args.list_devices:
        devices = list_audio_devices()
        print(f"\nAvailable Input Devices ({len(devices)}):\n")
        for d in devices:
            print(f"  ID={d['id']:3d}  {d['name']:40s}  ch={d['channels']}  sr={d['samplerate']}Hz")
        return 0

    if args.mode == "pure_noise":
        record_pure_noise(args)
    elif args.mode == "speech":
        record_speech(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
