#!/usr/bin/env python3
"""
噪声合成工具 - noise_mixer.py

将干净TTS音频与采集到的纯噪声按指定SNR混叠，生成噪声环境测试集。
用于 ASR 模型的噪声鲁棒性评估 (配合 noise_robustness_test.py 使用)。

用法:
    # 基础用法：用所有TTS音频 + 所有噪声文件生成全套SNR梯度测试集
    python noise_mixer.py \
        --clean-dir ../audio_output/ \
        --noise-dir ./noise_data/pure_noises/ \
        --output-dir ./noisy_test_set/

    # 指定SNR级别
    python noise_mixer.py \
        --clean-dir ../audio_output/ \
        --noise-dir idle_20260412_000000_30s.wav \
        --snr-levels clean 20 15 10 5 0 \
        --output-dir ./snr_test/

    # 只处理特定噪声类型
    python noise_mixer.py \
        --clean-dir ../audio_output/ \
        --noise-dir ./noise_data/pure_noises/ \
        --noise-filter idle moving \
        --output-dir ./filtered/

原理:
    SNR(dB) = 10 * log10(P_signal / P_noise)
    调整噪声幅度使目标SNR精确匹配：
        noise_scaled = noise * sqrt(P_signal / (P_noise * 10^(SNR/10)))
"""

import glob
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import soundfile as sf


# 默认SNR级别 (dB)，从干净到强噪声
DEFAULT_SNR_LEVELS = ["clean", 20, 15, 10, 5, 0]


def load_audio(path, target_sr=16000):
    """加载音频并重采样到目标采样率，返回 mono float64"""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # 转 mono
    if sr != target_sr:
        # 简单线性插值重采样
        import scipy.signal
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
    return data.astype(np.float64), target_sr


def compute_rms(audio):
    """计算 RMS 能量"""
    return np.sqrt(np.mean(audio ** 2))


def mix_snr(clean_audio, noise_audio, snr_db, eps=1e-10):
    """
    将 clean 和 noise 按 SNR 混合
    
    Args:
        clean_audio: 干净语音信号
        noise_audio: 噪声信号
        snr_db: 信噪比 (dB)，"clean" 表示不加噪声
        
    Returns:
        mixed_signal: 混合后的信号
        actual_snr: 实际达到的SNR (dB)
    """
    if isinstance(snr_db, str) and snr_db.lower() == "clean":
        return clean_audio.copy(), float("inf")

    snr_db = float(snr_db)
    
    # 如果噪声比信号短，循环扩展
    if len(noise_audio) < len(clean_audio):
        repeats = np.ceil(len(clean_audio) / len(noise_audio)).astype(int)
        noise_audio = np.tile(noise_audio, repeats)
    noise_audio = noise_audio[:len(clean_audio)]

    P_clean = compute_rms(clean_audio) ** 2
    P_noise_orig = compute_rms(noise_audio) ** 2

    if P_clean < eps:
        return clean_audio.copy(), float("inf")
    if P_noise_orig < eps:
        return clean_audio.copy(), float("inf")

    # 计算缩放因子: SNR = 10*log10(P_clean / (a^2 * P_noise))
    # => a = sqrt(P_clean / (P_noise * 10^(SNR/10)))
    scaling_factor = np.sqrt(P_clean / (P_noise_orig * 10 ** (snr_db / 10.0)))
    scaled_noise = noise_audio * scaling_factor

    mixed = clean_audio + scaled_noise

    # 防止削波（可选归一化）
    max_val = np.abs(mixed).max()
    if max_val > 0.99:
        mixed = mixed * (0.99 / max_val)

    # 验证实际SNR
    P_mixed_noise = compute_rms(scaled_noise) ** 2
    actual_snr = 10 * np.log10(P_clean / P_mixed_noise + eps)

    return mixed, actual_snr


def find_audio_files(directory_or_file, pattern="*.wav"):
    """查找音频文件，支持单个文件或目录"""
    path = Path(directory_or_file)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        files = sorted(glob.glob(os.path.join(str(path), pattern)))
        # 也检查子目录
        files += sorted(glob.glob(os.path.join(str(path), "*", pattern)))
        # 过滤掉非音频的json等文件
        files = [f for f in files if not f.endswith(".json")]
        return sorted(set(files))
    return []


def main():
    parser = ArgumentParser(description="Noise Mixer - 将干净音频与噪声按SNR混叠生成测试集")
    parser.add_argument("--clean-dir", required=True,
                       help="干净TTS音频目录或单个文件路径 (如 ../audio_output/)")
    parser.add_argument("--noise-dir", required=True,
                       help="采集的噪声目录或单个噪声文件路径 (如 ./noise_data/pure_noises/)")
    parser.add_argument("--output-dir", required=True,
                       help="输出目录")
    parser.add_argument("--snr-levels", nargs="+", default=DEFAULT_SNR_LEVELS,
                       help=f"SNR级别列表 (dB) (默认: {' '.join(map(str, DEFAULT_SNR_LEVELS))})")
    parser.add_argument("--noise-filter", nargs="*", default=[],
                       help="只使用文件名包含这些关键词的噪声 (如 idle moving)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="采样率 (默认: 16000)")
    args = parser.parse_args()

    clean_files = find_audio_files(args.clean_dir, "*.wav")
    noise_files = find_audio_files(args.noise_dir, "*.wav")

    if not clean_files:
        print(f"[ERROR] No WAV files found in: {args.clean_dir}")
        return 1
    if not noise_files:
        print(f"[ERROR] No WAV files found in: {args.noise_dir}")
        return 1

    # 过滤噪声文件
    if args.noise_filter:
        filtered = []
        for nf in noise_files:
            if any(f.lower() in os.path.basename(nf).lower() for f in args.noise_filter):
                filtered.append(nf)
        noise_files = filtered
        print(f"[INFO] Filtered to {len(noise_files)} noise file(s) matching: {args.noise_filter}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Noise Mixer - SNR Test Set Generator")
    print(f"{'='*60}")
    print(f"  Clean audios:  {len(clean_files)} file(s)")
    print(f"  Noises:        {len(noise_files)} file(s)")
    print(f"  SNR levels:    {args.snr_levels}")
    print(f"  Output dir:    {output_dir.resolve()}")
    print(f"{'='*60}\n")

    manifest = {
        "generator": "noise_mixer.py",
        "snr_levels": [str(s) for s in args.snr_levels],
        "sample_rate": args.sample_rate,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "combinations": [],
    }

    total_combinations = len(clean_files) * len(noise_files) * len(args.snr_levels)
    processed = 0

    for noise_path in noise_files:
        noise_name = Path(noise_path).stem
        print(f"\n>> Loading noise: {noise_name}")
        noise_audio, _ = load_audio(noise_path, args.sample_rate)

        for clean_path in clean_files:
            clean_name = Path(clean_path).stem
            print(f"  |-> Processing: {clean_name}")

            clean_audio, _ = load_audio(clean_path, args.sample_rate)

            for snr in args.snr_levels:
                snr_str = str(snr)
                # 输出命名: cleanName_noiseName_snrXXdB.wav
                safe_snr = "clean" if snr_str.lower() == "clean" else f"snr{snr_str}dB"
                out_name = f"{clean_name}_{noise_name}_{safe_snr}.wav"
                out_path = output_dir / out_name

                mixed, actual_snr = mix_snr(clean_audio, noise_audio, snr)
                sf.write(str(out_path), mixed.astype(np.float32), args.sample_rate)

                combo_entry = {
                    "output_file": out_name,
                    "source_clean": clean_name,
                    "source_noise": noise_name,
                    "target_snr_db": snr_str,
                    "actual_snr_db": round(actual_snr, 2),
                    "duration_sec": round(len(mixed) / args.sample_rate, 2),
                }
                manifest["combinations"].append(combo_entry)
                processed += 1

                progress = processed / total_combinations * 100
                print(f"      [{processed}/{total_combinations}] {out_name}  (actual SNR={actual_snr:.1f}dB)  [{progress:.0f}%]")

    # 保存清单
    manifest_path = output_dir / "mix_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Done! Generated {processed} noisy audio files.")
    print(f"  Manifest: {manifest_path.name}")
    print(f"  Output:   {output_dir.resolve()}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
