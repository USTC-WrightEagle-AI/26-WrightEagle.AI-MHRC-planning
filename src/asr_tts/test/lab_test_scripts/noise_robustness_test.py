#!/usr/bin/env python3
"""
噪声鲁棒性评估脚本 - noise_robustness_test.py

评估各 ASR 模型在不同信噪比(SNR)条件下的识别准确率变化趋势。
需要先运行 noise_mixer.py 生成带噪测试集，或者直接指定已生成的带噪音频目录。

用法:
    # 使用 noise_mixer.py 生成的带噪测试集进行评估
    python noise_robustness_test.py \
        --noisy-dir ./noisy_test_set/ \
        --gt-file ../test_ground_truth.txt

    # 只评估指定模型
    python noise_robustness_test.py \
        --noisy-dir ./noisy_test_set/ \
        --models whisper moonshine qwen3

    # 生成鲁棒性报告 Markdown
    python noise_robustness_test.py \
        --noisy-dir ./noisy_test_set/ \
        --generate-report

    # 指定模型根目录
    python noise_robustness_test.py \
        --noisy-dir ./noisy_test_set/ \
        --models-base-path ../../models/
"""

import json
import os
import re
import sys
import time
import traceback
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from glob import glob

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))

try:
    import sherpa_onnx
except ImportError:
    print("[FATAL] sherpa-onnx not installed.")
    sys.exit(1)


SAMPLE_RATE = 16000


# ============================================================
# 工具函数
# ============================================================
def char_accuracy(hyp, ref):
    if not ref:
        return 0.0
    h = hyp.lower().replace(" ", "")
    r = ref.lower().replace(" ", "")
    return SequenceMatcher(None, h, r).ratio() * 100


def parse_snr_from_filename(filename):
    """从文件名中提取SNR信息
    预期命名格式: cleanName_noiseName_snrXXdB.wav 或 cleanName_noiseName_clean.wav
    """
    basename = Path(filename).stem
    # 匹配 snrXXdB 或 clean
    match = re.search(r"(snr(-?\d+)dB|clean)", basename, re.IGNORECASE)
    if match:
        group = match.group(1)
        if group.lower() == "clean":
            return "clean", float("inf")
        snr_val = match.group(2)
        return f"snr{snr_val}dB", float(snr_val)
    return "unknown", None


def parse_source_name(filename):
    """从文件名提取原始干净音频名称"""
    basename = Path(filename).stem
    parts = basename.split("_")
    # 格式: cleanName_noiseName_snrXXdB -> 取第一部分作为clean name
    if len(parts) >= 1:
        return parts[0]
    return basename


def load_manifest(noisy_dir):
    """尝试加载 noise_mixer.py 生成的 manifest"""
    manifest_path = Path(noisy_dir) / "mix_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def group_files_by_snr(noisy_dir):
    """将带噪音频按 SNR 分组"""
    noisy_dir = Path(noisy_dir)
    wav_files = sorted(glob(str(noisy_dir / "*.wav")))
    groups = defaultdict(list)
    for fp in wav_files:
        snr_label, snr_val = parse_snr_from_filename(fp)
        groups[snr_label].append({
            "filepath": fp,
            "snr_value": snr_val,
            "source_clean": parse_source_name(fp),
        })
    return dict(groups)


# ============================================================
# 简化版 ASR 识别器（复用基准测试中的逻辑）
# ============================================================
class SimpleASREngine:
    """轻量级 ASR 引擎封装"""

    def __init__(self, model_key, model_dir):
        self.model_key = model_key
        self.model_dir = model_dir
        self.recognizer = None

    def init(self):
        """初始化模型"""
        t0 = time.perf_counter()
        md = self.model_dir

        if self.model_key == "whisper":
            cfg = sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    whisper=sherpa_offline_whisper(
                        encoder=str(md / "small.en-encoder.int8.onnx"),
                        decoder=str(md / "small.en-decoder.int8.onnx"),
                        tokens=str(md / "small.en-tokens.txt"),
                        language="en",
                ),
                tokens=str(md / "small.en-tokens.txt"),
            )
        elif self.model_key == "moonshine":
            cfg = sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    transducer=sherpa_offline_transducer(
                        encoder=str(md / "encoder_model.ort"),
                        decoder=str(md / "decoder_model_merged.ort"),
                        joiner="",
                ),
                tokens=str(md / "tokens.txt"),
            )
        elif self.model_key == "qwen3":
            cfg = sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    transducer=sherpa_offline_transducer(
                        encoder=str(md / "encoder.int8.onnx"),
                        decoder=str(md / "decoder.int8.onnx"),
                        joiner="",
                ),
                tokens_dir=str(md / "tokenizer/"),
            )
        elif self.model_key in ("fire_red", "medasr"):
            cfg = sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    transducer=sherpa_offline_transducer(
                        encoder=str(md / "model.int8.onnx"),
                        decoder="",
                        joiner="",
                ),
                tokens=str(md / "tokens.txt"),
            )
        else:
            raise ValueError(f"Unsupported model for robustness test: {self.model_key}")

        self.recognizer = sherpa_onnx.OfflineRecognizer(cfg)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"      [{self.model_key}] Loaded in {load_ms:.0f}ms")
        return load_ms

    def recognize(self, audio_path):
        """识别单个文件"""
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio.tolist())
        self.recognizer.decode(stream)
        return stream.result.text


# 支持的模型列表及目录映射
ROBUSTNESS_MODELS = {
    "whisper": "whisper",
    "moonshine": "moonshine",
    "qwen3": "qwen3",
    "fire_red": "fire_red",
    "medasr": "medasr",
}


def run_robustness_eval(models_to_test, models_base, snr_groups, gt_dict):
    """
    在各 SNR 级别上运行所有指定模型的鲁棒性评估

    Args:
        models_to_test: 模型key列表
        models_base: 模型根目录 Path
        snr_groups: {snr_label: [file_info_dict]} 
        gt_dict: {source_clean_name: ground_truth_text}

    Returns:
        结果字典
    """
    # 初始化所有模型引擎
    engines = {}
    for mk in models_to_test:
        if mk not in ROBUSTNESS_MODELS:
            print(f"    [SKIP] {mk} not supported in robustness test")
            continue
        model_dir = models_base / ROBUSTNESS_MODELS[mk]
        engine = SimpleASREngine(mk, model_dir)
        try:
            engine.init()
            engines[mk] = engine
        except Exception as e:
            print(f"    [FAIL] {mk}: {e}")

    if not engines:
        print("    [ERROR] No engines initialized!")
        return None

    # 按SNR级别排序（clean -> 高SNR -> 低SNR）
    snr_order = ["clean", "snr20dB", "snr15dB", "snr10dB", "snr5dB", "snr0dB"]
    sorted_snr_labels = [s for s in snr_order if s in snr_groups]
    remaining = [s for s in snr_groups.keys() if s not in snr_order]
    sorted_snr_labels += sorted(remaining)

    results = {mk: {} for mk in engines}

    for snr_label in sorted_snr_labels:
        files = snr_groups[snr_label]
        if not files:
            continue
        print(f"\n  >> SNR Level: {snr_label} ({len(files)} files)")

        for mk, engine in engines.items():
            cas = []
            for finfo in files:
                gt = gt_dict.get(finfo["source_clean"], "")
                try:
                    hyp = engine.recognize(finfo["filepath"])
                    ca = char_accuracy(hyp, gt)
                    cas.append(ca)
                except Exception as e:
                    print(f"      ERROR recognizing {finfo['filepath']}: {e}")
                    cas.append(0.0)

            avg_ca = sum(cas) / len(cas) if cas else 0
            results[mk][snr_label] = {
                "avg_char_accuracy": round(avg_ca, 1),
                "num_files": len(files),
                "per_file_ca": [round(c, 1) for c in cas],
            }
            bar_len = int(avg_ca / 2)
            bar = "#" * bar_len + "." * (50 - bar_len)
            print(f"      {mk:15s}  CA={avg_ca:5.1f}%  [{bar}]")

    return results


def generate_markdown_report(results, snr_groups, output_path):
    """生成 Markdown 鲁棒性报告"""
    lines = []
    lines.append("# ASR 噪声鲁棒性评估报告\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    # 表格: 各模型 x 各SNR级别的准确率
    models = list(results.keys())
    snr_labels = []
    if models:
        snr_labels = list(models[0].keys()) if results[models[0]] else []

    lines.append("## 准确率 vs SNR 表格 (%) \n")
    header = "| 模型 | " + " | ".join(snr_labels) + " |"
    sep = "|------|" + "|".join(["-------" for _ in snr_labels]) + "|"
    lines.append(header)
    lines.append(sep)

    for mk in models:
        row = f"| {mk} |"
        for sl in snr_labels:
            val = results[mk][sl]["avg_char_accuracy"] if sl in results[mk] else "N/A"
            row += f" {val} |"
        lines.append(row)

    lines.append("")
    lines.append("---\n")
    lines.append("## 分析要点\n")

    # 找出最鲁棒的模型
    best_overall = max(
        models,
        key=lambda m: sum(
            results[m].get(sl, {}).get("avg_char_accuracy", 0) for sl in snr_labels
        ) / len(snr_labels) if snr_labels else 0,
        default="N/A",
    )
    lines.append(f"- **综合最优**: {best_overall}\n")

    # 低SNR表现
    low_snr = [sl for sl in snr_labels if "0" in sl or "5" in sl]
    if low_snr:
        lines.append("- **低SNR环境 (<=5dB) 表现**:")
        for sl in low_snr:
            best_low = max(models, key=lambda m: results[m].get(sl, {}).get("avg_char_accuracy", 0), default="-")
            val = results.get(best_low, {}).get(sl, {}).get("avg_char_accuracy", "-")
            lines.append(f"  - {sl}: **{best_low}** ({val}%)\n")

    lines.append("")
    lines.append("*此报告由 `noise_robustness_test.py` 自动生成*\n")

    report_path = Path(output_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")


def build_gt_dict(noisy_dir, gt_texts=None):
    """构建 {source_clean_name: gt_text} 字典"""
    # 先尝试从 manifest 获取
    manifest = load_manifest(noisy_dir)
    if manifest:
        gt_map = {}
        seen_sources = set()
        for combo in manifest.get("combinations", []):
            src = combo.get("source_clean", "")
            gt = combo.get("ground_truth", "")
            if src and src not in seen_sources:
                gt_map[src] = gt
                seen_sources.add(src)
        if gt_map:
            return gt_map

    # 回退: 从 test_texts.py 匹配
    if gt_texts is None:
        from test_texts import ALL_TEST_TEXTS
        gt_texts = ALL_TEST_TEXTS

    # 从文件名推断源音频编号
    gt_map = {}
    for idx, text in enumerate(gt_texts):
        # 尝试匹配 tts_XXX 格式
        gt_map[f"tts_{idx:03d}"] = text
        gt_map[f"speech_{idx:03d}_*"] = text
    return gt_map


def main():
    parser = ArgumentParser(description="Noise Robustness Evaluation - ASR accuracy across SNR levels")
    parser.add_argument("--noisy-dir", required=True,
                       help="带噪测试集目录 (由 noise_mixer.py 生成)")
    parser.add_argument("--gt-file", default=None,
                       help="Ground truth 文件 (每行一句)")
    parser.add_argument("--models", nargs="*", default=list(ROBUSTNESS_MODELS.keys()),
                       help="要评估的模型 (默认全部离线模型)")
    parser.add_argument("--models-base-path", default=None,
                       help="模型根目录")
    parser.add_argument("--generate-report", action="store_true",
                       help="生成 Markdown 鲁棒性报告")
    parser.add_argument("--output-json", default="robustness_results.json",
                       help="结果JSON文件名")
    args = parser.parse_args()

    noisy_dir = Path(args.noisy_dir)
    models_base = Path(args.models_base_path) if args.models_base_path else (PROJECT_ROOT / "models")

    # 加载 GT 文本
    gt_texts = None
    if args.gt_file and Path(args.gt_file).exists():
        with open(args.gt_file, "r", encoding="utf-8") as f:
            gt_texts = [l.strip() for l in f.readlines() if l.strip()]

    print(f"\n{'='*65}")
    print(f"  Noise Robustness Evaluation Tool")
    print(f"{'='*65}")
    print(f"  Noisy audio dir: {noisy_dir}")
    print(f"  Models:          {', '.join(args.models)}")
    print(f"  Time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    # 分组
    snr_groups = group_files_by_snr(noisy_dir)
    print(f"  Found {sum(len(v) for v in snr_groups.values())} files across {len(snr_groups)} SNR levels:")
    for label, files in sorted(snr_groups.items()):
        print(f"    {label:12s}: {len(files)} files")

    if not snr_groups:
        print("[ERROR] No grouped audio files found.")
        return 1

    # 构建GT
    gt_dict = build_gt_dict(noisy_dir, gt_texts)
    print(f"\n  Ground truth entries: {len(gt_dict)}")

    # 过滤有效模型
    valid_models = [m for m in args.models if m in ROBUSTNESS_MODELS]
    print(f"  Models to evaluate:  {valid_models}")

    # 执行评估
    results = run_robustness_eval(valid_models, models_base, snr_groups, gt_dict)

    if not results:
        print("\n[ERROR] Evaluation failed.")
        return 1

    # 保存结果
    output = {
        "tool": "noise_robustness_test.py",
        "timestamp": datetime.now().isoformat(),
        "noisy_dir": str(noisy_dir),
        "snr_levels": list(snr_groups.keys()),
        "models_tested": list(results.keys()),
        "results": results,
    }

    output_path = SCRIPT_DIR / args.output_json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved: {output_path}")

    # 生成报告
    if args.generate_report:
        report_path = SCRIPT_DIR / "robustness_report.md"
        generate_markdown_report(results, snr_groups, report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
