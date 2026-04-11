#!/usr/bin/env python3
"""
多模型性能基准对比脚本 - robot_benchmark.py

用统一的测试集在机器人上运行全部ASR模型，记录：
  - 准确率指标 (Char Accuracy, Exact Match Rate)
  - 性能指标 (RTF, 推理时间, 首/尾延迟)
  - 资源占用 (GPU/CPU利用率)

输出: benchmark_results.json + 终端排名表 + 对比图表(如matplotlib可用)

用法:
    # 基准测试 (使用已有的TTS测试音频)
    python robot_benchmark.py

    # 指定测试音频目录
    python robot_benchmark.py --audio-dir ../audio_output/

    # 只测离线模型
    python robot_benchmark.py --mode offline

    # 只测特定模型
    python robot_benchmark.py --models whisper moonshine qwen3

    # 保存图表
    python robot_benchmark.py --plot-chart

    # 指定GT文本文件 (每行一句，对应音频文件字母序)
    python robot_benchmark.py --gt-file test_ground_truth.txt
"""

import json
import os
import sys
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from collections import OrderedDict

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from test_texts import ALL_TEST_TEXTS

try:
    import sherpa_onnx
except ImportError:
    print("[FATAL] sherpa-onnx not installed.")
    sys.exit(1)


SAMPLE_RATE = 16000

# ============================================================
# 模型配置
# ============================================================
ASR_MODEL_CONFIGS = OrderedDict([
    ("whisper", {
        "display": "Whisper small.en",
        "type": "offline",
        "dir": "whisper",
        "init_kwargs_type": "whisper",
        "encoder_file": "small.en-encoder.int8.onnx",
        "decoder_file": "small.en-decoder.int8.onnx",
        "tokens_file": "small.en-tokens.txt",
    }),
    ("moonshine", {
        "display": "Moonshine",
        "type": "offline",
        "dir": "moonshine",
        "encoder_file": "encoder_model.ort",
        "decoder_file": "decoder_model_merged.ort",
        "tokens_file": "tokens.txt",
    }),
    ("qwen3", {
        "display": "Qwen3-ASR",
        "type": "offline",
        "dir": "qwen3",
        "conv_frontend_file": "conv_frontend.onnx",
        "encoder_file": "encoder.int8.onnx",
        "decoder_file": "decoder.int8.onnx",
        "tokens_dir": "tokenizer/",
    }),
    ("fire_red", {
        "display": "FireRed",
        "type": "offline",
        "dir": "fire_red",
        "model_file": "model.int8.onnx",
        "tokens_file": "tokens.txt",
    }),
    ("medasr", {
        "display": "MedaSR",
        "type": "offline",
        "dir": "medasr",
        "model_file": "model.int8.onnx",
        "tokens_file": "tokens.txt",
    }),
    ("streaming_zipformer", {
        "display": "StreamingZipformer",
        "type": "streaming",
        "dir": "streaming_zipformer",
        "encoder_file": "encoder.int8.onnx",
        "decoder_file": "decoder.int8.onnx",
        "joiner_file": "joiner.int8.onnx",
        "tokens_file": "tokens.txt",
    }),
])


def find_test_audio(audio_dir):
    """查找测试音频文件"""
    audio_dir = Path(audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))
    return wav_files


def load_gt_texts(gt_file=None):
    """加载 ground truth 文本"""
    if gt_file and Path(gt_file).exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return lines
    return list(ALL_TEST_TEXTS)


def char_accuracy(hyp: str, ref: str) -> float:
    """字符级准确率 (CER 补集)"""
    if not ref:
        return 0.0
    matcher = SequenceMatcher(None, hyp.lower().replace(" ", ""), ref.lower().replace(" ", ""))
    return matcher.ratio() * 100


def exact_match(hyp: str, ref: str) -> float:
    """完全匹配率"""
    return 1.0 if hyp.strip().lower() == ref.strip().lower() else 0.0


def word_error_rate(hyp: str, ref: str) -> float:
    """简单 WER 计算 (基于词编辑距离)"""
    hyp_words = hyp.strip().lower().split()
    ref_words = ref.strip().lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 100.0
    matcher = SequenceMatcher(None, hyp_words, ref_words)
    return (1 - matcher.ratio()) * 100


class OfflineASRBenchmarker:
    """离线 ASR 基准测试器"""

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.recognizer = None
        self.load_time_ms = 0

    def load(self):
        """加载模型"""
        t0 = time.perf_counter()
        cfg = self.config
        md = self.model_dir

        # 根据模型类型构造不同的初始化参数
        if cfg.get("init_kwargs_type") == "whisper":
            recog_cfg = sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    whisper=sherpa_offline_whisper(
                        encoder=str(md / cfg["encoder_file"]),
                        decoder=str(md / cfg["decoder_file"]),
                        tokens=str(md / cfg["tokens_file"]),
                        language="en",
                ),
                tokens=str(md / cfg["tokens_file"]),
            )
        )
        elif "conv_frontend_file" in cfg:
            # Qwen3 style: transducer with conv frontend
            recog_cfg = sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    transducer=sherpa_offline_transducer(
                        encoder=str(md / cfg["encoder_file"]),
                        decoder=str(md / cfg["decoder_file"]),
                        joiner="",  # Qwen3 doesn't use joiner?
                ),
                tokens_dir=str(md / cfg["tokens_dir"]),
            )
        else:
            # General transducer (Moonshine, FireRed, MedaSR)
            kwargs = dict(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    transducer=sherpa_offline_transducer(
                        encoder=str(md / cfg.get("encoder_file", "")),
                        decoder=str(md / cfg.get("decoder_file", "")),
                        joiner=cfg.get("joiner_file", ""),
                ),
                tokens=str(md / cfg["tokens_file"]),
            )

        self.recognizer = sherpa_onnx.OfflineRecognizer(recog_cfg)
        self.load_time_ms = round((time.perf_counter() - t0) * 1000, 1)

    def recognize(self, audio_path):
        """识别单个音频文件，返回 (text, inference_ms, audio_duration_sec)"""
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_dur = len(audio) / sr

        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio.tolist())
        self.recognizer.decode(stream)
        text = stream.result.text
        infer_ms = (time.perf_counter() - t0) * 1000
        return text, round(infer_ms, 1), round(audio_dur, 2)


class StreamingASRBenchmarker:
    """流式 ASR 基准测试器"""

    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.recognizer = None
        self.load_time_ms = 0

    def load(self):
        """加载模型"""
        t0 = time.perf_counter()
        cfg = self.config
        md = self.model_dir

        recog_cfg = sherpa_online_recognizer_config(
            feat_config=online_feature_config(sample_rate=SAMPLE_RATE, feature_dim=80),
            model_config=sherpa_onnx.OnlineModelConfig(
                transducer=sherpa_onnx.OnlineTransducerModelConfig(
                    encoder_filename=str(md / cfg["encoder_file"]),
                    decoder_filename=str(md / cfg["decoder_file"]),
                    joiner_filename=str(md / cfg["joiner_file"]),
                ),
                tokens=str(md / cfg["tokens_file"]),
            ),
            endpoint_config=endpoint_config(rule1_min_utterance_length_ms=300),
        )
        self.recognizer = sherpa_onnx.OnlineRecognizer(recog_cfg)
        self.load_time_ms = round((time.perf_counter() - t0) * 1000, 1)

    def recognize(self, audio_path, chunk_ms=320):
        """流式识别音频文件"""
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_dur = len(audio) / sr
        chunk_samples = int(chunk_ms * sr / 1000)

        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        first_token_ms = None
        last_partial = ""

        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            stream.accept_waveform(sr, chunk.tolist())

            while self.recognizer.is_ready(stream):
                self.recognizer.decode_streams([stream])

            result = self.recognizer.get_result(stream).text
            if result != last_partial and not first_token_ms:
                first_token_ms = (time.perf_counter() - t0) * 1000
            last_partial = result

        # Finalize
        self.recognizer.decode_streams([stream])
        final_text = self.recognizer.finalize_stream(stream).text
        infer_total_ms = (time.perf_counter() - t0) * 1000

        return final_text, round(infer_total_ms, 1), round(audio_dur, 2), round(first_token_ms, 1) if first_token_ms else None


def run_benchmark(model_key, config, model_dir, audio_files, gt_texts, mode_filter="all"):
    """运行单个模型的完整基准测试"""
    model_type = config.get("type", "offline")
    if mode_filter != "all" and mode_filter != model_type:
        return None

    print(f"\n  Benchmarking: {config['display']} ({model_type})")

    try:
        if model_type == "offline":
            bench = OfflineASRBenchmarker(config, model_dir)
        else:
            bench = StreamingASRBenchmarker(config, model_dir)

        bench.load()
        print(f"    Loaded in {bench.load_time_ms}ms")

        per_file_results = []
        total_infer_ms = 0
        total_audio_dur = 0

        for idx, audio_path in enumerate(audio_files):
            gt = gt_texts[idx] if idx < len(gt_texts) else ""

            if model_type == "offline":
                text, infer_ms, audio_dur = bench.recognize(str(audio_path))
            else:
                text, infer_ms, audio_dur, first_tok = bench.recognize(str(audio_path))

            total_infer_ms += infer_ms
            total_audio_dur += audio_dur

            ca = char_accuracy(text, gt)
            em = exact_match(text, gt)
            wer = word_error_rate(text, gt)

            file_result = {
                "file": audio_path.name,
                "gt": gt,
                "hyp": text,
                "char_accuracy": round(ca, 1),
                "exact_match": em,
                "wer": round(wer, 1),
                "infer_ms": infer_ms,
                "audio_dur_sec": audio_dur,
            }
            if model_type == "streaming":
                file_result["first_token_ms"] = first_tok

            per_file_results.append(file_result)
            status_icon = "OK" if ca > 80 else ("~~" if ca > 50 else "XX")
            print(f"    [{idx+1:2d}] {status_icon} CA={ca:5.1f}% WER={wer:5.1f}% "
                  f"{infer_ms:7.1f}ms | \"{text[:50]}\"")

        # 汇总
        n = len(per_file_results)
        avg_ca = sum(r["char_accuracy"] for r in per_file_results) / n if n else 0
        avg_wer = sum(r["wer"] for r in per_file_results) / n if n else 0
        em_rate = sum(r["exact_match"] for r in per_file_results) / n if n else 0
        avg_rtf = sum(r["infer_ms"] for r in per_file_results) / (total_audio_dur * 1000) if total_audio_dur > 0 else 0
        avg_first_token = sum(r.get("first_token_ms", 0) or 0 for r in per_file_results) / n if n and model_type == "streaming" else None

        model_summary = {
            "key": model_key,
            "display": config["display"],
            "type": model_type,
            "load_time_ms": bench.load_time_ms,
            "num_files": n,
            "avg_char_accuracy": round(avg_ca, 1),
            "avg_wer": round(avg_wer, 1),
            "exact_match_rate": round(em_rate * 100, 1),
            "avg_rtf": round(avg_rtf, 3),
            "avg_inference_ms": round(total_infer_ms / n, 1) if n else 0,
            "per_file": per_file_results,
        }
        if avg_first_token is not None:
            model_summary["avg_first_token_ms"] = round(avg_first_token, 1)

        print(f"    >>> Avg: CA={avg_ca:.1f}%  WER={avg_wer:.1f}%  EM={em_rate*100:.1f}%  RTF={avg_rtf:.3f}")

        return model_summary

    except Exception as e:
        print(f"    [FAIL] {e}")
        return {
            "key": model_key,
            "display": config["display"],
            "type": config.get("type", "?"),
            "status": "FAIL",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def plot_results(all_summaries, output_dir):
    """绘制对比图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    [INFO] matplotlib not available, skipping chart generation")
        return

    valid = [s for s in all_summaries if s.get("status") != "FAIL"]
    if len(valid) < 2:
        print("    [INFO] Not enough valid results for chart")
        return

    names = [s["display"] for s in valid]
    ca_vals = [s["avg_char_accuracy"] for s in valid]
    rtf_vals = [s["avg_rtf"] for s in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 准确率柱状图
    colors = ["green" if ca > 90 else "orange" if ca > 75 else "red" for ca in ca_vals]
    bars = ax1.bar(names, ca_vals, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_ylabel("Char Accuracy (%)")
    ax1.set_title("ASR Model Accuracy Comparison")
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars, ca_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax1.tick_params(axis="x", rotation=30, labelsize=8)

    # 图2: RTF vs Accuracy 散点图
    scatter = ax2.scatter(rtf_vals, ca_vals, s=150, c=ca_vals, cmap="RdYlGn",
                          edgecolors="black", linewidths=1, vmin=0, vmax=100)
    for i, name in enumerate(names):
        ax2.annotate(name, (rtf_vals[i], ca_vals[i]), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("RTF (Real-Time Factor, lower=better)")
    ax2.set_ylabel("Char Accuracy (%)")
    ax2.set_title("Speed vs Accuracy Trade-off")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label="Accuracy (%)")

    plt.tight_layout()
    chart_path = Path(output_dir) / "benchmark_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {chart_path}")


def main():
    parser = ArgumentParser(description="ASR Multi-Model Benchmark on Robot Hardware")
    parser.add_argument("--audio-dir", default=None,
                       help="测试音频目录 (默认: ../audio_output/)")
    parser.add_argument("--gt-file", default=None,
                       help="Ground truth 文本文件 (每行一句)")
    parser.add_argument("--mode", choices=["all", "offline", "streaming"], default="all",
                       help="测试模式")
    parser.add_argument("--models", nargs="*", default=None,
                       help="只测试指定模型 (如 whisper moonshine qwen3)")
    parser.add_argument("--models-base-path", default=None,
                       help="模型根目录")
    parser.add_argument("--plot-chart", action="store_true",
                       help="生成对比图表")
    parser.add_argument("--output-json", default="benchmark_results.json",
                       help="结果输出文件名")
    args = parser.parse_args()

    # 参数解析
    audio_dir = Path(args.audio_dir) if args.audio_dir else (SCRIPT_DIR.parent / "audio_output")
    models_base = Path(args.models_base_path) if args.models_base_path else (PROJECT_ROOT / "models")

    audio_files = find_test_audio(audio_dir)
    gt_texts = load_gt_texts(args.gt_file)

    if not audio_files:
        print(f"[ERROR] No .wav files found in: {audio_dir}")
        return 1
    if not gt_texts:
        print("[ERROR] No ground truth texts available.")
        return 1

    print(f"\n{'='*65}")
    print(f"  ASR Benchmark Tool")
    print(f"{'='*65}")
    print(f"  Audio files: {len(audio_files)} (from {audio_dir.name}/)")
    print(f"  GT texts:    {len(gt_texts)}")
    print(f"  Mode:        {args.mode}")
    print(f"  Models base: {models_base}")
    print(f"  Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    # 确定要测试的模型
    if args.models:
        test_models = [m for m in args.models if m in ASR_MODEL_CONFIGS]
    else:
        test_models = list(ASR_MODEL_CONFIGS.keys())

    all_summaries = []

    for model_key in test_models:
        config = ASR_MODEL_CONFIGS[model_key]
        model_dir = models_base / config["dir"] if "dir" in config else models_base
        result = run_benchmark(model_key, config, model_dir, audio_files, gt_texts, args.mode)
        if result:
            all_summaries.append(result)

    # ---- 排名汇总 ----
    print(f"\n{'='*65}")
    print(f"  RANKING (by Char Accuracy)")
    print(f"{'='*65}")
    ranked = sorted(
        [s for s in all_summaries if s.get("status") != "FAIL"],
        key=lambda x: x.get("avg_char_accuracy", 0),
        reverse=True,
    )
    for rank, s in enumerate(ranked, 1):
        rtf_str = f"RTF={s['avg_rtf']:.3f}" if s.get("avg_rtf") else ""
        ft_str = f"FTok={s['avg_first_token_ms']:.0f}ms" if s.get("avg_first_token_ms") else ""
        extras = "  ".join(filter(None, [rtf_str, ft_str]))
        print(f"  #{rank:2d}  {s['display']:28s}  CA={s['avg_char_accuracy']:5.1f}%  "
              f"WER={s.get('avg_wer', 0):5.1f}%  {extras}")

    # ---- 保存结果 ----
    output_report = {
        "tool": "robot_benchmark.py",
        "timestamp": datetime.now().isoformat(),
        "hardware": "robot (see robot_env_report.json for details)",
        "settings": {
            "audio_dir": str(audio_dir),
            "num_files": len(audio_files),
            "mode": args.mode,
        },
        "ranked_results": ranked,
        "all_results": all_summaries,
    }

    output_path = SCRIPT_DIR / args.output_json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved: {output_path}")

    # ---- 图表 ----
    if args.plot_chart:
        plot_results(all_summaries, SCRIPT_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())
