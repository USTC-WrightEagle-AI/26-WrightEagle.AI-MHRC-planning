#!/usr/bin/env python3
"""
ASR/TTS 模块独立部署验证脚本 - robot_module_test.py

在机器人上逐一验证各模块是否能正常加载和推理：
  - VAD端点检测
  - 各离线ASR模型 (Whisper/Moonshine/Qwen3/FireRed/MedaSR/StreamingZipformer)
  - 各流式ASR模型 (NeMo Streaming / Streaming Zipformer)
  - TTS模型 (Kokoro)

每个模块跑一条最小验证样本，记录 PASS/FAIL、加载时间、推理延迟、内存占用。

用法:
    # 全模块验证
    python robot_module_test.py

    # 只验证ASR
    python robot_module_test.py --modules asr

    # 只验证 TTS
    python robot_module_test.py --modules tts

    # 只验证 VAD
    python robot_module_test.py --modules vad

    # 指定模型目录根路径
    python robot_module_test.py --models-base-path ../../models

    # 输出详细日志到文件
    python robot_module_test.py --log-file module_test.log
"""

import json
import os
import sys
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np

# 项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))

# 尝试导入 sherpa-onnx
try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    print("[ERROR] sherpa-onnx not available. Install first.")

# 尝试导入 soundfile
try:
    import soundfile as sf
except ImportError:
    SF_AVAILABLE = False
else:
    SF_AVAILABLE = True

SAMPLE_RATE = 16000


# ============================================================
# 模型配置 — 定义每个模型的文件路径规则
# ============================================================
MODEL_CONFIGS = {
    "vad": {
        "display": "VAD (Silero)",
        "type": "vad",
        "files": {"model": "silero_vad.onnx"},
    },
    "whisper": {
        "display": "Whisper small.en (int8)",
        "type": "offline_asr",
        "dir": "whisper",
        "files": {
            "encoder": "small.en-encoder.int8.onnx",
            "decoder": "small.en-decoder.int8.onnx",
            "tokens": "small.en-tokens.txt",
        },
    },
    "moonshine": {
        "display": "Moonshine (int8)",
        "type": "offline_asr",
        "dir": "moonshine",
        "files": {
            "encoder": "encoder_model.ort",
            "decoder": "decoder_model_merged.ort",
            "tokens": "tokens.txt",
        },
    },
    "qwen3": {
        "display": "Qwen3-ASR (int8)",
        "type": "offline_asr",
        "dir": "qwen3",
        "files": {
            "conv_frontend": "conv_frontend.onnx",
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "tokens_dir": "tokenizer/",
        },
    },
    "fire_red": {
        "display": "FireRed (int8)",
        "type": "offline_asr",
        "dir": "fire_red",
        "files": {"model": "model.int8.onnx", "tokens": "tokens.txt"},
    },
    "medasr": {
        "display": "MedaSR (int8)",
        "type": "offline_asr",
        "dir": "medasr",
        "files": {"model": "model.int8.onnx", "tokens": "tokens.txt"},
    },
    "streaming_zipformer": {
        "display": "StreamingZipformer (Kroko, int8)",
        "type": "streaming_asr",
        "dir": "streaming_zipformer",
        "files": {
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        },
    },
    "tts_kokoro": {
        "display": "TTS Kokoro (int8, en-v0.19)",
        "type": "tts",
        "dir": "kokoro-int8-en-v0_19",
        "files": {"model": "model.onnx", "voices": "voices.bin", "tokens": "tokens.txt", "dict": "espeak-ng-data/"},
    },
}


def get_models_base_path(custom_path=None):
    """获取模型根目录"""
    if custom_path:
        return Path(custom_path).resolve()
    # 默认: 项目根目录下的 models/
    base = PROJECT_ROOT / "models"
    if base.exists():
        return base.resolve()
    # 备选: 相对路径
    alt = PROJECT_ROOT / ".." / "models"
    if alt.exists():
        return alt.resolve()
    return base.resolve()


def check_model_files(model_key, models_base):
    """检查模型文件是否齐全"""
    config = MODEL_CONFIGS[model_key]
    model_dir = models_base
    if "dir" in config:
        model_dir = model_dir / config["dir"]

    missing = []
    for key, fname in config["files"].items():
        if fname.endswith("/"):
            # 目录
            full_path = model_dir / fname
            if not full_path.is_dir():
                missing.append(f"{key}:{fname}")
        else:
            full_path = model_dir / fname
            if not full_path.exists():
                missing.append(f"{key}:{fname}")

    return missing, str(model_dir)


def generate_test_audio(duration_sec=3.0, freq=440.0):
    """生成一个简单的测试正弦波信号 (模拟语音频段)"""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * freq * t)  # A4 note
    # 加一些包络避免爆音
    envelope = np.ones_like(signal)
    fade_samples = int(0.05 * SAMPLE_RATE)  # 50ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    signal *= envelope
    return signal


def generate_test_silence_with_voice(duration_sec=5.0, voice_start=1.0, voice_dur=2.0):
    """生成一段含静音区间的测试音频（用于VAD检测）"""
    total_samples = int(SAMPLE_RATE * duration_sec)
    silence_start = int(SAMPLE_RATE * voice_start)
    voice_samples = int(SAMPLE_RATE * voice_dur)
    audio = np.zeros(total_samples, dtype=np.float64)
    # 在中间插入一段"类语音"信号
    voice_sig = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, voice_dur, voice_samples))  # 440Hz
    end_idx = min(silence_start + voice_samples, total_samples)
    audio[silence_start:end_idx] = voice_sig[:end_idx - silence_start]
    return audio


def test_vad(model_dir_result, log):
    """测试 VAD 模块"""
    log["module"] = "VAD Silero"
    results = {}

    try:
        t0 = time.perf_counter()
        vad = sherpa_onnx.Vad.from_vad(
            silero_vad=model_dir_result + "/silero_vad.onnx"
        )
        results["load_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 用含静音的音频测VAD
        test_audio = generate_test_silence_with_voice(5.0, 1.0, 2.0)
        t1 = time.perf_counter()
        vad_stream = vad.create_stream()
        window_size = int(0.512 * SAMPLE_RATE)  # 512ms chunks
        detected_segments = []

        for start in range(0, len(test_audio), window_size):
            chunk = test_audio[start:start + window_size]
            vad_stream.accept_waveform(SAMPLE_RATE, chunk.tolist())
            while vad_stream.is_ready():
                vad_stream.pop()
            segment = vad_stream.finally_get_final_result()
            if segment and segment.start > 0:
                detected_segments.append({"start": segment.start, "duration": segment.duration})

        results["inference_time_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        results["detected_segments"] = len(detected_segments)
        results["status"] = "PASS" if len(detected_segments) > 0 else "WARN_NO_DETECTION"
        results["detail"] = f"Detected {len(detected_segments)} speech segment(s)"

    except Exception as e:
        results["status"] = "FAIL"
        results["detail"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


def test_offline_asr(model_key, model_dir_result, config, log):
    """测试离线 ASR 模型"""
    log["module"] = config["display"]
    results = {}

    try:
        # 构造 recognizer 配置
        t0 = time.perf_counter()
        
        if model_key == "whisper":
            recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    whisper=sherpa_offline_whisper(
                        encoder=str(Path(model_dir_result) / config["files"]["encoder"]),
                        decoder=str(Path(model_dir_result) / config["files"]["decoder"]),
                        tokens=str(Path(model_dir_result) / config["files"]["tokens"]),
                        language="en",
                ),
                tokens=str(Path(model_dir_result) / config["files"]["tokens"]),
            ),
        )
        recognizer = sherpa_onnx.OfflineRecognizer(recognizer_config)
        results["load_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 用测试音频做推理
        test_audio = generate_test_audio(3.0)
        t1 = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, test_audio.tolist())
        recognizer.decode(stream)
        text = stream.result.text
        results["inference_time_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        results["recognized_text"] = text
        results["status"] = "PASS"
        results["detail"] = f'Recognized: "{text}"'

    except Exception as e:
        results["status"] = "FAIL"
        results["detail"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


def test_streaming_asr(model_key, model_dir_result, config, log):
    """测试流式 ASR 模型"""
    log["module"] = config["display"]
    results = {}

    try:
        t0 = time.perf_counter()
        
        if model_key == "streaming_zipformer":
            recognizer_config = sherpa_online_recognizer_config(
                feat_config=online_feature_config(sample_rate=SAMPLE_RATE, feature_dim=80),
                model_config=sherpa_onnx.OnlineModelConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder_filename=str(Path(model_dir_result) / config["files"]["encoder"]),
                        decoder_filename=str(Path(model_dir_result) / config["files"]["decoder"]),
                        joiner_filename=str(Path(model_dir_result) / config["files"]["joiner"]),
                    ),
                    tokens=str(Path(model_dir_result) / config["files"]["tokens"]),
                ),
                endpoint_config=endpoint_config(rule1_min_utterance_length_ms=300),
            )
        recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
        results["load_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 流式推理
        test_audio = generate_test_audio(3.0)
        t1 = time.perf_counter()
        stream = recognizer.create_stream()
        chunk_size = int(0.32 * SAMPLE_RATE)  # 320ms chunks
        first_token_time = None
        partial_texts = []

        for start in range(0, len(test_audio), chunk_size):
            chunk = test_audio[start:start + chunk_size]
            stream.accept_waveform(SAMPLE_RATE, chunk.tolist())
            while recognizer.is_ready(stream):
                recognizer.decode_streams([stream])
            if recognizer.is_endpoint(stream):
                pass
            partial = recognizer.get_result(stream).text
            if partial and not first_token_time:
                first_token_time = time.perf_counter() - t1
            if partial:
                partial_texts.append(partial)

        recognizer.decode_streams([stream])
        final_text = recognizer.finalize_stream(stream).text
        results["inference_time_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        results["first_token_ms"] = round(first_token_time * 1000, 1) if first_token_time else None
        results["recognized_text"] = final_text
        results["status"] = "PASS"
        results["detail"] = f'Streaming result: "{final_text}"'

    except Exception as e:
        results["status"] = "FAIL"
        results["detail"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


def test_tts_kokoro(model_dir_result, config, log):
    """测试 TTS Kokoro 模型"""
    log["module"] = config["display"]
    results = {}

    try:
        t0 = time.perf_counter()
        tts = sherpa_onnx.OfflineTts(
            model_dir=model_dir_result,
            data_dir=str(Path(model_dir_result) / "espeak-ng-data"),
            tokens=str(Path(model_dir_result) / config["files"]["tokens"]),
            dict_dir=str(Path(model_dir_result) / config["files"]["dict"]),
            voices=str(Path(model_dir_result) / config["files"]["voices"]),
            language_code="en",
            provider="cpu",
            num_threads=4,
            speed=1.0,
        )
        results["load_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 生成一句测试文本
        test_text = "Hello, this is a quick test of the TTS system."
        t1 = time.perf_counter()
        audio = tts.generate(test_text, sid=0, speed=1.0)
        infer_ms = (time.perf_counter() - t1) * 1000
        samples = len(audio.samples)
        duration_sec = samples / SAMPLE_RATE if SAMPLE_RATE else samples / audio.sample_rate
        results["inference_time_ms"] = round(infer_ms, 1)
        results["generated_samples"] = samples
        results["generated_duration_sec"] = round(duration_sec, 2)
        results["rtf"] = round(infer_ms / 1000 / duration_sec, 3) if duration_sec > 0 else None
        results["status"] = "PASS"
        results["detail"] = f'Generated "{test_text[:40]}" -> {duration_sec:.2f}s audio, RTF={results["rtf"]}'

    except Exception as e:
        results["status"] = "FAIL"
        results["detail"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


# ============================================================
# 主流程
# ============================================================
MODULE_RUNNERS = {
    "vad": lambda mb, mk, cfg, log: test_vad(mb, log),
    "whisper": lambda mb, mk, cfg, log: test_offline_asr(mk, mb, cfg, log),
    "moonshine": lambda mb, mk, cfg, log: test_offline_asr(mk, mb, cfg, log),
    "qwen3": lambda mb, mk, cfg, log: test_offline_asr(mk, mb, cfg, log),
    "fire_red": lambda mb, mk, cfg, log: test_offline_asr(mk, mb, cfg, log),
    "medasr": lambda mb, mk, cfg, log: test_offline_asr(mk, mb, cfg, log),
    "streaming_zipformer": lambda mb, mk, cfg, log: test_streaming_asr(mk, mb, cfg, log),
    "tts_kokoro": lambda mb, mk, cfg, log: test_tts_kokoro(mb, cfg, log),
}

MODULE_CATEGORIES = {
    "vad": ["vad"],
    "asr": ["whisper", "moonshine", "qwen3", "fire_red", "medasr", "streaming_zipformer"],
    "tts": ["tts_kokoro"],
    "all": list(MODEL_CONFIGS.keys()),
}


def main():
    parser = ArgumentParser(description="Module Validation - Verify each ASR/TTS/VAD module on the robot")
    parser.add_argument("--modules", nargs="+", default=["all"],
                       choices=["all", "vad", "asr", "tts"] + list(MODEL_CONFIGS.keys()),
                       help="要验证的模块 (默认: all)")
    parser.add_argument("--models-base-path", default=None,
                       help="模型根目录 (默认自动查找项目下的 models/)")
    parser.add_argument("--log-file", default=None,
                       help="详细日志输出文件")
    parser.add_argument("--output-json", default="module_validation_results.json",
                       help="结果JSON输出文件名")
    args = parser.parse_args()

    if not SHERPA_AVAILABLE:
        print("[FATAL] sherpa-onnx is not installed. Cannot run tests.")
        return 1

    # 解析要测试的模块列表
    modules_to_test = []
    for m in args.modules:
        if m in MODULE_CATEGORIES:
            modules_to_test.extend(MODULE_CATEGORIES[m])
        elif m in MODEL_CONFIGS:
            modules_to_test.append(m)
    modules_to_test = list(dict.fromkeys(modules_to_test))  # 去重保序

    models_base = get_models_base_path(args.models_base_path)
    print(f"\n{'='*65}")
    print(f"  Robot Module Validation Tool")
    print(f"  Models Base: {models_base}")
    print(f"  Modules:     {', '.join(modules_to_test)}")
    print(f"  Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    all_results = []
    summary = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
    log_lines = []

    for model_key in modules_to_test:
        config = MODEL_CONFIGS[model_key]
        entry = {"key": model_key, "config": config["display"]}
        log_entry = {}
        print(f"--- [{config['display']}] ---")

        # 检查模型文件
        missing, model_dir = check_model_files(model_key, models_base)
        if missing:
            entry["status"] = "SKIP"
            entry["detail"] = f"Missing files: {missing}"
            entry["model_dir"] = model_dir
            summary["skip"] += 1
            print(f"  [SKIP] Missing model files in: {model_dir}")
            print(f"         Missing: {missing}")
            all_results.append(entry)
            continue

        entry["model_dir"] = model_dir
        print(f"  Model dir:  {model_dir}")

        # 执行测试
        runner = MODULE_RUNNERS.get(model_key)
        if runner:
            result = runner(model_dir, model_key, config, log_entry)
            entry.update(result)
            status = result.get("status", "UNKNOWN")
            if status == "PASS":
                summary["pass"] += 1
                print(f"  [PASS] {result.get('detail', '')}")
            elif status.startswith("WARN"):
                summary["warn"] += 1
                print(f"  [WARN] {result.get('detail', '')}")
            else:
                summary["fail"] += 1
                print(f"  [FAIL] {result.get('detail', '')}")
        else:
            entry["status"] = "SKIP"
            entry["detail"] = "No test runner defined"
            summary["skip"] += 1
            print(f"  [SKIP] No runner for {model_key}")

        all_results.append(entry)
        print()

    # ---- 汇总 ----
    total = summary["pass"] + summary["fail"] + summary["warn"] + summary["skip"]
    print(f"{'='*65}")
    print(f"  Summary:  PASS={summary['pass']}  FAIL={summary['fail']}  "
          f"WARN={summary['warn']}  SKIP={summary['skip']}  Total={total}")
    print(f"{''*65}")

    for r in all_results:
        icon = {"PASS": "+", "FAIL": "X", "WARN": "!", "SKIP": "-"}.get(r.get("status"), "?")
        detail = r.get("detail", "")[:70]
        load_ms = r.get("load_time_ms")
        infer_ms = r.get("inference_time_ms")
        extra = ""
        if load_ms:
            extra += f"  load={load_ms}ms"
        if infer_ms:
            extra += f"  infer={infer_ms}ms"
        print(f"  [{icon}] {r.get('config', r['key']):35s} | {detail}{extra}")

    # ---- 保存结果 ----
    output_report = {
        "timestamp": datetime.now().isoformat(),
        "models_base": str(models_base),
        "summary": summary,
        "results": all_results,
    }

    output_path = SCRIPT_DIR / args.output_json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved: {output_path}")

    if args.log_file:
        log_path = SCRIPT_DIR / args.log_file
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"  Log saved:     {log_path}")

    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
