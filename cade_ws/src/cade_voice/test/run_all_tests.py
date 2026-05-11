#!/usr/bin/env python3
"""
分步运行器 - 逐个模型独立运行，避免单个崩溃影响整体
每个模型的输出保存到单独的JSON文件，最后汇总生成报告
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "test"
    audio_output_dir = test_dir / "audio_output"
    sys.path.insert(0, str(test_dir))

    from path_utils import get_short_path, resolve_model_dir, cleanup_subst
    from test_texts import ALL_TEST_TEXTS, GENERAL_SENTENCES, ROBOCUP_COMMANDS
    gt_texts = {i+1: text for i, text in enumerate(ALL_TEST_TEXTS)}

    all_results = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "tts": None,
        "offline_asr": {},
        "online_asr": {},
    }

    # ============================================================
    # Step 1: TTS
    # ============================================================
    print("\n" + "=" * 70)
    print(" STEP 1/3: TTS Testing (Kokoro)")
    print("=" * 70)

    from test_tts import create_kokoro_tts, run_tts_test

    kokoro_model_dir = resolve_model_dir(str(project_root / "models" / "kokoro-int8-en-v0_19"))
    tts_result = None

    try:
        tts_instance = create_kokoro_tts(kokoro_model_dir, num_threads=2)
        tts_result = run_tts_test(tts_instance, str(audio_output_dir), sid=10)
        all_results["tts"] = tts_result
        print(f"\n[TTS] SUCCESS")
    except Exception as e:
        print(f"\n[TTS] FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["tts"] = {"error": str(e)}

    audio_files = sorted([str(f) for f in audio_output_dir.glob("tts_*.wav")])
    if not audio_files:
        print("\n[ERROR] No audio files! Cannot proceed.")
        _generate_report(all_results, test_dir)
        cleanup_subst()
        return

    audio_files_short = [get_short_path(f) for f in audio_files]
    print(f"[INFO] Found {len(audio_files)} audio files")

    # ============================================================
    # Step 2: 非流式 ASR (逐个模型独立运行)
    # ============================================================
    offline_models = [
        ("whisper", "sherpa-onnx-whisper-small.en", "Whisper Small EN"),
        ("moonshine", "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27", "Moonshine Tiny EN"),
        ("fire_red", "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25", "FireRed ASR2"),
        ("medasr", "sherpa-onnx-medasr-ctc-en-int8-2025-12-25", "MedaSR CTC EN"),
        ("qwen3", "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25", "Qwen3-ASR 0.6B"),
    ]

    print("\n" + "=" * 70)
    print(" STEP 2/3: Offline ASR Testing")
    print("=" * 70)

    for key, dir_name, desc in offline_models:
        _run_single_offline_asr(key, dir_name, desc, project_root,
                                audio_files_short, gt_texts, all_results, audio_output_dir)

    # ============================================================
    # Step 3: 流式 ASR
    # ============================================================
    online_models = [
        ("nemo_streaming", "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms-int8",
         "NeMo Streaming FastConformer"),
        ("streaming_zipformer", "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
         "Streaming Zipformer"),
    ]

    print("\n" + "=" * 70)
    print(" STEP 3/3: Streaming ASR Testing")
    print("=" * 70)

    for key, dir_name, desc in online_models:
        _run_single_online_asr(key, dir_name, desc, project_root,
                               audio_files_short, gt_texts, all_results, audio_output_dir)

    # ============================================================
    # Step 4: 保存结果 & 生成报告
    # ============================================================
    full_results_file = test_dir / "all_test_results.json"
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {full_results_file}")

    _generate_report(all_results, test_dir)

    cleanup_subst()


def _run_single_offline_asr(key, dir_name, desc, project_root, audio_files_short,
                             gt_texts, all_results, audio_output_dir):
    """运行单个非流式ASR模型测试"""
    from path_utils import resolve_model_dir
    print(f"\n--- Offline: {desc} ---")
    try:
        import importlib
        mod = importlib.import_module("test_offline_asr")
        recognizer = mod.create_offline_recognizer(
            key,
            str(resolve_model_dir(str(project_root / "models" / dir_name))),
            num_threads=2
        )
        result = mod.run_offline_asr_test(recognizer, key, audio_files_short, gt_texts)
        all_results["offline_asr"][key] = result
        json_file = audio_output_dir / f"offline_asr_{key}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Acc={result['avg_char_accuracy']:.1%} RTF={result['rtf']:.3f}")
    except FileNotFoundError as e:
        print(f"  [SKIP] File not found: {e}")
        all_results["offline_asr"][key] = {"error": str(e)}
    except SystemExit:
        print(f"  [CRASH] Native crash (SystemExit)")
        all_results["offline_asr"][key] = {"error": "Native crash / SystemExit"}
    except BaseException as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        all_results["offline_asr"][key] = {"error": f"{type(e).__name__}: {e}"}


def _run_single_online_asr(key, dir_name, desc, project_root, audio_files_short,
                           gt_texts, all_results, audio_output_dir):
    """运行单个流式ASR模型测试"""
    from path_utils import resolve_model_dir
    print(f"\n--- Online: {desc} ---")
    try:
        import importlib
        mod = importlib.import_module("test_online_asr")
        recognizer = mod.create_streaming_recognizer(
            key,
            str(resolve_model_dir(str(project_root / "models" / dir_name))),
            num_threads=2
        )
        result = mod.run_streaming_asr_test(recognizer, key, audio_files_short, gt_texts)
        all_results["online_asr"][key] = result
        json_file = audio_output_dir / f"online_asr_{key}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Acc={result['avg_char_accuracy']:.1%} RTF={result['rtf']:.3f}")
    except FileNotFoundError as e:
        print(f"  [SKIP] File not found: {e}")
        all_results["online_asr"][key] = {"error": str(e)}
    except SystemExit:
        print(f"  [CRASH] Native crash (SystemExit)")
        all_results["online_asr"][key] = {"error": "Native crash / SystemExit"}
    except BaseException as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        all_results["online_asr"][key] = {"error": f"{type(e).__name__}: {e}"}


def _generate_report(results: dict, test_dir: Path):
    """生成 Markdown 报告"""
    now = datetime.now()

    # 导入测试文本
    sys.path.insert(0, str(test_dir))
    from test_texts import GENERAL_SENTENCES, ROBOCUP_COMMANDS

    lines = []
    a = lines.append

    a("# RoboCup@Home Speech Model Test Report")
    a("")
    a(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    a("**Environment:** CPU | Windows | sherpa-onnx Python API")
    a("")
    a("## Table of Contents")
    a("")
    a("- [1. Overview](#1-overview)")
    a("- [2. Model List](#2-model-list)")
    a("- [3. Test Texts](#3-test-texts)")
    a("- [4. TTS Results](#4-tts-results)")
    a("- [5. Offline ASR Results](#5-offline-asr-results)")
    a("- [6. Streaming ASR Results](#6-streaming-asr-results)")
    a("- [7. Performance Comparison](#7-performance-comparison)")
    a("- [8. Audio Files](#8-audio-files)")
    a("- [9. Conclusions](#9-conclusions)")
    a("")

    # 1. Overview
    a("## 1. Overview")
    a("")
    a("| Item | Description |")
    a("|------|-------------|")
    a("| Goal | Evaluate speech models for RoboCup@Home Task 1 |")
    a("| Scope | TTS + Offline ASR (5) + Online ASR (2) |")
    a("| Hardware | CPU only |")
    a(f"| Date | {results.get('test_date', 'N/A')} |")
    a("")

    # 2. Model List
    a("## 2. Model List")
    a("")
    a("### 2.1 TTS Model")
    a("")
    a("| Model | Type | Language | Notes |")
    a("|-------|------|----------|-------|")
    a("| Kokoro int8 v0.19 | Offline TTS | English | Multi-voice support |")
    a("")
    a("### 2.2 Offline ASR Models")
    a("")
    a("| Model | Architecture | Language | Features |")
    a("|-------|-------------|----------|----------|")
    a("| Whisper Small EN | Transformer Enc-Dec | English | OpenAI, robust |")
    a("| Moonshine Tiny Quantized | E2E | English | Lightweight, edge-friendly |")
    a("| FireRed ASR2 | Paraformer-like | ZH+EN | Bilingual support |")
    a("| MedaSR CTC EN Int8 | NeMo CTC | English | Medical domain optimized |")
    a("| Qwen3-ASR 0.6B Int8 | LLM-based ASR | ZH+EN | Large language model based |")
    a("")
    a("### 2.3 Streaming ASR Models")
    a("")
    a("| Model | Architecture | Language | Latency |")
    a("|-------|-------------|----------|--------|")
    a("| NeMo FastConformer CTC 80ms | CTC Streaming | English | 80ms chunk |")
    a("| Streaming Zipformer Kroko | Transducer | English | Real-time |")
    a("")

    # 3. Test Texts
    a("## 3. Test Texts")
    a("")
    a(f"**Total: {len(GENERAL_SENTENCES) + len(ROBOCUP_COMMANDS)} texts** "
      f"(General: {len(GENERAL_SENTENCES)} + RoboCup Commands: {len(ROBOCUP_COMMANDS)}, all English)")
    a("")
    a("**General Conversation:**")
    a("")
    for i, t in enumerate(GENERAL_SENTENCES):
        a(f"{i+1}. {t}")
    a("")
    a("**RoboCup@Home Commands:**")
    a("")
    for i, t in enumerate(ROBOCUP_COMMANDS):
        a(f"{i+len(GENERAL_SENTENCES)+1}. {t}")
    a("")

    # 4. TTS Results
    a("## 4. TTS Results")
    a("")
    tts = results.get("tts", {})
    if tts and "error" not in tts:
        a(f"- **Model:** `{tts['model']}`")
        a(f"- **Speaker ID:** `{tts['speaker_id']}`")
        a(f"- **Generated:** `{tts.get('successful', '?')}/{tts.get('total_texts', '?')}` files")
        a(f"- **Total Duration:** `{tts.get('total_audio_duration', '?')}s`")
        a(f"- **Total Time:** `{tts.get('total_elapsed', '?')}s`")
        a(f"- **Avg RTF:** `{tts.get('avg_rtf', '?')}`")
        a("")
        a("**Details:**")
        a("")
        a("| # | File | Text Preview | Duration(s) | RTF |")
        a("|---|------|-------------|------------|-----|")
        for item in tts.get("results", []):
            preview = item['text'][:45].replace('|', '\\|')
            preview += '...' if len(item['text']) > 45 else ''
            a(f"| {item['index']} | {item['filename']} | {preview} | {item['audio_duration']} | {item['rtf']} |")
    else:
        err = tts.get("error", "No data") if tts else "Not run"
        a(f"**Error:** `{err}`")
    a("")

    # 5. Offline ASR
    a("## 5. Offline ASR Results")
    a("")
    offline = results.get("offline_asr", {})
    if offline:
        a("### 5.1 Summary")
        a("")
        a("| Model | Char Accuracy | Exact Match | RTF | Status |")
        a("|-------|-------------|-------------|-----|--------|")
        for k, v in offline.items():
            if "error" in v:
                short_err = v["error"][:50].replace('|', '\\|')
                a(f"| {k} | N/A | N/A | N/A | ERROR: `{short_err}` |")
            else:
                acc_pct = v['avg_char_accuracy'] / 100.0
                exact_pct = v['exact_match_rate'] / 100.0
                a(f"| {k} | {acc_pct:.1%} | {exact_pct:.1%} | "
                  f"{v['rtf']:.3f} | OK |")
        a("")

        for k, v in offline.items():
            if "error" in v:
                continue
            a(f"### 5.2 {k.upper()} Details")
            a("")
            a("| # | Ground Truth | Recognized | Match | Accuracy |")
            a("|---|-------------|------------|-------|----------|")
            for item in v.get("results", []):
                gt = item['ground_truth'].replace('|', '\\|')[:45]
                pred = item['recognized'].replace('|', '\\|')[:45]
                m = 'Y' if item['exact_match'] else 'N'
                a(f"| {item['index']} | {gt} | {pred} | {m} | - |")
            a("")
            a("")
    else:
        a("No data.")
    a("")

    # 6. Streaming ASR
    a("## 6. Streaming ASR Results")
    a("")
    online = results.get("online_asr", {})
    if online:
        a("### 6.1 Summary")
        a("")
        a("| Model | Char Accuracy | Exact Match | RTF | Status |")
        a("|-------|-------------|-------------|-----|--------|")
        for k, v in online.items():
            if "error" in v:
                short_err = v["error"][:50].replace('|', '\\|')
                a(f"| {k} | N/A | N/A | N/A | ERROR: `{short_err}` |")
            else:
                acc_pct = v['avg_char_accuracy'] / 100.0
                exact_pct = v['exact_match_rate'] / 100.0
                a(f"| {k} | {acc_pct:.1%} | {exact_pct:.1%} | "
                  f"{v['rtf']:.3f} | OK |")
        a("")

        for k, v in online.items():
            if "error" in v:
                continue
            a(f"### 6.2 {k.upper()} Details")
            a("")
            a("| # | Ground Truth | Recognized | Match |")
            a("|---|-------------|------------|-------|")
            for item in v.get("results", []):
                gt = item['ground_truth'].replace('|', '\\|')[:45]
                pred = item['recognized'].replace('|', '\\|')[:45]
                m = 'Y' if item['exact_match'] else 'N'
                a(f"| {item['index']} | {gt} | {pred} | {m} |")
            a("")
    else:
        a("No data.")
    a("")

    # 7. Comparison
    a("## 7. Performance Comparison")
    a("")
    all_models = []
    for k, v in offline.items():
        if "error" not in v:
            all_models.append({**v, "name": k, "mode": "Offline"})
    for k, v in online.items():
        if "error" not in v:
            all_models.append({**v, "name": k, "mode": "Streaming"})

    if all_models:
        all_models.sort(key=lambda x: x.get('avg_char_accuracy', 0), reverse=True)

        a("### 7.1 Ranked by Accuracy")
        a("")
        a("| Rank | Model | Mode | Char Accuracy | Exact Match | RTF |")
        a("|------|-------|------|---------------|-------------|-----|")
        for i, m in enumerate(all_models):
            acc_pct = m['avg_char_accuracy'] / 100.0
            exact_pct = m['exact_match_rate'] / 100.0
            a(f"| {i+1} | {m['name']} | {m['mode']} | {acc_pct:.1%} | "
              f"{exact_pct:.1%} | {m['rtf']:.3f} |")
        a("")

        speed_sorted = sorted(all_models, key=lambda x: x.get('rtf', float('inf')))
        a("### 7.2 Ranked by Speed (RTF)")
        a("")
        a("| Rank | Model | Mode | RTF | Char Accuracy |")
        a("|------|-------|------|-----|--------------|")
        for i, m in enumerate(speed_sorted):
            acc_pct = m['avg_char_accuracy'] / 100.0
            a(f"| {i+1} | {m['name']} | {m['mode']} | {m['rtf']:.3f} | {acc_pct:.1%} |")
        a("")

        best_acc = max(all_models, key=lambda x: x.get('avg_char_accuracy', 0))
        best_speed = min(all_models, key=lambda x: x.get('rtf', float('inf')))
        a("### 7.3 Recommendations")
        a("")
        acc_pct = best_acc['avg_char_accuracy'] / 100.0
        speed_acc_pct = best_speed['avg_char_accuracy'] / 100.0
        a(f"- **Best Accuracy:** `{best_acc['name']}` ({best_acc['mode']}) - "
          f"Accuracy: **{acc_pct:.1%}**")
        a(f"- **Fastest Speed:** `{best_speed['name']}` ({best_speed['mode']}) - "
          f"RTF: **{best_speed['rtf']:.3f}**")
        a("")
    else:
        a("No successful model tests to compare.")
    a("")

    # 8. Audio Files
    a("## 8. Audio Files")
    a("")
    a("All TTS-generated test audios:")
    a("``")
    a(str(test_dir.parent / "test" / "audio_output"))
    a("```")
    a("")
    a("**Naming:** `tts_kokoro_s{speaker_id}_{index:02d}.wav`")
    a("")
    a("These files can be used for:")
    a("- Future ASR model iteration testing")
    a("- RoboCup@Home Task 1 simulation")
    a("- Speech interaction system integration verification")
    a("")

    # 9. Conclusions
    a("## 9. Conclusions")
    a("")
    a("> Auto-generated report. Single-run CPU test results.")
    a("")
    a("### Test Methodology")
    a("")
    a("1. Kokoro TTS (Speaker ID=10) converts 16 test texts to audio")
    a("2. Each ASR model processes generated audio")
    a("3. Metrics:")
    a("   - **Char Accuracy**: SequenceMatcher similarity ratio")
    a("   - **Exact Match Rate**: Perfect match proportion")
    a("   - **RTF**: Processing time / audio duration (lower = faster)")
    a("")
    a("### Notes")
    a("")
    a("- All tests on **CPU**; GPU would significantly improve speed")
    a("- Synthetic speech (TTS-generated) may differ from real human voice")
    a("- Different models have different language strengths")
    a("- Consider multiple runs for stable averages")
    a("")

    total_ok = sum(1 for v in list(offline.values()) + list(online.values())
                    if "error" not in v)
    total_models = len(offline) + len(online)
    a("---")
    a(f"*Report: {total_ok}/{total_models} ASR models tested successfully*")
    a(f"*Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}*")

    report_path = test_dir / "TEST_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
