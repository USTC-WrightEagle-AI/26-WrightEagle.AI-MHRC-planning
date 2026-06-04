#!/usr/bin/env python3
"""
TTS 模型测试脚本 - Kokoro (英文)
使用 Kokoro TTS 将测试文本转换为音频，用于后续 ASR 测试
"""

import json
import os
import sys
import time
from pathlib import Path

import sherpa_onnx
import soundfile as sf

# 添加 test 目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from path_utils import resolve_model_dir
from test_texts import ALL_TEST_TEXTS


def create_kokoro_tts(model_dir: str, num_threads: int = 2):
    """创建 Kokoro TTS 实例"""
    # 使用短路径避免中文路径问题
    model_dir = resolve_model_dir(model_dir)
    model_path = Path(model_dir) / "model.int8.onnx"
    voices_path = Path(model_dir) / "voices.bin"
    tokens_path = Path(model_dir) / "tokens.txt"
    data_dir = Path(model_dir) / "espeak-ng-data"

    for p in [model_path, voices_path, tokens_path, data_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Kokoro model file not found: {p}")

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(model_path),
                voices=str(voices_path),
                tokens=str(tokens_path),
                data_dir=str(data_dir),
            ),
            provider="cpu",
            debug=False,
            num_threads=num_threads,
        )
    )
    if not tts_config.validate():
        raise ValueError("Invalid TTS config")

    return sherpa_onnx.OfflineTts(tts_config)


def run_tts_test(tts, output_dir: str, sid: int = 10):
    """
    运行 TTS 测试，生成所有音频文件
    返回结果字典列表
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"{'='*60}")
    print(f"Kokoro TTS Testing - Speaker ID: {sid}")
    print(f"Output directory: {output_dir}")
    print(f"Number of texts: {len(ALL_TEST_TEXTS)}")
    print(f"{'='*60}")

    total_elapsed = 0.0
    total_audio_duration = 0.0

    for idx, text in enumerate(ALL_TEST_TEXTS):
        filename = f"tts_kokoro_s{sid}_{idx:02d}.wav"
        filepath = os.path.join(output_dir, filename)

        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = sid
        gen_config.speed = 1.0
        gen_config.silence_scale = 0.2

        start = time.time()
        audio = tts.generate(text, gen_config)
        elapsed = time.time() - start

        if len(audio.samples) == 0:
            print(f"[FAIL] Text {idx}: Failed to generate audio")
            results.append({
                "index": idx,
                "text": text,
                "filename": filename,
                "filepath": filepath,
                "success": False,
                "audio_duration": None,
                "elapsed": elapsed,
                "rtf": None,
            })
            continue

        # 保存音频
        sf.write(
            filepath,
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
        )

        audio_duration = len(audio.samples) / audio.sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else float('inf')

        total_elapsed += elapsed
        total_audio_duration += audio_duration

        result = {
            "index": idx,
            "text": text,
            "filename": filename,
            "filepath": filepath,
            "success": True,
            "sample_rate": audio.sample_rate,
            "num_samples": len(audio.samples),
            "audio_duration": round(audio_duration, 3),
            "elapsed": round(elapsed, 3),
            "rtf": round(rtf, 3),
        }
        results.append(result)

        print(f"[OK] {filename} | Duration: {audio_duration:.2f}s | RTF: {rtf:.3f}")

    avg_rtf = total_elapsed / total_audio_duration if total_audio_duration > 0 else float('inf')

    summary = {
        "model": "kokoro-int8-en-v0_19",
        "provider": "cpu",
        "speaker_id": sid,
        "total_texts": len(ALL_TEST_TEXTS),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_audio_duration": round(total_audio_duration, 3),
        "total_elapsed": round(total_elapsed, 3),
        "avg_rtf": round(avg_rtf, 3),
        "results": results,
        "output_dir": output_dir,
    }

    print(f"\n--- Summary ---")
    print(f"Total: {summary['successful']}/{summary['total_texts']} succeeded")
    print(f"Total audio duration: {total_audio_duration:.2f}s")
    print(f"Total elapsed: {total_elapsed:.2f}s")
    print(f"Avg RTF: {avg_rtf:.3f}")

    return summary


def main():
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "kokoro-int8-en-v0_19"
    output_dir = project_root / "test" / "audio_output"

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    print("Creating Kokoro TTS instance...")
    tts = create_kokoro_tts(str(model_dir), num_threads=2)
    print("TTS instance created successfully.\n")

    summary = run_tts_test(tts, str(output_dir), sid=10)

    # 保存结果 JSON
    result_file = Path(output_dir) / "tts_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {result_file}")

    return summary


if __name__ == "__main__":
    main()
