"""
流式 ASR 模型测试模块

严格参照 python-api-examples 官方示例实现:
  - online-decode-files.py (transducer/zipformer)
  - online-nemo-ctc-decode-files.py (nemo_ctc streaming)

支持的模型类型:
  - nemo_streaming: NeMo Streaming FastConformer CTC
  - streaming_zipformer: Streaming Zipformer Transducer
"""

import json
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from path_utils import resolve_model_dir


def read_wave(wave_filename: str):
    """读取 wav 文件返回 (samples_float32, sample_rate)"""
    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    return audio[:, 0], sample_rate


def create_streaming_recognizer(model_type: str, model_dir: str, **kwargs):
    """
    创建流式 ASR 识别器
    
    Args:
        model_type: 模型类型 (nemo_streaming / streaming_zipformer)
        model_dir: 模型目录路径
    """
    # 转换为 ASCII 兼容路径
    model_dir = resolve_model_dir(model_dir)
    
    p = Path(model_dir)

    if model_type == "nemo_streaming":
        # 参照 online-nemo-ctc-decode-files.py
        # 使用 OnlineRecognizer.from_nemo_ctc
        model_path = p / "model.int8.onnx"
        tokens = p / "tokens.txt"
        
        for f in [model_path, tokens]:
            if not f.exists():
                raise FileNotFoundError(f"NeMo streaming file not found: {f}")

        print(f"  [INFO] Using OnlineRecognizer.from_nemo_ctc")
        return sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
            model=str(model_path),
            tokens=str(tokens),
            debug=False,
            num_threads=kwargs.get("num_threads", 2),
        )

    elif model_type == "streaming_zipformer":
        # 参照 online-decode-files.py (transducer 模式)
        encoder = p / "encoder.onnx"
        decoder = p / "decoder.onnx"
        joiner = p / "joiner.onnx"
        tokens = p / "tokens.txt"
        
        for f in [encoder, decoder, joiner, tokens]:
            if not f.exists():
                raise FileNotFoundError(f"Streaming zipformer file not found: {f}")
                
        print(f"  [INFO] Using OnlineRecognizer.from_transducer")
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=kwargs.get("num_threads", 2),
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )

    else:
        raise ValueError(f"Unknown streaming model type: {model_type}")


def run_streaming_asr_test(recognizer, model_type: str, audio_files: list, gt_texts: dict) -> dict:
    """
    运行流式 ASR 测试
    
    Args:
        recognizer: OnlineRecognizer 实例
        model_type: 模型类型名称
        audio_files: 音频文件路径列表
        gt_texts: {index: ground_truth_text} 字典
    
    Returns:
        结果字典
    """
    results = []
    total_time = 0
    total_audio_duration = 0
    exact_match_count = 0
    total_char_correct = 0
    total_char_total = 0
    
    n_files = len(audio_files)
    
    for i, wav_path in enumerate(audio_files):
        idx = i + 1
        try:
            audio, sample_rate = read_wave(wav_path)
            
            start_t = time.perf_counter()
            
            # 参照官方示例的流式解码方式
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            
            # 尾部填充（参照官方示例 0.66秒）
            tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
            stream.accept_waveform(sample_rate, tail_paddings)
            stream.input_finished()

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            elapsed = time.perf_counter() - start_t
            
            result_text = recognizer.get_result_all(stream).text.strip()
            duration = len(audio) / sample_rate
            total_time += elapsed
            total_audio_duration += duration
            
            gt = gt_texts.get(idx, "")
            
            # 计算字符级准确率 (大小写不敏感)
            gt_norm = gt.lower().strip()
            rec_norm = result_text.lower().strip()
            
            if gt_norm == rec_norm:
                exact_match_count += 1
                
            if gt_norm:
                correct = sum(1 for a, b in zip(gt_norm, rec_norm) if a == b)
                total_char_correct += correct
                total_char_total += max(len(gt_norm), len(rec_norm))
            
            results.append({
                "index": idx,
                "ground_truth": gt,
                "recognized": result_text,
                "time_sec": round(elapsed, 4),
                "audio_duration_sec": round(duration, 4),
                "exact_match": gt_norm == rec_norm,
            })
            
            status = "OK" if gt_norm == rec_norm else "DIFF"
            print(f"  [{i+1}/{n_files}] {status} | GT: \"{gt[:50]}\" -> ASR: \"{result_text[:50]}\" ({elapsed:.2f}s)")
            
        except Exception as e:
            print(f"  [{i+1}/{n_files}] ERROR: {e}")
            results.append({
                "index": idx,
                "ground_truth": gt_texts.get(idx, ""),
                "recognized": f"<ERROR: {str(e)}>",
                "time_sec": 0,
                "audio_duration_sec": 0,
                "exact_match": False,
            })

    avg_rtf = (total_time / total_audio_duration) if total_audio_duration > 0 else 0
    char_acc = (total_char_correct / total_char_total * 100) if total_char_total > 0 else 0
    exact_rate = (exact_match_count / n_files * 100) if n_files > 0 else 0

    result_data = {
        "model_type": model_type,
        "total_files": n_files,
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_rate, 2),
        "avg_char_accuracy": round(char_acc, 2),
        "total_time_sec": round(total_time, 2),
        "total_audio_duration_sec": round(total_audio_duration, 2),
        "rtf": round(avg_rtf, 4),
        "results": results,
    }
    
    print(f"\n  >>> {model_type.upper()}: Exact={exact_rate:.1f}% ({exact_match_count}/{n_files}), "
          f"CharAcc={char_acc:.1f}%, RTF={avg_rtf:.3f}")
    
    return result_data
