"""
非流式 ASR 模型测试模块

严格参照 python-api-examples 官方示例实现:
  - offline-whisper-decode-files.py
  - offline-moonshine-decode-files.py  
  - offline-fire-red-asr-decode-files.py
  - offline-qwen3-asr-decode-files.py
  - offline-decode-files.py (nemo_ctc / medasr_ctc)

支持的模型类型:
  - whisper: OpenAI Whisper
  - moonshine: Moonshine ASR (使用 from_moonshine_v2)
  - fire_red: FireRed ASR2 CTC (使用 from_fire_red_asr_ctc)
  - medasr: MedaSR CTC (使用 from_medasr_ctc)
  - qwen3: Qwen3-ASR
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


def create_offline_recognizer(model_type: str, model_dir: str, **kwargs) -> sherpa_onnx.OfflineRecognizer:
    """
    创建非流式 ASR 识别器
    
    Args:
        model_type: 模型类型 (whisper/moonshine/fire_red/medasr/qwen3)
        model_dir: 模型目录路径
        kwargs: 其他参数 (num_threads 等)
    
    Returns:
        OfflineRecognizer 实例
    """
    # 转换为 ASCII 兼容路径
    model_dir = resolve_model_dir(model_dir)
    
    p = Path(model_dir)
    
    if model_type == "whisper":
        # 参照 offline-whisper-decode-files.py
        encoder = p / "small.en-encoder.int8.onnx"
        decoder = p / "small.en-decoder.int8.onnx"
        tokens = p / "small.en-tokens.txt"
        
        for f in [encoder, decoder, tokens]:
            if not f.exists():
                raise FileNotFoundError(f"Whisper file not found: {f}")
                
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            debug=False,
            num_threads=kwargs.get("num_threads", 2),
        )

    elif model_type == "moonshine":
        # 使用 moonshine_v2 API（只需要 encoder + merged_decoder + tokens）
        # 模型文件: encoder_model.ort, decoder_model_merged.ort, tokens.txt
        encoder = None
        decoder = None
        tokens = p / "tokens.txt"
        
        if not tokens.exists():
            raise FileNotFoundError(f"Moonshine tokens not found: {tokens}")
        
        # 查找 encoder
        for name in ["encoder_model.ort", "encode.int8.onnx"]:
            if (p / name).exists():
                encoder = str(p / name)
                break
        
        if not encoder:
            raise FileNotFoundError(f"Moonshine encoder not found in {model_dir}")
        
        # 查找 merged decoder
        for name in ["decoder_model_merged.ort", "uncached_decode.int8.onnx"]:
            if (p / name).exists():
                decoder = str(p / name)
                break
                
        if not decoder:
            raise FileNotFoundError(f"Moonshine decoder not found in {model_dir}")
            
        print(f"  [INFO] Moonshine using from_moonshine_v2:")
        print(f"         encoder={Path(encoder).name}, decoder={Path(decoder).name}")

        return sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=encoder,
            decoder=decoder,
            tokens=str(tokens),
            debug=False,
            num_threads=kwargs.get("num_threads", 2),
        )

    elif model_type == "fire_red":
        # FireRed ASR2 CTC - 单文件模式，使用 from_fire_red_asr_ctc
        # 文件: model.int8.onnx, tokens.txt
        model_path = p / "model.int8.onnx"
        tokens = p / "tokens.txt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"FireRed model not found: {model_path}")
        if not tokens.exists():
            raise FileNotFoundError(f"FireRed tokens not found: {tokens}")

        print(f"  [INFO] FireRed using from_fire_red_asr_ctc (single-file mode)")
        return sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
            model=str(model_path),
            tokens=str(tokens),
            debug=False,
            num_threads=kwargs.get("num_threads", 2),
        )

    elif model_type == "medasr":
        # MedaSR CTC - 单文件模式，使用 from_medasr_ctc
        # 文件: model.int8.onnx, tokens.txt
        model_path = p / "model.int8.onnx"
        tokens = p / "tokens.txt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"MedaSR model not found: {model_path}")
        if not tokens.exists():
            raise FileNotFoundError(f"MedaSR tokens not found: {tokens}")

        print(f"  [INFO] MedaSR using from_medasr_ctc")
        return sherpa_onnx.OfflineRecognizer.from_medasr_ctc(
            model=str(model_path),
            tokens=str(tokens),
            debug=False,
            num_threads=kwargs.get("num_threads", 2),
        )

    elif model_type == "qwen3":
        # Qwen3-ASR - 参照 offline-qwen3-asr-decode-files.py
        conv_frontend = p / "conv_frontend.onnx"
        encoder = p / "encoder.int8.onnx"
        decoder = p / "decoder.int8.onnx"
        tokenizer = p / "tokenizer"
        
        for f in [conv_frontend, encoder, decoder]:
            if not f.exists():
                raise FileNotFoundError(f"Qwen3-ASR file not found: {f}")
        if not tokenizer.is_dir():
            raise FileNotFoundError(f"Qwen3-ASR tokenizer dir not found: {tokenizer}")
            
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(conv_frontend),
            encoder=str(encoder),
            decoder=str(decoder),
            tokenizer=str(tokenizer),
            debug=False,
            num_threads=kwargs.get("num_threads", 2),
            max_new_tokens=128,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_offline_asr_test(recognizer, model_type: str, audio_files: list, gt_texts: dict) -> dict:
    """
    运行非流式 ASR 测试
    
    Args:
        recognizer: OfflineRecognizer 实例
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
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            recognizer.decode_stream(stream)
            elapsed = time.perf_counter() - start_t
            
            recognized = stream.result.text.strip()
            duration = len(audio) / sample_rate
            total_time += elapsed
            total_audio_duration += duration
            
            gt = gt_texts.get(idx, "")
            
            # 计算字符级准确率 (大小写不敏感)
            gt_norm = gt.lower().strip()
            rec_norm = recognized.lower().strip()
            
            if gt_norm == rec_norm:
                exact_match_count += 1
            
            # 字符准确率
            if gt_norm:
                correct = sum(1 for a, b in zip(gt_norm, rec_norm) if a == b)
                total_char_correct += correct
                total_char_total += max(len(gt_norm), len(rec_norm))
            
            results.append({
                "index": idx,
                "ground_truth": gt,
                "recognized": recognized,
                "time_sec": round(elapsed, 4),
                "audio_duration_sec": round(duration, 4),
                "exact_match": gt_norm == rec_norm,
            })
            
            status = "OK" if gt_norm == rec_norm else "DIFF"
            print(f"  [{i+1}/{n_files}] {status} | GT: \"{gt[:50]}\" -> ASR: \"{recognized[:50]}\" ({elapsed:.2f}s)")
            
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
