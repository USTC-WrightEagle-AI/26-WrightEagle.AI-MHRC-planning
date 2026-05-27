#!/usr/bin/env python3
"""
端到端语音交互系统测试脚本 - e2e_voice_interaction_test.py

测试完整的语音交互链路:
  用户录音 -> VAD检测 -> ASR识别 -> LLM处理 -> TTS回复 -> 播放

支持功能:
  - 单轮/多轮对话测试
  - 各阶段耗时统计与端到端总延迟
  - Mock LLM模式 (无需实际LLM API)
  - 使用预录制音频回放模式 (无需麦克风)
  - 自动生成延迟报告

用法:
    # 单轮对话测试 (使用mock LLM, 从麦克风录音)
    python e2e_voice_interaction_test.py --rounds 1

    # 多轮对话 (3轮)
    python e2e_voice_interaction_test.py --rounds 3

    # 用已有TTS音频作为"用户输入"来测试ASR+TTS链路 (不需要麦克风)
    python e2e_voice_interaction_test.py --rounds 5 --use-recorded-audio ../audio_output/

    # 指定ASR模型和TTS音色
    python e2e_voice_interaction_test.py --asr-model whisper --tts-sid 0

    # 生成延迟报告
    python e2e_voice_interaction_test.py --rounds 5 --generate-report
"""

import json
import os
import sys
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import numpy as np
import sounddevice as sd
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))

try:
    import sherpa_onnx
except ImportError:
    print("[FATAL] sherpa-onnx not installed. Run: pip install sherpa-onnx")
    sys.exit(1)

SAMPLE_RATE = 16000


# ============================================================
# Mock LLM — 用于无API时测试完整链路延迟
# ============================================================
MOCK_RESPONSES = OrderedDict([
    ("hello", "Hello! I'm the robot assistant. How can I help you today?"),
    ("coke", "Sure! I'll go to the kitchen and get you a coke right away."),
    ("living room", "Okay, navigating to the living room now."),
    ("what do you see", "I can see a table with some objects on it, including a green cup."),
    ("follow me", "Following you. Please lead the way."),
    ("pick up", "I've picked up the object. Where should I place it?"),
    ("weather", "The weather looks great today, perfect for outdoor activities."),
    ("help me find", "Of course! I'll help you search for it. Can you tell me more details?"),
    ("default", "I understand your request. Let me process that for you."),
])


class MockLLMClient:
    """Mock LLM — 根据关键词返回预设回复"""

    def __init__(self, latency_ms=200):
        self.latency_ms = latency_ms

    def chat(self, user_text: str) -> dict:
        """模拟 LLM 推理延迟后返回回复"""
        t0 = time.perf_counter()
        time.sleep(self.latency_ms / 1000.0)
        lower = user_text.lower()
        for keyword, response in MOCK_RESPONSES.items():
            if keyword in lower and keyword != "default":
                return {"text": response, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return {
            "text": MOCK_RESPONSES["default"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


# ============================================================
# E2E Pipeline
# ============================================================
class VoiceInteractionPipeline:
    """端到端语音交互流水线"""

    def __init__(self, models_base, asr_model="whisper", tts_sid=0,
                 mock_llm=True, llm_latency_ms=200, device_id=None):
        self.models_base = models_base
        self.asr_model = asr_model
        self.tts_sid = tts_sid
        self.device_id = device_id
        self.vad = None
        self.asr_recognizer = None
        self.tts_engine = None
        self.llm = MockLLMClient(llm_latency_ms) if mock_llm else None
        self.timeline = []

    def _stamp(self, stage: str) -> float:
        t = time.perf_counter()
        self.timeline.append({"stage": stage, "time_abs": t})
        return t

    def _elapsed_since(self, prev_stage: str) -> float:
        """计算从上一个阶段到当前的时间差(ms)"""
        if len(self.timeline) < 2:
            return 0
        return (self.timeline[-1]["time_abs"] - self.timeline[-2]["time_abs"]) * 1000

    def initialize(self):
        """加载所有组件"""
        md = self.models_base
        print("\n[*] Initializing pipeline...")

        # VAD
        self._stamp("init_start")
        print("    [1/4] Loading VAD...")
        vad_path = md / "silero_vad.onnx"
        if not vad_path.exists():
            vad_path = md / ".." / "silero_vad.onnx"
        self.vad = sherpa_onnx.Vad(vad=sherpa_onnx.VadConfig(
        ))
        self._stamp("vad_loaded")

        # ASR
        print(f"    [2/4] Loading ASR ({self.asr_model})...")
        asr_md = md / self._get_asr_dir()
        if not asr_md.exists():
            asr_md = md.parent / self._get_asr_dir()
        self.asr_recognizer = self._create_asr(asr_md)
        self._stamp("asr_loaded")

        # TTS
        print(f"    [3/4] Loading TTS (SID={self.tts_sid})...")
        tts_md = md / "kokoro-int8-en-v0_19"
        if not tts_md.exists():
            tts_md = md.parent / "kokoro-int8-en-v0_19"
        self.tts_engine = sherpa_onnx.OfflineTts(
            model_dir=str(tts_md),
            data_dir=str(tts_md / "espeak-ng-data") if (tts_md / "espeak-ng-data").exists() else "",
            tokens=str(tts_md / "tokens.txt"),
            voices=str(tts_md / "voices.bin"),
            language_code="en",
            provider="cpu",
            num_threads=4,
            speed=1.0,
        )
        self._stamp("tts_loaded")

        # LLM (mock)
        if self.llm:
            print(f"    [4/4] Mock LLM ready (simulated latency={self.llm.latency_ms}ms)")
        self._stamp("init_done")
        total_init = (self.timeline[-1]["time_abs"] - self.timeline[0]["time_abs"]) * 1000
        print(f"\n[OK] Pipeline ready! Total init time: {total_init:.0f}ms\n")

    def _get_asr_dir(self):
        dirs = {
            "whisper": "whisper", "moonshine": "moonshine",
            "qwen3": "qwen3", "fire_red": "fire_red", "medasr": "medasr",
        }
        return dirs.get(self.asr_model, self.asr_model)

    def _create_asr(self, asr_md):
        """创建 ASR recognizer"""
        if self.asr_model == "whisper":
            return sherpa_onnx.OfflineRecognizer(sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    whisper=sherpa_offline_whisper(
                        encoder=str(asr_md / "small.en-encoder.int8.onnx"),
                        decoder=str(asr_md / "small.en-decoder.int8.onnx"),
                        tokens=str(asr_md / "small.en-tokens.txt"),
                        language="en",
                ),
                tokens=str(asr_md / "small.en-tokens.txt"),
            )
        elif self.asr_model == "moonshine":
            return sherpa_onnx.OfflineRecognizer(sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    transducer=sherpa_offline_transducer(
                        encoder=str(asr_md / "encoder_model.ort"),
                        decoder=str(asr_md / "decoder_model_merged.ort"),
                        joiner="",
                ),
                tokens=str(asr_md / "tokens.txt"),
            )
        elif self.asr_model == "qwen3":
            return sherpa_onnx.OfflineRecognizer(sherpa_offline_recognizer_config(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=SAMPLE_RATE),
                model_config=sherpa_onnx.OfflineModelConfig(
                    transducer=sherpa_offline_transducer(
                        encoder=str(asr_md / "encoder.int8.onnx"),
                        decoder=str(asr_md / "decoder.int8.onnx"),
                        joiner="",
                ),
                tokens_dir=str(asr_md / "tokenizer/"),
            )
        else:
            raise ValueError(f"E2E test does not yet support ASR model: {self.asr_model}")

    def run_one_round(self, audio_input=None, play_output=False, timeout_sec=15,
                      audio_idx=None):
        """
        执行一轮完整对话
        
        Args:
            audio_input: 预录制的numpy数组 (None则从麦克风实时录音)
            play_output: 是否播放TTS回复音频
            timeout_sec: 录制超时
            audio_idx: 当前轮次索引(用于多轮模式)
        
        Returns:
            result dict with per-stage timings
        """
        self.timeline = []
        result = {
            "round_idx": audio_idx,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }
        self._stamp("round_start")

        # ===== Stage 1: 获取音频 =====
        if audio_input is not None:
            user_audio = audio_input.astype(np.float64)
            audio_dur = len(user_audio) / SAMPLE_RATE
            result["stages"]["input"] = {
                "method": "preloaded_audio",
                "duration_sec": round(audio_dur, 2),
                "samples": len(user_audio),
            }
        else:
            print("    >>> Speak now! Recording... (press Ctrl+C to stop)")
            try:
                user_audio = sd.rec(
                    int(timeout_sec * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float64",
                    device=self.device_id,
                    blocking=True,
                ).flatten()
            except KeyboardInterrupt:
                pass
            audio_dur = len(user_audio) / SAMPLE_RATE
            result["stages"]["input"] = {
                "method": "microphone_live",
                "duration_sec": round(audio_dur, 2),
                "samples": len(user_audio),
            }

        self._stamp("input_done")
        result["stages"]["input"]["elapsed_ms"] = self._elapsed_since("round_start")
        print(f"    [Stage-1/Input] Got audio ({audio_dur:.2f}s)")

        # ===== Stage 2: VAD 检测 =====
        print("    [Stage-2/VAD] Detecting speech segments...")
        vad_stream = self.vad.create_stream()
        window_size = int(0.32 * SAMPLE_RATE)
        segments = []

        for start in range(0, len(user_audio), window_size):
            chunk = user_audio[start:start + window_size]
            vad_stream.accept_waveform(SAMPLE_RATE, chunk.tolist())
            while vad_stream.is_ready():
                vad_stream.pop()
            seg = vad_stream.finally_get_final_result()
            if seg and seg.start >= 0:
                segments.append({"start": seg.start, "dur": seg.duration})

        self._stamp("vad_done")
        result["stages"]["vad"] = {
            "segments_found": len(segments),
            "elapsed_ms": self._elapsed_since("input_done"),
        }
        print(f"    [Stage-2/VAD] Found {len(segments)} speech segment(s) [{result['stages']['vad']['elapsed_ms']:.0f}ms]")

        # 提取有效语音段（如果有VAD分段则用分段后的，否则用原始）
        if segments:
            # 简单拼接所有检测到的语音段
            speech_parts = []
            for seg in segments:
                s = int(seg.start * SAMPLE_RATE)
                e = s + int(seg.dur * SAMPLE_RATE)
                speech_parts.append(user_audio[s:e])
            processed_audio = np.concatenate(speech_parts) if len(speech_parts) > 1 else speech_parts[0]
        else:
            processed_audio = user_audio  # VAD没检测到时用原始音频

        # ===== Stage 3: ASR 识别 =====
        print(f"    [Stage-3/ASR] Recognizing with {self.asr_model}...")
        stream = self.asr_recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, processed_audio.tolist())
        self.asr_recognizer.decode(stream)
        recognized_text = stream.result.text

        self._stamp("asr_done")
        result["stages"]["asr"] = {
            "recognized_text": recognized_text,
            "elapsed_ms": self._elapsed_since("vad_done"),
        }
        print(f'    [Stage-3/ASR] "{recognized_text}" [{result["stages"]["asr"]["elapsed_ms"]:.0f}ms]')

        # ===== Stage 4: LLM 处理 =====
        if self.llm:
            print("    [Stage-4/LLM] Processing...")
            llm_result = self.llm.chat(recognized_text)
            llm_reply = llm_result["text"]
        else:
            llm_reply = f"(No LLM configured. You said: {recognized_text})"
            llm_result = {"text": llm_reply, "latency_ms": 0}

        self._stamp("llm_done")
        result["stages"]["llm"] = {
            "reply_text": llm_reply,
            "latency_ms": llm_result.get("latency_ms", 0),
            "elapsed_ms": self._elapsed_since("asr_done"),
        }
        print(f'    [Stage-4/LLM] "{llm_reply[:60]}" [{result["stages"]["llm"]["elapsed_ms"]:.0f}ms]')

        # ===== Stage 5: TTS 生成回复 =====
        print(f"    [Stage-5/TTS] Generating reply audio (SID={self.tts_sid})...")
        tts_audio = self.tts_engine.generate(llm_reply, sid=self.tts_sid, speed=1.0)
        tts_duration = len(tts_audio.samples) / tts_audio.sample_rate if hasattr(tts_audio, 'sample_rate') else len(tts_audio.samples) / SAMPLE_RATE

        self._stamp("tts_done")
        result["stages"]["tts"] = {
            "output_samples": len(tts_audio.samples),
            "output_duration_sec": round(tts_duration, 2),
            "elapsed_ms": self._elapsed_since("llm_done"),
            "rtf": round(result["stages"]["tts"]["elapsed_ms"] / 1000 / tts_duration, 3) if tts_duration > 0 else 0,
        }
        print(f"    [Stage-5/TTS] Generated {tts_duration:.2f}s audio [{result['stages']['tts']['elapsed_ms']:.0f}ms, RTF={result['stages']['tts']['rtf']}]")

        # ===== Stage 6: 可选播放 =====
        if play_output:
            print("    [Stage-6/Playback] Playing response...")
            sd.play(tts_audio.samples.astype(np.float32),
                     samplerate=getattr(tts_audio, 'sample_rate', SAMPLE_RATE))
            sd.wait()

        self._stamp("round_end")
        # 计算端到端总延迟（从输入结束到 TTS 完成）
        result["total_e2e_ms"] = self._elapsed_since("input_done")  # 不含输入阶段
        result["total_including_input_ms"] = (self.timeline[-1]["time_abs"] -
                                              self.timeline[0]["time_abs"]) * 1000
        result["timeline_raw"] = self.timeline[:]

        print(f"\n  === Round Result ===")
        print(f"  Input:     {result['stages'].get('input', {}).get('duration_sec', '?')}s")
        print(f"  VAD:       {result['stages'].get('vad', {}).get('elapsed_ms', 0):.0f}ms")
        print(f"  ASR:       {result['stages'].get('asr', {}).get('elapsed_ms', 0):.0f}ms -> '{recognized_text[:40]}'")
        print(f"  LLM:       {result['stages'].get('llm', {}).get('elapsed_ms', 0):.0f}ms -> '{llm_reply[:40]}'")
        print(f"  TTS:       {result['stages'].get('tts', {}).get('elapsed_ms', 0):.0f}ms ({tts_duration:.2f}s output)")
        print(f"  E2E delay: {result['total_e2e_ms']:.0f}ms  (incl. input: {result['total_including_input_ms']:.0f}ms)")
        print(f"  Target < 1500ms: {'PASS' if result['total_e2e_ms'] < 1500 else 'EXCEEDED'}")

        return result


def load_test_audios(audio_dir, max_files=None):
"""从目录加载测试音频文件"""
    audio_dir = Path(audio_dir)
    wav_files = sorted(audio_dir.glob("*.wav"))
    if max_files:
        wav_files = wav_files[:max_files]
    audios = []
    for wf in wav_files:
        data, sr = sf.read(wf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            resampled = np.interp(
                np.linspace(0, len(data)-1, int(len(data) * SAMPLE_RATE / sr)),
                np.arange(len(data)), data,
            )
            audios.append(resampled.astype(np.float64))
        else:
            audios.append(data.astype(np.float64))
    return audios, [wf.name for wf in wav_files]


def main():
    parser = ArgumentParser(description="End-to-End Voice Interaction Test (ASR+LLM+TTS pipeline)")
    parser.add_argument("--rounds", type=int, default=1,
                       help="测试轮数 (默认: 1)")
    parser.add_argument("--asr-model", default="whisper",
                       choices=["whisper", "moonshine", "qwen3"],
                       help="ASR模型 (默认: whisper)")
    parser.add_argument("--tts-sid", type=int, default=0,
                       help="TTS音色ID 0-10 (默认: 0)")
    parser.add_argument("--use-recorded-audio", default=None,
                       help="用已有音频目录代替麦克风 (如 ../audio_output/)")
    parser.add_argument("--device-id", type=int, default=None,
                       help="麦克风设备ID")
    parser.add_argument("--mock-llm-latency-ms", type=int, default=200,
                       help="Mock LLM模拟延迟ms (默认: 200)")
    parser.add_argument("--models-base-path", default=None,
                       help="模型根目录")
    parser.add_argument("--play-output", action="store_true",
                       help="播放TTS生成的回复音频")
    parser.add_argument("--generate-report", action="store_true",
                       help="生成Markdown延迟报告")
    parser.add_argument("--output-json", default="e2e_test_results.json",
                       help="结果JSON文件名")
    args = parser.parse_args()

    models_base = Path(args.models_base_path) if args.models_base_path else (PROJECT_ROOT / "models")
    use_preloaded = args.use_recorded_audio is not None

    print(f"\n{'='*65}")
    print(f"  End-to-End Voice Interaction Test")
    print(f"{'='*65}")
    print(f"  Mode:          {'Preloaded audio' if use_preloaded else 'Live microphone'}")
    print(f"  Rounds:        {args.rounds}")
    print(f"  ASR Model:     {args.asr_model}")
    print(f"  TTS Voice:     SID={args.tts_sid}")
    print(f"  Mock LLM:      yes ({args.mock_llm_latency_ms}ms sim latency)")
    print(f"  Models base:   {models_base}")
    print(f"  Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    # 加载预录制音频 (如果使用)
    preloaded_audios = []
    audio_names = []
    if use_preloaded:
        preloaded_audios, audio_names = load_test_audios(args.use_recorded_audio)
        actual_rounds = min(args.rounds, len(preloaded_audios))
        if actual_rounds < args.rounds:
            print(f"  [WARN] Only {len(preloaded_audios)} audio files available, reducing rounds to {actual_rounds}")
    else:
        actual_rounds = args.rounds

    # 初始化流水线
    pipeline = VoiceInteractionPipeline(
        models_base=models_base,
        asr_model=args.asr_model,
        tts_sid=args.tts_sid,
        mock_llm=True,
        llm_latency_ms=args.mock_llm_latency_ms,
        device_id=args.device_id,
    )

    try:
        pipeline.initialize()
    except Exception as e:
        print(f"[FATAL] Failed to initialize pipeline: {e}")
        traceback.print_exc()
        return 1

    # 执行测试轮次
    all_round_results = []

    for i in range(actual_rounds):
        print(f"\n{'#'*50}")
        print(f"# ROUND {i+1}/{actual_rounds}" + (f"  (file: {audio_names[i]})" if use_preloaded else ""))
        print(f"{'#'*50}")

        audio_src = preloaded_audios[i] if use_preloaded else None
        result = pipeline.run_one_round(
            audio_input=audio_src,
            play_output=args.play_output,
            audio_idx=i + 1,
        )
        all_round_results.append(result)

        if i < actual_rounds - 1 and not use_preloaded:
            print("\n    Next round starting in 2 seconds...")
            time.sleep(2)

    # ===== 汇总报告 =====
    print(f"\n\n{'='*65}")
    print(f"  END-to-END TEST SUMMARY")
    print(f"{'='*65}")

    avg_e2e = sum(r["total_e2e_ms"] for r in all_round_results) / len(all_round_results)
    max_e2e = max(r["total_e2e_ms"] for r in all_round_results)
    min_e2e = min(r["total_e2e_ms"] for r in all_round_results)

    avg_stages = {}
    for stage in ["vad", "asr", "llm", "tts"]:
        vals = [r["stages"].get(stage, {}).get("elapsed_ms", 0) for r in all_round_results]
        avg_stages[stage] = sum(vals) / len(vals) if vals else 0

    print(f"  Rounds completed: {len(all_round_results)}")
    print(f"  Avg E2E Delay:   {avg_e2e:.0f}ms  (target < 1500ms)")
    print(f"  Min E2E Delay:   {min_e2e:.0f}ms")
    print(f"  Max E2E Delay:   {max_e2e:.0f}ms")
    print(f"  Pass rate (<1.5s):{sum(1 for r in all_round_results if r['total_e2e_ms']<1500)}/{len(all_round_results)}")
    print(f"  Stage breakdown (avg):")
    for stage, ms in avg_stages.items():
        pct = ms / avg_e2e * 100 if avg_e2e > 0 else 0
        bar_len = int(pct / 2)
        bar = "#" * bar_len + "." * (50 - bar_len)
        print(f"    {stage:6s}: {ms:7.0f}ms  ({pct:5.1f}%)  [{bar}]")

    # 保存JSON结果
    output_report = {
        "tool": "e2e_voice_interaction_test.py",
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "asr_model": args.asr_model,
            "tts_sid": args.tts_sid,
            "mock_llm_latency_ms": args.mock_llm_latency_ms,
            "mode": "preloaded_audio" if use_preloaded else "live_mic",
            "rounds_completed": len(all_round_results),
        },
        "summary": {
            "avg_e2e_ms": round(avg_e2e, 1),
            "min_e2e_ms": round(min_e2e, 1),
            "max_e2e_ms": round(max_e2e, 1),
            "pass_rate_1_5s": f"{sum(1 for r in all_round_results if r['total_e2e_ms']<1500)}/{len(all_round_results)}",
            "avg_stage_ms": {k: round(v, 1) for k, v in avg_stages.items()},
        },
        "per_round_results": all_round_results,
    }

    output_path = SCRIPT_DIR / args.output_json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved: {output_path}")

    # 生成 Markdown 报告
    if args.generate_report:
        report_lines = [
            "# E2E Voice Interaction Test Report\n",
            f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "## Configuration\n",
            f"- ASR Model: `{args.asr_model}`\n",
            f"- TTS SID: `{args.tts_sid}`\n",
            f"- Mode: `{'Preloaded Audio' if use_preloaded else 'Live Microphone'}`\n",
            f"- Rounds: `{len(all_round_results)}`\n",
            "\n## Summary\n",
            f"| Metric | Value |\n|--------|-------|\n",
            f"| Avg E2E Delay | **{avg_e2e:.0f}** ms |\n",
            f"| Min E2E Delay | {min_e2e:.0f} ms |\n",
            f"| Max E2E Delay | {max_e2e:.0f} ms |\n",
            f"| Target <1.5s | {sum(1 for r in all_round_results if r['total_e2e_ms']<1500)}/{len(all_round_results)} PASS |\n",
            "\n## Per-Round Details\n",
            "| Round | VAD(ms) | ASR(ms) | LLM(ms) | TTS(ms) | E2E(ms) | Status |\n",
            "|-------|---------|---------|---------|---------|---------|--------|\n",
        ]
        for r in all_round_results:
            stages = r["stages"]
            v = stages.get("vad", {}).get("elapsed_ms", 0)
            a = stages.get("asr", {}).get("elapsed_ms", 0)
            l = stages.get("llm", {}).get("elapsed_ms", 0)
            t = stages.get("tts", {}).get("elapsed_ms", 0)
            e2e = r["total_e2e_ms"]
            ok = "**OK**" if e2e < 1500 else "SLOW"
            report_lines.append(
                f"| {r.get('round_idx', '?')} | {v:.0f} | {a:.0f} | {l:.0f} | {t:.0f} | {e2e:.0f} | {ok} |\n"
            )

        report_lines.append("\n---\n*Auto-generated by `e2e_voice_interaction_test.py`*\n")

        report_path = SCRIPT_DIR / "e2e_test_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("".join(report_lines))
        print(f"  Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
