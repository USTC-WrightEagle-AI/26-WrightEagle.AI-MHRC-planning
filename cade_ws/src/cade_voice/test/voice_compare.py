"""
Kokoro TTS 音色对比生成器
用所有 11 个音色 (SID 0~10) 生成相同文本的音频，方便对比

输出: test/voice_comparison/
  s00_*.wav ~ s10_*.wav
"""

import os
import sys
import time
from pathlib import Path

import sherpa_onnx
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from path_utils import resolve_model_dir


# 选 3-4 句代表性文本覆盖不同场景（不要太长，方便快速试听）
COMPARISON_TEXTS = [
    {
        "id": "greeting",
        "text": "Hello, how are you doing today? I hope everything is going well.",
        "label": "日常问候",
    },
    {
        "id": "command",
        "text": "Bring me a coke from the kitchen, please.",
        "label": "RoboCup指令",
    },
    {
        "id": "question",
        "text": "What objects do you see on the dining table?",
        "label": "询问",
    },
    {
        "id": "narration",
        "text": "I am learning about artificial intelligence and machine learning these days.",
        "label": "叙述句",
    },
]

# 所有可用音色 (kokoro-int8-en-v0_19 支持 SID 0~10)
ALL_VOICES = list(range(11))  # 0, 1, 2, ..., 10


def create_tts(model_dir: str):
    """创建 Kokoro TTS 实例"""
    model_dir = resolve_model_dir(model_dir)
    m = Path(model_dir)

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(m / "model.int8.onnx"),
                voices=str(m / "voices.bin"),
                tokens=str(m / "tokens.txt"),
                data_dir=str(m / "espeak-ng-data"),
            ),
            provider="cpu",
            num_threads=2,
        )
    )
    if not tts_config.validate():
        raise ValueError("Invalid TTS config")
    return sherpa_onnx.OfflineTts(tts_config)


def main():
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "kokoro-int8-en-v0_19"
    output_dir = project_root / "test" / "voice_comparison"

    if not model_dir.exists():
        print(f"ERROR: Model dir not found: {model_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("Creating Kokoro TTS instance...")
    tts = create_tts(str(model_dir))
    print(f"TTS ready. Voices: {ALL_VOICES}")
    print(f"Texts:   {len(COMPARISON_TEXTS)} samples")
    print(f"Output:  {output_dir}")
    print()

    total = len(ALL_VOICES) * len(COMPARISON_TEXTS)
    done = 0
    start_time = time.time()

    for sid in ALL_VOICES:
        for item in COMPARISON_TEXTS:
            filename = f"s{sid:02d}_{item['id']}.wav"
            filepath = output_dir / filename

            gen_config = sherpa_onnx.GenerationConfig()
            gen_config.sid = sid
            gen_config.speed = 1.0

            t0 = time.time()
            audio = tts.generate(item["text"], gen_config)
            elapsed = time.time() - t0

            if len(audio.samples) == 0:
                print(f"[FAIL] {filename}")
                done += 1
                continue

            sf.write(str(filepath), audio.samples, audio.sample_rate, subtype="PCM_16")
            duration = len(audio.samples) / audio.sample_rate
            done += 1
            print(f"[{done:3d}/{total}] {filename} | {duration:.2f}s | {elapsed:.2f}s gen")

    elapsed_total = time.time() - start_time
    print(f"\nDone! {done} files in {elapsed_total:.1f}s")
    print(f"Output directory: {output_dir}")

    # 打印文件清单
    print("\n--- File listing ---")
    for f in sorted(output_dir.glob("*.wav")):
        info = sf.info(str(f))
        print(f"  {f.name}  ({info.duration:.2f}s)")


if __name__ == "__main__":
    main()
