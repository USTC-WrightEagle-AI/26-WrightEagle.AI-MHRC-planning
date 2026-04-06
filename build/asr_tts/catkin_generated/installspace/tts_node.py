#！/home/tiger/miniforge3/envs/ros/bin/python
import rospy
from std_msgs.msg import String
import rospkg
import os
pkg_path = rospkg.RosPack().get_path('asr_tts')
import argparse
import time
import sounddevice as sd

import sherpa_onnx
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # TTS 相关参数
    parser.add_argument(
        "--kitten-model",
        type=str,
        default=os.path.join(pkg_path, "models/kitten-nano-en-v0_1-fp16/model.fp16.onnx"),
        help="Path to kitten TTS model",
    )

    parser.add_argument(
        "--kitten-voices",
        type=str,
        default=os.path.join(pkg_path, "models/kitten-nano-en-v0_1-fp16/voices.bin"),
        help="Path to kitten TTS voices",
    )

    parser.add_argument(
        "--kitten-tokens",
        type=str,
        default=os.path.join(pkg_path, "models/kitten-nano-en-v0_1-fp16/tokens.txt"),
        help="Path to kitten TTS tokens",
    )

    parser.add_argument(
        "--kitten-data-dir",
        type=str,
        default=os.path.join(pkg_path, "models/kitten-nano-en-v0_1-fp16/espeak-ng-data"),
        help="Path to kitten TTS data directory",
    )

    parser.add_argument(
        "--sid",
        type=int,
        default=0,
        help="Speaker ID for TTS",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="TTS speech speed",
    )

    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Text to convert to speech",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(pkg_path, "output.wav"),
        help="Path to save generated audio",
    )

    parser.add_argument(
        "--play",
        action="store_false",
        default=True,
        help="Play the generated audio (default: True)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Inference provider: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads",
    )

    return parser.parse_known_args()


def create_tts(args):
    """创建 TTS 引擎"""
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
                model=args.kitten_model,
                voices=args.kitten_voices,
                tokens=args.kitten_tokens,
                data_dir=args.kitten_data_dir,
            ),
            provider=args.provider,
            num_threads=args.num_threads,
        ),
    )
    if not tts_config.validate():
        raise ValueError("Invalid TTS config")
    
    tts = sherpa_onnx.OfflineTts(tts_config)
    return tts


def play_audio(audio, sample_rate):
    """播放音频"""
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


def generate_speech(tts, text, sid=0, speed=1.0):
    """生成语音
    
    Args:
        tts: TTS引擎实例
        text: 要转换的文本
        sid: 说话人ID
        speed: 语速
    
    Returns:
        audio: 生成的音频数据
        sample_rate: 采样率
    """
    start = time.time()
    audio = tts.generate(text, sid=sid, speed=speed)
    end = time.time()
    
    if len(audio.samples) == 0:
        raise ValueError("Error in generating audio")
    
    elapsed_seconds = end - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration
    
    print(f"TTS generation time: {elapsed_seconds:.3f}s")
    print(f"Audio duration: {audio_duration:.3f}s")
    print(f"RTF: {real_time_factor:.3f}")
    
    return audio.samples, audio.sample_rate

class TTSNode:
    def __init__(self, node_name:str):
        rospy.init_node(node_name)
        self.args, _ = get_args()
        print("Initializing TTS engine...")
        self.tts = create_tts(self.args) # 初始化 TTS 引擎
        self.tts_subscription_ = rospy.Subscriber('tts', String, self.TTS, queue_size=10)
    
    def TTS(self, msg: String):
        print(f"Generating speech for: '{msg.data}'")
        audio, sample_rate = generate_speech(self.tts, msg.data, self.args.sid, self.args.speed)
        # 保存音频
        sf.write(
            self.args.output,
            audio,
            samplerate=sample_rate,
            subtype="PCM_16",
        )
        print(f"Saved to {self.args.output}")

        # 播放音频
        if self.args.play:
            print("Playing audio...")
            play_audio(audio, sample_rate)


    

def main():
    node = TTSNode("tts_node")
    rospy.spin()

if __name__ == "__main__":
    main()
