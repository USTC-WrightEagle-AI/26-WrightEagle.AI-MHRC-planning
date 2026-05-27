#!/usr/bin/env python3
"""
Voice Bridge Node - CADE 语音桥接节点 (Refactored)

纯语音层职责：
1. 监听 ASR 识别结果（从 /asr 话题）
2. 将结果转发给大脑处理
3. 接收大脑回复并发布到 /tts 播放

与旧版关键区别：
- 不直接引用 RobotController（通过 /asr -> BrainNode -> /tts 解耦）
- 纯语音管道，不包含任何 LLM 或控制逻辑
- 负责回声抑制和状态管理
"""

import sys
import argparse
import threading
import json

try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class VoiceBridgeNode:
    """
    Voice Bridge - 纯语音管道

    架构（与 BrainNode 解耦）：
    ASR input -> /asr -> [BrainNode processes] -> /tts -> VoiceBridge plays audio

    本节点可选功能：
    - 回声抑制：在 TTS 播放期间忽略 ASR 输入
    - 状态转发：将语音状态发布给其他节点
    """

    def __init__(self, echo_suppression: bool = True):
        if ROS_AVAILABLE:
            rospy.init_node('cade_voice_bridge', anonymous=True)

        self.echo_suppression = echo_suppression
        self.is_speaking = False
        self._lock = threading.Lock()

        if ROS_AVAILABLE:
            # 订阅 /asr 接收语音识别结果，转发到 /brain/asr_input
            self.asr_sub = rospy.Subscriber(
                '/asr', String, self._on_asr, queue_size=10
            )

            # 订阅 /tts 接收大脑的语音合成请求
            self.tts_sub = rospy.Subscriber(
                '/tts', String, self._on_tts, queue_size=10
            )

            # 发布 TTS 播放状态
            self.tts_status_pub = rospy.Publisher(
                '/tts/status', String, queue_size=10
            )

            rospy.loginfo("=" * 60)
            rospy.loginfo("Voice Bridge Node initialized")
            rospy.loginfo(f"  Echo suppression: {echo_suppression}")
            rospy.loginfo(f"  Subscribing: /asr, /tts")
            rospy.loginfo(f"  Publishing: /tts/status")
            rospy.loginfo("=" * 60)

        self.total_asr = 0
        self.total_tts = 0
        self.suppressed = 0

    def _on_asr(self, msg: String):
        """ASR 识别结果回调"""
        text = msg.data.strip()
        if not text:
            return

        self.total_asr += 1

        # 回声抑制
        with self._lock:
            if self.echo_suppression and self.is_speaking:
                self.suppressed += 1
                rospy.logdebug(f"[VoiceBridge] ASR suppressed (TTS active): '{text}'")
                return

        print(f"\n[VoiceBridge] ASR: '{text}'")
        # ASR 消息会自然流向 BrainNode（BrainNode 也订阅 /asr）

    def _on_tts(self, msg: String):
        """TTS 播放请求回调"""
        text = msg.data.strip()
        if not text:
            return

        self.total_tts += 1
        print(f"[VoiceBridge] TTS: '{text}'")

        with self._lock:
            self.is_speaking = True

        # 通知状态
        self._publish_tts_status("playing", text=text)

        # 模拟 TTS 播放时长（实际应由 TTS 播放完成事件驱动）
        estimated_duration = max(1.0, len(text) * 0.1)

        # 异步等待播放完成
        threading.Thread(
            target=self._tts_complete_after,
            args=(estimated_duration,),
            daemon=True
        ).start()

    def _tts_complete_after(self, duration: float):
        """TTS 播放完成后的清理"""
        import time
        time.sleep(duration)
        with self._lock:
            self.is_speaking = False
        self._publish_tts_status("idle")
        print(f"[VoiceBridge] TTS playback complete")

    def _publish_tts_status(self, status: str, text: str = ""):
        """发布 TTS 播放状态"""
        if not ROS_AVAILABLE:
            return
        msg = String()
        msg.data = json.dumps({"status": status, "text": text})
        self.tts_status_pub.publish(msg)

    def spin(self):
        """进入主循环"""
        if ROS_AVAILABLE:
            rospy.loginfo("Voice bridge spinning...")
            rospy.spin()
        else:
            print("ROS not available, exiting")
            sys.exit(1)

    def print_statistics(self):
        print(f"\nVoice Bridge Statistics:")
        print(f"  ASR received: {self.total_asr}")
        print(f"  ASR suppressed (echo): {self.suppressed}")
        print(f"  TTS played: {self.total_tts}")


def main():
    parser = argparse.ArgumentParser(description="CADE Voice Bridge")
    parser.add_argument('--no-echo-suppression', action='store_true',
                       help='Disable echo suppression')

    args = parser.parse_args()

    bridge = VoiceBridgeNode(
        echo_suppression=not args.no_echo_suppression
    )

    try:
        bridge.spin()
    except KeyboardInterrupt:
        bridge.print_statistics()
    except Exception as e:
        print(f"Voice bridge error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
