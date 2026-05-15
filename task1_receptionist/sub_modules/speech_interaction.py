"""
语音交互接口 — SpeechInterface

封装 TTS 播报 + ASR 识别的交互模式, 提供两套后端:
  - MockSpeechInterface   纯 Python, 无 ROS 依赖, 用于离线开发测试
  - ROSSpeechInterface    通过 /tts 和 /asr 话题与真实语音节点通信

典型用法:

    speech = ROSSpeechInterface()
    speech.say("请问您的名字是什么?")
    name = speech.listen(timeout=10.0)
    print(f"识别结果: {name}")

    # 或一步完成
    drink = speech.ask("请问您想喝什么饮料?", timeout=10.0)
"""

import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


# ============================================================
# 抽象基类
# ============================================================

class SpeechInterface(ABC):
    """语音交互抽象接口"""

    @abstractmethod
    def say(self, text: str):
        """TTS 播报一段文本"""
        ...

    @abstractmethod
    def listen(self, timeout_sec: float = 10.0) -> Optional[str]:
        """监听 ASR, 返回识别到的文本 (超时返回 None)"""
        ...

    def ask(self, question: str, timeout_sec: float = 10.0) -> Optional[str]:
        """播报问题并等待回答 — say + listen 组合"""
        self.say(question)
        return self.listen(timeout_sec)

    def close(self):
        """清理资源 (可选)"""
        pass


# ============================================================
# Mock 后端 — 纯 Python, 控制台输入输出
# ============================================================

class MockSpeechInterface(SpeechInterface):
    """
    离线开发 / 测试用后端

    say()  → 打印到控制台
    listen() → 从本轮预置回复中取值, 或阻塞等待键盘输入
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Args:
            responses: 预置回复映射 {question_keyword → answer}
                       例如 {"名字": "Alice", "饮料": "橙汁"}
        """
        self._responses = responses or {}
        self._input_queue = queue.Queue()
        self._listen_thread = None

    def say(self, text: str):
        print(f"  🎤 [TTS] {text}")

    def listen(self, timeout_sec: float = 10.0) -> Optional[str]:
        # 先检查预置回复
        for keyword, answer in self._responses.items():
            if keyword in str(self._last_question):
                result = answer
                print(f"  👂 [ASR] → \"{result}\" (预置)")
                return result

        # 无预置回复时从 stdin 读取
        print(f"  👂 [ASR] 等待输入 ({timeout_sec}s)...")
        try:
            result = input("  > ").strip()
            return result if result else None
        except (EOFError, KeyboardInterrupt):
            return None

    def ask(self, question: str, timeout_sec: float = 10.0) -> Optional[str]:
        self._last_question = question
        return super().ask(question, timeout_sec)


# ============================================================
# 简单的语音对话脚本 — 用于 Task1 的固定交互
# ============================================================

class ScriptedSpeechInterface(SpeechInterface):
    """
    按预设脚本逐步返回回复, 跳过真实 ASR。

    用于反复测试 Task1 流程, 无需真人说话。
    """

    def __init__(self, script: list):
        """
        Args:
            script: 预设回复列表, 按顺序每次 listen() 返回一个
                    例如 ["Alice", "橙汁", "Bob", "可乐"]
        """
        self._script = script
        self._cursor = 0

    def say(self, text: str):
        print(f"  🎤 [TTS] {text}")

    def listen(self, timeout_sec: float = 10.0) -> Optional[str]:
        if self._cursor < len(self._script):
            result = self._script[self._cursor]
            self._cursor += 1
            print(f"  👂 [ASR] → \"{result}\" (脚本)")
            return result
        print(f"  👂 [ASR] 脚本用尽, 返回 None")
        return None


# ============================================================
# ROS 后端 — 真实 TTS + ASR
# ============================================================

class ROSSpeechInterface(SpeechInterface):
    """
    ROS 语音后端

    需要 roscore + tts_node + asr_node 已启动。

    say(text)    → 发布 std_msgs/String 到 /tts
    listen(t)    → 阻塞等待 /asr 消息, 超时返回 None

    线程安全: listen() 可在任意线程调用。
    """

    def __init__(self, tts_topic: str = "/tts", asr_topic: str = "/asr"):
        import rospy
        from std_msgs.msg import String

        self._rospy = rospy
        self._String = String

        # 检查 rospy 是否已初始化
        if not rospy.core.is_initialized():
            rospy.init_node("task1_controller", anonymous=True, disable_signals=True)

        self._pub_tts = rospy.Publisher(tts_topic, String, queue_size=10)

        # ASR 结果队列
        self._asr_queue = queue.Queue()
        self._sub_asr = rospy.Subscriber(asr_topic, String, self._on_asr)

        # 给 subscriber 一点时间注册
        time.sleep(0.3)

    def _on_asr(self, msg):
        text = msg.data.strip()
        if text:
            self._asr_queue.put(text)

    def say(self, text: str):
        rospy = self._rospy
        rospy.loginfo(f"[TTS] {text}")
        self._pub_tts.publish(self._String(data=text))

    def listen(self, timeout_sec: float = 10.0) -> Optional[str]:
        # 清空旧消息
        self._drain_queue()

        try:
            text = self._asr_queue.get(timeout=timeout_sec)
            rospy = self._rospy
            rospy.loginfo(f"[ASR] → \"{text}\"")
            return text
        except queue.Empty:
            rospy = self._rospy
            rospy.logwarn(f"[ASR] 超时 ({timeout_sec}s), 无识别结果")
            return None

    def _drain_queue(self):
        while True:
            try:
                self._asr_queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        if self._sub_asr:
            self._sub_asr.unregister()


# ============================================================
# 工厂函数
# ============================================================

def create_speech_interface(use_ros: bool = False, **kwargs) -> SpeechInterface:
    """根据参数创建合适的语音接口"""
    if use_ros:
        return ROSSpeechInterface(**kwargs)
    else:
        script = kwargs.pop("script", None)
        if script:
            return ScriptedSpeechInterface(script)
        return MockSpeechInterface(**kwargs)
