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
import os
import queue
import re
import select
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from task1_receptionist.sub_modules.topic_names import (
    TTS_TOPIC,
    TTS_PLAYING_TOPIC,
    ASR_TOPIC,
    ASR_SEGMENT_TOPIC,
)

NON_SPEECH_PATTERN = re.compile(r'^(?:\([^)]*\)?|\[[^\]]*\]?|<[^>]*>?)$')


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    def listen(self, timeout_sec: Optional[float] = 10.0) -> Optional[str]:
        """监听 ASR, 返回识别到的文本 (超时返回 None)"""
        ...

    def ask(self, question: str, timeout_sec: Optional[float] = 10.0) -> Optional[str]:
        """播报问题并等待回答 — say + listen 组合"""
        self.say(question)
        return self.listen(timeout_sec)

    def close(self):
        """清理资源 (可选)"""
        pass

    @property
    def last_audio_path(self) -> Optional[str]:
        """最近一次有效 ASR 结果对应的 WAV 文件。"""
        return None


# ============================================================
# Mock 后端 — 纯 Python, 控制台输入输出
# ============================================================

class MockSpeechInterface(SpeechInterface):
    """
    离线开发 / 测试用后端

    say()  → 打印到控制台
    listen() → 从本轮预置回复中取值, 或阻塞等待键盘输入
    """

    def __init__(self,
                 responses: Optional[Dict[str, str]] = None,
                 interactive: Optional[bool] = None):
        """
        Args:
            responses: 预置回复映射 {question_keyword → answer}
                       例如 {"名字": "Alice", "饮料": "橙汁"}
        """
        self._responses = responses or {}
        self._interactive = _env_flag("TASK1_INTERACTIVE", False) if interactive is None else bool(interactive)
        self._last_question = ""
        self._input_queue = queue.Queue()
        self._listen_thread = None

    def say(self, text: str):
        print(f"  🎤 [TTS] {text}")

    def listen(self, timeout_sec: Optional[float] = 10.0) -> Optional[str]:
        # 先检查预置回复
        for keyword, answer in self._responses.items():
            if keyword in str(self._last_question):
                result = answer
                print(f"  👂 [ASR] → \"{result}\" (预置)")
                return result

        if not self._interactive:
            print("  👂 [ASR] Mock 非交互模式: 未配置预置回复，返回 None")
            return None

        # 无预置回复时从 stdin 读取
        print(f"  👂 [ASR] 等待控制台输入 ({timeout_sec}s)...")
        try:
            result = input("  > ").strip()
            return result if result else None
        except (EOFError, KeyboardInterrupt):
            return None

    def ask(self, question: str, timeout_sec: Optional[float] = 10.0) -> Optional[str]:
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

    def listen(self, timeout_sec: Optional[float] = 10.0) -> Optional[str]:
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
    listen(t)    → 等待 /asr/segment 消息；仅交互模式可按 Enter 跳过

    线程安全: listen() 可在任意线程调用。
    """

    def __init__(self, tts_topic: str = TTS_TOPIC, asr_topic: str = ASR_TOPIC,
                 asr_segment_topic: str = ASR_SEGMENT_TOPIC):
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
        self._last_audio_path = None
        self._tts_status = "idle"
        self._tts_status_time = 0.0
        self._tts_status_cv = threading.Condition()
        self._tts_start_timeout = self._env_float("TASK1_TTS_START_TIMEOUT", 8.0)
        self._tts_wait_timeout = self._env_float("TASK1_TTS_WAIT_TIMEOUT", 20.0)
        self._default_asr_timeout = max(
            0.0,
            self._env_float("TASK1_ASR_LISTEN_TIMEOUT_SEC", 10.0),
        )
        self._interactive = _env_flag("TASK1_INTERACTIVE", False)
        self._tts_min_say_duration = max(
            0.0,
            self._env_float("TASK1_TTS_MIN_SAY_DURATION_SEC", 0.0),
        )
        self._sub_asr = rospy.Subscriber(asr_topic, String, self._on_asr)
        self._sub_asr_segment = rospy.Subscriber(
            asr_segment_topic, String, self._on_asr_segment
        )
        self._sub_tts_status = rospy.Subscriber(
            TTS_PLAYING_TOPIC, String, self._on_tts_status
        )

        # 给 subscriber 一点时间注册
        time.sleep(0.3)

    def _on_asr(self, msg):
        text = msg.data.strip()
        if not text:
            return
        if NON_SPEECH_PATTERN.match(text.lower()):
            self._rospy.loginfo(f"[ASR] 过滤非语音标记: '{text}'")
            return
        # /asr/segment 是 Task1 的主输入。保留 /asr 订阅用于过滤日志，
        # 但不重复入队，避免同一句回答被消费两次。

    def _on_asr_segment(self, msg):
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self._rospy.logwarn(f"[ASR] 无效片段消息: '{msg.data}'")
            return

        text = str(data.get("text", "")).strip()
        if not text:
            return
        if NON_SPEECH_PATTERN.match(text.lower()):
            self._rospy.loginfo(f"[ASR] 过滤非语音标记: '{text}'")
            return

        self._asr_queue.put({
            "text": text,
            "wav_path": data.get("wav_path"),
        })

    def say(self, text: str):
        self._say_impl(text, enforce_min_duration=True)

    def ask(self, question: str, timeout_sec: Optional[float] = 10.0) -> Optional[str]:
        self._drain_queue()
        self._say_impl(question, enforce_min_duration=True)
        return self._listen_impl(timeout_sec, drain_stale=False)

    def _say_impl(self, text: str, enforce_min_duration: bool):
        rospy = self._rospy
        publish_time = time.time()
        rospy.loginfo(f"[TTS] {text}")
        self._pub_tts.publish(self._String(data=text))
        wait_for_finished = True
        try:
            if self._pub_tts.get_num_connections() <= 0:
                rospy.logwarn("[TTS] /tts 暂无订阅者，跳过播放完成等待")
                wait_for_finished = False
        except Exception:
            pass
        if wait_for_finished:
            self._wait_for_tts_finished(publish_time)
        if enforce_min_duration:
            self._wait_for_min_say_duration(publish_time)

    def listen(self, timeout_sec: Optional[float] = None) -> Optional[str]:
        return self._listen_impl(timeout_sec, drain_stale=True)

    def _listen_impl(
        self,
        timeout_sec: Optional[float] = None,
        drain_stale: bool = True,
    ) -> Optional[str]:
        # 清空旧消息
        if drain_stale:
            self._drain_queue()
        self._last_audio_path = None
        if timeout_sec is None:
            timeout_sec = self._default_asr_timeout
        timeout_sec = max(0.0, float(timeout_sec))
        deadline = time.time() + timeout_sec if timeout_sec > 0.0 else time.time()

        if self._interactive:
            print(f"  👂 [ASR-ROS] 等待语音输入，最多 {timeout_sec:.1f}s；按 Enter 跳过")
        else:
            print(f"  👂 [ASR-ROS] 等待语音输入，最多 {timeout_sec:.1f}s")
        while True:
            try:
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0.0:
                    print("  👂 [ASR-ROS] 等待语音输入超时，继续流程")
                    return None
                result = self._asr_queue.get(timeout=min(0.2, remaining))
                text = result["text"]
                self._last_audio_path = result.get("wav_path")
                rospy = self._rospy
                rospy.loginfo(f"[ASR] → \"{text}\" (wav={self._last_audio_path})")
                return text
            except queue.Empty:
                if self._stdin_skip_requested():
                    print("  👂 [ASR-ROS] 已按 Enter 跳过语音输入")
                    return None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def _on_tts_status(self, msg):
        status = str(msg.data or "").strip().lower()
        if status not in {"playing", "idle"}:
            return
        with self._tts_status_cv:
            self._tts_status = status
            self._tts_status_time = time.time()
            self._tts_status_cv.notify_all()

    def _wait_for_tts_finished(self, publish_time: float):
        if self._tts_wait_timeout <= 0:
            return

        start_deadline = time.time() + max(0.0, self._tts_start_timeout)
        seen_playing = False
        with self._tts_status_cv:
            while time.time() < start_deadline:
                if (
                    self._tts_status == "playing"
                    and self._tts_status_time >= publish_time - 0.05
                ):
                    seen_playing = True
                    break
                self._tts_status_cv.wait(timeout=0.05)

            if not seen_playing:
                self._rospy.logwarn(
                    "[TTS] 未在 %.1fs 内收到 playing 状态，继续执行",
                    self._tts_start_timeout,
                )
                return

            finish_deadline = time.time() + max(0.0, self._tts_wait_timeout)
            while time.time() < finish_deadline:
                if (
                    self._tts_status == "idle"
                    and self._tts_status_time >= publish_time
                ):
                    return
                self._tts_status_cv.wait(timeout=0.05)

        self._rospy.logwarn(
            "[TTS] 等待播放完成超过 %.1fs，继续执行",
            self._tts_wait_timeout,
        )

    def _wait_for_min_say_duration(self, publish_time: float):
        remaining = self._tts_min_say_duration - (time.time() - publish_time)
        if remaining <= 0.0:
            return
        self._rospy.loginfo(
            "[TTS] 发布后等待 %.1fs 再继续下一步动作",
            remaining,
        )
        time.sleep(remaining)

    def _stdin_skip_requested(self) -> bool:
        if not self._interactive:
            return False
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
        except (OSError, TypeError, ValueError):
            return False
        if not readable:
            return False
        sys.stdin.readline()
        return True

    @property
    def last_audio_path(self) -> Optional[str]:
        return self._last_audio_path

    def _drain_queue(self):
        while True:
            try:
                self._asr_queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        if self._sub_asr:
            self._sub_asr.unregister()
        if self._sub_asr_segment:
            self._sub_asr_segment.unregister()


# ============================================================
# 工厂函数
# ============================================================

def create_speech_interface(use_ros: bool = False, **kwargs) -> SpeechInterface:
    """根据参数创建合适的语音接口"""
    if use_ros:
        kwargs.pop("script", None)
        return ROSSpeechInterface(**kwargs)
    else:
        script = kwargs.pop("script", None)
        if script:
            return ScriptedSpeechInterface(script)
        return MockSpeechInterface(**kwargs)
