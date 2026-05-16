"""
LLM 接口 — LLMInterface

封装 LLM 信息提取能力, 提供两套后端:
  - MockLLMInterface      纯 Python, 无 ROS 依赖, 用于离线开发测试
  - ROSLLMInterface       通过 /llm/request + /llm/response 话题与 LLM ROS 节点通信

ROS 话题协议:
  请求话题: /llm/request   (std_msgs/String, JSON)
  响应话题: /llm/response  (std_msgs/String, JSON)

请求格式:
  {
    "request_id": "uuid",
    "task": "extract_guest_info" | "extract_name" | "extract_drink",
    "text": "My name is Alice and I'd like some orange juice",
    "context": {"role": "guest1"}
  }

响应格式:
  {
    "request_id": "uuid",
    "status": "success" | "error",
    "result": {"name": "Alice", "drink": "orange juice"},
    "error": null
  }

典型用法:

    llm = ROSLLMInterface()
    info = llm.extract_guest_info("My name is Alice, I'd like orange juice", role="guest1")
    print(info)  # {"name": "Alice", "drink": "orange juice"}
"""

import json
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from task1_receptionist.sub_modules.topic_names import (
    LLM_REQUEST_TOPIC,
    LLM_RESPONSE_TOPIC,
)


EXTRACT_GUEST_INFO_PROMPT = (
    "You are an information extraction assistant for a robot receptionist. "
    "Extract the guest's name and preferred drink from the following text. "
    "If the name cannot be determined, set it to null. "
    "If the drink cannot be determined, set it to null. "
    "Output ONLY a JSON object with keys \"name\" and \"drink\". "
    "Do not output anything else.\n\n"
    "Examples:\n"
    "Input: \"My name is Alice and I would like some orange juice\"\n"
    "Output: {\"name\": \"Alice\", \"drink\": \"orange juice\"}\n\n"
    "Input: \"I'm Bob, coffee please\"\n"
    "Output: {\"name\": \"Bob\", \"drink\": \"coffee\"}\n\n"
    "Input: \"Hi, I'm Sarah\"\n"
    "Output: {\"name\": \"Sarah\", \"drink\": null}\n\n"
    "Input: \"Can I get some cola?\"\n"
    "Output: {\"name\": null, \"drink\": \"cola\"}\n\n"
    "Now extract from:\n"
)

EXTRACT_NAME_PROMPT = (
    "Extract the person's name from the following text. "
    "Output ONLY a JSON object with key \"name\". "
    "If no name is found, set it to null. "
    "Do not output anything else.\n\n"
    "Examples:\n"
    "Input: \"My name is Alice\"\n"
    "Output: {\"name\": \"Alice\"}\n\n"
    "Input: \"I'm Bob\"\n"
    "Output: {\"name\": \"Bob\"}\n\n"
    "Input: \"Hi there\"\n"
    "Output: {\"name\": null}\n\n"
    "Now extract from:\n"
)

EXTRACT_DRINK_PROMPT = (
    "Extract the preferred drink from the following text. "
    "Output ONLY a JSON object with key \"drink\". "
    "If no drink is found, set it to null. "
    "Do not output anything else.\n\n"
    "Examples:\n"
    "Input: \"I'd like some orange juice\"\n"
    "Output: {\"drink\": \"orange juice\"}\n\n"
    "Input: \"Coffee please\"\n"
    "Output: {\"drink\": \"coffee\"}\n\n"
    "Input: \"I'm fine thanks\"\n"
    "Output: {\"drink\": null}\n\n"
    "Now extract from:\n"
)


class LLMInterface(ABC):
    """LLM 信息提取抽象接口"""

    @abstractmethod
    def extract_guest_info(self, text: str, role: str = "guest") -> Dict[str, Any]:
        """
        从文本中提取客人姓名和饮品偏好

        Args:
            text: ASR 识别的原始文本
            role: 客人角色 (guest1 / guest2), 用于日志

        Returns:
            {"name": "Alice", "drink": "orange juice"} 或 {"name": None, "drink": None}
        """
        ...

    @abstractmethod
    def extract_name(self, text: str, role: str = "guest") -> Optional[str]:
        """
        从文本中提取客人姓名

        Args:
            text: ASR 识别的原始文本
            role: 客人角色

        Returns:
            提取到的姓名, 或 None
        """
        ...

    @abstractmethod
    def extract_drink(self, text: str, role: str = "guest") -> Optional[str]:
        """
        从文本中提取饮品偏好

        Args:
            text: ASR 识别的原始文本
            role: 客人角色

        Returns:
            提取到的饮品, 或 None
        """
        ...

    def close(self):
        pass


class MockLLMInterface(LLMInterface):
    """
    离线开发 / 测试用后端

    使用简单的正则/规则匹配模拟 LLM 信息提取, 无需真实 LLM 服务。
    """

    def __init__(self):
        self._name_patterns = [
            "my name is ", "i am ", "i'm ", "call me ", "this is ",
        ]
        self._drink_patterns = [
            "i'd like ", "i would like ", "can i have ", "i want ",
            "give me ", "please give me ", "i'll have ", "could i get ",
        ]

    def extract_guest_info(self, text: str, role: str = "guest") -> Dict[str, Any]:
        text_lower = text.lower().strip()
        name = self._extract_name_impl(text_lower)
        drink = self._extract_drink_impl(text_lower)
        print(f"  🧠 [LLM-Mock] 提取 {role} 信息: name={name}, drink={drink} (from: \"{text}\")")
        return {"name": name, "drink": drink}

    def extract_name(self, text: str, role: str = "guest") -> Optional[str]:
        text_lower = text.lower().strip()
        name = self._extract_name_impl(text_lower)
        print(f"  🧠 [LLM-Mock] 提取 {role} 姓名: {name} (from: \"{text}\")")
        return name

    def extract_drink(self, text: str, role: str = "guest") -> Optional[str]:
        text_lower = text.lower().strip()
        drink = self._extract_drink_impl(text_lower)
        print(f"  🧠 [LLM-Mock] 提取 {role} 饮品: {drink} (from: \"{text}\")")
        return drink

    def _extract_name_impl(self, text_lower: str) -> Optional[str]:
        for pattern in self._name_patterns:
            idx = text_lower.find(pattern)
            if idx >= 0:
                rest = text_lower[idx + len(pattern):].strip()
                for sep in [",", ".", " and ", " but ", " so "]:
                    if sep in rest:
                        rest = rest[:rest.find(sep)].strip()
                if rest:
                    return rest.capitalize()
        return None

    def _extract_drink_impl(self, text_lower: str) -> Optional[str]:
        for pattern in self._drink_patterns:
            idx = text_lower.find(pattern)
            if idx >= 0:
                rest = text_lower[idx + len(pattern):].strip()
                for sep in [",", ".", " and ", " but ", " so "]:
                    if sep in rest:
                        rest = rest[:rest.find(sep)].strip()
                if rest:
                    if rest.endswith(" please"):
                        rest = rest[:-7].strip()
                    for prefix in ["some ", "a ", "the "]:
                        if rest.startswith(prefix):
                            rest = rest[len(prefix):]
                    return rest
        if "please" in text_lower:
            idx = text_lower.find("please")
            before = text_lower[:idx].strip()
            words = before.split()
            if words:
                return words[-1].rstrip(".,")
        return None


class ROSLLMInterface(LLMInterface):
    """
    ROS LLM 后端

    通过 /llm/request + /llm/response 话题与 LLM ROS 节点通信。
    需要启动 llm_node (asr_tts/scripts/llm_node.py)。

    请求/响应使用 request_id 匹配, 支持并发调用。
    """

    def __init__(self,
                 request_topic: str = LLM_REQUEST_TOPIC,
                 response_topic: str = LLM_RESPONSE_TOPIC,
                 timeout_sec: float = 30.0):
        import rospy
        from std_msgs.msg import String

        self._rospy = rospy
        self._String = String
        self._timeout = timeout_sec

        if not rospy.core.is_initialized():
            rospy.init_node("task1_llm_client", anonymous=True, disable_signals=True)

        self._pending: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

        self._pub_request = rospy.Publisher(request_topic, String, queue_size=10)
        self._sub_response = rospy.Subscriber(
            response_topic, String, self._on_response, queue_size=10
        )

        time.sleep(0.3)
        rospy.loginfo("[LLM-ROS] 客户端就绪, 请求话题: %s, 响应话题: %s",
                      request_topic, response_topic)

    def _on_response(self, msg):
        try:
            data = json.loads(msg.data)
            req_id = data.get("request_id")
            if not req_id:
                return
            with self._lock:
                q = self._pending.get(req_id)
            if q is not None:
                q.put(data)
        except (json.JSONDecodeError, Exception):
            pass

    def _call_llm(self, task: str, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())
        payload = {
            "request_id": req_id,
            "task": task,
            "text": text,
            "context": context or {},
        }

        q = queue.Queue()
        with self._lock:
            self._pending[req_id] = q

        self._pub_request.publish(self._String(data=json.dumps(payload)))

        try:
            result = q.get(timeout=self._timeout)
            if result.get("status") == "error":
                raise RuntimeError(f"LLM 返回错误: {result.get('error', 'unknown')}")
            return result.get("result", {})
        except queue.Empty:
            raise TimeoutError(f"LLM 请求超时 ({self._timeout}s), task={task}")
        finally:
            with self._lock:
                self._pending.pop(req_id, None)

    def extract_guest_info(self, text: str, role: str = "guest") -> Dict[str, Any]:
        result = self._call_llm("extract_guest_info", text, {"role": role})
        print(f"  🧠 [LLM-ROS] 提取 {role} 信息: {result} (from: \"{text}\")")
        return result

    def extract_name(self, text: str, role: str = "guest") -> Optional[str]:
        result = self._call_llm("extract_name", text, {"role": role})
        name = result.get("name")
        print(f"  🧠 [LLM-ROS] 提取 {role} 姓名: {name} (from: \"{text}\")")
        return name

    def extract_drink(self, text: str, role: str = "guest") -> Optional[str]:
        result = self._call_llm("extract_drink", text, {"role": role})
        drink = result.get("drink")
        print(f"  🧠 [LLM-ROS] 提取 {role} 饮品: {drink} (from: \"{text}\")")
        return drink

    def close(self):
        if self._sub_response:
            self._sub_response.unregister()


def create_llm_interface(use_ros: bool = False, **kwargs) -> LLMInterface:
    """根据参数创建合适的 LLM 接口"""
    if use_ros:
        return ROSLLMInterface(**kwargs)
    return MockLLMInterface(**kwargs)
