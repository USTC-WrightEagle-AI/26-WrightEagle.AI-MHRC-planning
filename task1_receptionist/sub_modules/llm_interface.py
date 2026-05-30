"""
LLM 接口 — LLMInterface

封装 LLM 信息提取能力, 提供两套后端:
  - LocalLLMInterface   直接调用 brain.llm_client, 无需 ROS, 使用大模型做语义理解
  - ROSLLMInterface     通过 /llm/request + /llm/response 话题与通用 LLM ROS 节点通信

ROS 话题协议 (通用, 与 llm_node.py 对接):
  请求话题: /llm/request   (std_msgs/String, JSON)
  响应话题: /llm/response  (std_msgs/String, JSON)

请求格式:
  {
    "request_id": "uuid",
    "messages": [
      {"role": "system", "content": "You are an information extraction assistant..."},
      {"role": "user", "content": "My name is Alice and I'd like some orange juice"}
    ],
    "temperature": 0.1,
    "max_tokens": 512
  }

响应格式:
  {
    "request_id": "uuid",
    "status": "success" | "error",
    "text": "raw LLM output string",
    "error": null
  }

典型用法:

    llm = LocalLLMInterface()
    info = llm.extract_guest_info("My name is Alice, I'd like orange juice", role="guest1")
    print(info)  # {"name": "Alice", "drink": "orange juice"}
"""

import json
import queue
import re
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from task1_receptionist.sub_modules.topic_names import (
    LLM_REQUEST_TOPIC,
    LLM_RESPONSE_TOPIC,
)


# ============================================================
# Prompt 定义 (由调用方管理)
# ============================================================

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


# ============================================================
# JSON 提取工具
# ============================================================

def extract_json_from_text(text: str) -> dict:
    """
    从 LLM 原始输出中提取 JSON 对象。

    按优先级尝试:
      1. 直接 json.loads
      2. ```json ... ``` 代码块
      3. ``` ... ``` 代码块
      4. 正则匹配第一个 {...}
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ============================================================
# 抽象接口
# ============================================================

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


# ============================================================
# 本地 LLM 后端 (直接调用 OpenAI 兼容接口, 无需 ROS)
# ============================================================

class LocalLLMInterface(LLMInterface):
    """
    本地 LLM 后端

    直接调用 OpenAI 兼容接口, 无需 ROS 和 llm_node。
    使用大模型做语义理解, 能处理 ASR 识别错误和各种自然表达。
    内置 qwen3 思考模式兼容: 当 content 为空时自动从 reasoning 提取答案。

    需要 Ollama 或云端 API 服务可用。
    """

    def __init__(self):
        from config import Config
        import httpx
        from openai import OpenAI

        llm_config = Config.get_llm_config()
        self._model = llm_config["model"]
        self._is_qwen3 = "qwen3" in self._model.lower()
        self._openai_client = OpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            timeout=120,
            http_client=httpx.Client(trust_env=False),
        )
        print("  🧠 [LLM-Local] 本地 LLM 后端就绪")

    def _call_llm(self, messages: List[Dict[str, str]],
                  temperature: float = 0.1,
                  max_tokens: int = 1024) -> str:
        msgs = [dict(m) for m in messages]
        if self._is_qwen3:
            if msgs and msgs[-1].get("role") == "user":
                msgs.append({
                    "role": "assistant",
                    "content": "<think/>\n\n</think\n\n"
                })

        response = self._openai_client.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message
        content = message.content or ""

        if not content and hasattr(message, "reasoning") and message.reasoning:
            content = self._extract_from_reasoning(message.reasoning)

        return content

    @staticmethod
    def _extract_from_reasoning(reasoning: str) -> str:
        json_match = re.search(r'\{[^{}]*\}', reasoning)
        if json_match:
            return json_match.group(0)
        return ""

    def extract_guest_info(self, text: str, role: str = "guest") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": EXTRACT_GUEST_INFO_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = self._call_llm(messages)
        parsed = extract_json_from_text(raw)
        result = {
            "name": parsed.get("name"),
            "drink": parsed.get("drink"),
        }
        for key in result:
            if result[key] is not None:
                result[key] = str(result[key])
        print(f"  🧠 [LLM-Local] 提取 {role} 信息: {result} (from: \"{text}\")")
        return result

    def extract_name(self, text: str, role: str = "guest") -> Optional[str]:
        messages = [
            {"role": "system", "content": EXTRACT_NAME_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = self._call_llm(messages)
        parsed = extract_json_from_text(raw)
        name = parsed.get("name")
        if name is not None:
            name = str(name)
        print(f"  🧠 [LLM-Local] 提取 {role} 姓名: {name} (from: \"{text}\")")
        return name

    def extract_drink(self, text: str, role: str = "guest") -> Optional[str]:
        messages = [
            {"role": "system", "content": EXTRACT_DRINK_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = self._call_llm(messages)
        parsed = extract_json_from_text(raw)
        drink = parsed.get("drink")
        if drink is not None:
            drink = str(drink)
        print(f"  🧠 [LLM-Local] 提取 {role} 饮品: {drink} (from: \"{text}\")")
        return drink

    def close(self):
        pass


# ============================================================
# ROS 后端
# ============================================================

class ROSLLMInterface(LLMInterface):
    """
    ROS LLM 后端

    通过 /llm/request + /llm/response 话题与通用 LLM ROS 节点通信。
    Prompt 构造和输出解析由本类负责, llm_node 只做纯代理转发。

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

    def _call_llm(self, messages: List[Dict[str, str]],
                  temperature: float = 0.1,
                  max_tokens: int = 512) -> str:
        """
        发送 messages 到 LLM 节点, 返回原始文本输出。

        Args:
            messages: OpenAI 格式的消息列表
            temperature: 采样温度
            max_tokens: 最大输出 token 数

        Returns:
            LLM 原始文本输出

        Raises:
            RuntimeError: LLM 返回错误
            TimeoutError: 请求超时
        """
        req_id = str(uuid.uuid4())
        payload = {
            "request_id": req_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        q = queue.Queue()
        with self._lock:
            self._pending[req_id] = q

        self._pub_request.publish(self._String(data=json.dumps(payload)))

        try:
            result = q.get(timeout=self._timeout)
            if result.get("status") == "error":
                raise RuntimeError(f"LLM 返回错误: {result.get('error', 'unknown')}")
            return result.get("text", "")
        except queue.Empty:
            raise TimeoutError(f"LLM 请求超时 ({self._timeout}s)")
        finally:
            with self._lock:
                self._pending.pop(req_id, None)

    # ============================================================
    # 业务方法 (Prompt 构造 + 输出解析)
    # ============================================================

    def extract_guest_info(self, text: str, role: str = "guest") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": EXTRACT_GUEST_INFO_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = self._call_llm(messages)
        parsed = extract_json_from_text(raw)
        result = {
            "name": parsed.get("name"),
            "drink": parsed.get("drink"),
        }
        for key in result:
            if result[key] is not None:
                result[key] = str(result[key])
        print(f"  🧠 [LLM-ROS] 提取 {role} 信息: {result} (from: \"{text}\")")
        return result

    def extract_name(self, text: str, role: str = "guest") -> Optional[str]:
        messages = [
            {"role": "system", "content": EXTRACT_NAME_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = self._call_llm(messages)
        parsed = extract_json_from_text(raw)
        name = parsed.get("name")
        if name is not None:
            name = str(name)
        print(f"  🧠 [LLM-ROS] 提取 {role} 姓名: {name} (from: \"{text}\")")
        return name

    def extract_drink(self, text: str, role: str = "guest") -> Optional[str]:
        messages = [
            {"role": "system", "content": EXTRACT_DRINK_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = self._call_llm(messages)
        parsed = extract_json_from_text(raw)
        drink = parsed.get("drink")
        if drink is not None:
            drink = str(drink)
        print(f"  🧠 [LLM-ROS] 提取 {role} 饮品: {drink} (from: \"{text}\")")
        return drink

    def close(self):
        if self._sub_response:
            self._sub_response.unregister()


# ============================================================
# 工厂函数
# ============================================================

def create_llm_interface(use_ros: bool = False, **kwargs) -> LLMInterface:
    """根据参数创建合适的 LLM 接口

    Args:
        use_ros: True 使用 ROS 话题与 llm_node 通信,
                 False 直接调用本地 brain.llm_client
    """
    if use_ros:
        return ROSLLMInterface(**kwargs)
    return LocalLLMInterface(**kwargs)
