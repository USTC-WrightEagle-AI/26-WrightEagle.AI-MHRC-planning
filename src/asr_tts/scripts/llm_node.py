#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Node - 通用大模型 ROS 代理节点

纯代理: 接收 messages → 调用 LLM → 返回原始文本输出。
不关心任务类型、Prompt 构造、输出格式解析, 这些由调用方负责。

订阅话题:
  /llm/request  (std_msgs/String, JSON)  LLM 请求

发布话题:
  /llm/response (std_msgs/String, JSON)  LLM 响应

请求格式:
  {
    "request_id": "uuid",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant..."},
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
    "error": null | "error message"
  }

参数:
  ~request_topic   (str,  default: /llm/request)
  ~response_topic  (str,  default: /llm/response)
  ~max_concurrent  (int,  default: 3)       最大并发请求数
  ~default_temperature (float, default: 0.1)
  ~default_max_tokens  (int,   default: 512)

依赖:
  - brain.llm_client.LLMClient (OpenAI 兼容接口)

启动:
  rosrun asr_tts llm_node.py
  或: python src/asr_tts/scripts/llm_node.py
"""

import json
import os
import sys
import threading

import rospy
from std_msgs.msg import String

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class LLMNode:
    """通用 LLM ROS 代理节点"""

    def __init__(self):
        rospy.init_node("llm_node", anonymous=False)

        self._request_topic = rospy.get_param("~request_topic", "/llm/request")
        self._response_topic = rospy.get_param("~response_topic", "/llm/response")
        self._max_concurrent = rospy.get_param("~max_concurrent", 3)
        self._default_temperature = rospy.get_param("~default_temperature", 0.1)
        self._default_max_tokens = rospy.get_param("~default_max_tokens", 512)

        self._semaphore = threading.Semaphore(self._max_concurrent)

        self._pub_response = rospy.Publisher(self._response_topic, String, queue_size=10)
        self._sub_request = rospy.Subscriber(
            self._request_topic, String, self._on_request, queue_size=10
        )

        self._llm_client = None
        self._init_llm_client()

        rospy.loginfo("[LLM-Node] 就绪, 请求话题: %s, 响应话题: %s",
                      self._request_topic, self._response_topic)
        rospy.loginfo("[LLM-Node] 参数: max_concurrent=%d, default_temperature=%.2f, default_max_tokens=%d",
                      self._max_concurrent, self._default_temperature, self._default_max_tokens)

    # ============================================================
    # LLM Client 初始化
    # ============================================================

    def _init_llm_client(self):
        try:
            from brain.llm_client import LLMClient
            self._llm_client = LLMClient()
            rospy.loginfo("[LLM-Node] LLM Client 初始化成功")
        except Exception as e:
            rospy.logwarn("[LLM-Node] LLM Client 初始化失败: %s, 将在收到请求时重试", e)
            self._llm_client = None

    def _ensure_llm_client(self):
        if self._llm_client is not None:
            return True
        try:
            from brain.llm_client import LLMClient
            self._llm_client = LLMClient()
            rospy.loginfo("[LLM-Node] LLM Client 延迟初始化成功")
            return True
        except Exception as e:
            rospy.logerr("[LLM-Node] LLM Client 延迟初始化失败: %s", e)
            return False

    # ============================================================
    # 请求处理
    # ============================================================

    def _on_request(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            rospy.logwarn("[LLM-Node] 无效 JSON 请求: %s", e)
            return

        req_id = payload.get("request_id", "unknown")
        messages = payload.get("messages")
        temperature = payload.get("temperature", self._default_temperature)
        max_tokens = payload.get("max_tokens", self._default_max_tokens)

        if not messages or not isinstance(messages, list):
            self._publish_error(req_id, "缺少 messages 字段或格式错误 (需要 list of {role, content})")
            return

        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                self._publish_error(req_id, "messages 中每项必须包含 role 和 content 字段")
                return

        t = threading.Thread(
            target=self._process_request,
            args=(req_id, messages, temperature, max_tokens),
        )
        t.daemon = True
        t.start()

    def _process_request(self, req_id: str, messages: list, temperature: float, max_tokens: int):
        acquired = self._semaphore.acquire(timeout=60.0)
        if not acquired:
            self._publish_error(req_id, "并发请求超限, 请稍后重试")
            return

        try:
            if not self._ensure_llm_client():
                self._publish_error(req_id, "LLM Client 不可用")
                return

            user_preview = ""
            for m in messages:
                if m["role"] == "user":
                    user_preview = m["content"][:80]
                    break

            rospy.loginfo("[LLM-Node] 处理请求 %s: messages=%d, temperature=%.2f, max_tokens=%d, user=\"%s\"",
                          req_id[:8], len(messages), temperature, max_tokens, user_preview)

            response_text = self._llm_client.chat(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            rospy.loginfo("[LLM-Node] LLM 原始输出: %s", response_text[:200])

            self._publish_success(req_id, response_text)

        except Exception as e:
            rospy.logerr("[LLM-Node] 处理请求 %s 异常: %s", req_id[:8], e)
            self._publish_error(req_id, str(e))
        finally:
            self._semaphore.release()

    # ============================================================
    # 发布
    # ============================================================

    def _publish_success(self, req_id: str, text: str):
        response = {
            "request_id": req_id,
            "status": "success",
            "text": text,
            "error": None,
        }
        self._pub_response.publish(String(data=json.dumps(response, ensure_ascii=False)))

    def _publish_error(self, req_id: str, error_msg: str):
        response = {
            "request_id": req_id,
            "status": "error",
            "text": None,
            "error": error_msg,
        }
        self._pub_response.publish(String(data=json.dumps(response, ensure_ascii=False)))

    # ============================================================
    # 主循环
    # ============================================================

    def run(self):
        rospy.spin()


def main():
    node = LLMNode()
    node.run()


if __name__ == "__main__":
    main()
