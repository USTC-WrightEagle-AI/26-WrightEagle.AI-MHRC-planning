#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Node - 大模型 ROS 服务节点

订阅 /llm/request (std_msgs/String, JSON), 调用 LLM 进行信息提取,
将结果发布到 /llm/response (std_msgs/String, JSON)。

支持的任务类型 (task 字段):
  - extract_guest_info: 提取客人姓名和饮品偏好
  - extract_name:       提取客人姓名
  - extract_drink:      提取饮品偏好

依赖:
  - brain.llm_client.LLMClient (OpenAI 兼容接口)
  - config.Config (LLM 配置)

启动:
  rosrun asr_tts llm_node.py
  或: python src/asr_tts/scripts/llm_node.py

话题协议:
  请求: /llm/request
  {
    "request_id": "uuid",
    "task": "extract_guest_info",
    "text": "My name is Alice and I'd like some orange juice",
    "context": {"role": "guest1"}
  }

  响应: /llm/response
  {
    "request_id": "uuid",
    "status": "success",
    "result": {"name": "Alice", "drink": "orange juice"},
    "error": null
  }
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

TASK_PROMPTS = {
    "extract_guest_info": EXTRACT_GUEST_INFO_PROMPT,
    "extract_name": EXTRACT_NAME_PROMPT,
    "extract_drink": EXTRACT_DRINK_PROMPT,
}


class LLMNode:
    """LLM ROS 服务节点"""

    def __init__(self):
        rospy.init_node("llm_node", anonymous=False)

        self._request_topic = rospy.get_param("~request_topic", "/llm/request")
        self._response_topic = rospy.get_param("~response_topic", "/llm/response")
        self._max_concurrent = rospy.get_param("~max_concurrent", 3)

        self._semaphore = threading.Semaphore(self._max_concurrent)

        self._pub_response = rospy.Publisher(self._response_topic, String, queue_size=10)
        self._sub_request = rospy.Subscriber(
            self._request_topic, String, self._on_request, queue_size=10
        )

        self._llm_client = None
        self._init_llm_client()

        rospy.loginfo("[LLM-Node] 就绪, 请求话题: %s, 响应话题: %s",
                      self._request_topic, self._response_topic)

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

    def _on_request(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            rospy.logwarn("[LLM-Node] 无效 JSON 请求: %s", e)
            return

        req_id = payload.get("request_id", "unknown")
        task = payload.get("task", "")
        text = payload.get("text", "")
        context = payload.get("context", {})

        if task not in TASK_PROMPTS:
            self._publish_error(req_id, f"未知任务类型: {task}")
            return

        if not text.strip():
            self._publish_error(req_id, "空文本输入")
            return

        t = threading.Thread(target=self._process_request, args=(req_id, task, text, context))
        t.daemon = True
        t.start()

    def _process_request(self, req_id: str, task: str, text: str, context: dict):
        acquired = self._semaphore.acquire(timeout=60.0)
        if not acquired:
            self._publish_error(req_id, "并发请求超限, 请稍后重试")
            return

        try:
            if not self._ensure_llm_client():
                self._publish_error(req_id, "LLM Client 不可用")
                return

            prompt = TASK_PROMPTS[task]
            role = context.get("role", "guest")
            rospy.loginfo("[LLM-Node] 处理请求 %s: task=%s, role=%s, text=\"%s\"",
                          req_id[:8], task, role, text[:80])

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]

            response_text = self._llm_client.chat(messages, temperature=0.1, max_tokens=128)
            rospy.loginfo("[LLM-Node] LLM 原始输出: %s", response_text[:200])

            result = self._parse_llm_response(response_text, task)
            self._publish_success(req_id, result)

        except Exception as e:
            rospy.logerr("[LLM-Node] 处理请求 %s 异常: %s", req_id[:8], e)
            self._publish_error(req_id, str(e))
        finally:
            self._semaphore.release()

    def _parse_llm_response(self, response_text: str, task: str) -> dict:
        result = {}
        try:
            parsed = self._extract_json(response_text)
        except json.JSONDecodeError:
            rospy.logwarn("[LLM-Node] 无法解析 LLM 输出为 JSON: %s", response_text[:200])
            return result

        if task == "extract_guest_info":
            result = {
                "name": parsed.get("name"),
                "drink": parsed.get("drink"),
            }
        elif task == "extract_name":
            result = {"name": parsed.get("name")}
        elif task == "extract_drink":
            result = {"drink": parsed.get("drink")}

        for key in result:
            if result[key] is not None:
                result[key] = str(result[key])

        return result

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return json.loads(text[start:end].strip())

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return json.loads(text[start:end].strip())

        import re
        brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group())

        raise json.JSONDecodeError("无法从文本中提取 JSON", text, 0)

    def _publish_success(self, req_id: str, result: dict):
        response = {
            "request_id": req_id,
            "status": "success",
            "result": result,
            "error": None,
        }
        self._pub_response.publish(String(data=json.dumps(response, ensure_ascii=False)))

    def _publish_error(self, req_id: str, error_msg: str):
        response = {
            "request_id": req_id,
            "status": "error",
            "result": None,
            "error": error_msg,
        }
        self._pub_response.publish(String(data=json.dumps(response, ensure_ascii=False)))

    def run(self):
        rospy.spin()


def main():
    node = LLMNode()
    node.run()


if __name__ == "__main__":
    main()
