"""
LLM Client - 大模型调用客户端

封装OpenAI兼容接口，支持云端/本地无缝切换
"""

import json
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from openai import OpenAI, AsyncOpenAI
from cade_brain.llm_core.config import Config
from cade_brain.schemas import RobotDecision, parse_action


class LLMServiceError(RuntimeError):
    """Raised when the configured LLM service cannot be reached or returns an API error."""


class LLMConfigError(RuntimeError):
    """Raised when LLM runtime configuration is incomplete or invalid."""


def _extract_decision_dict(text: str) -> dict:
    """Parse either strict JSON or the common Thought/Action/Reply fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_block = _extract_code_block(text)
    if code_block is not None:
        return json.loads(code_block)

    tagged = _extract_tagged_react(text)
    if tagged is not None:
        return tagged

    embedded = _extract_embedded_decision_object(text)
    if embedded is not None:
        return embedded

    raise json.JSONDecodeError("无法从文本中提取JSON", text, 0)


def _extract_code_block(text: str) -> Optional[str]:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_tagged_react(text: str) -> Optional[dict]:
    thought = _extract_tagged_field(text, "Thought")
    reply = _extract_tagged_field(text, "Reply")
    action_text = _extract_tagged_field(text, "Action")

    if thought is None and reply is None and action_text is None:
        return None

    action = None
    if action_text is not None:
        stripped = action_text.strip()
        if stripped.lower() not in ("", "none", "null"):
            action_json = _balanced_json_prefix(stripped)
            action = json.loads(action_json)

    return {
        "thought": thought or "",
        "reply": reply or "",
        "action": action,
    }


def _extract_tagged_field(text: str, label: str) -> Optional[str]:
    pattern = (
        rf"(?ims)^\s*{re.escape(label)}\s*:\s*"
        rf"(.*?)"
        rf"(?=^\s*(?:Thought|Action|Reply)\s*:|\Z)"
    )
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def _balanced_json_prefix(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("Action字段中没有JSON对象", text, 0)

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    raise json.JSONDecodeError("Action字段中的JSON对象不完整", text, start)


def _extract_embedded_decision_object(text: str) -> Optional[dict]:
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            candidate = _balanced_json_prefix(text[start:])
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and any(k in parsed for k in ("thought", "reply", "action")):
            return parsed
    return None


def _ollama_chat_url(base_url: str) -> Optional[str]:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    if parsed.port != 11434:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/api/chat"


def _copy_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(message) for message in messages]


def _openai_messages_for_model(
    messages: List[Dict[str, str]],
    model: str,
    enable_thinking: bool,
) -> List[Dict[str, str]]:
    request_messages = _copy_messages(messages)
    if not enable_thinking and "qwen3" in model.lower():
        if request_messages and request_messages[-1].get("role") == "user":
            request_messages[-1]["content"] = (
                request_messages[-1].get("content", "") + " /no_think"
            )
    return request_messages


def _supports_response_format_retry(error: Exception) -> bool:
    text = str(error).lower()
    return "response_format" in text and any(
        marker in text
        for marker in ("unsupported", "not support", "unknown", "extra", "invalid")
    )


class LLMClient:
    """
    LLM 客户端（同步版本）

    支持：
    - 云端API（DeepSeek, DashScope等）
    - 本地Ollama
    - 自动JSON解析和重试
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化LLM客户端

        Args:
            config: 自定义配置，如果为None则使用Config.get_llm_config()
        """
        if config is None:
            config = Config.get_llm_config()

        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        self.timeout = config.get("timeout", 30)
        self.ollama_chat_url = _ollama_chat_url(self.base_url)

        if Config.is_cloud_mode() and not self.api_key:
            raise LLMConfigError(
                "CADE_MODE=CLOUD but CADE_CLOUD_API_KEY is empty. "
                "Export CADE_CLOUD_API_KEY before starting cade_brain."
            )

        # 初始化OpenAI客户端（禁用代理以避免SOCKS问题）
        import httpx
        self.http_client = httpx.Client(trust_env=False, timeout=self.timeout)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=self.http_client  # 禁用代理
        )

        print(f"✓ LLM Client 初始化成功")
        print(f"  模式: {'云端' if Config.is_cloud_mode() else '本地'}")
        print(f"  模型: {self.model}")
        print(f"  Base URL: {self.base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False
    ) -> str:
        """
        调用LLM进行对话

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度系数（覆盖默认值）
            max_tokens: 最大token数（覆盖默认值）
            enable_thinking: 是否启用思考模式（仅 Qwen3 支持）

        Returns:
            str: LLM的回复文本
        """
        temperature_value = self.temperature if temperature is None else temperature
        max_tokens_value = self.max_tokens if max_tokens is None else max_tokens

        if self.ollama_chat_url:
            return self._chat_ollama_native(
                messages,
                temperature_value,
                max_tokens_value,
                enable_thinking,
            )

        request_messages = _openai_messages_for_model(
            messages,
            self.model,
            enable_thinking,
        )
        request = {
            "model": self.model,
            "messages": request_messages,
            "temperature": temperature_value,
            "max_tokens": max_tokens_value,
            "response_format": {"type": "json_object"},
        }

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as e:
            if not _supports_response_format_retry(e):
                raise
            request.pop("response_format", None)
            response = self.client.chat.completions.create(**request)

        return response.choices[0].message.content or ""

    def _chat_ollama_native(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": _copy_messages(messages),
            "stream": False,
            "think": bool(enable_thinking),
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = self.http_client.post(self.ollama_chat_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "") or ""

    def get_decision(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3
    ) -> RobotDecision:
        """
        获取机器人决策（核心方法）

        Args:
            user_input: 用户输入
            system_prompt: 系统提示词
            conversation_history: 对话历史
            max_retries: JSON解析失败时的最大重试次数

        Returns:
            RobotDecision: 解析后的决策对象

        Raises:
            ValueError: 如果重试后仍无法解析
        """
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话
        if conversation_history:
            messages.extend(conversation_history)

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        # 重试机制
        last_error = None
        for attempt in range(max_retries):
            response = None
            try:
                # 调用LLM
                try:
                    response = self.chat(messages)
                except Exception as e:
                    raise LLMServiceError(
                        f"LLM service request failed for {self.base_url}: {e}"
                    ) from e

                # 调试：打印 LLM 原始输出
                if attempt == 0:  # 只在第一次尝试时打印
                    print(f"\n📄 LLM 原始输出:\n{'-'*60}")
                    print(response)
                    print(f"{'-'*60}\n")

                # 尝试解析JSON
                decision_dict = self._extract_json(response)

                # 如果action字段存在且不为None，解析动作
                if decision_dict.get("action") is not None:
                    action_dict = decision_dict["action"]
                    decision_dict["action"] = parse_action(action_dict)

                # 使用Pydantic验证
                decision = RobotDecision(**decision_dict)
                return decision

            except LLMServiceError:
                raise
            except Exception as e:
                last_error = e
                print(f"⚠ 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    # 将错误信息反馈给LLM，让它重新生成
                    error_msg = (
                        f"Internal parser error: {str(e)}\n"
                        f"Respond with one valid JSON object only. "
                        f"Do not use Thought:/Action:/Reply: labels or markdown."
                    )
                    if response is not None:
                        messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "system", "content": error_msg})

        # 所有重试都失败
        raise ValueError(
            f"LLM输出解析失败，已重试{max_retries}次。最后错误: {last_error}"
        )

    def _extract_json(self, text: str) -> dict:
        """
        从文本中提取决策JSON（支持markdown代码块和Thought/Action/Reply文本）

        Args:
            text: 包含JSON的文本

        Returns:
            dict: 解析后的字典

        Raises:
            json.JSONDecodeError: 如果无法解析
        """
        return _extract_decision_dict(text)


class AsyncLLMClient:
    """
    LLM 客户端（异步版本）

    用于需要异步调用的场景（如Web服务、ROS节点）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = Config.get_llm_config()

        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        self.timeout = config.get("timeout", 30)
        self.ollama_chat_url = _ollama_chat_url(self.base_url)

        if Config.is_cloud_mode() and not self.api_key:
            raise LLMConfigError(
                "CADE_MODE=CLOUD but CADE_CLOUD_API_KEY is empty. "
                "Export CADE_CLOUD_API_KEY before starting cade_brain."
            )

        # 初始化异步OpenAI客户端（禁用代理以避免SOCKS问题）
        import httpx
        self.http_client = httpx.AsyncClient(trust_env=False, timeout=self.timeout)
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=self.http_client  # 禁用代理
        )

        print(f"✓ Async LLM Client 初始化成功")
        print(f"  模式: {'云端' if Config.is_cloud_mode() else '本地'}")
        print(f"  模型: {self.model}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False
    ) -> str:
        """异步聊天"""
        temperature_value = self.temperature if temperature is None else temperature
        max_tokens_value = self.max_tokens if max_tokens is None else max_tokens

        if self.ollama_chat_url:
            return await self._chat_ollama_native(
                messages,
                temperature_value,
                max_tokens_value,
                enable_thinking,
            )

        request_messages = _openai_messages_for_model(
            messages,
            self.model,
            enable_thinking,
        )
        request = {
            "model": self.model,
            "messages": request_messages,
            "temperature": temperature_value,
            "max_tokens": max_tokens_value,
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self.client.chat.completions.create(**request)
        except Exception as e:
            if not _supports_response_format_retry(e):
                raise
            request.pop("response_format", None)
            response = await self.client.chat.completions.create(**request)

        return response.choices[0].message.content or ""

    async def _chat_ollama_native(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": _copy_messages(messages),
            "stream": False,
            "think": bool(enable_thinking),
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = await self.http_client.post(self.ollama_chat_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "") or ""

    async def get_decision(
        self,
        user_input: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 3
    ) -> RobotDecision:
        """异步获取决策"""
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_input})

        last_error = None
        for attempt in range(max_retries):
            response = None
            try:
                try:
                    response = await self.chat(messages)
                except Exception as e:
                    raise LLMServiceError(
                        f"LLM service request failed for {self.base_url}: {e}"
                    ) from e
                decision_dict = self._extract_json(response)

                if decision_dict.get("action") is not None:
                    action_dict = decision_dict["action"]
                    decision_dict["action"] = parse_action(action_dict)

                decision = RobotDecision(**decision_dict)
                return decision

            except LLMServiceError:
                raise
            except Exception as e:
                last_error = e
                print(f"⚠ 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    error_msg = (
                        f"Internal parser error: {str(e)}\n"
                        f"Respond with one valid JSON object only. "
                        f"Do not use Thought:/Action:/Reply: labels or markdown."
                    )
                    if response is not None:
                        messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "system", "content": error_msg})

        raise ValueError(
            f"LLM输出解析失败，已重试{max_retries}次。最后错误: {last_error}"
        )

    def _extract_json(self, text: str) -> dict:
        """从文本中提取决策JSON"""
        return _extract_decision_dict(text)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 测试 LLM Client ===\n")

    # 创建客户端
    client = LLMClient()

    # 简单对话测试
    print("\n--- 测试1: 简单对话 ---")
    messages = [
        {"role": "system", "content": "你是一个友好的助手"},
        {"role": "user", "content": "你好"}
    ]

    try:
        response = client.chat(messages)
        print(f"回复: {response}\n")
    except Exception as e:
        print(f"错误: {e}\n")

    print("✓ 基础功能测试完成")
    print("\n提示：运行前请先配置 config.py 中的 API 密钥")
