"""
LLM Client - 大模型调用客户端

封装OpenAI兼容接口，支持云端/本地无缝切换
"""

import json
from typing import Optional, List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from config import Config
from brain.schemas import RobotDecision, parse_action


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

        # 初始化OpenAI客户端（禁用代理以避免SOCKS问题）
        import httpx
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=httpx.Client(trust_env=False)  # 禁用代理
        )

        print(f"✓ LLM Client 初始化成功")
        print(f"  模式: {'云端' if Config.is_cloud_mode() else '本地'}")
        print(f"  模型: {self.model}")
        print(f"  Base URL: {self.base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        调用LLM进行对话

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度系数（覆盖默认值）
            max_tokens: 最大token数（覆盖默认值）

        Returns:
            str: LLM的回复文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

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
            try:
                # 调用LLM
                response = self.chat(messages)

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

            except Exception as e:
                last_error = e
                print(f"⚠ 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    # 将错误信息反馈给LLM，让它重新生成
                    error_msg = (
                        f"你的输出格式有误，错误信息：{str(e)}\n"
                        f"请严格按照JSON格式重新输出。"
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        # 所有重试都失败
        raise ValueError(
            f"LLM输出解析失败，已重试{max_retries}次。最后错误: {last_error}"
        )

    def _extract_json(self, text: str) -> dict:
        """
        从文本中提取JSON（支持markdown代码块）

        Args:
            text: 包含JSON的文本

        Returns:
            dict: 解析后的字典

        Raises:
            json.JSONDecodeError: 如果无法解析
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取markdown代码块中的JSON
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        # 尝试提取普通代码块
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        # 都失败了，直接抛出原始错误
        raise json.JSONDecodeError("无法从文本中提取JSON", text, 0)


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

        # 初始化异步OpenAI客户端（禁用代理以避免SOCKS问题）
        import httpx
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=httpx.AsyncClient(trust_env=False)  # 禁用代理
        )

        print(f"✓ Async LLM Client 初始化成功")
        print(f"  模式: {'云端' if Config.is_cloud_mode() else '本地'}")
        print(f"  模型: {self.model}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """异步聊天"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        return response.choices[0].message.content

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
            try:
                response = await self.chat(messages)
                decision_dict = self._extract_json(response)

                if decision_dict.get("action") is not None:
                    action_dict = decision_dict["action"]
                    decision_dict["action"] = parse_action(action_dict)

                decision = RobotDecision(**decision_dict)
                return decision

            except Exception as e:
                last_error = e
                print(f"⚠ 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    error_msg = (
                        f"你的输出格式有误，错误信息：{str(e)}\n"
                        f"请严格按照JSON格式重新输出。"
                    )
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": error_msg})

        raise ValueError(
            f"LLM输出解析失败，已重试{max_retries}次。最后错误: {last_error}"
        )

    def _extract_json(self, text: str) -> dict:
        """从文本中提取JSON"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
            return json.loads(json_str)

        raise json.JSONDecodeError("无法从文本中提取JSON", text, 0)


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
