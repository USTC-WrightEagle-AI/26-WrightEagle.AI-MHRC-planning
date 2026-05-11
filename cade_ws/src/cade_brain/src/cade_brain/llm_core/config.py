"""
配置抽象层 - Configuration Layer

支持云端(CLOUD)和本地(LOCAL)模式无缝切换
"""

from enum import Enum
import os
from dotenv import load_dotenv

# 加载 .env 文件
# override=False 表示优先使用系统环境变量，.env 作为补充
load_dotenv(override=False)


class RunMode(str, Enum):
    """运行模式"""
    CLOUD = "CLOUD"  # 云端API（当前PC开发）
    LOCAL = "LOCAL"  # 本地Ollama（未来Orin部署）


class Config:
    """
    全局配置类

    使用方法：
    1. 开发阶段：保持 MODE = RunMode.CLOUD，配置云端API
    2. 部署阶段：改为 MODE = RunMode.LOCAL，无需修改其他代码
    """

    # ==================== 运行模式 ====================
    # 优先从环境变量读取，默认为 CLOUD
    MODE: RunMode = RunMode(os.getenv("CADE_MODE", "CLOUD").upper())

    # ==================== 云端配置 (DeepSeek/DashScope) ====================
    CLOUD_BASE_URL = os.getenv("CADE_CLOUD_BASE_URL", "https://api.deepseek.com")
    CLOUD_API_KEY = os.getenv("CADE_CLOUD_API_KEY", "")
    CLOUD_MODEL = os.getenv("CADE_CLOUD_MODEL", "deepseek-chat")

    # ==================== 本地配置 (Ollama) ====================
    LOCAL_BASE_URL = os.getenv("CADE_LOCAL_BASE_URL", "http://localhost:11434/v1")
    LOCAL_API_KEY = os.getenv("CADE_LOCAL_API_KEY", "ollama")
    LOCAL_MODEL = os.getenv("CADE_LOCAL_MODEL", "qwen2.5:3b")

    # ==================== LLM 运行参数 ====================
    # 注意：从 env 读取的都是字符串，需要转换类型
    TEMPERATURE = float(os.getenv("CADE_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("CADE_MAX_TOKENS", "512"))
    TIMEOUT = int(os.getenv("CADE_TIMEOUT", "30"))

    # ==================== 机器人基础配置 ====================
    ROBOT_NAME = os.getenv("CADE_ROBOT_NAME", "LARA")

    # ==================== 日志配置 ====================
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE = "logs/robot.log"

    @classmethod
    def get_llm_config(cls) -> dict:
        """
        获取当前模式下的LLM配置

        Returns:
            dict: 包含 base_url, api_key, model 的配置字典
        """
        if cls.MODE == RunMode.CLOUD:
            return {
                "base_url": cls.CLOUD_BASE_URL,
                "api_key": cls.CLOUD_API_KEY,
                "model": cls.CLOUD_MODEL,
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS,
                "timeout": cls.TIMEOUT,
            }
        else:  # LOCAL
            return {
                "base_url": cls.LOCAL_BASE_URL,
                "api_key": cls.LOCAL_API_KEY,
                "model": cls.LOCAL_MODEL,
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS,
                "timeout": cls.TIMEOUT,
            }

    @classmethod
    def is_cloud_mode(cls) -> bool:
        """判断是否为云端模式"""
        return cls.MODE == RunMode.CLOUD

    @classmethod
    def is_local_mode(cls) -> bool:
        """判断是否为本地模式"""
        return cls.MODE == RunMode.LOCAL
