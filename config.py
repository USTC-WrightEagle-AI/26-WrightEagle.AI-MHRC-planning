"""
配置抽象层 - Configuration Layer

支持云端(CLOUD)和本地(LOCAL)模式无缝切换
"""

from enum import Enum
from typing import Literal
import os


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
    MODE: RunMode = RunMode.CLOUD

    # ==================== 云端配置 ====================
    # DeepSeek API
    CLOUD_BASE_URL = "https://api.deepseek.com"
    CLOUD_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 从环境变量 DEEPSEEK_API_KEY 读取
    CLOUD_MODEL = "deepseek-chat"

    # 阿里 DashScope (可选)
    # CLOUD_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # CLOUD_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    # CLOUD_MODEL = "qwen-turbo"

    # ==================== 本地配置 ====================
    LOCAL_BASE_URL = "http://localhost:11434/v1"
    LOCAL_API_KEY = "ollama"  # Ollama不需要真实key
    LOCAL_MODEL = "qwen2.5:3b"  # 推荐3B量化版本

    # ==================== LLM 参数 ====================
    TEMPERATURE = 0.7  # 温度系数（0-1，越高越随机）
    MAX_TOKENS = 512  # 最大生成长度
    TIMEOUT = 30  # 请求超时时间（秒）

    # ==================== 机器人配置 ====================
    ROBOT_NAME = "LARA"
    ENABLE_MOCK = True  # True=Mock模式，False=真实ROS

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
