"""
本地配置示例

1. 将此文件重命名为 config_local.py
2. 填入你的 API 密钥
3. config_local.py 会自动覆盖 config.py 中的默认配置
"""

from config import RunMode

# ==================== 运行模式 ====================
MODE = RunMode.CLOUD  # 云端API
# MODE = RunMode.LOCAL  # 本地Ollama

# ==================== 云端配置 ====================

# DeepSeek API（推荐）
# 注册地址: https://platform.deepseek.com/
CLOUD_BASE_URL = "https://api.deepseek.com/v1"
CLOUD_API_KEY = "sk-512b7892b08546efbcab268d150f624d"
CLOUD_MODEL = "deepseek-chat"

# 阿里云 DashScope（可选）
# 注册地址: https://dashscope.console.aliyun.com/
# CLOUD_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# CLOUD_API_KEY = "sk-your-dashscope-api-key-here"
# CLOUD_MODEL = "qwen-turbo"

# OpenAI（可选）
# CLOUD_BASE_URL = "https://api.openai.com/v1"
# CLOUD_API_KEY = "sk-your-openai-api-key-here"
# CLOUD_MODEL = "gpt-3.5-turbo"

# ==================== 本地配置 ====================
# 如果使用本地 Ollama，无需修改以下配置
LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_API_KEY = "ollama"
LOCAL_MODEL = "qwen3:8b"

# ==================== 其他配置 ====================
TEMPERATURE = 0.7
MAX_TOKENS = 1024
ROBOT_NAME = "LARA"
