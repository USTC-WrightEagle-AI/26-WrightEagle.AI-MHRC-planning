"""
本地配置示例 - 云服务器部署 Qwen3-8B

使用方法：
1. 复制此文件为 config_local.py
2. 根据你的实际情况修改配置
3. config_local.py 会自动覆盖 config.py 中的默认值
"""

from config import RunMode

# ==================== 运行模式 ====================
# 切换为本地模式（使用 Ollama）
MODE = RunMode.LOCAL

# ==================== 本地 Ollama 配置 ====================
# 如果 Ollama 运行在同一台机器上
LOCAL_BASE_URL = "http://localhost:11434/v1"

# 如果 Ollama 运行在远程云服务器上（从本地Mac访问）
# LOCAL_BASE_URL = "http://YOUR_SERVER_IP:11434/v1"  # 替换为云服务器IP

# 如果需要外网访问，需要配置防火墙和 Ollama 监听地址
# LOCAL_BASE_URL = "http://your-domain.com:11434/v1"

LOCAL_API_KEY = "ollama"  # Ollama 不需要真实 API key，保持不变
LOCAL_MODEL = "qwen3:8b"  # 使用 Qwen3-8B 模型

# ==================== LLM 参数调优 ====================
TEMPERATURE = 0.7     # 温度系数，0.7 适合对话，0.3 适合精确任务
MAX_TOKENS = 1024     # Qwen3-8B 可以支持更长的输出
TIMEOUT = 60          # 本地推理可能需要更长时间，增加到 60 秒

# ==================== 机器人配置 ====================
ROBOT_NAME = "LARA"
ENABLE_MOCK = True    # 保持 Mock 模式进行测试

# ==================== 日志配置 ====================
LOG_LEVEL = "DEBUG"   # 调试阶段使用 DEBUG 级别
LOG_FILE = "logs/robot.log"
