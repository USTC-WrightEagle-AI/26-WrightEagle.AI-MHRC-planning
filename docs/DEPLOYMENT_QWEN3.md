# Qwen3-8B 云服务器部署指南

> **目标**: 在云服务器上部署 Qwen3-8B 模型，供 CADE 机器人系统使用
> **环境**: Linux 云服务器（推荐 Ubuntu 22.04+）
> **模型**: Qwen3-8B-Instruct (Q4 量化)
> **更新时间**: 2025-12-12

---

## 📋 前置检查

### 系统要求

- **操作系统**: Ubuntu 20.04+ / CentOS 7+ / Debian 11+
- **内存**: 至少 16GB RAM（推荐 32GB+）
- **存储**: 至少 20GB 可用空间（用于存储模型）
- **GPU**: 可选（有 GPU 可大幅提升推理速度）

### 检查系统信息

```bash
# 查看系统版本
uname -a
cat /etc/os-release

# 查看 GPU（如果有）
nvidia-smi

# 查看可用内存
free -h

# 查看磁盘空间
df -h
```

---

## 步骤 1: 安装 Ollama

### 方法 A：一键安装（推荐）

```bash
# 下载并安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 验证安装
ollama --version
```

**预期输出**：
```
ollama version is 0.x.x
```

### 方法 B：手动安装

```bash
# 下载二进制文件
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
sudo mv ollama /usr/local/bin/
sudo chmod +x /usr/local/bin/ollama

# 创建服务用户
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama

# 创建 systemd 服务
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=default.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

### 验证 Ollama 服务

```bash
# 检查服务状态
sudo systemctl status ollama

# 检查是否监听 11434 端口
curl http://localhost:11434
```

**预期输出**：
```
Ollama is running
```

---

## 步骤 2: 配置模型存储路径（重要！）

### 默认存储路径

Ollama 默认将模型下载到：
- `/usr/share/ollama/.ollama/models`
- 或 `~/.ollama/models`

### 修改为自定义路径（如 /data）

```bash
# 停止 Ollama 服务
sudo systemctl stop ollama

# 创建自定义模型目录
sudo mkdir -p /data/ollama/models
sudo chown -R ollama:ollama /data/ollama

# 编辑 systemd 服务配置
sudo nano /etc/systemd/system/ollama.service
```

**在 `[Service]` 部分添加环境变量**：

```ini
[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_MODELS=/data/ollama/models"
```

**重新加载并启动服务**：

```bash
sudo systemctl daemon-reload
sudo systemctl start ollama
sudo systemctl status ollama

# 验证环境变量
sudo systemctl show ollama | grep OLLAMA_MODELS
```

**预期输出**：
```
Environment=OLLAMA_MODELS=/data/ollama/models
```

---

## 步骤 3: 下载 Qwen3-8B 模型

```bash
# 拉取 Qwen3-8B 模型（Q4 量化版本）
ollama pull qwen3:8b

# 验证模型已下载
ollama list
```

**预期输出**：
```
NAME          ID            SIZE      MODIFIED
qwen3:8b      abc123...     4.5 GB    2 minutes ago
```

### 验证模型存储位置

```bash
# 查看模型文件
ls -lh /data/ollama/models/

# 查看模型占用空间
du -sh /data/ollama/models/*
```

**预期结构**：
```
/data/ollama/models/
├── blobs/
│   └── sha256-xxxx... (4.5GB)
└── manifests/
    └── registry.ollama.ai/
        └── library/
            └── qwen3/
                └── 8b
```

---

## 步骤 4: 测试 Qwen3-8B

### 交互式测试

```bash
# 启动交互式对话
ollama run qwen3:8b
```

**测试对话**：
```
>>> 你好，请介绍一下你自己

>>> 解释什么是具身智能机器人

>>> 如何抓取桌上的苹果？给出详细步骤

>>> 退出
/bye
```

### API 测试

```bash
# 使用 curl 测试 API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "什么是机器人操作系统ROS？",
  "stream": false
}'
```

### Python 客户端测试

创建测试脚本 `test_qwen3.py`：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama 不需要真实 API key
)

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": "你是一个家庭服务机器人助手"},
        {"role": "user", "content": "如何抓取桌上的苹果？"}
    ]
)

print(response.choices[0].message.content)
```

运行测试：
```bash
python test_qwen3.py
```

---

## 步骤 5: 配置 CADE 项目

### 创建本地配置文件

```bash
# 进入项目目录
cd /home/orin/huyanshen/CADE

# 创建本地配置
nano config_local.py
```

**粘贴以下内容**：

```python
"""本地配置 - 云服务器 Qwen3-8B"""
from config import RunMode

# ==================== 运行模式 ====================
MODE = RunMode.LOCAL

# ==================== 本地 Ollama 配置 ====================
LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_API_KEY = "ollama"
LOCAL_MODEL = "qwen3:8b"

# ==================== LLM 参数调优 ====================
TEMPERATURE = 0.7     # 温度系数
MAX_TOKENS = 1024     # Qwen3-8B 可以支持更长的输出
TIMEOUT = 60          # 增加超时时间

# ==================== 机器人配置 ====================
ROBOT_NAME = "LARA"
ENABLE_MOCK = True    # Mock 模式测试

# ==================== 日志配置 ====================
LOG_LEVEL = "DEBUG"   # 调试阶段使用 DEBUG
LOG_FILE = "logs/robot.log"
```

保存并退出（Ctrl+O, Enter, Ctrl+X）

### 测试 CADE 项目

```bash
# 运行演示模式
python main.py --mode demo
```

**预期输出**：
```
✓ 已加载本地配置 config_local.py

🤖 CADE - 具身智能机器人系统

运行模式: 🏠 本地
模型: qwen3:8b

✓ LLM Client 初始化成功
  模式: 本地
  模型: qwen3:8b
  Base URL: http://localhost:11434/v1
```

---

## 步骤 6: 配置远程访问（可选）

如果需要从其他机器访问 Ollama 服务：

### 修改 Ollama 监听地址

```bash
# 编辑 systemd 服务
sudo nano /etc/systemd/system/ollama.service
```

**在 `[Service]` 部分添加**：

```ini
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

**完整配置示例**：

```ini
[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_MODELS=/data/ollama/models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

**重启服务**：

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 验证监听地址
sudo netstat -tlnp | grep 11434
```

**预期输出**：
```
tcp  0.0.0.0:11434  0.0.0.0:*  LISTEN  12345/ollama
```

### 配置防火墙

```bash
# Ubuntu/Debian
sudo ufw allow 11434/tcp
sudo ufw status

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=11434/tcp
sudo firewall-cmd --reload
```

### 云服务商安全组

在云服务商控制台添加安全组规则：
- **协议**: TCP
- **端口**: 11434
- **来源**: 你的 IP 地址或 0.0.0.0/0

### 从本地 Mac 访问

修改本地 Mac 的 `config_local.py`：

```python
# 将 localhost 改为云服务器 IP
LOCAL_BASE_URL = "http://YOUR_SERVER_IP:11434/v1"
```

测试连接：

```bash
# 从本地 Mac 执行
curl http://YOUR_SERVER_IP:11434/api/version
```

---

## 性能测试

### 创建性能测试脚本

创建 `benchmark_qwen3.py`：

```python
"""Qwen3-8B 性能基准测试"""
import time
from brain.llm_client import LLMClient

def benchmark():
    print("=== Qwen3-8B 性能基准测试 ===\n")

    client = LLMClient()

    test_cases = [
        "你好，请介绍你自己",
        "解释什么是具身智能",
        "如何让机器人抓取苹果？给出详细步骤",
        "设计一个家庭服务机器人的导航算法",
    ]

    total_time = 0
    total_chars = 0

    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {prompt[:30]}...")
        print('='*60)

        start_time = time.time()

        response = client.chat([
            {"role": "user", "content": prompt}
        ])

        elapsed = time.time() - start_time
        chars = len(response)
        speed = chars / elapsed if elapsed > 0 else 0

        total_time += elapsed
        total_chars += chars

        print(f"\n✓ 响应时间: {elapsed:.2f}s")
        print(f"✓ 输出长度: {chars} 字符")
        print(f"✓ 推理速度: {speed:.1f} 字符/秒")
        print(f"\n回复预览:\n{response[:200]}...")

    # 统计信息
    avg_time = total_time / len(test_cases)
    avg_speed = total_chars / total_time

    print(f"\n{'='*60}")
    print("统计信息")
    print('='*60)
    print(f"总测试数: {len(test_cases)}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均响应时间: {avg_time:.2f}s")
    print(f"平均推理速度: {avg_speed:.1f} 字符/秒")

if __name__ == "__main__":
    benchmark()
```

运行测试：

```bash
python benchmark_qwen3.py
```

---

## 常见问题排查

### 问题 1: Ollama 服务无法启动

```bash
# 查看日志
sudo journalctl -u ollama -f

# 手动启动调试
ollama serve

# 检查端口占用
sudo lsof -i :11434
```

### 问题 2: 模型下载失败

```bash
# 检查网络连接
curl -I https://ollama.com

# 检查存储空间
df -h /data

# 手动重试
ollama pull qwen3:8b
```

### 问题 3: 推理速度很慢

```bash
# 检查 GPU 是否被识别
nvidia-smi

# 检查 Ollama 是否使用 GPU
ollama ps

# 查看系统资源
htop

# 尝试更小的模型
ollama pull qwen3:4b
```

### 问题 4: CADE 连接失败

**检查配置**：

```python
# config_local.py
LOCAL_BASE_URL = "http://localhost:11434/v1"  # 确保有 /v1 后缀
TIMEOUT = 120  # 增加超时时间
```

**测试连接**：

```bash
# 测试 Ollama API
curl http://localhost:11434/v1/models

# 测试 OpenAI 兼容接口
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "你好"}]
}'
```

### 问题 5: 内存不足

```bash
# 查看内存使用
free -h

# 停止其他服务释放内存
sudo systemctl stop nginx  # 示例

# 使用更小的模型
ollama pull qwen3:3b  # 3B 模型仅需 ~2GB 内存
```

---

## 模型管理

### 查看已安装模型

```bash
ollama list
```

### 删除模型

```bash
ollama rm qwen3:8b
```

### 更新模型

```bash
ollama pull qwen3:8b
```

### 切换不同版本

```bash
# Q4 量化（默认，4.5GB）
ollama pull qwen3:8b-instruct-q4_K_M

# Q8 量化（更高精度，8GB）
ollama pull qwen3:8b-instruct-q8_0

# FP16 完整精度（16GB）
ollama pull qwen3:8b-instruct-fp16
```

---

## 性能优化建议

### 1. 使用 GPU

如果有 NVIDIA GPU，确保安装了 CUDA：

```bash
# 检查 GPU
nvidia-smi

# Ollama 会自动检测并使用 GPU
```

### 2. 调整并发设置

```bash
# 编辑服务配置
sudo nano /etc/systemd/system/ollama.service

# 添加环境变量
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
```

### 3. 优化系统参数

```python
# config_local.py
TEMPERATURE = 0.3      # 降低温度，提高确定性和速度
MAX_TOKENS = 512       # 限制输出长度
TIMEOUT = 30           # 根据实际情况调整
```

---

## 快速命令清单

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 配置模型路径
sudo mkdir -p /data/ollama/models
sudo chown -R ollama:ollama /data/ollama
sudo nano /etc/systemd/system/ollama.service
# 添加: Environment="OLLAMA_MODELS=/data/ollama/models"
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 下载模型
ollama pull qwen3:8b

# 测试模型
ollama run qwen3:8b

# 配置 CADE
cd /home/orin/huyanshen/CADE
nano config_local.py
# 设置 MODE = RunMode.LOCAL, LOCAL_MODEL = "qwen3:8b"

# 运行测试
python main.py --mode demo
```

---

## 参考资源

- [Ollama 官方文档](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [Qwen3 模型介绍](https://github.com/QwenLM/Qwen)
- [OpenAI API 兼容性](https://github.com/ollama/ollama/blob/main/docs/openai.md)
- [CADE 项目文档](../README.md)

---

**最后更新**: 2025-12-12
**维护者**: CADE Team
**版本**: 1.0
