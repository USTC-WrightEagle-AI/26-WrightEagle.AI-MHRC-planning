# Tree View:
```
.
├── __pycache__
│   ├── config.cpython-310.pyc
│   ├── config.cpython-311.pyc
│   ├── config.cpython-312.pyc
│   ├── config_local.cpython-310.pyc
│   ├── config_local.cpython-311.pyc
│   ├── config_local.cpython-312.pyc
│   ├── robot_controller.cpython-311.pyc
│   └── robot_controller.cpython-312.pyc
├── body
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── mock_robot.cpython-311.pyc
│   │   └── robot_interface.cpython-311.pyc
│   ├── mock_robot.py
│   └── robot_interface.py
├── brain
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── __init__.cpython-312.pyc
│   │   ├── llm_client.cpython-311.pyc
│   │   ├── llm_client.cpython-312.pyc
│   │   ├── prompts.cpython-311.pyc
│   │   └── schemas.cpython-311.pyc
│   ├── llm_client.py
│   ├── prompts.py
│   └── schemas.py
├── compare_modes.py
├── config.py
├── config_local.example.py
├── config_local.py
├── config_local_example.py
├── demo_navigate.py
├── demo_retry.py
├── docs
│   ├── COMPACT_MODE_GUIDE.md
│   ├── DEPLOYMENT_QWEN3.md
│   ├── HOW_ROBOT_UNDERSTANDS.md
│   ├── LLM_RETRY_MECHANISM.md
│   └── NO_THOUGHT_MODE.md
├── GUIDELINE.md
├── main.py
├── QUICKSTART.md
├── README.md
├── requirements.txt
├── robot_controller.py
├── setup.sh
├── SETUP_GUIDE.md
├── test_no_thought.py
├── tests
│   └── test_basic.py
├── TODO.md
└── TODO_V2.md

```

# Content:

## GUIDELINE.md

````md
# 🤖 具身智能机器人开发行动指南 (Project LARA/CADE)

**核心策略**：云端借脑开发逻辑 $\rightarrow$ 本地部署替换接口 $\rightarrow$ 真机联调。

-----

## 📅 第一阶段：PC 端开发（当前可行）

**目标**：在没有 GPU 的情况下，完成“大脑”逻辑、意图识别、决策解析和模拟执行代码。

### 1\. 架构搭建：双层解耦设计

  - [ ] **建立项目结构**：将代码严格分为 `brain/` (LLM 交互) 和 `body/` (硬件控制)。
  - [ ] **配置抽象层**：编写 `config.py`，通过开关控制运行模式。

<!-- end list -->

```python
# config.py
class Config:
    # MODE: "CLOUD" (当前PC) 或 "LOCAL" (未来Orin)
    MODE = "CLOUD" 
    
    # 云端配置 (使用 DeepSeek/阿里 DashScope 等兼容 OpenAI 协议的服务)
    CLOUD_BASE_URL = "https://api.deepseek.com"
    CLOUD_API_KEY = "sk-xxxxxxxx"
    
    # 本地配置 (Ollama)
    LOCAL_BASE_URL = "http://localhost:11434/v1"
    LOCAL_API_KEY = "ollama"
```

### 2\. 大脑模块 (The Brain)：Prompt 与 接口

  - [ ] **编写 System Prompt**：核心是定义“动作菜单”。
    > "你是一个服务机器人。请根据用户指令输出 JSON。可用动作：`Maps(target)`, `pick(object)`, `speak(content)`..."
  - [ ] **实现 LLM Client**：使用 `openai` 标准库封装调用类。
  - [ ] **防御性编程 (Pydantic)**：编写 JSON 校验器，防止 LLM 胡乱输出。

<!-- end list -->

```python
# brain/parser.py
from pydantic import BaseModel, Field
from typing import Literal

class RobotAction(BaseModel):
    # 强制动作必须在枚举范围内
    type: Literal["navigate", "pick", "place", "speak"] 
    # 参数可以是字符串或坐标列表
    params: str | list[float] | None = None 
```

### 3\. 躯干模块 (The Body)：模拟层 (Mock)

  - [ ] **编写 Mock Robot 类**：模拟 ROS 接口，打印日志代替真实运动。
  - [ ] **状态机 (FSM) 开发**：使用 `transitions` 库或手写逻辑，管理 `IDLE` -\> `THINKING` -\> `EXECUTING` 的状态流转。

-----

## 🚀 第二阶段：迁移至 Orin（硬件到货后）

**目标**：零代码修改，仅切换配置，让机器人“活”过来。

### 1\. 环境准备 (On Orin)

  - [ ] **系统安装**：刷写 JetPack 6.x (Ubuntu 22.04)。
  - [ ] **部署 Ollama**：
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
  - [ ] **拉取模型**：
    ```bash
    # 推荐 Qwen 2.5 3B (Int4 量化)，速度快效果好
    ollama pull qwen2.5:3b
    ```

### 2\. 切换大脑

  - [ ] **修改配置**：在 `config.py` 中将 `MODE` 改为 `"LOCAL"`。
  - [ ] **测试验证**：运行 PC 上写好的测试脚本，确认 Orin 的 GPU (通过 `jtop` 查看) 正在进行推理。

### 3\. 注入灵魂 (Real Body)

  - [ ] **替换 Mock 类**：将 `MockRobot` 替换为真实的 `RosRobot` 类。
  - [ ] **对接 ROS 2**：在 `Maps` 函数中填入 `Nav2` 的 Action Client 代码；在 `speak` 函数中填入 TTS 代码。

-----

## 🛠️ 技术栈清单 (Tech Stack)

| 模块 | 推荐技术/库 | 说明 |
| :--- | :--- | :--- |
| **推理引擎** | **Ollama** | 负责加载模型，提供 HTTP API。PC/Orin 通用。 |
| **基础模型** | **Qwen 2.5 (3B-Instruct)** | 中文理解力强，指令遵循好，适合边缘端。 |
| **API 客户端** | **OpenAI (Python SDK)** | 标准化调用，方便在云端/本地间切换。 |
| **数据校验** | **Pydantic** | 确保 JSON 结构 100% 正确，避免程序崩溃。 |
| **机器人通信** | **ROS 2 (Humble)** | 工业级标准，处理导航和硬件驱动。 |
| **视觉识别** | **YOLOv8 / CLIP** | 这里的 `find` 动作需要调用的视觉模型。 |

-----

## ⚠️ 避坑指南 (Do's and Don'ts)

1.  **Do Not** 在 PC 上编译 TensorRT Engine 文件（硬件不兼容）。
2.  **Do Not** 在 Python 代码中直接 `import torch` 加载大模型（会与 ROS 环境冲突，且阻塞主线程）。
3.  **Do** 始终要求 LLM 输出 **JSON** 格式，并设置重试机制（如果解析失败，把错误信息喂回给 LLM 让它重写）。
4.  **Do** 使用 **异步 (Async/Await)** 调用 LLM 接口，保证机器人在“思考”时，急停按钮依然有效。

-----

### 下一步建议

现在你可以直接打开 IDE，安装 `openai` 和 `pydantic` 库，开始编写你的 `MockRobot` 和 `LLMClient` 类了。你需要我为你提供这两部分的**样板代码 (Boilerplate Code)** 吗？

````


## QUICKSTART.md

````md
# 快速开始指南

## 第一步：安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 第二步：配置 API 密钥

### 方案 A：使用云端 API（推荐用于开发）

1. 注册DeepSeek账号并获取API密钥
   - 访问: https://platform.deepseek.com/
   - 创建 API Key

2. 配置密钥
```bash
# 复制配置示例
cp config_local.example.py config_local.py

# 编辑 config_local.py，填入你的API密钥
# 取消注释相关行并填写：
# MODE = RunMode.CLOUD
# CLOUD_API_KEY = "sk-your-actual-key-here"
```

### 方案 B：使用本地 Ollama（无需API密钥）

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉取模型
ollama pull qwen2.5:3b

# 3. 配置为本地模式
# 在 config_local.py 中设置:
# MODE = RunMode.LOCAL
```

## 第三步：运行测试

```bash
# 运行基础模块测试（不需要API）
python tests/test_basic.py

# 运行端到端测试（需要API）
python main.py --test

# 运行演示
python main.py --demo
```

## 第四步：交互式使用

```bash
# 启动交互模式
python main.py

# 然后你可以输入：
# "你好"
# "去厨房"
# "帮我找苹果"
# "status" - 查看机器人状态
# "stats" - 查看统计信息
# "quit" - 退出
```

## 项目结构说明

```
CADE/
├── brain/              # 🧠 大脑模块
│   ├── llm_client.py   # LLM 调用客户端
│   ├── prompts.py      # 系统提示词
│   └── schemas.py      # 数据模型（动作定义）
├── body/               # 🦾 躯干模块
│   ├── robot_interface.py  # 机器人接口
│   └── mock_robot.py   # Mock 实现
├── tests/              # 🧪 测试
│   └── test_basic.py
├── config.py           # ⚙️ 配置（默认值）
├── config_local.py     # ⚙️ 本地配置（你的密钥）
├── robot_controller.py # 🎮 主控制器
└── main.py             # 🚀 入口程序
```

## 常见问题

### Q: 没有API密钥怎么办？

A: 使用本地 Ollama 方案（见方案B），完全免费且离线运行。

### Q: API调用失败？

A: 检查：
1. API密钥是否正确
2. 网络是否正常
3. 账户余额是否充足

### Q: 如何切换模型？

A: 在 `config_local.py` 中修改：
- 云端: 修改 `CLOUD_MODEL`
- 本地: 修改 `LOCAL_MODEL`

### Q: 如何部署到真实机器人？

A: 见 `GUIDELINE.md` 第二阶段说明：
1. 在 Orin 上部署 Ollama
2. 修改 `MODE = RunMode.LOCAL`
3. 替换 `MockRobot` 为 `RealRobot`（对接ROS）

## 下一步

- 查看 `GUIDELINE.md` 了解完整开发计划
- 查看 `TODO.md` 了解架构设计思路
- 阅读代码中的注释和文档字符串
- 根据需求自定义提示词（`brain/prompts.py`）
- 添加新的动作类型（`brain/schemas.py`）

## 获取帮助

遇到问题？
1. 检查各个模块的 `if __name__ == "__main__"` 部分的测试代码
2. 运行 `python -m pydoc <module_name>` 查看文档
3. 查看代码注释

祝开发顺利！ 🚀

````


## README.md

````md
# Project CADE - 具身智能机器人

基于云端大脑的服务机器人开发框架

## 架构设计

```
CADE/
├── brain/          # 大脑模块（LLM交互、意图识别、决策生成）
├── body/           # 躯干模块（硬件控制、Mock/Real切换）
├── tests/          # 测试用例
├── config.py       # 配置抽象层（支持云端/本地切换）
└── main.py         # 主入口
```

## 开发阶段

### 第一阶段：PC端开发（当前）
- [x] 建立项目结构
- [ ] 配置抽象层
- [ ] LLM Client 实现
- [ ] Mock Robot 实现
- [ ] 端到端测试

### 第二阶段：Orin部署（硬件到货后）
- [ ] 部署 Ollama
- [ ] 切换为本地模型
- [ ] 对接真实 ROS 2

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API（创建 config.local.py）
cp config.py config.local.py
# 编辑 config.local.py，填入你的API密钥

# 运行测试
python main.py
```

## 技术栈

- **推理引擎**: DeepSeek/DashScope (云端) → Ollama (本地)
- **基础模型**: Qwen 2.5 (3B-Instruct)
- **API标准**: OpenAI Compatible
- **数据校验**: Pydantic
- **机器人通信**: ROS 2 Humble (未来)

````


## SETUP_GUIDE.md

````md
# 环境配置指南

## 系统信息

- 操作系统: WSL2 (Linux)
- 架构: x86_64

## 第一步：安装 Miniforge

Miniforge 是一个轻量级的 conda 发行版，包含 conda-forge 作为默认 channel。

### 1.1 下载 Miniforge

```bash
# 下载 Miniforge 安装脚本
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# 如果 wget 不可用，可以使用 curl
# curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

### 1.2 运行安装脚本

```bash
# 添加执行权限
chmod +x Miniforge3-Linux-x86_64.sh

# 运行安装（会有交互式提示）
bash Miniforge3-Linux-x86_64.sh
```

安装过程中：
- 按 Enter 开始阅读许可协议
- 输入 `yes` 接受许可
- 确认安装路径（默认 `~/miniforge3`，直接按 Enter）
- 选择 `yes` 让安装程序初始化 miniforge（会修改 `.bashrc`）

### 1.3 激活 conda

```bash
# 重新加载 shell 配置
source ~/.bashrc

# 或者直接激活 miniforge
source ~/miniforge3/bin/activate

# 验证安装
conda --version
python --version
```

## 第二步：创建项目环境

### 2.1 创建专用虚拟环境

```bash
# 进入项目目录
cd /mnt/c/Users/pc/Projects/CADE

# 创建名为 'cade' 的环境，使用 Python 3.11
conda create -n cade python=3.11 -y

# 激活环境
conda activate cade

# 验证
which python
python --version
```

### 2.2 安装项目依赖

```bash
# 确保在 cade 环境中（命令行前缀应显示 (cade)）
pip install -r requirements.txt

# 验证安装
pip list
```

## 第三步：配置 API（可选）

### 方案 A：使用云端 API（需要网络和密钥）

```bash
# 复制配置示例
cp config_local.example.py config_local.py

# 使用你喜欢的编辑器编辑（vim/nano/code）
nano config_local.py
# 或
code config_local.py
```

在 `config_local.py` 中取消注释并填写：

```python
from config import RunMode

MODE = RunMode.CLOUD
CLOUD_API_KEY = "sk-your-actual-deepseek-key-here"
```

获取 DeepSeek API 密钥：
1. 访问 https://platform.deepseek.com/
2. 注册/登录
3. 创建 API Key
4. 新用户通常有免费额度

### 方案 B：使用本地 Ollama（免费，离线）

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取推荐模型（约 2GB）
ollama pull qwen2.5:3b

# 验证
ollama list

# 配置为本地模式
echo 'from config import RunMode
MODE = RunMode.LOCAL' > config_local.py
```

## 第四步：运行测试

### 4.1 基础模块测试（无需API）

```bash
# 激活环境
conda activate cade

# 运行测试
python tests/test_basic.py
```

应该看到类似输出：
```
=== 测试 Config ===
✓ 配置模块正常
...
✅ 所有测试通过！
```

### 4.2 端到端测试（需要API/Ollama）

```bash
# 演示模式（完整服务流程）
python main.py --demo

# 测试模式（多个测试用例）
python main.py --test

# 交互模式
python main.py
```

## 常用命令

```bash
# 激活项目环境
conda activate cade

# 退出环境
conda deactivate

# 查看所有环境
conda env list

# 更新依赖
pip install --upgrade -r requirements.txt

# 删除环境（如需重建）
conda env remove -n cade
```

## 故障排查

### 问题 1: conda 命令找不到

```bash
# 手动激活 miniforge
source ~/miniforge3/bin/activate

# 或添加到 PATH
export PATH="$HOME/miniforge3/bin:$PATH"
```

### 问题 2: pip 安装失败

```bash
# 更新 pip
pip install --upgrade pip

# 使用国内镜像（可选，加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 3: API 调用失败

检查：
1. `config_local.py` 是否存在且配置正确
2. API 密钥是否有效
3. 网络是否正常

```bash
# 查看配置
python -c "from config import Config; print(Config.get_llm_config())"
```

### 问题 4: Ollama 模型下载慢

```bash
# 使用代理（如果有）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

ollama pull qwen2.5:3b
```

## 下一步

环境配置完成后：
1. 阅读 `QUICKSTART.md` 开始使用
2. 查看 `GUIDELINE.md` 了解开发路线图
3. 运行 `python main.py` 开始体验

## 自动化配置脚本（可选）

创建一个一键配置脚本：

```bash
cat > setup.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 开始配置 CADE 环境..."

# 创建 conda 环境
conda create -n cade python=3.11 -y

# 激活环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cade

# 安装依赖
pip install -r requirements.txt

# 复制配置示例
if [ ! -f config_local.py ]; then
    cp config_local.example.py config_local.py
    echo "⚠️  请编辑 config_local.py 填入你的 API 密钥"
fi

# 运行测试
echo "🧪 运行基础测试..."
python tests/test_basic.py

echo "✅ 环境配置完成！"
echo "下一步: 编辑 config_local.py，然后运行 python main.py"
EOF

chmod +x setup.sh
```

使用方法：
```bash
./setup.sh
```

````


## TODO.md

````md
这是一个非常经典的 **具身智能（Embodied AI）** 任务。在这个场景下，你的核心挑战在于如何让同一个“大脑”（通常是 LLM 或 VLM）既能像人一样自然聊天，又能像机器一样精准执行指令。

为了解决这个问题，我建议采用 **“路由+规划”（Router + Planning）** 的架构思路。

以下是为你拟定的分步行动思路：

-----

### 第一阶段：架构设计（分流与统一）

你需要决定是让一个模型做完所有事，还是拆分开来。为了稳定性和可控性，建议采用**逻辑分层**：

1.  **意图识别层 (Intent Router):**
      * 首先判断用户的输入是单纯的“闲聊（Chit-chat）”还是“任务指令（Instruction）”。
      * *例子：* 用户说“你好”，分类为“闲聊”；用户说“去拿苹果”，分类为“任务”。
2.  **双轨输出机制:**
      * **Track A (对话):** 负责生成自然语言回复。
      * **Track B (决策):** 负责生成结构化的 Action Code (JSON 或 API 调用)。
      * *高级技巧：* 现代 LLM 支持 Function Calling（函数调用），你可以将动作定义为 Function，让 LLM 决定是否调用。

-----

### 第二阶段：动作空间定义 (Action Space Alignment)

为了让决策与动作分类对齐，你需要建立一个严格的 **API 词汇表**。LLM 不能随意创造动作，必须从“菜单”里点菜。

**建议的动作定义 (Schema):**

  * `Maps(target: str | coords: [x,y,z])`: 导航到语义地点或坐标。
  * `pick(object_name: str, object_id: int)`: 抓取物体。
  * `place(target_location: str)`: 放置物体。
  * `scan/search(object_name: str)`: 寻找物体。
  * `speak(content: str)`: **关键点**——将“说话”也视为一种动作。

> **提示：** 对于 `[x, y, z]` 这种坐标，LLM 通常无法凭空产生准确的数字。你需要一个中间件（如语义地图 Semantic Map），让 LLM 输出 "Kitchen"，然后由下层系统将 "Kitchen" 翻译成 `[12.5, 3.0, 0]`。

-----

### 第三阶段：Prompt Engineering (系统提示词构建)

这是实现任务的核心。你需要设计一个 System Prompt，强制模型遵循特定的思考模式，例如 **ReAct (Reasoning + Acting)** 模式。

**Prompt 结构示例：**

> **Role:** 你是一个服务机器人。
>
> **Constraints:**
>
> 1.  如果用户只是聊天，请直接用自然语言回答。
> 2.  如果用户下达指令，必须输出特定的 JSON 格式动作。
> 3.  不要幻想你可以做物理动作以外的事情。
>
> **Action List:**
>
>   - `Maps_to(x, y, z)`
>   - `find(object)`
>   - `place(object)`
>
> **Output Format (Decision):**
>
> ```json
> {
>   "thought": "分析用户的意图...",
>   "reply": "给用户的口头回应 (可选)",
>   "action": "find",
>   "params": ["apple"]
> }
> ```

-----

### 第四阶段：处理流程模拟 (Workflow)

我们来看两个具体的 Case，帮助你理清代码逻辑：

#### Case 1: 纯对话

  * **用户输入:** “你今天感觉怎么样？”
  * **模型思考:** 用户在问候，无需物理动作。
  * **输出应答:** “我感觉很好，时刻准备为您服务。”
  * **输出决策:** `null` 或 `Wait`。

#### Case 2: 复合任务

  * **用户输入:** “帮我把那瓶水拿过来。”
  * **模型思考 (CoT):**
    1.  用户想要水。
    2.  我不知道水在哪 -\> 需要先 `find`。
    3.  找到后需要 `Maps` 过去。
    4.  然后 `pick`。
  * **输出应答:** “好的，我这就去帮您找水。”
  * **输出决策:** (输出序列中的第一步)
    ```json
    {
      "action": "find",
      "params": ["water_bottle"]
    }
    ```
    *(注意：机器人通常是一步一步执行的，执行完一步后，将环境反馈再次输入给 LLM，生成下一步。)*

-----

### 第五阶段：关键技术难点与对策

1.  **坐标对齐问题 (Grounding):**

      * *问题:* LLM 输出 "navigate to [10, 20, 0]"，但那里是墙怎么办？
      * *对策:* 尽量让 LLM 输出语义标签（如 `Maps("table")`），让底层的导航算法（如 ROS MoveBase）去解算具体坐标。

2.  **上下文记忆:**

      * 机器必须记住之前的对话和动作。比如我刚让你“把苹果放下”，下一句我说“把它切开”，你得知道“它”指的是苹果。

3.  **多模态融合 (Vision):**

      * 如果任务包含 `find [object]`，你需要一个视觉模型（如 YOLO 或 CLIP）来告诉 LLM 现在的摄像头里看到了什么。

-----


````


## TODO_V2.md

````md
# CADE 项目迭代计划 V2.0

> **当前状态**: 基础架构完成，Mock模式运行良好
> **目标**: 从原型系统升级为可部署的应用级机器人系统
> **时间规划**: 12-16周（分4个阶段）

---

## 📊 当前项目评估

### ✅ 已完成 (约 2700 行代码)
- [x] **Brain模块**: LLM客户端、Prompt工程、Schema定义、ReAct决策流程
- [x] **Body模块**: Mock Robot、抽象接口、状态机基础
- [x] **Controller**: 完整的感知-决策-执行循环
- [x] **配置层**: 支持云端/本地模式切换
- [x] **多种运行模式**: 交互/测试/演示
- [x] **基础文档**: README、QUICKSTART、GUIDELINE

### ⚠️ 待完善方向
1. **稳定性**: 缺少完整的错误处理和恢复机制
2. **硬件集成**: 尚未对接真实ROS 2系统
3. **感知能力**: 缺少视觉、语音等多模态输入
4. **测试覆盖**: 单元测试不足
5. **用户体验**: 缺少可视化界面和监控工具
6. **性能优化**: 响应速度、本地部署待优化

---

## 🎯 迭代路线图

### **第一阶段：稳定基座** (3-4周)
**目标**: 让系统更健壮、可测试、可维护

#### 1.1 测试框架 (Priority: 🔴 High)
```
tests/
├── unit/
│   ├── test_schemas.py          # 数据模型验证
│   ├── test_llm_client.py       # LLM客户端测试
│   ├── test_robot_controller.py # 控制器逻辑测试
│   └── test_prompts.py          # Prompt生成测试
├── integration/
│   ├── test_end_to_end.py       # 端到端场景测试
│   └── test_conversation.py     # 多轮对话测试
├── performance/
│   └── benchmark.py             # 性能基准测试
└── fixtures/
    └── test_data.json           # 测试数据集
```

**任务清单**:
- [ ] 安装 pytest, pytest-cov, pytest-asyncio
- [ ] 编写单元测试（目标覆盖率 >80%）
- [ ] 建立回归测试集（50个标准场景）
- [ ] 添加性能benchmark（响应时间、token使用）
- [ ] 配置 GitHub Actions CI

#### 1.2 错误处理增强 (Priority: 🔴 High)
```python
# 新增模块
brain/
└── exceptions.py       # 自定义异常类型

body/
└── recovery.py         # 自动恢复策略

utils/
├── retry.py           # 重试装饰器
└── circuit_breaker.py # 熔断器
```

**任务清单**:
- [ ] 定义异常分类体系（网络/硬件/逻辑错误）
- [ ] 实现重试机制（exponential backoff）
- [ ] 添加熔断器（防止级联失败）
- [ ] 设计降级策略（如LLM失败时使用规则引擎）
- [ ] 实现动作超时和自动中断

#### 1.3 日志和监控 (Priority: 🟡 Medium)
```python
# 改进现有日志
utils/
├── logger.py          # 结构化日志配置
└── metrics.py         # 性能指标收集

# 输出格式
logs/
├── robot_YYYYMMDD.log      # 每日日志
├── error_YYYYMMDD.log      # 错误日志
└── performance_YYYYMMDD.json # 性能数据
```

**任务清单**:
- [ ] 配置 loguru 结构化日志
- [ ] 添加日志级别和过滤
- [ ] 实现性能指标收集（latency, token usage, success rate）
- [ ] 日志轮转和归档策略
- [ ] （可选）集成 Prometheus + Grafana

#### 1.4 任务队列和状态机 (Priority: 🟡 Medium)
```python
brain/
└── task_manager.py    # 任务队列和规划器

# 支持复杂任务分解
User: "把苹果放到桌子上"
→ Task Queue: [search(apple), navigate(apple), pick(apple), navigate(table), place(table)]
```

**任务清单**:
- [ ] 实现任务队列（支持暂停/恢复）
- [ ] 添加任务优先级管理
- [ ] 设计状态机（支持中断和异常处理）
- [ ] 实现任务执行历史记录
- [ ] 支持多步骤任务的断点续传

---

### **第二阶段：硬件对接** (4-6周)
**目标**: 在真实机器人上运行

#### 2.1 ROS 2 接口实现 (Priority: 🔴 High)
```
body/
├── real_robot.py          # ROS 2 实现
├── ros_bridge.py          # ROS消息转换
└── calibration/
    ├── camera_calib.yaml  # 相机标定
    └── tf_tree.yaml       # 坐标系变换
```

**技术栈**:
- ROS 2 Humble
- rclpy (Python ROS客户端)
- Nav2 (导航栈)
- MoveIt2 (抓取规划)

**任务清单**:
- [ ] 搭建ROS 2开发环境（Docker或虚拟机）
- [ ] 实现 `RealRobot` 类（继承 `RobotInterface`）
- [ ] 对接导航：`navigate()` → Nav2 ActionClient
- [ ] 对接抓取：`pick()` → MoveIt2 规划器
- [ ] 实现传感器数据订阅（LaserScan, PointCloud）
- [ ] 坐标系转换（map ↔ base_link ↔ camera）
- [ ] 编写ROS 2 launch文件

#### 2.2 视觉模块集成 (Priority: 🔴 High)
```
perception/
├── __init__.py
├── detector.py           # 目标检测
├── segmentation.py       # 实例分割
└── models/
    ├── yolo_config.yaml
    └── clip_config.yaml
```

**技术选型**:
- 目标检测: YOLOv8 / YOLOv10
- 语义理解: CLIP / GroundingDINO
- 位姿估计: FoundationPose

**任务清单**:
- [ ] 集成YOLOv8进行物体检测
- [ ] 实现 `search()` 的视觉实现
- [ ] 3D位姿估计（RGB-D → 物体坐标）
- [ ] 物体跟踪（Kalman滤波）
- [ ] 与ROS 2集成（发布Detection消息）

#### 2.3 导航和建图 (Priority: 🟡 Medium)
```
mapping/
├── semantic_map.py       # 语义地图
└── location_manager.py   # 地点管理

# 数据格式
maps/
├── office_map.yaml       # 占用栅格地图
└── semantic_locations.json  # 语义标签
{
  "kitchen": {"coords": [2.5, 3.0, 0.0], "type": "room"},
  "table": {"coords": [1.0, 1.5, 0.8], "type": "furniture"}
}
```

**任务清单**:
- [ ] 使用SLAM构建地图（Cartographer/SLAM Toolbox）
- [ ] 建立语义地图（room → coords映射）
- [ ] 实现语义导航（"去厨房" → 查表 → Nav2）
- [ ] 动态障碍物避障
- [ ] 充电桩/Home位置标定

#### 2.4 机械臂和抓取 (Priority: 🟡 Medium)
```
manipulation/
├── grasp_planner.py     # 抓取规划
└── gripper_control.py   # 夹爪控制
```

**任务清单**:
- [ ] MoveIt2抓取规划
- [ ] 夹爪开合控制
- [ ] 碰撞检测
- [ ] 力反馈控制（避免损坏物体）

---

### **第三阶段：智能增强** (3-4周)
**目标**: 多模态交互，更智能的决策

#### 3.1 多模态LLM集成 (Priority: 🔴 High)
```python
brain/
├── vlm_client.py         # 视觉语言模型客户端
└── multimodal_prompt.py  # 多模态Prompt

# 使用场景
User: "这是什么？" [同时发送相机图像]
VLM: "这是一个红色的苹果"
```

**技术选型**:
- Qwen2-VL (本地部署)
- GPT-4V / Claude 3.5 (云端备选)

**任务清单**:
- [ ] 集成VLM API客户端
- [ ] 实现图像+文本的联合Prompt
- [ ] 支持"看图识物"场景
- [ ] 视觉问答（VQA）
- [ ] 场景理解（场景描述生成）

#### 3.2 语音交互 (Priority: 🟡 Medium)
```
voice/
├── asr.py               # 语音识别
├── tts.py               # 语音合成
└── wake_word.py         # 唤醒词检测
```

**技术选型**:
- ASR: Whisper / FunASR
- TTS: Edge-TTS / PaddleSpeech
- 唤醒词: Porcupine

**任务清单**:
- [ ] 集成Whisper进行语音识别
- [ ] 集成TTS引擎
- [ ] 实现唤醒词检测（"你好LARA"）
- [ ] 语音端点检测（VAD）
- [ ] 噪声抑制

#### 3.3 本地模型部署 (Priority: 🔴 High)
```bash
# Jetson Orin 部署方案
├── Ollama (推理引擎)
│   └── Qwen2.5:3B (量化版)
├── TensorRT (加速)
└── vLLM (可选，batch推理)
```

**任务清单**:
- [ ] 在Orin上安装Ollama
- [ ] 部署Qwen2.5-3B-Instruct-Q4
- [ ] 性能测试（latency, throughput）
- [ ] 量化优化（INT8/INT4）
- [ ] 切换config.py为LOCAL模式
- [ ] 对比云端/本地性能差异

#### 3.4 记忆和上下文管理 (Priority: 🟢 Low)
```python
brain/
├── memory.py            # 长期记忆
└── context_manager.py   # 上下文窗口管理

# 支持跨对话记忆
Session 1: "我叫张三"
Session 2: "我是谁？" → "您是张三"
```

**任务清单**:
- [ ] 实现向量数据库（Chroma/FAISS）
- [ ] 存储对话历史
- [ ] 实现RAG（检索增强生成）
- [ ] 上下文窗口滑动策略
- [ ] 用户偏好记忆

---

### **第四阶段：用户体验** (2-3周)
**目标**: 可视化、易用、可演示

#### 4.1 Web监控面板 (Priority: 🔴 High)
```
web/
├── backend/
│   └── server.py        # FastAPI服务
├── frontend/
│   ├── index.html
│   ├── app.js          # Vue.js / React
│   └── styles.css
└── static/
    └── assets/
```

**功能设计**:
- 实时机器人状态显示
- 相机视频流
- 地图可视化（ROS 2地图显示）
- 对话历史
- 性能监控图表
- 远程控制（发送指令）

**任务清单**:
- [ ] 搭建FastAPI后端
- [ ] 实现WebSocket实时通信
- [ ] 前端UI开发（推荐Vue 3）
- [ ] 视频流转发（WebRTC/MJPEG）
- [ ] ROS 2地图可视化（rosbridge）
- [ ] 移动端适配

#### 4.2 RViz可视化 (Priority: 🟡 Medium)
```bash
# RViz配置
rviz/
└── robot_view.rviz     # 预配置的RViz布局

# 显示内容
- 机器人模型（URDF）
- 地图（OccupancyGrid）
- 激光雷达数据
- 相机图像
- 检测框（BoundingBox）
- 导航路径
```

**任务清单**:
- [ ] 配置RViz显示
- [ ] 添加自定义Marker（检测结果）
- [ ] 路径可视化
- [ ] 交互式导航（点击地图发送目标）

#### 4.3 移动端App (Priority: 🟢 Low)
```
mobile/
└── flutter_app/        # 跨平台App
    ├── lib/
    └── pubspec.yaml
```

**功能**:
- 语音控制
- 远程查看
- 快捷指令
- 推送通知

**任务清单**:
- [ ] Flutter项目初始化
- [ ] 接入Web API
- [ ] 实现语音输入
- [ ] 视频流显示

#### 4.4 文档和演示 (Priority: 🟡 Medium)
```
docs/
├── API.md              # API文档
├── DEPLOYMENT.md       # 部署指南
├── ARCHITECTURE.md     # 架构文档
└── videos/
    └── demo.mp4        # 演示视频
```

**任务清单**:
- [ ] 补充API文档（Swagger自动生成）
- [ ] 编写部署教程
- [ ] 录制演示视频
- [ ] 制作PPT演示材料

---

## 🔧 技术债务清理

### 代码质量
- [ ] 添加类型注解（mypy检查）
- [ ] 代码格式化（black, isort）
- [ ] Lint检查（pylint, flake8）
- [ ] 代码审查流程

### 安全性
- [ ] API密钥加密存储
- [ ] 用户输入验证
- [ ] SQL注入防护（如果使用数据库）
- [ ] HTTPS部署
- [ ] 安全模式（限制危险动作）

### 性能优化
- [ ] 异步化（asyncio）
- [ ] 请求缓存
- [ ] 数据库查询优化
- [ ] 内存泄漏检查

---

## 📈 里程碑和验收标准

### Milestone 1: 稳定基座 (Week 3-4)
**验收标准**:
- [ ] 测试覆盖率 >80%
- [ ] CI/CD流程运行正常
- [ ] 错误恢复机制完善
- [ ] 日志系统完整

### Milestone 2: 硬件对接 (Week 8-10)
**验收标准**:
- [ ] 在真实机器人上完成导航任务
- [ ] 视觉检测准确率 >85%
- [ ] 抓取成功率 >70%
- [ ] ROS 2集成无故障

### Milestone 3: 智能增强 (Week 12-13)
**验收标准**:
- [ ] 支持视觉问答
- [ ] 语音交互流畅
- [ ] 本地模型推理延迟 <2s
- [ ] 多轮对话理解准确

### Milestone 4: 用户体验 (Week 15-16)
**验收标准**:
- [ ] Web面板功能完整
- [ ] 演示视频制作完成
- [ ] 文档齐全
- [ ] 用户测试反馈良好

---

## 🚀 快速启动建议

### 立即开始（本周）
1. **搭建测试框架** - 安装pytest，编写前10个测试用例
2. **设计ROS 2接口** - 绘制接口UML图，定义消息格式
3. **创建Web面板雏形** - FastAPI + 简单前端

### 优先级排序
```
🔴 High Priority (必须):
- 测试框架
- 错误处理
- ROS 2接口
- 视觉模块
- 本地模型部署
- Web监控面板

🟡 Medium Priority (重要):
- 日志系统
- 任务队列
- 导航建图
- 语音交互
- RViz可视化

🟢 Low Priority (可选):
- 移动端App
- 记忆系统
- 高级功能
```

---

## 📝 附录

### 技术栈总览
```yaml
Language: Python 3.10+
Framework:
  Backend: FastAPI, ROS 2 Humble
  Frontend: Vue 3 / React
  Mobile: Flutter (可选)

AI/ML:
  LLM: Qwen2.5-3B (Ollama)
  VLM: Qwen2-VL
  Vision: YOLOv8, CLIP
  Speech: Whisper, Edge-TTS

Robotics:
  Navigation: Nav2
  Manipulation: MoveIt2
  SLAM: Cartographer

Infrastructure:
  Test: pytest, pytest-cov
  Log: loguru
  Monitor: Prometheus (可选)
  DB: SQLite / PostgreSQL (可选)
```

### 参考资源
- [ROS 2 Humble Docs](https://docs.ros.org/en/humble/)
- [Nav2 Documentation](https://navigation.ros.org/)
- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)

---

**最后更新**: 2025-12-05
**版本**: V2.0
**维护者**: Project CADE Team

````


## body/__init__.py

```py
"""
Body Module - 躯干模块

负责硬件控制、动作执行（支持Mock模式和真实ROS模式）
"""

__version__ = "0.1.0"

```


## body/mock_robot.py

```py
"""
Mock Robot - 模拟机器人

在没有真实硬件的情况下，模拟机器人的行为
用于PC端开发和测试
"""

import time
import random
from typing import Optional, List
from body.robot_interface import RobotInterface, RobotState


class MockRobot(RobotInterface):
    """
    模拟机器人类

    模拟所有硬件操作，用日志代替真实动作
    """

    def __init__(self, name: str = "MockRobot"):
        super().__init__()
        self.name = name
        self.current_position = "home"  # 初始位置
        self.holding_object = None      # 当前抓取的物体

        # 模拟的环境地图
        self.known_locations = {
            "home": [0.0, 0.0, 0.0],
            "起点": [0.0, 0.0, 0.0],  # 中文别名
            "start_point": [0.0, 0.0, 0.0],  # 英文别名
            "kitchen": [5.0, 2.0, 0.0],
            "厨房": [5.0, 2.0, 0.0],
            "living_room": [3.0, -1.0, 0.0],
            "客厅": [3.0, -1.0, 0.0],
            "bedroom": [-2.0, 4.0, 0.0],
            "卧室": [-2.0, 4.0, 0.0],
            "table": [4.0, 1.0, 0.0],
            "桌子": [4.0, 1.0, 0.0],
            "desk": [1.0, 3.0, 0.0],
            "书桌": [1.0, 3.0, 0.0],
        }

        # 模拟的物体数据库
        self.known_objects = {
            "apple": {"name": "apple", "location": "table", "position": [4.0, 1.0, 0.8]},
            "bottle": {"name": "bottle", "location": "kitchen", "position": [5.0, 2.0, 1.0]},
            "cup": {"name": "cup", "location": "table", "position": [4.2, 1.0, 0.8]},
            "book": {"name": "book", "location": "desk", "position": [1.0, 3.0, 0.9]},
        }

        print(f"🤖 {self.name} 初始化成功")
        print(f"   当前位置: {self.current_position}")
        print(f"   已知位置: {list(self.known_locations.keys())}")
        print(f"   已知物体: {list(self.known_objects.keys())}")

    def navigate(self, target) -> bool:
        """模拟导航"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n🚗 [导航] 从 {self.current_position} 前往 {target}")

        # 模拟移动延迟
        time.sleep(0.5)

        # 检查目标是否存在
        if isinstance(target, str):
            if target in self.known_locations:
                self.current_position = target
                coords = self.known_locations[target]
                print(f"✓ 已到达 {target} (坐标: {coords})")
                self.set_state(RobotState.IDLE)
                return True
            else:
                print(f"✗ 未知位置: {target}")
                self.set_state(RobotState.ERROR)
                return False
        elif isinstance(target, list) and len(target) == 3:
            # 直接坐标导航
            self.current_position = f"坐标{target}"
            print(f"✓ 已到达坐标 {target}")
            self.set_state(RobotState.IDLE)
            return True
        else:
            print(f"✗ 无效的目标格式: {target}")
            self.set_state(RobotState.ERROR)
            return False

    def search(self, object_name: str) -> Optional[dict]:
        """模拟搜索物体"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n🔍 [搜索] 正在寻找: {object_name}")

        # 模拟搜索延迟
        time.sleep(0.8)

        # 查找物体
        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            print(f"✓ 找到 {object_name} 在 {obj['location']}")
            print(f"   位置: {obj['position']}")
            self.set_state(RobotState.IDLE)
            return obj
        else:
            # 模拟一定概率找不到
            if random.random() < 0.3:
                print(f"✗ 未找到 {object_name}")
                self.set_state(RobotState.IDLE)
                return None
            else:
                # 创建一个"发现"的物体
                new_obj = {
                    "name": object_name,
                    "location": self.current_position,
                    "position": [
                        random.uniform(-5, 5),
                        random.uniform(-5, 5),
                        random.uniform(0.5, 1.5)
                    ]
                }
                self.known_objects[object_name] = new_obj
                print(f"✓ 找到 {object_name} 在当前位置")
                self.set_state(RobotState.IDLE)
                return new_obj

    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        """模拟抓取物体"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n🤏 [抓取] 尝试抓取: {object_name}")

        # 检查是否已经抓着东西
        if self.holding_object:
            print(f"✗ 手中已有物体: {self.holding_object}")
            self.set_state(RobotState.ERROR)
            return False

        # 模拟抓取延迟
        time.sleep(0.6)

        # 检查物体是否存在
        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            # 简化版：不检查距离，假设已经在附近
            self.holding_object = object_name
            print(f"✓ 成功抓取 {object_name}")
            self.set_state(RobotState.IDLE)
            return True
        else:
            print(f"✗ 物体不存在: {object_name}")
            self.set_state(RobotState.ERROR)
            return False

    def place(self, location) -> bool:
        """模拟放置物体"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n📦 [放置] 将物体放置到: {location}")

        # 检查是否抓着东西
        if not self.holding_object:
            print(f"✗ 手中没有物体")
            self.set_state(RobotState.ERROR)
            return False

        # 模拟放置延迟
        time.sleep(0.5)

        print(f"✓ 已将 {self.holding_object} 放置到 {location}")
        # 更新物体位置
        if self.holding_object in self.known_objects:
            self.known_objects[self.holding_object]["location"] = str(location)

        self.holding_object = None
        self.set_state(RobotState.IDLE)
        return True

    def speak(self, content: str) -> bool:
        """模拟语音输出"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n💬 [语音] {self.name}: \"{content}\"")

        # 模拟TTS延迟
        time.sleep(0.3)

        self.set_state(RobotState.IDLE)
        return True

    def wait(self, reason: Optional[str] = None) -> bool:
        """等待/无操作"""
        msg = f"\n⏸️  [等待]"
        if reason:
            msg += f" 原因: {reason}"
        print(msg)

        self.set_state(RobotState.IDLE)
        return True

    def get_status(self) -> dict:
        """获取机器人状态"""
        return {
            "name": self.name,
            "state": self.state.value,
            "position": self.current_position,
            "holding": self.holding_object,
        }

    def print_status(self):
        """打印当前状态"""
        status = self.get_status()
        print(f"\n📊 机器人状态:")
        print(f"   状态: {status['state']}")
        print(f"   位置: {status['position']}")
        print(f"   手持: {status['holding'] or '无'}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 测试 Mock Robot ===\n")

    # 创建机器人
    robot = MockRobot(name="LARA")

    # 测试导航
    robot.navigate("kitchen")
    robot.print_status()

    # 测试搜索
    result = robot.search("bottle")
    print(f"搜索结果: {result}")

    # 测试抓取
    robot.pick("bottle")
    robot.print_status()

    # 测试导航 + 放置
    robot.navigate("table")
    robot.place("table")
    robot.print_status()

    # 测试语音
    robot.speak("任务完成！")

    print("\n✓ 所有测试通过！")

```


## body/robot_interface.py

```py
"""
Robot Interface - 机器人接口定义

定义统一的机器人抽象接口，Mock和Real类都需要实现这个接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum


class RobotState(str, Enum):
    """机器人状态"""
    IDLE = "IDLE"              # 空闲
    THINKING = "THINKING"      # 思考中（LLM推理）
    EXECUTING = "EXECUTING"    # 执行中
    ERROR = "ERROR"            # 错误状态


class RobotInterface(ABC):
    """
    机器人抽象接口

    所有机器人类（Mock/Real）都必须实现这些方法
    """

    def __init__(self):
        self.state = RobotState.IDLE
        self.current_position: Optional[str] = None
        self.holding_object: Optional[str] = None

    @abstractmethod
    def navigate(self, target) -> bool:
        """
        导航到目标位置

        Args:
            target: 目标位置（语义标签或坐标）

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def search(self, object_name: str) -> Optional[dict]:
        """
        搜索物体

        Args:
            object_name: 物体名称

        Returns:
            dict: 找到的物体信息，如 {"name": "apple", "position": [x,y,z]}
            None: 未找到
        """
        pass

    @abstractmethod
    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        """
        抓取物体

        Args:
            object_name: 物体名称
            object_id: 物体ID（如果有多个同名物体）

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def place(self, location) -> bool:
        """
        放置物体

        Args:
            location: 放置位置

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def speak(self, content: str) -> bool:
        """
        语音输出

        Args:
            content: 要说的内容

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def wait(self, reason: Optional[str] = None) -> bool:
        """
        等待/无操作

        Args:
            reason: 等待原因

        Returns:
            bool: 是否成功
        """
        pass

    def get_state(self) -> RobotState:
        """获取当前状态"""
        return self.state

    def set_state(self, state: RobotState):
        """设置状态"""
        self.state = state

```


## brain/__init__.py

```py
"""
Brain Module - 大脑模块

负责LLM交互、意图识别、决策生成
"""

__version__ = "0.1.0"

```


## brain/llm_client.py

````py
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

````


## brain/prompts.py

````py
"""
System Prompts - 系统提示词

定义机器人的行为规范、动作空间和输出格式
"""

from config import Config


# ==================== 核心系统提示词 ====================

ROBOT_SYSTEM_PROMPT = f"""你是 {Config.ROBOT_NAME}，一个智能服务机器人。你的任务是理解用户的指令，并做出合理的决策。

## 核心能力

你拥有以下物理动作能力：

1. **navigate** - 导航到指定位置
   - 参数: target (语义位置如"kitchen"或坐标[x,y,z])
   - 示例: {{"type": "navigate", "target": "kitchen"}}

2. **search** - 搜索物体
   - 参数: object_name (物体名称)
   - 示例: {{"type": "search", "object_name": "apple"}}

3. **pick** - 抓取物体
   - 参数: object_name (物体名称), object_id (可选)
   - 示例: {{"type": "pick", "object_name": "bottle", "object_id": 1}}

4. **place** - 放置物体
   - 参数: location (位置)
   - 示例: {{"type": "place", "location": "table"}}

5. **speak** - 语音输出
   - 参数: content (要说的内容)
   - 示例: {{"type": "speak", "content": "好的，我明白了"}}

6. **wait** - 等待/无需动作
   - 参数: reason (可选)
   - 示例: {{"type": "wait", "reason": "用户只是在聊天"}}

## 行为规则

1. **意图识别**：首先判断用户是在"闲聊"还是"下达任务指令"
   - 闲聊示例："你好"、"今天天气怎么样"、"你叫什么名字"
   - 任务示例："帮我拿苹果"、"去厨房"、"找到水杯"

2. **思考模式（CoT）**：
   - 分析用户的真实意图
   - 判断需要执行哪些步骤
   - 考虑当前状态和限制
   - **重要**：一次只输出一个动作，执行后会收到反馈再决定下一步

3. **输出约束**：
   - 不要幻想你能做物理动作之外的事情（如查天气、算数学题）
   - 只能使用上述定义的6种动作
   - 坐标尽量使用语义标签（如"kitchen"），除非明确给出数字坐标
   - 保持谦逊和礼貌

## 输出格式

你**必须**严格按照以下JSON格式输出（可以用markdown代码块包裹）：

```json
{{
  "thought": "你的思考过程（中文，详细说明你的推理）",
  "reply": "给用户的自然语言回复（可选，如果不需要说话可以为null）",
  "action": {{
    "type": "动作类型",
    "参数名": "参数值"
  }}
}}
```

### 特殊情况

- **纯对话**（无需动作）：action 设为 {{"type": "wait", "reason": "闲聊"}}
- **需要说话**：如果要告知用户你在做什么，使用 speak 动作而不是 reply 字段
- **多步任务**：只输出第一步动作，等待执行结果反馈后再决定下一步

## 示例

### 示例1：闲聊
用户："你好呀"
输出：
```json
{{
  "thought": "用户在问候我，这是社交性对话，不需要执行物理动作",
  "reply": "您好！我是{Config.ROBOT_NAME}，很高兴为您服务。有什么我可以帮您的吗？",
  "action": {{"type": "wait", "reason": "闲聊"}}
}}
```

### 示例2：简单任务
用户："去厨房"
输出：
```json
{{
  "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
  "reply": "好的，我这就去厨房",
  "action": {{"type": "navigate", "target": "kitchen"}}
}}
```

### 示例3：复杂任务（多步骤）
用户："帮我把桌子上的水杯拿过来"
输出：
```json
{{
  "thought": "用户要我拿水杯。完整流程应该是：1)导航到桌子 2)搜索水杯 3)抓取水杯 4)导航回用户。但我一次只能执行一个动作，所以先导航到桌子",
  "reply": "好的，我去拿水杯",
  "action": {{"type": "navigate", "target": "table"}}
}}
```

### 示例4：需要搜索
用户："找到苹果"
输出：
```json
{{
  "thought": "用户要我找苹果，我需要使用搜索功能",
  "reply": "我开始搜索苹果",
  "action": {{"type": "search", "object_name": "apple"}}
}}
```

## 重要提醒

- 永远不要输出超出上述6种动作的内容
- 每次只输出一个动作，不要试图输出动作序列
- 如果用户要求你做不了的事（如"帮我订外卖"），礼貌地说明你的能力限制
- 保持JSON格式的严格正确，否则系统会无法解析
"""


# ==================== 其他提示词变体 ====================

# 简化版提示词（用于测试）
SIMPLE_PROMPT = """你是一个服务机器人。根据用户指令，输出JSON格式的决策。

可用动作：navigate, search, pick, place, speak, wait

输出格式：
{{
  "thought": "思考过程",
  "reply": "回复（可选）",
  "action": {{"type": "动作类型", ...参数}}
}}
"""


# 精简版提示词（无思考过程，直接执行）
COMPACT_PROMPT = f"""你是 {Config.ROBOT_NAME}，一个智能服务机器人。你的任务是理解用户的指令，并做出合理的决策。

## 核心能力

你拥有以下物理动作能力：

1. **navigate** - 导航到指定位置
   - 参数: target (语义位置如"kitchen"或坐标[x,y,z])
   - 示例: {{"type": "navigate", "target": "kitchen"}}

2. **search** - 搜索物体
   - 参数: object_name (物体名称)
   - 示例: {{"type": "search", "object_name": "apple"}}

3. **pick** - 抓取物体
   - 参数: object_name (物体名称), object_id (可选)
   - 示例: {{"type": "pick", "object_name": "bottle"}}

4. **place** - 放置物体
   - 参数: location (位置)
   - 示例: {{"type": "place", "location": "table"}}

5. **speak** - 语音输出
   - 参数: content (要说的内容)
   - 示例: {{"type": "speak", "content": "好的"}}

6. **wait** - 等待/无需动作
   - 参数: reason (可选)
   - 示例: {{"type": "wait"}}

## 行为规则

1. **意图识别**：判断用户是"闲聊"还是"下达任务指令"
2. **直接执行**：不需要详细的思考过程，直接输出动作
3. **一次一步**：一次只输出一个动作

## 输出格式（精简）

你**必须**严格按照以下JSON格式输出（**不需要** thought 字段）：

```json
{{
  "reply": "给用户的回复（可选）",
  "action": {{
    "type": "动作类型",
    "参数名": "参数值"
  }}
}}
```

## 示例

### 示例1：闲聊
用户："你好呀"
输出：
```json
{{
  "reply": "您好！我是{Config.ROBOT_NAME}，很高兴为您服务。",
  "action": {{"type": "wait"}}
}}
```

### 示例2：简单任务
用户："去厨房"
输出：
```json
{{
  "reply": "好的",
  "action": {{"type": "navigate", "target": "kitchen"}}
}}
```

### 示例3：复杂任务
用户："帮我把桌子上的水杯拿过来"
输出：
```json
{{
  "reply": "好的",
  "action": {{"type": "navigate", "target": "table"}}
}}
```

## 重要提醒

- 永远不要输出超出上述6种动作的内容
- 每次只输出一个动作，不要试图输出动作序列
- **不要输出 thought 字段**，直接输出 reply 和 action
- 保持JSON格式的严格正确
"""


# 调试模式提示词（会输出更详细的思考过程）
DEBUG_PROMPT = ROBOT_SYSTEM_PROMPT + """

## 调试模式已启用

请在 thought 字段中输出极其详细的推理过程，包括：
- 对用户意图的理解
- 考虑的所有可能方案
- 为什么选择当前动作
- 预期的执行结果
"""


# ==================== 提示词工具函数 ====================

def get_system_prompt(mode: str = "default") -> str:
    """
    获取系统提示词

    Args:
        mode: 提示词模式
            - "default": 标准提示词（包含 thought）
            - "simple": 简化提示词
            - "compact": 精简提示词（不需要 thought）⭐
            - "debug": 调试提示词

    Returns:
        str: 对应的系统提示词
    """
    prompts = {
        "default": ROBOT_SYSTEM_PROMPT,
        "simple": SIMPLE_PROMPT,
        "compact": COMPACT_PROMPT,  # ← 新增
        "debug": DEBUG_PROMPT,
    }

    if mode not in prompts:
        raise ValueError(f"未知的提示词模式: {mode}. 可用: {list(prompts.keys())}")

    return prompts[mode]


def add_context(base_prompt: str, context: str) -> str:
    """
    向基础提示词中添加上下文信息

    Args:
        base_prompt: 基础提示词
        context: 要添加的上下文（如当前位置、已知物体等）

    Returns:
        str: 增强后的提示词
    """
    return f"""{base_prompt}

## 当前环境信息

{context}

请根据上述环境信息做出决策。
"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 系统提示词预览 ===\n")
    print(ROBOT_SYSTEM_PROMPT)
    print("\n" + "=" * 50)
    print(f"提示词长度: {len(ROBOT_SYSTEM_PROMPT)} 字符")
    print(f"机器人名称: {Config.ROBOT_NAME}")

````


## brain/schemas.py

```py
"""
数据模型定义 - Data Schemas

使用 Pydantic 严格定义机器人的动作空间和决策输出格式
"""

from typing import Literal, Union, Optional, List
from pydantic import BaseModel, Field, field_validator


# ==================== 动作类型定义 ====================

class NavigateAction(BaseModel):
    """导航动作 - 移动到指定位置"""
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]] = Field(
        ...,
        description="目标位置，可以是语义标签(如'kitchen')或坐标[x,y,z]"
    )

    @field_validator('target')
    @classmethod
    def validate_target(cls, v):
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError("坐标必须是 [x, y, z] 格式")
            if not all(isinstance(i, (int, float)) for i in v):
                raise ValueError("坐标必须是数字")
        return v


class PickAction(BaseModel):
    """抓取动作 - 拾取物体"""
    type: Literal["pick"] = "pick"
    object_name: str = Field(..., description="物体名称")
    object_id: Optional[int] = Field(None, description="物体ID（如果有多个同名物体）")


class PlaceAction(BaseModel):
    """放置动作 - 将物体放到指定位置"""
    type: Literal["place"] = "place"
    location: Union[str, List[float]] = Field(
        ...,
        description="放置位置，可以是语义标签(如'table')或坐标"
    )


class SearchAction(BaseModel):
    """搜索动作 - 寻找物体"""
    type: Literal["search"] = "search"
    object_name: str = Field(..., description="要搜索的物体名称")


class SpeakAction(BaseModel):
    """说话动作 - 语音输出"""
    type: Literal["speak"] = "speak"
    content: str = Field(..., description="要说的内容")


class WaitAction(BaseModel):
    """等待动作 - 保持当前状态"""
    type: Literal["wait"] = "wait"
    reason: Optional[str] = Field(None, description="等待原因")


# ==================== 联合动作类型 ====================

# 所有可能的动作类型
RobotAction = Union[
    NavigateAction,
    PickAction,
    PlaceAction,
    SearchAction,
    SpeakAction,
    WaitAction
]


# ==================== 决策输出格式 ====================

class RobotDecision(BaseModel):
    """
    机器人决策输出格式（LLM的输出结构）

    这是LLM必须遵循的输出格式，包含思考过程、回复和动作
    """
    thought: Optional[str] = Field(  # ← 改为可选
        None,
        description="内部思考过程（CoT - Chain of Thought）"
    )
    reply: Optional[str] = Field(
        None,
        description="给用户的自然语言回复（可选）"
    )
    action: Optional[RobotAction] = Field(
        None,
        description="要执行的动作（如果是纯对话则为None）"
    )

    class Config:
        # 允许任意类型（用于处理联合类型）
        arbitrary_types_allowed = True


# ==================== 辅助函数 ====================

def parse_action(action_dict: dict) -> RobotAction:
    """
    根据 type 字段解析动作

    Args:
        action_dict: 包含 type 字段的字典

    Returns:
        对应的 Action 对象

    Raises:
        ValueError: 如果 type 不合法
    """
    action_type = action_dict.get("type")

    action_map = {
        "navigate": NavigateAction,
        "pick": PickAction,
        "place": PlaceAction,
        "search": SearchAction,
        "speak": SpeakAction,
        "wait": WaitAction,
    }

    if action_type not in action_map:
        raise ValueError(
            f"未知的动作类型: {action_type}. "
            f"可用动作: {list(action_map.keys())}"
        )

    return action_map[action_type](**action_dict)


# ==================== 示例数据 ====================

if __name__ == "__main__":
    # 测试用例
    print("=== 测试动作定义 ===\n")

    # 1. 导航动作
    nav_action = NavigateAction(target="kitchen")
    print(f"1. 导航（语义）: {nav_action.model_dump_json(indent=2)}\n")

    nav_action_coords = NavigateAction(target=[1.5, 2.3, 0.0])
    print(f"2. 导航（坐标）: {nav_action_coords.model_dump_json(indent=2)}\n")

    # 2. 完整决策
    decision = RobotDecision(
        thought="用户想要苹果，我需要先找到它",
        reply="好的，我这就去找苹果",
        action=SearchAction(object_name="apple")
    )
    print(f"3. 完整决策: {decision.model_dump_json(indent=2)}\n")

    # 3. 纯对话（无动作）
    chat_decision = RobotDecision(
        thought="用户在问候我",
        reply="您好！我是服务机器人LARA，很高兴为您服务",
        action=None
    )
    print(f"4. 纯对话: {chat_decision.model_dump_json(indent=2)}\n")

    print("✓ 所有测试通过！")

```


## compare_modes.py

```py
#!/usr/bin/env python3
"""
对比测试：标准模式 vs 精简模式

测试 DeepSeek 是否真的不生成 thought 字段
"""

import time
from robot_controller import RobotController


def test_mode(mode_name: str, prompt_mode: str, show_thought: bool = True):
    """测试指定模式"""

    print("="*70)
    print(f"🧪 测试模式: {mode_name}")
    print(f"   Prompt: {prompt_mode}")
    print(f"   显示思考: {show_thought}")
    print("="*70)

    controller = RobotController(
        prompt_mode=prompt_mode,
        show_thought=show_thought
    )

    test_inputs = [
        "你好",
        "去厨房",
        "找到苹果",
    ]

    total_time = 0
    for user_input in test_inputs:
        start = time.time()
        decision = controller.process_input(user_input)
        elapsed = time.time() - start

        total_time += elapsed

        # 检查是否生成了 thought
        has_thought = decision.thought is not None
        thought_info = "✅ 有" if has_thought else "❌ 无"

        print(f"\n📊 结果分析:")
        print(f"   thought 字段: {thought_info}")
        if has_thought:
            print(f"   thought 长度: {len(decision.thought)} 字符")
        print(f"   耗时: {elapsed:.2f}s")
        print()

    print(f"\n⏱️  总耗时: {total_time:.2f}s")
    print(f"📊 平均耗时: {total_time/len(test_inputs):.2f}s/次")

    return total_time


def main():
    """主函数 - 对比测试"""

    print("\n" + "🎯"*35)
    print("对比测试：标准模式 vs 精简模式")
    print("🎯"*35 + "\n")

    # 测试1：标准模式（default）
    time1 = test_mode(
        mode_name="标准模式（包含思考）",
        prompt_mode="default",
        show_thought=True
    )

    input("\n按 Enter 继续下一个测试...")

    # 测试2：精简模式（compact）
    time2 = test_mode(
        mode_name="精简模式（无思考）",
        prompt_mode="compact",
        show_thought=False
    )

    # 对比结果
    print("\n\n" + "="*70)
    print("📊 对比结果")
    print("="*70)

    print(f"\n标准模式总耗时: {time1:.2f}s")
    print(f"精简模式总耗时: {time2:.2f}s")
    print(f"时间差: {abs(time1-time2):.2f}s")

    if time2 < time1:
        speedup = ((time1 - time2) / time1) * 100
        print(f"⚡ 精简模式快了 {speedup:.1f}%")
    else:
        slowdown = ((time2 - time1) / time1) * 100
        print(f"⚠️  精简模式慢了 {slowdown:.1f}%")

    print("\n💡 观察重点:")
    print("   1. 精简模式的 LLM 输出中是否真的没有 thought 字段？")
    print("   2. 精简模式是否节省了推理时间？")
    print("   3. 精简模式的动作执行是否同样准确？")

    print("\n📖 查看详细分析:")
    print("   - LLM 原始输出在 '📄 LLM 原始输出' 部分")
    print("   - thought 字段状态在 '📊 结果分析' 部分")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "default":
            test_mode("标准模式", "default", True)
        elif mode == "compact":
            test_mode("精简模式", "compact", False)
        else:
            print(f"未知模式: {mode}")
            print("用法: python compare_modes.py [default|compact]")
    else:
        main()

```


## config.py

```py
"""
配置抽象层 - Configuration Layer

支持云端(CLOUD)和本地(LOCAL)模式无缝切换
"""

from enum import Enum
from typing import Literal


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
    # DeepSeek API (推荐，性价比高)
    CLOUD_BASE_URL = "https://api.deepseek.com"
    CLOUD_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 替换为你的API Key
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


# 尝试导入本地配置（用于覆盖默认值，不提交到git）
try:
    import config_local

    # 动态更新 Config 类的属性
    for attr in dir(config_local):
        if not attr.startswith('_'):  # 跳过私有属性
            value = getattr(config_local, attr)
            if hasattr(Config, attr):
                setattr(Config, attr, value)

    print("✓ 已加载本地配置 config_local.py")
except ImportError:
    print("⚠ 未找到 config_local.py，使用默认配置")
    print("提示：复制 config.py 为 config_local.py 并填入你的API密钥")

```


## config_local.example.py

```py
"""
本地配置示例

1. 将此文件重命名为 config_local.py
2. 填入你的 API 密钥
3. config_local.py 会自动覆盖 config.py 中的默认配置
"""

from config import RunMode

# ==================== 运行模式 ====================
# MODE = RunMode.CLOUD  # 云端API
# MODE = RunMode.LOCAL  # 本地Ollama

# ==================== 云端配置 ====================

# DeepSeek API（推荐）
# 注册地址: https://platform.deepseek.com/
# CLOUD_BASE_URL = "https://api.deepseek.com"
# CLOUD_API_KEY = "sk-your-deepseek-api-key-here"
# CLOUD_MODEL = "deepseek-chat"

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
# LOCAL_BASE_URL = "http://localhost:11434/v1"
# LOCAL_API_KEY = "ollama"
# LOCAL_MODEL = "qwen2.5:3b"

# ==================== 其他配置 ====================
# TEMPERATURE = 0.7
# MAX_TOKENS = 512
# ROBOT_NAME = "LARA"

```


## config_local.py

```py
"""
本地配置示例

1. 将此文件重命名为 config_local.py
2. 填入你的 API 密钥
3. config_local.py 会自动覆盖 config.py 中的默认配置
"""

from config import RunMode

# ==================== 运行模式 ====================
# MODE = RunMode.CLOUD  # 云端API
MODE = RunMode.LOCAL  # 本地Ollama

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

```


## config_local_example.py

```py
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

```


## demo_navigate.py

```py
#!/usr/bin/env python3
"""
navigate 指令完整流程演示

展示从用户输入到机器人执行的每一步
"""

import json
from brain.schemas import NavigateAction, parse_action
from body.mock_robot import MockRobot


def demo_flow():
    """完整流程演示"""

    print("="*70)
    print("🎬 navigate 指令完整流程演示")
    print("="*70)

    # ==================== 阶段1：用户输入 ====================
    print("\n【阶段1：用户输入（自然语言）】")
    print("─"*70)
    user_input = "去厨房"
    print(f"👤 用户说: \"{user_input}\"")
    print(f"\n💭 这只是普通的人类语言，机器人不能直接理解")

    input("\n按 Enter 继续...")

    # ==================== 阶段2：System Prompt ====================
    print("\n\n【阶段2：System Prompt 提供\"说明书\"】")
    print("─"*70)

    prompt_snippet = """
你拥有以下动作能力：

1. **navigate** - 导航到指定位置
   - 参数: target (如"kitchen"或[x,y,z])
   - 示例: {"type": "navigate", "target": "kitchen"}

输出格式：
{
  "thought": "思考过程",
  "reply": "回复",
  "action": {"type": "动作类型", "参数": "值"}
}
    """
    print(f"📖 System Prompt（片段）:\n{prompt_snippet}")

    print(f"\n💡 这告诉 LLM:")
    print(f"   ✅ 有哪些动作可以用")
    print(f"   ✅ 每个动作需要什么参数")
    print(f"   ✅ 如何输出 JSON 格式")

    input("\n按 Enter 继续...")

    # ==================== 阶段3：LLM 推理 ====================
    print("\n\n【阶段3：LLM 的推理过程】")
    print("─"*70)

    print(f"\n🧠 LLM 的内心独白:")
    print(f"   1. '用户说\"去厨房\"'")
    print(f"   2. '这是一个位置移动的需求'")
    print(f"   3. '查看动作列表... navigate 可以用！'")
    print(f"   4. 'navigate 需要 target 参数'")
    print(f"   5. '\"厨房\"就是 target'")
    print(f"   6. '输出 JSON'")

    llm_output = {
        "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
        "reply": "好的，我这就去厨房",
        "action": {
            "type": "navigate",
            "target": "kitchen"
        }
    }

    print(f"\n📤 LLM 输出:")
    print(json.dumps(llm_output, indent=2, ensure_ascii=False))

    input("\n按 Enter 继续...")

    # ==================== 阶段4：Schema 验证 ====================
    print("\n\n【阶段4：Pydantic Schema 验证】")
    print("─"*70)

    print(f"\n📋 Schema 定义 (brain/schemas.py):")
    schema_code = '''
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]

    @field_validator('target')
    def validate_target(cls, v):
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError("坐标必须是 [x, y, z]")
        return v
    '''
    print(schema_code)

    print(f"\n🔍 验证过程:")

    try:
        action_data = llm_output["action"]
        print(f"   输入: {action_data}")

        # 验证
        action = parse_action(action_data)
        print(f"   ✅ type 检查: '{action.type}' == 'navigate'")
        print(f"   ✅ target 检查: '{action.target}' 是字符串")
        print(f"   ✅ 验证通过！")

        print(f"\n📦 创建的对象:")
        print(f"   action.type = '{action.type}'")
        print(f"   action.target = '{action.target}'")

    except Exception as e:
        print(f"   ❌ 验证失败: {e}")

    input("\n按 Enter 继续...")

    # ==================== 阶段5：动作派发 ====================
    print("\n\n【阶段5：动作派发（Dispatch）】")
    print("─"*70)

    dispatch_code = '''
def _execute_action(self, action: RobotAction) -> bool:
    action_type = action.type  # "navigate"

    if action_type == "navigate":
        return self.robot.navigate(action.target)
                                    ↑
                            传入 "kitchen"
    '''
    print(f"📝 代码逻辑 (robot_controller.py):")
    print(dispatch_code)

    print(f"\n🔀 派发流程:")
    print(f"   1. 读取 action.type = '{action.type}'")
    print(f"   2. 匹配到 if action_type == \"navigate\"")
    print(f"   3. 调用 robot.navigate('{action.target}')")

    input("\n按 Enter 继续...")

    # ==================== 阶段6：机器人执行 ====================
    print("\n\n【阶段6：机器人执行】")
    print("─"*70)

    print(f"\n🤖 创建机器人实例:")
    robot = MockRobot(name="DEMO")

    print(f"\n📍 当前位置: {robot.current_position}")
    print(f"🗺️  已知位置: {list(robot.known_locations.keys())[:6]}...")

    print(f"\n⚡ 执行 robot.navigate('kitchen'):")
    success = robot.navigate("kitchen")

    if success:
        print(f"\n✅ 执行成功！")
        print(f"📍 新位置: {robot.current_position}")
    else:
        print(f"\n❌ 执行失败")

    # ==================== 完整流程图 ====================
    print("\n\n" + "="*70)
    print("📊 完整流程总结")
    print("="*70)

    flow_diagram = '''
┌────────────────────────────────────────────────────┐
│  用户: "去厨房"                                     │
│  (自然语言)                                        │
└─────────────┬──────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  System Prompt                                      │
│  告诉 LLM: navigate 是什么，怎么用                  │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  LLM 推理                                           │
│  "用户要移动" → navigate 动作                       │
│  输出: {"type": "navigate", "target": "kitchen"}    │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  Pydantic Schema 验证                               │
│  ✓ type 是 "navigate"                              │
│  ✓ target 是字符串 "kitchen"                       │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  代码派发                                           │
│  if type == "navigate":                             │
│      robot.navigate(target)                         │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  机器人执行                                         │
│  1. 查找 "kitchen" 的坐标: [5.0, 2.0, 0.0]         │
│  2. 移动到目标位置                                  │
│  3. 更新当前位置                                    │
└─────────────┬───────────────────────────────────────┘
              ↓
         ✅ 完成！
    '''
    print(flow_diagram)


def demo_variations():
    """演示不同的 navigate 变体"""

    print("\n\n" + "="*70)
    print("🔬 navigate 的不同变体")
    print("="*70)

    robot = MockRobot(name="DEMO")

    variations = [
        {
            "name": "语义导航（推荐）",
            "json": {"type": "navigate", "target": "kitchen"},
            "description": "使用语义标签，LLM 容易理解"
        },
        {
            "name": "中文语义",
            "json": {"type": "navigate", "target": "厨房"},
            "description": "支持中文地点名"
        },
        {
            "name": "坐标导航",
            "json": {"type": "navigate", "target": [1.5, 2.3, 0.0]},
            "description": "直接使用坐标（精确但 LLM 难生成）"
        },
    ]

    for i, var in enumerate(variations, 1):
        print(f"\n变体 {i}: {var['name']}")
        print(f"  说明: {var['description']}")
        print(f"  JSON: {json.dumps(var['json'], ensure_ascii=False)}")

        try:
            action = NavigateAction(**var['json'])
            print(f"  ✅ Schema 验证通过")

            # 模拟执行（不真的移动）
            target = action.target
            if isinstance(target, str):
                if target in robot.known_locations:
                    coords = robot.known_locations[target]
                    print(f"  🎯 目标坐标: {coords}")
                else:
                    print(f"  ⚠️  未知位置: {target}")
            else:
                print(f"  🎯 直接坐标: {target}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")


def demo_invalid_cases():
    """演示非法输入的处理"""

    print("\n\n" + "="*70)
    print("❌ 非法输入示例（Schema 会拦截）")
    print("="*70)

    invalid_cases = [
        {
            "name": "缺少 target 参数",
            "json": {"type": "navigate"},
            "expected_error": "field required"
        },
        {
            "name": "错误的 type",
            "json": {"type": "fly", "target": "sky"},
            "expected_error": "Input should be 'navigate'"
        },
        {
            "name": "坐标不是3个",
            "json": {"type": "navigate", "target": [1.0, 2.0]},
            "expected_error": "坐标必须是 [x, y, z]"
        },
        {
            "name": "坐标包含字符串",
            "json": {"type": "navigate", "target": [1, 2, "three"]},
            "expected_error": "Input should be a valid number"
        },
    ]

    for i, case in enumerate(invalid_cases, 1):
        print(f"\n案例 {i}: {case['name']}")
        print(f"  输入: {json.dumps(case['json'], ensure_ascii=False)}")

        try:
            action = parse_action(case['json'])
            print(f"  ⚠️  意外：验证通过了（不应该发生）")
        except Exception as e:
            error_msg = str(e)
            print(f"  ✅ 正确拦截: {error_msg[:60]}...")


if __name__ == "__main__":
    demo_flow()

    print("\n\n" + "="*70)
    input("按 Enter 查看更多示例...")

    demo_variations()
    demo_invalid_cases()

    print("\n\n" + "="*70)
    print("✅ 演示完成！")
    print("\n💡 查看详细文档: docs/HOW_ROBOT_UNDERSTANDS.md")
    print("="*70)

```


## demo_retry.py

````py
#!/usr/bin/env python3
"""
演示 LLM 重试机制

模拟 LLM 第一次输出错误，第二次输出正确的过程
"""

import json
from brain.schemas import RobotDecision, parse_action


def simulate_llm_retry():
    """模拟 LLM 的重试过程"""

    print("="*70)
    print("🎬 模拟场景：用户说 '去桌子那里'")
    print("="*70)

    # ==================== 第1次尝试 ====================
    print("\n【第1次尝试】")
    print("─"*70)

    # LLM 第一次可能的错误输出
    response_1 = "好的，我这就去桌子那里。我会立即执行导航动作。"

    print(f"\n📄 LLM 原始输出:")
    print(f"   {response_1}")

    print(f"\n🔍 尝试解析 JSON...")

    try:
        # 尝试解析
        data = json.loads(response_1)
        print(f"   ✅ 解析成功")
    except json.JSONDecodeError as e:
        print(f"   ❌ 解析失败: {e}")
        print(f"   ⚠ 警告: 无法从文本中提取JSON")

        # 系统会把错误反馈给 LLM
        print(f"\n💬 系统反馈给 LLM:")
        print(f"   '你的输出格式有误，错误信息：{e}'")
        print(f"   '请严格按照JSON格式重新输出。'")

    # ==================== 第2次尝试 ====================
    print("\n\n【第2次尝试】")
    print("─"*70)

    # LLM 第二次的正确输出
    response_2 = '''```json
{
  "thought": "用户明确指示我去桌子那里。这是一个直接的导航指令。我应该执行导航动作。",
  "reply": "好的，我这就去桌子那里。",
  "action": {
    "type": "navigate",
    "target": "table"
  }
}
```'''

    print(f"\n📄 LLM 原始输出:")
    print(response_2)

    print(f"\n🔍 尝试解析 JSON...")

    try:
        # 提取 JSON（支持 markdown 代码块）
        if "```json" in response_2:
            start = response_2.find("```json") + 7
            end = response_2.find("```", start)
            json_str = response_2[start:end].strip()
        else:
            json_str = response_2

        # 解析 JSON
        data = json.loads(json_str)
        print(f"   ✅ JSON 解析成功！")

        # 打印解析结果
        print(f"\n📋 解析结果:")
        print(f"   thought: {data['thought'][:50]}...")
        print(f"   reply: {data['reply']}")
        print(f"   action.type: {data['action']['type']}")
        print(f"   action.target: {data['action']['target']}")

        # 验证并创建 Pydantic 模型
        action = parse_action(data['action'])
        decision = RobotDecision(**data)

        print(f"\n✅ Pydantic 验证通过！")

        # 模拟执行动作
        print(f"\n🤖 执行动作:")
        print(f"   类型: {action.type}")
        print(f"   目标: {action.target}")
        print(f"   → 调用 robot.navigate('table')")
        print(f"   ✅ 导航成功！")

    except Exception as e:
        print(f"   ❌ 失败: {e}")

    # ==================== 总结 ====================
    print("\n\n" + "="*70)
    print("📊 总结")
    print("="*70)
    print("""
流程：
  1️⃣  用户输入 → "去桌子那里"
  2️⃣  LLM 第1次尝试 → 输出纯文本（失败）
  3️⃣  系统检测到错误 → 反馈给 LLM
  4️⃣  LLM 第2次尝试 → 输出正确 JSON（成功）
  5️⃣  解析 JSON → 提取 action 字段
  6️⃣  调用对应函数 → robot.navigate('table')
  7️⃣  机器人执行动作 → 移动到桌子 ✅

关键点：
  • JSON 是大脑和躯体的"通信协议"
  • action 字段决定了调用哪个函数
  • 重试机制保证了即使第一次失败也能成功
  • 最终只要有一次成功，动作就会执行
    """)

    print("="*70)


def show_json_structure():
    """展示 JSON 结构的重要性"""

    print("\n\n" + "="*70)
    print("🧩 JSON 输出和机器人动作的映射关系")
    print("="*70)

    mappings = [
        {
            "json": {
                "action": {"type": "navigate", "target": "kitchen"}
            },
            "function": "robot.navigate('kitchen')",
            "result": "🚗 机器人移动到厨房"
        },
        {
            "json": {
                "action": {"type": "search", "object_name": "apple"}
            },
            "function": "robot.search('apple')",
            "result": "🔍 机器人搜索苹果"
        },
        {
            "json": {
                "action": {"type": "pick", "object_name": "bottle"}
            },
            "function": "robot.pick('bottle')",
            "result": "🤏 机器人抓取瓶子"
        },
        {
            "json": {
                "action": {"type": "speak", "content": "任务完成"}
            },
            "function": "robot.speak('任务完成')",
            "result": "💬 机器人说话：任务完成"
        },
    ]

    for i, mapping in enumerate(mappings, 1):
        print(f"\n示例 {i}:")
        print(f"  JSON: {json.dumps(mapping['json'], ensure_ascii=False)}")
        print(f"  ↓")
        print(f"  调用: {mapping['function']}")
        print(f"  ↓")
        print(f"  结果: {mapping['result']}")
        print()

    print("="*70)


if __name__ == "__main__":
    simulate_llm_retry()
    show_json_structure()

    print("\n💡 提示:")
    print("   查看详细文档: docs/LLM_RETRY_MECHANISM.md")
    print()

````


## docs/COMPACT_MODE_GUIDE.md

````md
# 精简模式使用指南

## ✅ 已完成的修改

### 1. Schema 修改
- ✅ `thought` 字段改为可选 (`Optional[str]`)
- ✅ LLM 可以选择不生成 thought

### 2. 新增 Prompt
- ✅ 创建 `COMPACT_PROMPT`
- ✅ 明确告诉 LLM："**不要输出 thought 字段**"
- ✅ 提供无 thought 的示例

### 3. 命令行支持
- ✅ 添加 `--mode compact` 参数
- ✅ 支持与 `--no-thought` 组合使用

---

## 🚀 使用方法

### 方式1：命令行（推荐）

```bash
# 激活环境
conda activate cade

# 精简模式演示
python main.py --demo --mode compact

# 精简模式交互
python main.py --mode compact

# 精简模式测试
python main.py --test --mode compact

# 精简模式 + 不显示（双保险）
python main.py --mode compact --no-thought
```

### 方式2：代码方式

```python
from robot_controller import RobotController

# 创建精简模式控制器
controller = RobotController(
    prompt_mode="compact",   # ← 使用精简 Prompt
    show_thought=False       # ← 即使有也不显示
)

# 使用
controller.process_input("去厨房")
```

---

## 📊 效果对比

### 标准模式（default）

**LLM 输出：**
```json
{
  "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
  "reply": "好的，我这就去厨房",
  "action": {
    "type": "navigate",
    "target": "kitchen"
  }
}
```

**终端显示：**
```
🧠 [大脑思考中...]

💭 思考过程: 用户要求我移动到厨房，这是一个明确的导航指令
💬 回复: 好的，我这就去厨房
⚡ 计划动作: navigate
```

---

### 精简模式（compact）

**LLM 输出（预期）：**
```json
{
  "reply": "好的",
  "action": {
    "type": "navigate",
    "target": "kitchen"
  }
}
```

**终端显示：**
```
🧠 [大脑思考中...]

💬 回复: 好的
⚡ 计划动作: navigate
```

---

## 🔍 验证方法

### 1. 查看 LLM 原始输出

在 `brain/llm_client.py` 中已添加调试输出：

```
📄 LLM 原始输出:
------------------------------------------------------------
{"reply": "好的", "action": {"type": "navigate", "target": "kitchen"}}
------------------------------------------------------------
```

如果看到这里**没有 thought 字段**，说明成功了！

### 2. 运行对比测试

```bash
# 完整对比（包括性能测试）
python compare_modes.py

# 只测试标准模式
python compare_modes.py default

# 只测试精简模式
python compare_modes.py compact
```

### 3. 观察关键指标

```
📊 结果分析:
   thought 字段: ❌ 无        ← 这里应该显示"无"
   耗时: 1.23s
```

---

## 📈 预期优化效果

### Token 节省

假设一次对话：
- 标准模式 thought: ~50 字符（中文）
- 精简模式: 0 字符

**节省比例：** ~15-25%（取决于 thought 长度）

### 推理时间

理论上：
- ✅ 减少生成的 token 数量
- ✅ 减少推理计算量
- ⚠️ 实际效果取决于 LLM 的优化

**预期：** 5-15% 的时间节省

### 注意事项

⚠️ **LLM 可能仍然生成 thought**

即使 Prompt 中明确说"不要输出 thought"，某些模型可能：
1. 忽略指令
2. 仍然生成但不输出
3. 内部思考但省略输出

这取决于：
- 模型的训练方式
- 模型对指令的遵循程度
- DeepSeek 的具体实现

---

## 🧪 测试建议

### 测试1：基础功能

```bash
python main.py --mode compact

# 测试输入：
你好
去厨房
找到苹果
回到起点
```

**检查：** 所有功能是否正常

### 测试2：JSON 格式

运行后查看 `📄 LLM 原始输出` 部分

**期望：**
```json
{
  "reply": "...",
  "action": {...}
}
```

**没有 thought 字段！**

### 测试3：性能对比

```bash
time python main.py --demo --mode default
time python main.py --demo --mode compact
```

对比两次的执行时间。

### 测试4：准确率

运行相同的测试用例，对比：
- 标准模式的成功率
- 精简模式的成功率

**期望：** 成功率相同或接近

---

## 🎯 所有可用模式

| 模式 | Prompt 类型 | 包含 thought | 适用场景 |
|------|------------|-------------|----------|
| `default` | 标准 | ✅ 必须 | 开发、调试 |
| `simple` | 简化 | ✅ 必须 | 快速测试 |
| `compact` | 精简 | ❌ 禁止 | 生产、性能优化 ⭐ |
| `debug` | 调试 | ✅ 详细 | 深度调试 |

---

## 💡 使用建议

### 开发阶段
```bash
python main.py --mode default
```
保留完整思考过程，便于调试。

### 性能测试
```bash
python main.py --mode compact
python compare_modes.py
```
对比优化效果。

### 生产部署
```bash
python main.py --mode compact --no-thought
```
最精简的输出，最快的速度。

---

## 🐛 可能遇到的问题

### 问题1：LLM 仍然输出 thought

**现象：**
```
📊 结果分析:
   thought 字段: ✅ 有
```

**原因：**
- DeepSeek 可能忽略了 Prompt 中的"不要输出 thought"指令
- 模型训练时强制要求 CoT（思维链）

**解决：**
1. 尝试更强调 Prompt
2. 尝试其他模型（如 Qwen）
3. 使用方案1（只隐藏显示）

### 问题2：解析失败

**现象：**
```
⚠ 解析失败: field required
```

**原因：**
- LLM 理解错了，输出了错误的 JSON
- 网络波动

**解决：**
- 自动重试机制会处理
- 如果频繁失败，回退到 default 模式

### 问题3：性能没有明显提升

**原因：**
- DeepSeek 可能有内部优化
- 网络延迟占主要时间
- thought 字段本身不大

**不要担心：**
- 仍然节省了 Token（成本）
- 输出更简洁（体验）
- 生产环境更合适

---

## ✅ 快速开始

```bash
# 1. 激活环境
conda activate cade

# 2. 测试精简模式
python main.py --mode compact

# 3. 输入测试指令
去厨房
找到苹果
回到起点

# 4. 观察输出
# - 是否看到 "💭 思考过程"？（不应该看到）
# - 动作是否正常执行？（应该正常）

# 5. 运行对比测试
python compare_modes.py
```

---

## 📚 相关文档

- `docs/NO_THOUGHT_MODE.md` - 方案对比
- `brain/prompts.py` - Prompt 定义
- `brain/schemas.py` - Schema 定义
- `compare_modes.py` - 对比测试脚本

---

## 🎉 总结

**已实现：**
- ✅ Schema 支持可选 thought
- ✅ 精简版 Prompt
- ✅ 命令行参数支持
- ✅ 完整的显示逻辑
- ✅ 对比测试工具

**立即可用：**
```bash
python main.py --mode compact
```

**下一步：**
测试 DeepSeek 是否真的不生成 thought，以及性能提升如何！

````


## docs/DEPLOYMENT_QWEN3.md

````md
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

````


## docs/HOW_ROBOT_UNDERSTANDS.md

````md
# 机器人如何"理解"指令？

## 以 navigate 为例的完整流程

### 🎯 核心问题

用户说："去厨房"
机器人怎么知道要执行 `robot.navigate("kitchen")` ？

---

## 📖 完整流程拆解

### 阶段1：用户输入（自然语言）

```
用户: "去厨房"
```

这只是普通的人类语言，机器人不能直接理解。

---

### 阶段2：System Prompt 的作用（关键！）

在 `brain/prompts.py` 中，我们给 LLM 提供了一个"说明书"：

```python
ROBOT_SYSTEM_PROMPT = """
你是 LARA，一个智能服务机器人。

## 核心能力

你拥有以下物理动作能力：

1. **navigate** - 导航到指定位置
   - 参数: target (语义位置如"kitchen"或坐标[x,y,z])
   - 示例: {"type": "navigate", "target": "kitchen"}

2. **search** - 搜索物体
   ...

## 输出格式

你必须严格按照以下JSON格式输出：

{
  "thought": "你的思考过程",
  "reply": "给用户的回复",
  "action": {
    "type": "navigate",  ← 这里指定动作类型
    "target": "kitchen"  ← 这里指定参数
  }
}

## 示例

用户："去厨房"
输出：
{
  "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
  "reply": "好的，我这就去厨房",
  "action": {"type": "navigate", "target": "kitchen"}
}
"""
```

**关键点：**
- ✅ 告诉 LLM 有哪些"动作菜单"（navigate, search, pick...）
- ✅ 告诉 LLM 每个动作需要什么参数
- ✅ 给出大量示例，教 LLM 如何把自然语言转换为 JSON

---

### 阶段3：LLM 的推理过程

LLM 看到用户说 "去厨房"，会这样思考：

```
LLM 的内心独白：

1. "用户说'去厨房'"
2. "这明显是一个位置移动的需求"
3. "查看我的动作列表... 有 navigate 可以用！"
4. "navigate 需要一个 target 参数"
5. "用户说的'厨房'就是 target"
6. "我应该输出：
   {
     "action": {
       "type": "navigate",
       "target": "厨房"  或 "kitchen"
     }
   }
```

输出结果：
```json
{
  "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
  "reply": "好的，我这就去厨房",
  "action": {
    "type": "navigate",
    "target": "kitchen"
  }
}
```

---

### 阶段4：Pydantic Schema 的验证

在 `brain/schemas.py` 中定义的 `NavigateAction`：

```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"  # ← 强制 type 必须是 "navigate"
    target: Union[str, List[float]]         # ← target 可以是字符串或坐标数组

    @field_validator('target')
    def validate_target(cls, v):
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError("坐标必须是 [x, y, z] 格式")
        return v
```

**作用：**
```python
# ✅ 合法输入
NavigateAction(type="navigate", target="kitchen")
NavigateAction(type="navigate", target=[1.0, 2.0, 0.0])

# ❌ 非法输入 - 会抛出错误
NavigateAction(type="fly", target="sky")       # type 错误
NavigateAction(type="navigate", target=[1, 2]) # 坐标不是3个
NavigateAction(type="navigate")                # 缺少 target
```

这是**防御性编程**：
- 即使 LLM 输出错了，也能立即捕获
- 不会让错误的指令传递到机器人

---

### 阶段5：动作派发（Dispatch）

在 `robot_controller.py` 中：

```python
def _execute_action(self, action: RobotAction) -> bool:
    action_type = action.type  # 读取 "navigate"

    if action_type == "navigate":
        return self.robot.navigate(action.target)  # ← 调用真实函数
                                   ↑
                        把 "kitchen" 传进去

    elif action_type == "search":
        return self.robot.search(action.object_name)

    elif action_type == "pick":
        return self.robot.pick(action.object_name)
    # ...
```

**这就是"理解"的本质：**
```
JSON 中的 "type": "navigate"
    ↓
通过 if-elif 链条匹配
    ↓
调用对应的 Python 函数 robot.navigate()
```

---

### 阶段6：机器人执行

在 `body/mock_robot.py` 中：

```python
def navigate(self, target) -> bool:
    """模拟导航"""

    # target 就是 LLM 传过来的 "kitchen"

    if isinstance(target, str):
        if target in self.known_locations:  # 检查是否认识这个地方
            coords = self.known_locations[target]  # [5.0, 2.0, 0.0]

            # 模拟移动（真实机器人会调用 ROS 导航）
            self.current_position = target
            print(f"✓ 已到达 {target} (坐标: {coords})")
            return True
        else:
            print(f"✗ 未知位置: {target}")
            return False
```

**最终效果：**
```
🚗 [导航] 从 home 前往 kitchen
✓ 已到达 kitchen (坐标: [5.0, 2.0, 0.0])
```

---

## 🧠 为什么"简单的 Schema"就够用？

### 1. 你看到的"简单"，其实不简单

```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]
```

这短短3行代码，实际做了：

✅ **类型约束**
- `type` 必须是字符串 "navigate"
- `target` 可以是字符串或3个浮点数的列表

✅ **自动验证**
- Pydantic 自动检查类型
- 自动转换兼容类型（如 `1` → `1.0`）

✅ **错误提示**
- 如果格式错误，立即抛出详细的错误信息

✅ **文档生成**
- 可以自动生成 JSON Schema
- 可以自动生成 API 文档

### 2. LLM 负责"智能"，Schema 负责"安全"

**分工明确：**

| 模块 | 职责 | 举例 |
|------|------|------|
| **System Prompt** | 教 LLM 如何把自然语言转为 JSON | "去厨房" → `{"type": "navigate", "target": "kitchen"}` |
| **LLM** | 理解用户意图，生成指令 | 判断用户是要移动、抓取还是搜索 |
| **Pydantic Schema** | 验证格式，防止错误 | 确保 target 不是 `null` 或奇怪的值 |
| **Robot 函数** | 真实执行动作 | 调用硬件、ROS、传感器 |

### 3. Schema 是"合同"，不是"智能"

Schema 的作用类似于：
```
你：去厨房
合同：确保"厨房"是一个有效的地点名称
执行者：按照合同去厨房
```

不需要 Schema "理解"什么是厨房，只需要：
- ✅ 检查格式对不对
- ✅ 检查参数齐不齐
- ✅ 检查类型符不符合

---

## 🔍 深入 navigate 的两种模式

### 模式1：语义导航（推荐）

```json
{
  "action": {
    "type": "navigate",
    "target": "kitchen"  ← 语义标签
  }
}
```

**优点：**
- ✅ LLM 容易理解（厨房、客厅、卧室）
- ✅ 不需要精确坐标
- ✅ 可以动态调整（厨房的位置改了，只需更新地图）

**机器人的处理：**
```python
if target == "kitchen":
    coords = self.known_locations["kitchen"]  # [5.0, 2.0, 0.0]
    # 调用底层导航系统
    self.ros_nav.navigate_to(coords)
```

### 模式2：坐标导航

```json
{
  "action": {
    "type": "navigate",
    "target": [1.5, 2.3, 0.0]  ← 直接坐标
  }
}
```

**缺点：**
- ❌ LLM 很难凭空生成准确的数字
- ❌ 用户不知道坐标

**适用场景：**
- 从视觉系统获得物体位置
- 从地图上计算出的中间点

---

## 🚀 如果需要更复杂的功能？

### 示例：导航时避开障碍物

#### 当前 Schema（简单）
```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]
```

#### 扩展后的 Schema（复杂）
```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]

    # 新增参数
    speed: Optional[float] = 1.0          # 移动速度
    avoid_obstacles: bool = True          # 是否避障
    max_retries: int = 3                  # 最大重试次数
    path_planning: Literal["shortest", "safest"] = "shortest"
```

#### 更新 System Prompt
```python
1. **navigate** - 导航到指定位置
   - 参数:
     - target (必需): 目标位置
     - speed (可选): 移动速度，默认1.0
     - avoid_obstacles (可选): 是否避障，默认true
   - 示例:
     {"type": "navigate", "target": "kitchen", "speed": 0.5}
```

#### 更新机器人函数
```python
def navigate(self, target, speed=1.0, avoid_obstacles=True, **kwargs):
    # 根据参数调整导航策略
    if avoid_obstacles:
        self.enable_obstacle_avoidance()

    self.set_speed(speed)
    # ...
```

---

## 💡 关键洞察

### 机器人"理解"的本质

机器人并不是真的"理解"你说的话，而是：

1. **LLM 做翻译**
   - 自然语言 → 结构化指令（JSON）

2. **Schema 做校验**
   - 确保指令格式正确

3. **代码做映射**
   - JSON 的 `type` 字段 → 对应的函数名

4. **函数做执行**
   - 调用真实的硬件/算法

### 类比：餐厅点餐

```
你：我要一份宫保鸡丁        ← 用户输入（自然语言）
    ↓
服务员：好的，点了一份      ← LLM（理解意图）
         宫保鸡丁（辣度中等）
    ↓
订单系统：                  ← Schema（验证格式）
  {
    "dish": "宫保鸡丁",
    "spicy_level": "medium",
    "quantity": 1
  }
    ↓
厨房：收到订单，开始做菜     ← Robot 函数（执行）
```

厨房不需要"理解"你为什么想吃宫保鸡丁，只需要：
- ✅ 知道要做什么菜
- ✅ 知道用什么食材
- ✅ 按照菜谱执行

---

## 📚 总结

| 问题 | 答案 |
|------|------|
| 机器人怎么理解指令？ | LLM 把自然语言翻译成 JSON，代码根据 JSON 调用函数 |
| Schema 为什么这么简单？ | 因为"智能"在 LLM，Schema 只负责格式验证 |
| navigate 怎么工作的？ | `{"type": "navigate", "target": "kitchen"}` → `if type == "navigate": robot.navigate(target)` |
| 能扩展更复杂的功能吗？ | 可以！在 Schema 中添加参数，更新 Prompt 和函数 |

**核心思想：**
> 分层设计 = LLM（理解） + Schema（验证） + 代码（执行）

每一层只做自己擅长的事，整体就能处理复杂任务！

````


## docs/LLM_RETRY_MECHANISM.md

````md
# LLM 重试机制详解

## ❓ 你的问题

为什么看到"解析失败"的警告，但最后还是执行成功了？

## 📖 答案

这是 **自动重试机制** 在工作！

## 🔄 完整流程

### 场景：用户说 "去桌子那里"

#### 第1次尝试（失败）

**LLM 可能的输出：**
```
好的，我这就去桌子那里。我会立即执行导航动作。
```

**问题：** 这不是 JSON 格式！

**系统反应：**
```
⚠ 解析失败 (尝试 1/3): 无法从文本中提取JSON
```

系统会自动把这个错误信息反馈给 LLM：
```
你的输出格式有误，错误信息：无法从文本中提取JSON
请严格按照JSON格式重新输出。
```

---

#### 第2次尝试（成功）

**LLM 的输出：**
```json
{
  "thought": "用户明确指示我去桌子那里。这是一个直接的导航指令。我应该执行导航动作。",
  "reply": "好的，我这就去桌子那里。",
  "action": {
    "type": "navigate",
    "target": "table"
  }
}
```

**系统反应：**
```
✅ 解析成功！
💭 思考过程: 用户明确指示我去桌子那里...
💬 回复: 好的，我这就去桌子那里。
⚡ 计划动作: navigate
```

执行动作 ✅

---

## 🎯 核心代码逻辑

在 `brain/llm_client.py` 的 `get_decision()` 方法中：

```python
for attempt in range(max_retries):  # 最多重试 3 次
    try:
        # 1. 调用 LLM
        response = self.chat(messages)

        # 2. 尝试解析 JSON
        decision_dict = self._extract_json(response)

        # 3. 验证格式
        decision = RobotDecision(**decision_dict)

        # ✅ 成功！返回结果
        return decision

    except Exception as e:
        # ❌ 失败！打印警告
        print(f"⚠ 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")

        # 把错误反馈给 LLM，让它重新生成
        messages.append({
            "role": "assistant",
            "content": response
        })
        messages.append({
            "role": "user",
            "content": f"你的输出格式有误，错误信息：{e}\n请严格按照JSON格式重新输出。"
        })
        # 继续下一次循环...
```

---

## 🧩 JSON 输出和机器人动作的关系

### 数据流图

```
用户输入 "去桌子那里"
    ↓
┌─────────────────────┐
│  LLM 大脑思考       │
│  (System Prompt)    │
└─────────────────────┘
    ↓
生成 JSON 决策
{
  "thought": "...",    ← 📝 LLM 的思考过程（供调试）
  "reply": "...",      ← 💬 说给用户听的话
  "action": {          ← ⚡ 关键！机器人要执行的动作
    "type": "navigate",
    "target": "table"
  }
}
    ↓
解析 action 字段
    ↓
┌─────────────────────┐
│  根据 type 调用     │
│  对应的函数：       │
│  robot.navigate()   │
└─────────────────────┘
    ↓
🤖 机器人移动到桌子
```

### 关键点

1. **JSON 是协议**
   - LLM（大脑）和 Robot（躯体）之间的"通信协议"
   - 就像人的"大脑发出神经信号"，必须是特定格式

2. **action 字段是核心**
   - 决定了调用哪个函数：`navigate`, `pick`, `search`...
   - 包含了函数参数：去哪里？拿什么？

3. **Pydantic 保证安全**
   - 严格校验 JSON 结构
   - 防止 LLM 输出奇怪的格式导致程序崩溃

---

## 🐛 为什么会解析失败？

### 常见原因

1. **LLM 忘记了 JSON 格式**
   ```
   # ❌ 错误输出
   好的，我去桌子！
   ```

2. **JSON 语法错误**
   ```json
   # ❌ 缺少引号
   {
     thought: "...",
     action: {...}
   }
   ```

3. **使用了未定义的动作类型**
   ```json
   # ❌ 没有 "fly" 这个动作
   {
     "thought": "...",
     "action": {"type": "fly", "target": "sky"}
   }
   ```

4. **参数格式错误**
   ```json
   # ❌ navigate 需要 target 参数
   {
     "thought": "...",
     "action": {"type": "navigate"}
   }
   ```

---

## 🛠️ 如何减少解析失败？

### 方法 1：优化 System Prompt

在 `brain/prompts.py` 中：
- ✅ 给出**大量正确示例**
- ✅ 强调"必须严格按照 JSON 格式"
- ✅ 明确列出所有可用动作

### 方法 2：增加重试次数

在调用时：
```python
decision = llm_client.get_decision(
    user_input=user_input,
    system_prompt=system_prompt,
    max_retries=5  # 从 3 次改为 5 次
)
```

### 方法 3：使用更强大的模型

```python
# config_local.py
CLOUD_MODEL = "deepseek-chat"  # 当前
# 或
CLOUD_MODEL = "gpt-4"  # 更准确，但更贵
```

### 方法 4：添加 JSON Schema

使用 OpenAI 的 Function Calling：
```python
# 未来改进：使用结构化输出
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    response_format={"type": "json_object"}  # 强制 JSON 输出
)
```

---

## 📊 统计数据建议

你可以在 `RobotController` 中添加失败统计：

```python
class RobotController:
    def __init__(self):
        # ...
        self.parse_failures = 0  # 记录解析失败次数
        self.parse_retries = 0   # 记录重试次数
```

然后在 `print_statistics()` 中显示：
```
📊 统计信息:
   总交互次数: 10
   解析失败次数: 2
   平均重试次数: 1.2
   成功率: 80%
```

---

## ✅ 结论

**"解析失败但最终成功"是正常现象！**

- 🎯 这是**容错机制**，不是 bug
- 🔄 系统会自动重试，把错误反馈给 LLM
- 🧠 LLM 从错误中学习，第二次通常会成功
- ⚡ 只要最终成功，动作就会正常执行

这就像人说话一样：
- 第一次表达不清楚 → 对方听不懂
- 重新组织语言 → 对方理解了
- 最终完成沟通 ✅

---

## 🔧 调试工具

如果你想看 LLM 的原始输出，可以：

1. 运行修改后的代码（已添加调试输出）
2. 查看 `📄 LLM 原始输出` 部分
3. 对比第1次（失败）和第2次（成功）的输出

示例：
```bash
python main.py --mode debug  # 使用调试模式的 prompt
```

````


## docs/NO_THOUGHT_MODE.md

````md
# 无思考模式配置指南

## 🎯 目标

测试 DeepSeek 在"不显示思考过程"时的性能表现。

---

## 📋 三种方案对比

| 方案 | 改动大小 | LLM 是否生成 thought | 适用场景 |
|------|---------|---------------------|----------|
| **方案1** ⭐ | 最小 | ✅ 生成，但不显示 | 生产环境、演示 |
| **方案2** | 中等 | ❌ 可选生成 | 节省 Token、测试 |
| **方案3** | 较大 | ❌ 完全不生成 | 极致性能优化 |

---

## ✅ 方案1：只隐藏显示（已实现）

### 使用方法

```bash
# 命令行方式
python main.py --no-thought
python main.py --demo --no-thought
python main.py --test --no-thought

# 代码方式
controller = RobotController(show_thought=False)
```

### 效果对比

**标准模式：**
```
============================================================
👤 用户: 去厨房
============================================================

🧠 [大脑思考中...]

💭 思考过程: 用户要求我移动到厨房，这是一个明确的导航指令  ← 显示
💬 回复: 好的，我这就去厨房
⚡ 计划动作: navigate

🚗 [导航] 从 home 前往 kitchen
✓ 已到达 kitchen
```

**无思考模式：**
```
============================================================
👤 用户: 去厨房
============================================================

🧠 [大脑思考中...]

💬 回复: 好的，我这就去厨房  ← 思考过程不显示
⚡ 计划动作: navigate

🚗 [导航] 从 home 前往 kitchen
✓ 已到达 kitchen
```

### 优点

- ✅ **改动最小** - 只修改了显示逻辑
- ✅ **完全兼容** - 不影响任何功能
- ✅ **立即可用** - 无需修改 Prompt 或 Schema
- ✅ **可随时切换** - 通过参数控制

### 缺点

- ❌ LLM 仍然会生成 thought 字段
- ❌ 不节省 API Token
- ❌ 不减少 LLM 推理时间

---

## 🔧 方案2：让 thought 变为可选

### 修改步骤

#### 1. 修改 Schema

```python
# brain/schemas.py
class RobotDecision(BaseModel):
    thought: Optional[str] = None  # ← 改为可选
    reply: Optional[str] = None
    action: Optional[RobotAction] = None
```

#### 2. 修改 Prompt

```python
# brain/prompts.py

# 添加一个"精简模式"提示词
COMPACT_PROMPT = """
你是 LARA，一个智能服务机器人。

## 输出格式（精简）

你必须输出 JSON，但可以省略 thought 字段：

{
  "reply": "给用户的回复（可选）",
  "action": {"type": "动作类型", "参数": "值"}
}

示例：
用户："去厨房"
输出：
{
  "reply": "好的",
  "action": {"type": "navigate", "target": "kitchen"}
}
"""

# 在 get_system_prompt 中添加
def get_system_prompt(mode: str = "default") -> str:
    prompts = {
        "default": ROBOT_SYSTEM_PROMPT,
        "simple": SIMPLE_PROMPT,
        "debug": DEBUG_PROMPT,
        "compact": COMPACT_PROMPT,  # ← 新增
    }
    return prompts.get(mode, ROBOT_SYSTEM_PROMPT)
```

#### 3. 使用方法

```bash
python main.py --mode compact
```

或代码方式：

```python
controller = RobotController(
    prompt_mode="compact",
    show_thought=False  # 双保险
)
```

### 优点

- ✅ 可能节省 Token（LLM 可以选择不生成 thought）
- ✅ 输出更简洁
- ✅ 保持灵活性（可以选择生成或不生成）

### 缺点

- ⚠️ LLM 可能仍然会生成 thought（取决于模型训练）
- ⚠️ 调试困难（看不到思考过程）

---

## 🚀 方案3：强制不生成 thought

### 修改步骤

#### 1. 创建新的 Schema

```python
# brain/schemas.py

class CompactRobotDecision(BaseModel):
    """精简版决策（无思考过程）"""
    reply: Optional[str] = None
    action: Optional[RobotAction] = None
    # 完全移除 thought 字段
```

#### 2. 修改 Prompt（强制要求）

```python
COMPACT_PROMPT = """
你是 LARA。

重要：你的输出中不能包含 thought 字段，只能包含 reply 和 action。

输出格式：
{
  "reply": "...",
  "action": {...}
}

禁止输出：
{
  "thought": "...",  ← 不允许
  "reply": "...",
  "action": {...}
}
"""
```

#### 3. 修改 LLMClient

```python
# brain/llm_client.py

def get_decision(self, ..., compact_mode=False):
    if compact_mode:
        # 使用 CompactRobotDecision
        decision = CompactRobotDecision(**decision_dict)
    else:
        decision = RobotDecision(**decision_dict)
    return decision
```

### 优点

- ✅ 真正节省 Token
- ✅ 减少 LLM 推理时间
- ✅ 输出最简洁

### 缺点

- ❌ **改动较大** - 需要修改多个文件
- ❌ **调试困难** - 完全看不到思考过程
- ❌ **维护成本** - 需要维护两套 Schema

---

## 📊 性能对比测试

### 测试脚本

```bash
# 方案1：只隐藏显示
time python main.py --demo --no-thought

# 方案2/3：完全不生成（需先实现）
time python main.py --demo --mode compact
```

### 预期结果

| 指标 | 标准模式 | 方案1 | 方案2/3 |
|------|---------|-------|---------|
| Token 消耗 | 100% | 100% | ~70% |
| 推理时间 | 100% | 100% | ~80% |
| 显示速度 | 100% | 110% | 110% |
| 可调试性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 💡 推荐方案

### 开发阶段
使用**标准模式**，保留完整的思考过程：
```bash
python main.py  # 默认显示 thought
```

### 演示/测试阶段
使用**方案1**，隐藏思考过程：
```bash
python main.py --no-thought
```

### 生产部署
根据需求选择：
- 需要调试 → 方案1（可随时开启 thought）
- 极致性能 → 方案2/3（需要先实现）

---

## 🔍 DeepSeek 特性测试

DeepSeek 可能有特殊优化，建议测试：

### 1. 测试是否支持 JSON Mode

```python
# 测试代码
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={"type": "json_object"}  # ← DeepSeek 可能支持
)
```

如果支持，可以强制输出 JSON，减少解析失败。

### 2. 测试 thought 对性能的影响

```python
# 分别测试：
# A. 包含 thought 的 Prompt
# B. 不包含 thought 的 Prompt

# 对比：
# - API 返回时间
# - Token 消耗
# - 准确率
```

---

## 📝 实施建议

**立即可用（推荐）：**
```bash
# 方案1 已经实现，直接使用：
python main.py --no-thought
```

**如需真正节省 Token：**
1. 先测试 DeepSeek 是否支持 JSON Mode
2. 实现方案2（thought 可选）
3. 对比性能数据
4. 根据结果决定是否需要方案3

**测试脚本：**
```bash
# 对比测试
python test_no_thought.py

# 演示模式（无思考）
python main.py --demo --no-thought

# 交互模式（无思考）
python main.py --no-thought
```

---

## ✅ 总结

| 需求 | 推荐方案 | 命令 |
|------|---------|------|
| 快速测试 | 方案1 | `python main.py --no-thought` |
| 演示展示 | 方案1 | `python main.py --demo --no-thought` |
| 节省 Token | 方案2/3 | 需要先实现 |
| 开发调试 | 标准模式 | `python main.py` |

**现在就可以用方案1测试！** ⭐

````


## main.py

```py
#!/usr/bin/env python3
"""
CADE - 具身智能机器人主程序

运行模式：
1. 交互模式: python main.py
2. 测试模式: python main.py --test
3. 演示模式: python main.py --demo
"""

import sys
import argparse
from robot_controller import RobotController
from config import Config


def print_banner():
    """打印欢迎信息"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 CADE - 具身智能机器人系统                            ║
║   Cognitive Agent for Domestic Environment                ║
║                                                           ║
║   项目: Project LARA                                      ║
║   版本: 0.1.0                                             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"运行模式: {'☁️  云端' if Config.is_cloud_mode() else '💻 本地'}")
    print(f"模型: {Config.get_llm_config()['model']}")
    print(f"机器人: {Config.ROBOT_NAME}")
    print(f"Mock模式: {'✓' if Config.ENABLE_MOCK else '✗'}")
    print()


def interactive_mode(args):
    """交互模式"""
    print_banner()
    controller = RobotController(
        prompt_mode=args.mode,
        show_thought=not args.no_thought
    )
    controller.interactive_mode()


def test_mode(args):
    """测试模式"""
    print_banner()
    print("🧪 运行测试场景\n")

    controller = RobotController(
        prompt_mode=args.mode,
        show_thought=not args.no_thought
    )

    # 测试用例
    test_cases = [
        # 1. 闲聊测试
        "你好呀",
        "你叫什么名字？",
        "你能做什么？",

        # 2. 简单导航
        "去厨房",
        "回到起点",

        # 3. 搜索任务
        "帮我找苹果",
        "找到水杯",

        # 4. 抓取任务
        "拿起苹果",

        # 5. 复合任务
        "把苹果放到桌子上",

        # 6. 边界情况
        "帮我订个外卖",  # 无法完成的任务
    ]

    controller.run_test_scenario(test_cases)


def demo_mode(args):
    """演示模式 - 展示一个完整的服务流程"""
    print_banner()
    print("🎬 演示模式：展示完整服务流程\n")

    controller = RobotController(
        prompt_mode=args.mode,
        show_thought=not args.no_thought
    )

    demo_scenario = [
        "你好",
        "我想要桌子上的苹果",
        "去桌子那里",
        "找到苹果",
        "拿起苹果",
        "回到起点",
        "谢谢你",
    ]

    print("📋 演示场景：用户请求获取桌子上的苹果\n")
    controller.run_test_scenario(demo_scenario)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CADE 具身智能机器人系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py              # 交互模式
  python main.py --test       # 测试模式
  python main.py --demo       # 演示模式
  python main.py --mode debug # 使用调试提示词

提示:
  - 首次运行请先配置 config.py 中的 API 密钥
  - 交互模式下输入 'quit' 退出
  - 输入 'status' 查看机器人状态
  - 输入 'stats' 查看统计信息
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='运行测试模式'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行演示模式'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'simple', 'compact', 'debug'],
        default='default',
        help='提示词模式（default=标准, compact=精简无思考, debug=调试）'
    )

    parser.add_argument(
        '--no-thought',
        action='store_true',
        help='不显示 LLM 的思考过程（测试纯执行性能）'
    )

    args = parser.parse_args()

    try:
        if args.test:
            test_mode(args)
        elif args.demo:
            demo_mode(args)
        else:
            interactive_mode(args)

    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

```


## requirements.txt

```txt
# LLM 相关
openai>=1.0.0
pydantic>=2.0.0

# 异步支持
aiohttp>=3.9.0

# 日志和调试
loguru>=0.7.0

# 未来本地部署需要（可选）
# ollama-python>=0.1.0

# 状态机（可选）
# transitions>=0.9.0

```


## robot_controller.py

```py
"""
Robot Controller - 机器人控制器

整合大脑(LLM)和躯体(Robot)，实现完整的感知-决策-执行循环
"""

from typing import List, Dict, Optional
from config import Config
from brain.llm_client import LLMClient
from brain.prompts import get_system_prompt
from brain.schemas import RobotDecision, RobotAction
from body.mock_robot import MockRobot
from body.robot_interface import RobotInterface, RobotState


class RobotController:
    """
    机器人主控制器

    负责：
    1. 接收用户输入
    2. 调用LLM进行决策
    3. 执行动作
    4. 管理对话历史
    """

    def __init__(
        self,
        robot: Optional[RobotInterface] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_mode: str = "default",
        show_thought: bool = True  # ← 新增参数
    ):
        """
        初始化控制器

        Args:
            robot: 机器人实例（如果为None则创建MockRobot）
            llm_client: LLM客户端（如果为None则创建默认客户端）
            prompt_mode: 提示词模式（default/simple/debug）
            show_thought: 是否显示 LLM 的思考过程（默认 True）
        """
        # 初始化机器人
        self.robot = robot or MockRobot(name=Config.ROBOT_NAME)

        # 初始化LLM客户端
        self.llm_client = llm_client or LLMClient()

        # 系统提示词
        self.system_prompt = get_system_prompt(prompt_mode)

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

        # 显示选项
        self.show_thought = show_thought  # ← 保存参数

        # 统计信息
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0

        print(f"\n{'='*60}")
        print(f"🚀 机器人控制器初始化成功")
        print(f"{'='*60}")

    def process_input(self, user_input: str) -> RobotDecision:
        """
        处理用户输入（完整流程）

        Args:
            user_input: 用户输入文本

        Returns:
            RobotDecision: 决策对象
        """
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*60}")

        self.total_interactions += 1

        # 1. 让LLM思考
        self.robot.set_state(RobotState.THINKING)
        print(f"\n🧠 [大脑思考中...]")

        try:
            decision = self.llm_client.get_decision(
                user_input=user_input,
                system_prompt=self.system_prompt,
                conversation_history=self.conversation_history
            )

            # 打印决策
            if self.show_thought and decision.thought:  # ← 检查是否存在且需要显示
                print(f"\n💭 思考过程: {decision.thought}")
            if decision.reply:
                print(f"💬 回复: {decision.reply}")
            if decision.action:
                print(f"⚡ 计划动作: {decision.action.type}")

            # 2. 执行动作
            if decision.action:
                success = self._execute_action(decision.action)
                if success:
                    self.successful_actions += 1
                else:
                    self.failed_actions += 1

            # 3. 更新对话历史
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })

            # 构建助手回复（包含思考、回复和动作）
            assistant_response_parts = []
            if decision.thought:  # ← 只在有 thought 时添加
                assistant_response_parts.append(f"思考: {decision.thought}")
            if decision.reply:
                assistant_response_parts.append(f"回复: {decision.reply}")
            if decision.action:
                assistant_response_parts.append(f"动作: {decision.action.model_dump_json()}")

            assistant_response = "\n".join(assistant_response_parts)

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })

            return decision

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            self.robot.set_state(RobotState.ERROR)
            raise

    def _execute_action(self, action: RobotAction) -> bool:
        """
        执行具体动作

        Args:
            action: 动作对象

        Returns:
            bool: 是否成功
        """
        action_type = action.type

        try:
            if action_type == "navigate":
                return self.robot.navigate(action.target)

            elif action_type == "search":
                result = self.robot.search(action.object_name)
                return result is not None

            elif action_type == "pick":
                return self.robot.pick(action.object_name, action.object_id)

            elif action_type == "place":
                return self.robot.place(action.location)

            elif action_type == "speak":
                return self.robot.speak(action.content)

            elif action_type == "wait":
                return self.robot.wait(action.reason)

            else:
                print(f"⚠ 未知动作类型: {action_type}")
                return False

        except Exception as e:
            print(f"❌ 动作执行失败: {e}")
            return False

    def interactive_mode(self):
        """
        交互模式（命令行对话）
        """
        print(f"\n{'='*60}")
        print(f"🤖 进入交互模式")
        print(f"提示：输入 'quit' 或 'exit' 退出")
        print(f"{'='*60}\n")

        while True:
            try:
                user_input = input("👤 你: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再见！")
                    self.print_statistics()
                    break

                if user_input.lower() == 'status':
                    self.robot.print_status()
                    continue

                if user_input.lower() == 'stats':
                    self.print_statistics()
                    continue

                # 处理输入
                self.process_input(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                self.print_statistics()
                break

            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()

    def run_test_scenario(self, scenarios: List[str]):
        """
        运行测试场景

        Args:
            scenarios: 测试用例列表
        """
        print(f"\n{'='*60}")
        print(f"🧪 开始测试场景 (共 {len(scenarios)} 个)")
        print(f"{'='*60}")

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n\n{'─'*60}")
            print(f"测试 {i}/{len(scenarios)}")
            print(f"{'─'*60}")

            try:
                self.process_input(scenario)
                # 等待一下，模拟真实交互
                import time
                time.sleep(1)

            except Exception as e:
                print(f"❌ 测试失败: {e}")

        print(f"\n\n{'='*60}")
        print(f"🏁 测试完成")
        print(f"{'='*60}")
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        print(f"\n📊 统计信息:")
        print(f"   总交互次数: {self.total_interactions}")
        print(f"   成功动作: {self.successful_actions}")
        print(f"   失败动作: {self.failed_actions}")
        if self.total_interactions > 0:
            success_rate = (self.successful_actions / self.total_interactions) * 100
            print(f"   成功率: {success_rate:.1f}%")

    def reset(self):
        """重置控制器状态"""
        self.conversation_history.clear()
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        print("✓ 控制器已重置")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 测试机器人控制器 ===\n")

    # 创建控制器
    controller = RobotController()

    # 测试场景
    test_scenarios = [
        "你好",                          # 闲聊
        "你叫什么名字？",                # 闲聊
        "去厨房",                        # 简单导航
        "帮我找苹果",                    # 搜索
        "把苹果拿过来",                  # 复合任务
    ]

    # 运行测试
    controller.run_test_scenario(test_scenarios)

```


## setup.sh

```sh
#!/bin/bash
# CADE 项目自动配置脚本

set -e  # 遇到错误立即退出

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  🤖 CADE 环境配置脚本                                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否已安装 conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ 未检测到 conda${NC}"
    echo ""
    echo "请先安装 Miniforge:"
    echo "  1. 下载: wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "  2. 安装: bash Miniforge3-Linux-x86_64.sh"
    echo "  3. 重启终端或运行: source ~/.bashrc"
    echo ""
    echo "详细步骤请查看 SETUP_GUIDE.md"
    exit 1
fi

echo -e "${GREEN}✓${NC} 检测到 conda: $(conda --version)"
echo ""

# 询问是否创建新环境
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "是否创建新的 conda 环境 'cade'? (y/n) " -n 1 -r
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 检查环境是否已存在
    if conda env list | grep -q "^cade "; then
        echo -e "${YELLOW}⚠️  环境 'cade' 已存在${NC}"
        read -p "是否删除并重建? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  删除旧环境..."
            conda env remove -n cade -y
        else
            echo "跳过创建环境"
            ENV_EXISTS=true
        fi
    fi

    if [ "$ENV_EXISTS" != "true" ]; then
        echo "📦 创建 conda 环境 'cade' (Python 3.11)..."
        conda create -n cade python=3.11 -y
        echo -e "${GREEN}✓${NC} 环境创建完成"
    fi
fi

echo ""

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cade

# 验证 Python
echo -e "${GREEN}✓${NC} Python: $(python --version)"
echo -e "${GREEN}✓${NC} Python 路径: $(which python)"
echo ""

# 安装依赖
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "是否安装项目依赖? (y/n) " -n 1 -r
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 安装依赖包..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} 依赖安装完成"
    echo ""
fi

# 配置文件
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ! -f config_local.py ]; then
    echo "📝 创建配置文件..."
    cp config_local.example.py config_local.py
    echo -e "${GREEN}✓${NC} 已创建 config_local.py"
    echo ""
    echo -e "${YELLOW}⚠️  重要提示:${NC}"
    echo "   请编辑 config_local.py 填入你的 API 密钥"
    echo "   或者安装 Ollama 使用本地模型"
    echo ""
else
    echo -e "${GREEN}✓${NC} config_local.py 已存在"
    echo ""
fi

# 运行测试
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "是否运行基础测试? (y/n) " -n 1 -r
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 运行测试..."
    python tests/test_basic.py
fi

# 完成
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ✅ 环境配置完成！                                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "下一步:"
echo "  1. 激活环境: conda activate cade"
echo "  2. 配置 API: nano config_local.py"
echo "  3. 运行演示: python main.py --demo"
echo "  4. 交互模式: python main.py"
echo ""
echo "查看帮助:"
echo "  - SETUP_GUIDE.md  : 详细环境配置步骤"
echo "  - QUICKSTART.md   : 快速开始指南"
echo "  - GUIDELINE.md    : 开发路线图"
echo ""

```


## test_no_thought.py

```py
#!/usr/bin/env python3
"""
测试 DeepSeek 的"无思考"模式性能

对比显示/不显示思考过程的输出差异
"""

from robot_controller import RobotController


def test_with_thought():
    """标准模式 - 显示思考过程"""
    print("="*70)
    print("🧠 标准模式：显示思考过程")
    print("="*70)

    controller = RobotController(show_thought=True)

    test_inputs = [
        "你好",
        "去厨房",
        "找到苹果"
    ]

    for user_input in test_inputs:
        controller.process_input(user_input)
        print()


def test_without_thought():
    """无思考模式 - 不显示思考过程"""
    print("\n\n" + "="*70)
    print("⚡ 无思考模式：不显示思考过程")
    print("="*70)

    controller = RobotController(show_thought=False)

    test_inputs = [
        "你好",
        "去厨房",
        "找到苹果"
    ]

    for user_input in test_inputs:
        controller.process_input(user_input)
        print()


def compare():
    """对比说明"""
    print("\n\n" + "="*70)
    print("📊 对比说明")
    print("="*70)

    print("""
标准模式（show_thought=True）输出：
  ✅ 💭 思考过程: ...
  ✅ 💬 回复: ...
  ✅ ⚡ 计划动作: ...
  ✅ 🤖 执行动作

无思考模式（show_thought=False）输出：
  ❌ 💭 思考过程: ...（不显示）
  ✅ 💬 回复: ...
  ✅ ⚡ 计划动作: ...
  ✅ 🤖 执行动作

注意：
  • LLM 仍然会生成 thought 字段（遵循 Prompt）
  • 只是在显示时被隐藏了
  • 适合：
    - 生产环境（用户不需要看到机器人的"内心独白"）
    - 演示展示（更简洁的输出）
    - 性能测试（减少终端输出）

如果想让 LLM 完全不生成 thought，需要修改 Schema 和 Prompt（见方案2）
    """)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "with":
            test_with_thought()
        elif sys.argv[1] == "without":
            test_without_thought()
        else:
            print("用法: python test_no_thought.py [with|without]")
    else:
        print("🎬 完整对比测试\n")
        compare()
        print("\n运行示例:")
        print("  python test_no_thought.py with      # 只测试标准模式")
        print("  python test_no_thought.py without   # 只测试无思考模式")

```


## tests/test_basic.py

```py
"""
基础测试用例

测试各个模块的基本功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    """测试配置模块"""
    print("=== 测试 Config ===")
    from config import Config

    config = Config.get_llm_config()
    assert "base_url" in config
    assert "api_key" in config
    assert "model" in config

    print(f"✓ 配置模块正常")
    print(f"  模式: {Config.MODE}")
    print(f"  模型: {config['model']}\n")


def test_schemas():
    """测试数据模型"""
    print("=== 测试 Schemas ===")
    from brain.schemas import (
        NavigateAction, SearchAction, RobotDecision, parse_action
    )

    # 测试导航动作
    nav = NavigateAction(target="kitchen")
    assert nav.type == "navigate"
    assert nav.target == "kitchen"

    # 测试搜索动作
    search = SearchAction(object_name="apple")
    assert search.type == "search"
    assert search.object_name == "apple"

    # 测试决策
    decision = RobotDecision(
        thought="测试思考",
        reply="测试回复",
        action=nav
    )
    assert decision.thought == "测试思考"

    # 测试解析
    action_dict = {"type": "navigate", "target": "kitchen"}
    action = parse_action(action_dict)
    assert action.type == "navigate"

    print("✓ 数据模型正常\n")


def test_mock_robot():
    """测试Mock Robot"""
    print("=== 测试 Mock Robot ===")
    from body.mock_robot import MockRobot
    from body.robot_interface import RobotState

    robot = MockRobot(name="TestBot")

    # 测试导航
    success = robot.navigate("kitchen")
    assert success is True
    assert robot.current_position == "kitchen"
    assert robot.get_state() == RobotState.IDLE

    # 测试搜索
    result = robot.search("apple")
    assert result is not None
    assert "name" in result

    # 测试语音
    success = robot.speak("测试语音")
    assert success is True

    print("✓ Mock Robot 正常\n")


def test_prompts():
    """测试提示词"""
    print("=== 测试 Prompts ===")
    from brain.prompts import get_system_prompt, add_context

    # 测试获取提示词
    prompt = get_system_prompt("default")
    assert len(prompt) > 0
    assert "LARA" in prompt or "机器人" in prompt

    # 测试简化提示词
    simple = get_system_prompt("simple")
    assert len(simple) > 0

    # 测试添加上下文
    enhanced = add_context(prompt, "当前位置: 厨房")
    assert "当前位置" in enhanced

    print("✓ 提示词模块正常\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 开始运行基础测试")
    print("="*60 + "\n")

    tests = [
        test_config,
        test_schemas,
        test_mock_robot,
        test_prompts,
    ]

    failed = []

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_func.__name__} 失败: {e}\n")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)

    print("="*60)
    if not failed:
        print("✅ 所有测试通过！")
    else:
        print(f"❌ {len(failed)} 个测试失败: {', '.join(failed)}")
    print("="*60)

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

```

