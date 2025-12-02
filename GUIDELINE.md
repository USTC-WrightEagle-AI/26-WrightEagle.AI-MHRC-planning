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
