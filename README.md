# CADE: 具身智能机器人系统

CADE (Embodied AI Robot System) 是一个集成语音交互（ASR/TTS）与大模型决策驱动的机器人控制系统。支持在 Fedora (开发环境) 下进行静默链路测试，以及在 NVIDIA Jetson Orin / Ubuntu 20.04 (部署环境) 上进行实机运行。

## 1. 基础环境安装

### A. Python & ROS 核心环境 (Conda)
建议使用 Conda 隔离环境，防止项目依赖与系统自带的机器人驱动产生版本冲突。

```bash
# 1. 创建环境 (推荐 Python 3.10)
conda create -n CADE -y python=3.10
conda activate CADE

# 2. 安装 ROS 构建工具
# Fedora/开发环境: catkin_make 是由这个包提供的，不是 pip
conda install -c robostack-release ros-noetic-catkin ros-noetic-rosbash -y

# 3. 安装项目依赖
pip install -r requirements.txt
```

### B. ROS 桥接依赖
为了让 Conda 隔离环境识别 ROS 资源，项目依赖 rospkg 与 catkin_pkg（已包含在 requirements.txt 中，无需手动执行 pip）。
* **Ubuntu 20.04 (实机)**: 
  1. 系统层安装：`sudo apt install ros-noetic-desktop-full`
  2. Conda 环境内：`pip install rospkg catkin_pkg` (允许隔离环境访问系统 ROS 资源)
* **Fedora (开发)**: 
  1. 直接在 Conda 环境内：`pip install rospkg catkin_pkg`

---

## 2. ROS 工作空间编译与环境激活

CADE 的源码位于项目根目录的 `src/` 文件夹下。在运行任何节点前，必须先完成编译并激活环境。

### 核心步骤：编译并 Source
1. **编译 (仅需执行一次)**:
   ```bash
   conda activate CADE
   # 在 CADE 项目根目录下运行，生成 build/ 和 devel/ 文件夹
   catkin_make -DCMAKE_BUILD_TYPE=Release
   ```

2. **激活环境 (每个新终端必做)**:
   * **Ubuntu 20.04**:
     ```bash
     source /opt/ros/noetic/setup.zsh    # 1. 激活系统 ROS
     source devel/setup.zsh             # 2. 激活 CADE 工作空间
     ```
   * **Fedora (Conda)**:
     ```bash
     source devel/setup.zsh             # 直接激活当前目录生成的配置
     ```

---

## 3. 环境变量配置 (`.env`)

项目使用 `.env` 文件隔离配置。禁止在代码中硬编码任何 API Key。

1. 执行 `cp .env.example .env`。
2. 配置参数：
   * `CADE_MODE`: `CLOUD` (调用 API) 或 `LOCAL` (调用本地 Ollama)。
   * `CADE_CLOUD_API_KEY`: 你的 DeepSeek 或 DashScope 密钥。
   * `CADE_LOCAL_MODEL`: 本地运行的模型名称 (默认 `qwen3:8b`)。

---

## 4. 语音链路配置

### A. 开发环境：Fedora + 虚拟音频 (静默测试)
建立虚拟线缆链路：`测试脚本 -> 虚拟扬声器 (CADE_Speaker) -> 虚拟麦克风 (Monitor) -> CADE ASR`。

**1. 准备工作**:
```bash
sudo dnf install pulseaudio-utils pavucontrol
bash scripts/setup_virtual_audio.sh
```

**2. 运行静默测试**:
* **终端 1**: 启动语音节点
  ```bash
  roslaunch asr_tts speech.launch mic_id:=CADE_Speaker.monitor speaker_id:=CADE_Speaker
  ```
* **终端 2**: 运行注入脚本
  ```bash
  bash test_me.sh
  ```
*注意：在 `pavucontrol` 中确保 `asr_node` 正在从 "Monitor of CADE_Speaker" 录制。*

### B. 部署环境：Ubuntu 20.04 (实机)
**1. 确认硬件 ID**: 使用 `arecord -l` 确定录音设备 ID（如 `hw:1,0`）。
**2. 启动实机语音**:
```bash
roslaunch asr_tts speech.launch mic_id:="hw:1,0" speaker_id:="hw:1,0"
```

---

## 5. CADE 主控制器启动

编译并配置完成后，启动 CADE 大脑控制器以接收语音指令。

### A. ROS 语音模式（生产环境）

启动顺序：
```bash
# 终端 1: 启动 ROS core
roscore

# 终端 2: 启动 ASR/TTS 语音节点
roslaunch asr_tts speech.launch

# 终端 3: 启动 CADE 主控制器
python main.py
```

`main.py` 支持的参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode` | 提示词模式: `default`(标准) / `compact`(精简) / `simple`(简化) / `debug`(调试) | `--mode compact` |
| `--no-thought` | 不显示 LLM 的思考过程 | `--no-thought` |
| `--env` | 自定义环境上下文，注入到系统提示词中 | `--env "你在实验室桌子上"` |

完整示例：
```bash
python main.py --mode debug --env "你正坐在 Fedora 实验室的桌子上，目前只能通过语音与人交流。"
```

启动后，CADE 进入等待状态。对着麦克风说话（参考终端输出的测试案例），ASR 识别后交由 LLM 决策，机器人执行动作并通过 TTS 语音回复。

### B. 控制台交互模式（开发调试）

无需 ROS，直接在终端中与 LLM 对话测试决策链路：

```python
python -c "
from robot_controller import RobotController
controller = RobotController()
controller.interactive_mode()
"
```

在交互模式下输入 quit/exit 退出，输入 status 查看机器人状态，输入 stats 查看统计。

### C. 快捷测试：静默链路

如果已按第 4 节完成虚拟音频配置：

```bash
# 终端 1: 语音节点
roslaunch asr_tts speech.launch mic_id:=CADE_Speaker.monitor speaker_id:=CADE_Speaker

# 终端 2: CADE 控制器
python main.py

# 终端 3: 注入测试语音
bash test_me.sh
```

---

## 6. 本地 LLM 加速 (Ollama)

在边缘端（Orin）部署时，推荐使用 Ollama：
1. **启动模型**: `ollama pull qwen3:8b && ollama serve`。
2. **协议兼容**: CADE 通过 OpenAI 协议连接本地端口 `11434`。代码内已内置 `trust_env=False` 逻辑，**自动跳过系统代理干扰**。

---

## 7. 项目结构与工具

* **`main.py`**: ROS 语音模式入口，启动 `RosVoiceBridge` 进入 ASR→LLM→TTS 循环。
* **`robot_controller.py`**: 主控制器，整合 Brain(LLM) 和 Body(Robot) 的感知-决策-执行循环。
* **`test_me.sh`**: 交互式测试分发脚本，支持 10+ 种机器人指令注入。
* **`ARCHITECTURE.md`**: 详细说明 `body` (硬件接口), `brain` (LLM 决策), `bridge` (通信层) 的代码实现。
* **`config.py`**: 配置抽象层，负责从 `.env` 自动读取并验证参数。

---

## 常见问题 (FAQ)

> **Q: `catkin_make` 命令找不到？**
> A: 这通常是因为你没有在 Conda 环境中安装 `ros-noetic-catkin` 或者没有 `conda activate CADE`。请检查第一步安装步骤。

> **Q: 为什么要在 Conda 里 pip 安装 `rospkg`？**
> A: Conda 环境是物理隔离的。安装 `rospkg` 相当于在你的隔离环境里安装了一个“导航仪”，让 Python 能准确找到并加载系统路径下的 ROS 消息类型。

> **Q: 连接本地 Ollama 报错？**
> A: 请检查 `.env` 中的 `CADE_LOCAL_BASE_URL` 是否包含 `/v1` 后缀（如 `http://localhost:11434/v1`）。
