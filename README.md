# 🤖 CADE (Cognitive Agent for Domestic Environment)

**CADE** 是一个基于 **ROS (Noetic)** 的具身智能机器人实验平台，专注于实现**“感知-推理-动作”**的闭环交互。项目采用模块化的“线性演化”架构，集成了实时语音交互 (ASR/TTS) 与大语言模型 (LLM) 推理大脑。



---

## 🛠️ 环境要求 (Environment)

* **OS**: Fedora Linux (推荐) / Ubuntu 20.04
* **Middleware**: ROS Noetic
* **Language**: Python 3.8+ (Conda 环境名建议: `CADE`)
* **Audio Backend**: PipeWire / PulseAudio

---

## 🚀 快速开始 (Quick Start)

### 1. 基础设施：音频虚拟线缆
为了解决 Linux 下音频设备索引漂移及“静默调试”需求，必须先建立虚拟环回链路。
```bash
# 创建 CADE_Speaker (Sink) 和 CADE_Mic (Source) 并完成内部桥接
bash scripts/setup_virtual_audio.sh
```

### 2. 启动语音交互节点
启动 ASR (SenseVoice) 与 TTS (VITS) 节点。系统会自动通过字符串匹配锁定虚拟设备。
```bash
# mic_id 和 speaker_id 传入 "Mic" 和 "Speaker" 即可精准锁定虚拟接口
roslaunch asr_tts speech.launch mic_id:="Mic" speaker_id:="Speaker"
```

### 3. 运行测试分发器
在不方便说话的环境下，通过预录制指令文件测试全链路：
```bash
bash test_me.sh
```

---

## 🏗️ 核心架构特性 (Key Features)

### 1. 生产者-消费者音频流
为了解决 `sd.rec()` 同步录音带来的盲区丢包问题，ASR 节点实现了**不间断流式架构**：
* **Producer (后台线程)**: `sd.InputStream` 零间断采集采样点并推入队列。
* **Consumer (主线程)**: 从队列提取数据进行 Silero VAD 检测与文本转译，确保语音连贯性。

### 2. 线性演化模型 (Linear Evolution)
项目摒弃了复杂的类继承，采用版本线性更替。`body/robot.py` 集成了 V1 的空间记忆与 V2 的状态机管理，严格遵循 `IDLE -> THINKING -> EXECUTING -> SPEAKING` 状态切换。

---

## 🚧 当前的“让步”与未来路线图 (Roadmap)

### 🌙 当前的“工程让步”
目前的系统为了在 PC 开发环境下快速跑通全链路，做了一些务实的折衷：
* **虚拟音频环回**: 使用 `null-sink` 模拟麦克风，解决了物理环境噪音和调试不便的问题，但尚未处理真实麦克风下的回声消除 (AEC)。
* **Mock 执行器**: 机器人动作在 `robot.py` 中主要是语义层面的模拟（如逻辑移动、物体状态变更），暂未接入物理底盘驱动。
* **云端 LLM**: 当前默认通过 API 调用云端大模型，依赖网络稳定性。

### ☀️ 未来演进方向
* **[感知升级]**: 接入视觉 VLM 模块，将 ASR 识别出的指令与视觉空间特征（Vision-Language Alignment）进行深度对齐。
* **[端侧部署]**: 将 Brain 模块迁移至 **NVIDIA Jetson Orin**，实现基于 Ollama 的纯本地推理，保障隐私与响应延迟。
* **[物理接入]**: 将 `robot_interface.py` 的具体实现从 Mock 切换为真实的硬件驱动（如移动底盘、机械臂控制）。
* **[多模态反馈]**: 实现 TTS 语调与机器人表情/动作的同步。

---

## 📂 项目结构 (Structure)

* `brain/`: LLM 客户端、Prompt 工程与数据 Schema
* `body/`: 机器人抽象接口与具体实现
* `bridge/`: ROS 消息桥接与业务逻辑协调
* `src/asr_tts/`: 基于 Sherpa-ONNX 的底层语音处理节点
* `scripts/`: 系统初始化与环境配置脚本

---

## 👨‍💻 开发者
**Huyanshen** (University of Science and Technology of China)

---

