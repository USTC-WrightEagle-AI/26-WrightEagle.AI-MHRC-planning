# 🤖 ROS 语音识别与合成功能包 (ASR & TTS)

这是一个基于 ROS (Robot Operating System) 的功能包，实现了语音识别 (ASR) 和语音合成 (TTS) 功能。它包含两个核心节点，分别用于发布语音识别结果和订阅文本进行语音播报。

---

## 📁 项目结构

请确保将本功能包放置在你的工作空间的 `src` 目录下：

```bash
your_workspace/
    src/
        asr_tts/  # 本功能包
```

---

## 🚀 快速开始

### 1. 下载模型文件
本功能包需要模型文件才能运行，请在 `asr_tts` 目录下新建 `models` 文件夹，并将下载的模型放入其中。

- **完整版模型 (推荐):** [科大云盘 - models](https://pan.ustc.edu.cn/share/index/914a5d8730ab4abea628)
- **小型模型 (轻量级):** [科大云盘 - models](https://pan.ustc.edu.cn/share/index/8fabcbe366de45ed8072)

### 2. 安装依赖
进入功能包目录，安装 Python 依赖：

```bash
pip install -r requirements.txt
```

### 3. 编译与运行（在你的工作空间）
```bash
# 编译
catkin_make

# 激活环境 (source)
source devel/setup.bash

# 启动功能包
roslaunch asr_tts speech.launch
```

---

## 🧩 功能节点说明

本功能包包含以下两个核心节点：

| 节点名称     | 功能描述     | 话题 (Topic) | 类型              |
| :----------- | :----------- | :----------- | :---------------- |
| **asr_node** | 负责语音识别 | `/asr`       | 发布 (Publisher)  |
| **tts_node** | 负责语音合成 | `/tts`       | 订阅 (Subscriber) |

### 话题交互逻辑
- **ASR 节点**：监听麦克风输入，识别后将文本结果发布到 `/asr` 话题。
- **TTS 节点**：监听 `/tts` 话题，接收到文本后进行语音合成并播放。
