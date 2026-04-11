# Robot Lab Test Scripts - 机器人现场测试脚本集

> 为明天实验室机器人部署测试准备的完整脚本工具链，覆盖环境探测、噪声采集、模块验证、基准测试、噪声鲁棒性、端到端交互全流程。

## 文件清单

| 脚本 | 功能 | 对应待办 |
|------|------|----------|
| `robot_env_probe.py` | 系统环境探测 (GPU/CPU/OS/音频设备/Docker) | #1 系统架构信息 |
| `noise_collector.py` | 噪声采集 + 带标注语音采集 | #2 采集机器人噪声数据 |
| `noise_mixer.py` | SNR噪声合成工具 (干净音频+纯噪声→带噪测试集) | #2 补充: 合成噪声环境 |
| `robot_module_test.py` | ASR/TTS/VAD 模块独立验证 (加载+推理PASS/FAIL) | #4, #5 部署验证 |
| `robot_benchmark.py` | 多模型性能基准对比 (准确率/RTF/延迟排名) | #6 性能测试 |
| `noise_robustness_test.py` | 噪声鲁棒性评估 (准确率 vs SNR 曲线) | #6 抗噪性能 |
| `e2e_voice_interaction_test.py` | 端到端语音交互测试 (VAD→ASR→LLM→TTS 全链路延迟) | #7 交互系统原型 |

---

## 推荐使用顺序

```
到达实验室后按以下顺序执行:

1. [环境探测]   robot_env_probe.py          ← 先确认硬件条件
2. [模块验证]   robot_module_test.py         ← 确认各模块能跑
3. [噪声采集]   noise_collector.py           ← 采集纯噪声(空闲时 + 运行时)
4. [基准测试]   robot_benchmark.py           ← 跑全部模型性能对比
5. [E2E测试]    e2e_voice_interaction_test.py ← 测完整交互链路延迟
```

离开实验室后（在本地电脑上）:

```
6. [噪声合成]   noise_mixer.py              ← 用采集的噪声合成多SNR测试集
7. [鲁棒性评估] noise_robustness_test.py     ← 跑抗噪性能曲线
```

---

## 各脚本详细说明

### 1. robot_env_probe.py — 系统环境探测

**用途**: 一键收集机器人的全部软硬件环境信息，确认部署条件是否满足。

```bash
cd test/lab_test_scripts/

# 基本用法 — 自动探测并保存报告
python robot_env_probe.py

# 指定输出目录
python robot_env_probe.py --output-dir ./reports/

# 只显示不保存文件
python robot_env_probe.py --no-save
```

**输出**:
- 终端打印: GPU型号/显存、CPU核心数/内存、OS版本、音频输入输出设备列表、Docker状态、sherpa-onnx版本、磁盘空间
- 文件: `robot_env_report.json` (完整的结构化JSON)

**检查项**:
- GPU Available? (nvidia-smi)
- Audio Input Device? (麦克风)
- Audio Output Device? (扬声器)
- sherpa-onnx Installed?
- Disk Free > 5GB?

---

### 2. noise_collector.py — 噪声/语音采集

**用途**: 在机器人上录制环境噪声或带标注的语音。

#### 模式A: 录制纯噪声 (推荐!)

用于后续 SNR 合成测试。在不同场景下各录一段:

```bash
# 机器人空闲时 (风扇声、空调背景音)
python noise_collector.py --mode pure_noise --scene idle --duration 30

# 机器人移动时 (电机噪声)
python noise_collector.py --mode pure_noise --scene moving --duration 30

# 实验室有人活动时 (人声背景)
python noise_collector.py --mode pure_noise --scene lab_busy --duration 30

# 风扇高转速时
python noise_collector.py --mode pure_noise --scene fan_high --duration 30
```

> **关键提示**: 录制时**不要说话**，保持安静。这些纯噪声会与干净的TTS音频混叠，生成不同SNR级别的测试音频。

**输出**: `noise_data/pure_noises/{scene}_{timestamp}_30s.wav` + 同名 `.json` 元信息文件

#### 模式B: 带标注语音采集 (可选)

对着机器人麦克风念预设文本，自动记录 ground truth：

```bash
python noise_collector.py --mode speech --speaker user_01
```

会逐条显示16条测试文本，每条按回车开始录音。
**输出**: `noise_data/speech_collection/{speaker}_{timestamp}/manifest.json`

#### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | `pure_noise` 或 `speech` | **必填** |
| `--scene` | 场景标签 (仅 pure_noise) | unknown |
| `--duration` | 每段时长(秒) | 30 |
| `--speaker` | 说话人标识 (仅 speech) | unknown |
| `--device-id` | 麦克风设备ID | 自动选择默认 |
| `--output-dir` | 输出根目录 | ./noise_data/ |
| `--list-devices` | 列出可用音频设备后退出 | - |

先用 `--list-devices` 查看可用的麦克风设备:
```bash
python noise_collector.py --list-devices
```

---

### 3. noise_mixer.py — SNR噪声合成工具 (本地用, 不需要在机器人上跑)

**用途**: 将干净的TTS音频与采集的纯噪声按指定SNR混叠，生成带噪测试集。

**前提**: 已完成步骤2的噪声采集 (`noise_data/pure_noises/` 下有 .wav 文件)，且 `test/audio_output/` 下有 TTS 测试音频。

```bash
cd test/lab_test_scripts/

# 基础用法 — 用所有TTS音频 x 所有噪声文件 x 默认SNR级别
python noise_mixer.py \
    --clean-dir ../audio_output/ \
    --noise-dir ./noise_data/pure_noises/ \
    --output-dir ./noisy_test_set/

# 自定义SNR级别 (clean / 20dB / 15dB / 10dB / 5dB / 0dB)
python noise_mixer.py \
    --clean-dir ../audio_output/ \
    --noise-dir ./noise_data/pure_noises/ \
    --snr-levels clean 20 15 10 5 0 -5 \
    --output-dir ./noisy_test_set/

# 只用特定场景的噪声 (如只用 idle 和 moving)
python noise_mixer.py \
    --clean-dir ../audio_output/ \
    --noise-dir ./noise_data/pure_noises/ \
    --noise-filter idle moving \
    --output-dir ./noisy_test_set/
```

**输出命名规则**:
```
{cleanName}_{noiseName}_{snrXXdB}.wav
例如: tts_000_idle_snr10dB.wav
     tts_001_moving_clean.wav
```
同时生成 `mix_manifest.json` 清单文件，包含每个文件的源文件和实际SNR值。

**原理**:
$$\text{SNR(dB)} = 10 \cdot \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$

通过调整噪声幅度精确匹配目标SNR。噪声长度不足时自动循环扩展。

---

### 4. robot_module_test.py — 模块独立验证

**用途**: 在机器人上逐个验证 ASR/TTS/VAD 模块是否能正常加载和推理。

```bash
cd test/lab_test_scripts/

# 验证所有模块
python robot_module_test.py

# 只验证 ASR 模型
python robot_module_test.py --modules asr

# 只验证 VAD
python robot_module_test.py --modules vad

# 只验证特定模型
python robot_module_test.py --modules whisper moonshine qwen3

# 指定自定义模型目录
python robot_module_test.py --models-base-path /path/to/models/
```

**检测内容**:
- 模型文件完整性检查 (缺少文件则 SKIP)
- 加载时间 (ms)
- 推理时间 (ms) — 用生成的正弦波信号做最小推理
- 输出 PASS / WARN / FAIL

**输出**: `module_validation_results.json`

---

### 5. robot_benchmark.py — 多模型性能基准对比

**用途**: 用统一的16条TTS测试音频，在机器人上运行全部ASR模型并排名。

```bash
cd test/lab_test_scripts/

# 全量基准测试
python robot_benchmark.py

# 指定测试音频目录
python robot_benchmark.py --audio-dir ../audio_output/

# 只测离线模型
python robot_benchmark.py --mode offline

# 只测指定模型 (加速)
python robot_benchmark.py --models whisper moonshine qwen3

# 生成对比图表 (需要 matplotlib)
python robot_benchmark.py --plot-chart
```

**指标**:
| 指标 | 说明 |
|------|------|
| Char Accuracy (%) | 字符级准确率 (CER 补集) |
| WER (%) | 词错误率 |
| Exact Match Rate (%) | 完全匹配率 |
| RTF | 实时率 (<1.0 = 实时可行) |
| Avg Inference (ms) | 平均推理耗时 |

**输出**:
- 终端: 按准确率排序的排名表
- 文件: `benchmark_results.json` (含每个模型的逐文件详细结果)
- 图表: `benchmark_charts.png` (如果加 `--plot-chart`, 准确率柱状图 + RTF散点图)

---

### 6. noise_robustness_test.py — 噪声鲁棒性评估

**用途**: 评估各ASR模型在不同信噪比(SNR)下的准确率变化趋势。

**前提**: 先运行 `noise_mixer.py` 生成了带噪测试集 (`noisy_test_set/`)。

```bash
cd test/lab_test_scripts/

# 使用已合成的带噪测试集进行评估
python noise_robustness_test.py \
    --noisy-dir ./noisy_test_set/

# 只评估部分模型
python noise_robustness_test.py \
    --noisy-dir ./noisy_test_set/ \
    --models whisper moonshine qwen3

# 生成 Markdown 报告
python noise_robustness_test.py \
    --noisy-dir ./noisy_test_set/ \
    --generate-report
```

**输出**:
- 终端: 每个模型在每个SNR级别上的准确率 + ASCII 进度条
- 文件: `robustness_results.json`
- 报告: `robustness_report.md` (含准确率-SNR表格和分析要点, 如果加了 `--generate-report`)

---

### 7. e2e_voice_interaction_test.py — 端到端语音交互测试

**用途**: 测量完整语音交互链路 (录音 → VAD → ASR → LLM → TTS) 的各阶段延迟和总延迟。

```bash
cd test/lab_test_scripts/

# 单轮实时对话测试 (从麦克风录音)
python e2e_voice_interaction_test.py --rounds 1

# 多轮对话循环 (3轮)
python e2e_voice_interaction_test.py --rounds 3

# ★ 回放模式: 用已有的TTS音频作为"用户输入" (不需要麦克风!)
python e2e_voice_interaction_test.py --rounds 5 \
    --use-recorded-audio ../audio_output/

# 指定ASR模型和TTS音色
python e2e_voice_interaction_test.py --asr-model whisper --tts-sid 0

# 播放TTS回复 (需接扬声器)
python e2e_voice_interaction_test.py --rounds 1 --play-output

# 生成延迟报告
python e2e_voice_interaction_test.py --rounds 5 \
    --use-recorded-audio ../audio_output/ \
    --generate-report
```

**关于 Mock LLM**: 
- 默认使用内置 MockLLMClient (模拟200ms延迟，根据关键词返回预设回复)
- 无需真实 LLM API 即可测量完整链路延迟
- 可通过 `--mock-llm-latency-ms` 调整模拟延迟

**延迟指标**:
| 阶段 | 说明 | 目标 |
|------|------|------|
| Input | 音频采集 | - |
| VAD | 语音端点检测 | < 50ms |
| ASR | 语音识别 | < 500ms |
| LLM | 语言处理 (mock) | ~200ms |
| TTS | 语音合成 | 视文本长度 |
| **E2E Total** | **端到端总延迟** | **< 1500ms** |

**输出**:
- 终端: 每轮各阶段耗时明细
- 文件: `e2e_test_results.json`
- 报告: `e2e_test_report.md` (如果加了 `--generate-report`)

---

## 快速参考卡

```bash
# ====== 到达实验室后立即执行 ======

# Step 0: 安装依赖 (如尚未安装)
pip install sounddevice soundfile scipy matplotlib wmi

# Step 1: 探查环境
python robot_env_probe.py

# Step 2: 验证模块
python robot_module_test.py

# Step 3: 采集噪声 (不同场景各录一段!)
python noise_collector.py --mode pure_noise --scene idle --duration 30
python noise_collector.py --mode pure_noise --scene moving --duration 30

# Step 4: 性能基准测试
python robot_benchmark.py --plot-chart

# Step 5: E2E交互测试
python e2e_voice_interaction_test.py --rounds 5 --use-recorded-audio ../audio_output/ --generate-report

# ====== 回到本地后执行 ======

# Step 6: 合成噪声测试集
python noise_mixer.py --clean-dir ../audio_output/ --noise-dir ./noise_data/pure_noises/ --output-dir ./noisy_test_set/

# Step 7: 鲁棒性评估
python noise_robustness_test.py --noisy-dir ./noisy_test_set/ --generate-report
```

---

## 依赖项

```
sherpa-onnx      (必须, 核心依赖)
numpy            (必须)
sounddevice      (噪声采集 + E2E测试需要)
soundfile        (噪声合成 + 基准测试需要)
scipy            (noise_mixer.py 重采样需要)
matplotlib       (可选, 生成对比图表)
wmi              (可选, Windows下robot_env_probe.py获取CPU信息)
```

## 注意事项

1. **采样率统一为 16000Hz** — 所有脚本均使用此采样率，与ASR模型一致
2. **路径问题** — 脚本中使用了 `path_utils.py` 的逻辑来处理中文路径，但在Linux机器人上通常不需要
3. **GPU/CPU** — 所有脚本默认使用 CPU 推理；如果有 GPU 可通过修改代码中的 `provider="cpu"` 改为 `"cuda"`
4. **噪声采集建议** — 至少采集 2 种场景 (idle + moving)，每种 30 秒，后续合成空间最大
