# Robot Speaker DOA Bundle

这个目录是给机器人部署用的精简版，只包含：

- 说话人录入与识别
- M2 板子的原始音频导出
- 本地 DOA 估计
- 一个可直接运行的主程序

不包含历史调试脚本、测试入口和无关文档。

## 目录结构

- `live_speaker_doa.py`
  主程序。按 `T` 录入 5 秒，说话人识别成功时输出 `speaker_id` 和 `angle_deg`
- `m2_mic_array.py`
  对 M2 板子的精简封装，只保留导出音频和 DOA 估计
- `m2_serial_cmd.py`
  M2 官方串口封包命令
- `m2_doa_origin.py`
  从 `origin.pcm` 估计 DOA
- `speaker_manager.py`
  说话人特征提取、注册、分类
- `requirements.txt`
  Python 依赖

## 系统依赖

机器人上至少需要：

- Python 3.10+
- `adb`
- 能访问串口设备，例如 `/dev/wheeltec_mic`、`/dev/ttyACM0`、`/dev/ttyUSB0`

Ubuntu/Debian 常见安装：

```bash
sudo apt update
sudo apt install -y android-sdk-platform-tools
```

如果你的系统里 `adb` 不在 PATH，可以运行前指定：

```bash
export ADB=/path/to/adb
```

## Python 环境

建议使用单独虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1. 先安装 PyTorch / torchaudio

这一步和你的机器人平台有关。

如果是常见 x86 Linux CPU，可以先尝试：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如果是 ARM / Jetson，请按设备对应的 PyTorch 安装方式先装好 `torch` 和 `torchaudio`。

### 2. 再安装其余依赖

```bash
pip install -r requirements.txt
```

## 离线模型

这个离线包会直接附带：

- `workdir/pretrained_models/spkrec-ecapa-voxceleb`

所以在另一台电脑上如果不能访问 `huggingface`，也可以直接运行。

默认 `work-dir` 就是包内的：

- `./workdir`

默认说话人数据库仍然是“本次运行临时库”，每次启动都会重新开始。

## 运行

```bash
python live_speaker_doa.py
```

常用参数：

```bash
python live_speaker_doa.py \
  --baud 115200 \
  --threshold 0.3 \
  --detect-seconds 0.5 \
  --enroll-seconds 5
```

默认会自动探测串口设备，优先顺序大致是：

- `/dev/wheeltec_mic`
- `/dev/ttyACM*`
- `/dev/ttyUSB*`

如果自动探测不对，再手动指定：

```bash
python live_speaker_doa.py --serial-device /dev/ttyACM0
```

如果环境音较多，程序现在会先做一层轻量人声检测：

- 录入时默认要求至少约 `0.8s` 的明显人声
- 日常检测时默认要求至少约 `0.15s` 的明显人声

需要调松或调严时：

```bash
python live_speaker_doa.py \
  --min-enroll-speech-seconds 0.6 \
  --min-detect-speech-seconds 0.1
```

## 使用方式

- 按 `T`
  录入 5 秒当前说话人的声音
- 按 `Q`
  退出程序
- 平时程序持续短窗采集
  如果识别到已录入说话人，就输出：
  - `speaker_id`
  - `angle_deg`
- `detect_*` 临时目录会自动清理，不会一直堆积
- `enroll_*` 目录默认保留，便于你回看录入样本

## 输出示例

```json
{"status":"ready","detect_seconds":0.5,"enroll_seconds":5.0,"threshold":0.3}
{"status":"empty_db","message":"当前还没有录入说话人。按 T 开始录入 5 秒样本。"}
{"status":"enroll_requested","message":"收到按键 T，立即开始录入 5 秒说话人样本。"}
{"status":"enroll_rejected_non_speech","reason":"too_little_active_speech", ...}
{"status":"enrolled","speaker_id":"spk_ab12cd34","action":"new","score":null}
{"status":"recognized","speaker_id":"spk_ab12cd34","speaker_name":"spk_ab12cd34","score":0.82,"angle_deg":186.0}
```

## 部署建议

- 首次在机器人上运行前，先单独确认：
  - 串口设备存在
  - `adb devices` 能看到板子
- 如果 DOA 零点和机器人坐标不一致，可以后续再在 `m2_doa_origin.py` 中做统一角度映射
- 如果你想持久化说话人库，可以加：

```bash
python live_speaker_doa.py --db ./speakers_db.pkl
```
