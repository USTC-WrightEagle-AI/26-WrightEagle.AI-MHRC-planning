# Task1 — Receptionist 接待任务

RoboCup\@Home 接待任务：机器人迎接两位客人，引导入座、介绍、拿包并跟随 host 放包。

## 目录结构

```
task1_receptionist/
├── __init__.py                  — 模块入口，导出核心类
├── state_definitions.py         — 状态定义、消息格式、流转表
├── task1_controller.py          — 总控执行器 (状态机编排)
├── readme.md                    — 本文档
└── sub_modules/                 — 各子模块
    ├── __init__.py              — 子模块包导出
    ├── topic_names.py           — 所有 ROS 话题名称和指令常量 (集中管理)
    ├── base_module.py           — ROSTopicBridge + DoorbellModule + NavigationModule + SpeechModule + VisionModule + ManipulationModule + SpeakerDOAModule
    ├── speech_interaction.py    — SpeechInterface (TTS + ASR)，Mock/ROS/Scripted 三后端
    └── llm_interface.py         — LLMInterface (信息提取)，Mock/ROS 双后端
```

## 状态机流程 (16 个执行状态)

```
IDLE
  │
  ▼
[1]  WAIT_FOR_DOORBELL_1 → 等待门铃声 (guest1 到达)
[2]  GO_TO_DOOR          → 移动到门口
[3]  ASK_GUEST1_INFO     → 询问 guest1 姓名和饮料 (ASR + LLM 提取)
[4]  GUIDE_GUEST1        → 带 guest1 去客厅
[5]  POINT_EMPTY_SEAT    → 指向空座请 guest1 入座 (语音 + 操作)
[6]  RETURN_TO_START     → 返回起点
[7]  WAIT_FOR_DOORBELL_2 → 等待门铃声 (guest2 到达)
[8]  PICK_UP_GUEST2      → 到门口接 guest2 (导航 + ASR + LLM 提取姓名)
[9]  DESCRIBE_GUEST1     → 向 guest2 描述 guest1 外貌 (视觉 + 语音)
[10] SEAT_GUEST2         → 带 guest2 入座 (导航 + ASR + LLM 提取饮品)
[11] INTRODUCE_GUESTS    → 相互介绍两位客人
[12] REQUEST_GUEST2_BAG  → 请求 guest2 放包到托盘 (语音 + 操作)
[13] FIND_HOST           → 找到 host (视觉 + 声源定位)
[14] FOLLOW_HOST         → 跟随 host (声源跟踪 + 导航)
[15] PLACE_BAG           → 把包放到指定位置
[16] TASK_COMPLETE       → 播报结束语
  │
  ▼
IDLE
```

## 状态 → 模块调度表

| 状态 | 模块调用顺序 | 说明 |
|------|------------|------|
| WAIT_FOR_DOORBELL_1 | doorbell | 等待 guest1 门铃 |
| GO_TO_DOOR | navigation | 导航到门口 |
| ASK_GUEST1_INFO | speaker_doa → speech | 面向说话人 + 语音对话 + LLM 提取 |
| GUIDE_GUEST1 | navigation | 导航到客厅 |
| POINT_EMPTY_SEAT | speech → manipulation | 语音引导 + 手臂指向 |
| RETURN_TO_START | navigation | 返回起点 |
| WAIT_FOR_DOORBELL_2 | doorbell | 等待 guest2 门铃 |
| PICK_UP_GUEST2 | navigation → speaker_doa → speech | 导航到门口 + 面向说话人 + 语音对话 |
| DESCRIBE_GUEST1 | vision → speaker_doa → speech | 视觉识别 + 面向说话人 + 语音播报 |
| SEAT_GUEST2 | navigation → speaker_doa → speech | 导航到客厅 + 面向说话人 + 语音对话 |
| INTRODUCE_GUESTS | speaker_doa → speech | 面向说话人 + 语音播报 |
| REQUEST_GUEST2_BAG | speaker_doa → speech → manipulation | 面向说话人 + 语音请求 + 等待放包 |
| FIND_HOST | vision → speaker_doa | 视觉搜索 + 声源定位 |
| FOLLOW_HOST | speaker_doa → navigation | 声源跟踪 + 导航跟随 |
| PLACE_BAG | manipulation | 放包到指定位置 |
| TASK_COMPLETE | speech | 语音播报 |

## ROS 话题接口

> **所有话题名称和指令常量集中在 `sub_modules/topic_names.py` 中定义，修改话题名只需改该文件。**

### 话题总览

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/doorbell/detected` | std_msgs/String (JSON) | 门铃节点 → Task1 | 门铃检测信号 (asr_tts/doorbell_node.py 发布) |
| `/navigation/command` | std_msgs/String | Task1 → 导航节点 | 导航指令 |
| `/navigation/result` | std_msgs/String (JSON) | 导航节点 → Task1 | 导航结果 |
| `/vision/command` | std_msgs/String | Task1 → 视觉节点 | 视觉识别指令 |
| `/vision/result` | std_msgs/String (JSON) | 视觉节点 → Task1 | 视觉识别结果 |
| `/manipulation/command` | std_msgs/String | Task1 → 操作节点 | 手臂/夹爪/托盘指令 |
| `/manipulation/result` | std_msgs/String (JSON) | 操作节点 → Task1 | 操作结果 |
| `/speaker_doa/enroll` | std_msgs/String (JSON) | Task1 → 声纹节点 | 声纹录入指令 |
| `/speaker_doa/command` | std_msgs/String | Task1 → 声纹节点 | 声源定位/跟踪指令 |
| `/speaker_doa/result` | std_msgs/String (JSON) | 声纹节点 → Task1 | 声源定位/跟踪结果 |
| `/tts` | std_msgs/String | Task1 → TTS 节点 | TTS 播报文本 |
| `/asr` | std_msgs/String | ASR 节点 → Task1 | ASR 识别结果 |
| `/tts/playing` | std_msgs/Bool | TTS 节点 → ASR 节点 | TTS 播放状态 (回声消除) |
| `/llm/request` | std_msgs/String (JSON) | Task1 → LLM 节点 | LLM 信息提取请求 |
| `/llm/response` | std_msgs/String (JSON) | LLM 节点 → Task1 | LLM 提取结果 |
| `/task1/start` | std_msgs/Empty | 外部 → Task1 | 启动任务 |
| `/task1/abort` | std_msgs/Empty | 外部 → Task1 | 中止任务 |
| `/task1/status` | std_msgs/String | Task1 → 外部 | 任务状态播报 |
| `/task1/result` | std_msgs/String (JSON) | Task1 → 外部 | 任务最终结果 |

### 通用结果格式

所有 `*/result` 话题返回 JSON 字符串，格式约定：

```json
// 成功
{"status": "success", "data": {...}}

// 失败
{"status": "failed", "error": "error message"}
```

如果外部节点返回非 JSON 字符串 (如 `"done"`)，将被解析为：

```json
{"status": "success", "raw": "done"}
```

### 门铃接口

**信号话题**: `/doorbell/detected` (std_msgs/String, JSON)

门铃节点 (`asr_tts/doorbell_node.py`) 检测到门铃声时，发布 JSON 消息到该话题。

**消息格式**:
```json
{"detected": true, "label": "Doorbell", "probability": 0.85, "timestamp": 1716234567.89}
```

Task1 订阅该话题，收到 `detected: true` 即表示有人到达门口。非 JSON 消息也会被接受（降级处理）。

超时常量: `DOORBELL_TIMEOUT` = 300 秒

### 导航接口

**指令话题**: `/navigation/command` (std_msgs/String)

| 指令常量 | 值 | 说明 |
|---------|---|------|
| `NAV_CMD_GO_TO_DOOR` | `"导航到门口"` | 导航到门口接客位置 |
| `NAV_CMD_GO_TO_LIVING_ROOM` | `"导航到客厅"` | 导航到客厅 |
| `NAV_CMD_GO_TO_START` | `"导航到起点"` | 返回起始位置 |
| `NAV_CMD_FOLLOW_PERSON` | `"跟随人物"` | 跟随前方人物 |
| `NAV_CMD_TURN_TO_ANGLE` | `"转向角度"` | 原地转向指定角度 (指令格式: `"转向角度 45"`) |

**结果话题**: `/navigation/result` (JSON)

```json
{"status": "success", "data": {"destination": "door"}}
{"status": "failed", "error": "obstacle_blocked"}
```

### 视觉接口

**指令话题**: `/vision/command` (std_msgs/String)

| 指令常量 | 值 | 说明 |
|---------|---|------|
| `VISION_CMD_DESCRIBE_PERSON` | `"识别人物外貌"` | 识别当前人物外貌特征 |
| `VISION_CMD_FIND_HOST` | `"寻找host"` | 搜索 host |
| `VISION_CMD_TRACK_PERSON` | `"跟踪人物"` | 视觉跟踪人物 |

**结果话题**: `/vision/result` (JSON)

```json
{"status": "success", "data": {"clothing": "red jacket", "position": "seat 1"}}
{"status": "success", "data": {"location": "living_room"}}
```

### 操作接口

**指令话题**: `/manipulation/command` (std_msgs/String)

| 指令常量 | 值 | 说明 |
|---------|---|------|
| `MANIP_CMD_POINT_SEAT` | `"指向空座"` | 手臂指向空座位 |
| `MANIP_CMD_WAIT_FOR_BAG` | `"等待放包"` | 等待客人将包放到托盘 |
| `MANIP_CMD_PLACE_BAG` | `"放包到指定位置"` | 将包放到指定位置 |

**结果话题**: `/manipulation/result` (JSON)

```json
{"status": "success", "data": {}}
{"status": "failed", "error": "no_bag_detected"}
```

### 声纹 + 声源定位接口

**录入话题**: `/speaker_doa/enroll` (JSON)

```json
{"command": "enroll", "speaker_name": "guest1_Alice", "duration": 5.0}
```

**指令话题**: `/speaker_doa/command` (std_msgs/String)

| 指令常量 | 值 | 说明 |
|---------|---|------|
| `SPEAKER_DOA_CMD_LOCATE` | `"声源定位"` | 定位声源方向 |
| `SPEAKER_DOA_CMD_TRACK` | `"声源跟踪"` | 持续跟踪声源 |
| `SPEAKER_DOA_CMD_FACE_SPEAKER` | `"面向说话人"` | 面向当前说话人 (对话状态自动调用) |

**结果话题**: `/speaker_doa/result` (JSON)

```json
{"status": "success", "data": {"angle_deg": 45, "speaker_name": "host"}}
{"status": "success", "data": {"destination": "storage_room"}}
```

**面向说话人机制**: 在对话状态 (ASK_GUEST1_INFO, PICK_UP_GUEST2, DESCRIBE_GUEST1, SEAT_GUEST2, INTRODUCE_GUESTS, REQUEST_GUEST2_BAG) 执行时, SpeakerDOAModule 会:

1. 订阅 `/speaker_doa/result`, 等待 `status == "recognized"` 的消息
2. 从消息中提取 `angle_deg` (声源到达角)
3. 通过 `/navigation/command` 发送 `"转向角度 <angle>"` 指令
4. 等待 `/navigation/result` 确认转向完成

超时常量: `SPEAKER_DOA_FACE_TIMEOUT = 10.0` (秒, 超时后跳过转向, 继续对话)

### LLM 接口

**请求话题**: `/llm/request` (JSON)

```json
{
  "request_id": "uuid-string",
  "task": "extract_guest_info",
  "text": "My name is Alice and I'd like some orange juice",
  "context": {"role": "guest1"}
}
```

支持的 `task` 类型：

- `extract_guest_info` — 同时提取姓名和饮品
- `extract_name` — 仅提取姓名
- `extract_drink` — 仅提取饮品

**响应话题**: `/llm/response` (JSON)

```json
{
  "request_id": "uuid-string",
  "status": "success",
  "result": {"name": "Alice", "drink": "orange juice"},
  "error": null
}
```

### 语音接口

| 话题 | 类型 | 说明 |
|------|------|------|
| `/tts` | std_msgs/String | TTS 播报文本 |
| `/asr` | std_msgs/String | ASR 识别结果 |
| `/tts/playing` | std_msgs/Bool | TTS 播放状态 (用于回声消除) |

## 超时配置

| 模块 | 常量 | 默认值 (秒) | 说明 |
|------|------|-----------|------|
| 导航 | `NAV_TIMEOUT` | 120 | 一般导航超时 |
| 视觉 | `VISION_TIMEOUT` | 120 | 视觉识别超时 |
| 操作 | `MANIP_TIMEOUT` | 60 | 操作执行超时 |
| 声纹 | `SPEAKER_DOA_TIMEOUT` | 60 | 声源定位超时 |

> 特定状态可覆盖默认超时，如 `FOLLOW_HOST` 导航使用 180s，`WAIT_FOR_BAG` 操作使用 120s。

## ROS 降级策略

所有子模块通过 `ROSTopicBridge` 与外部 ROS 节点通信。当 ROS 不可用时（roscore 未启动、话题无响应），系统自动降级为模拟行为：

1. **门铃**: 控制台等待用户按 Enter 模拟门铃
2. **导航**: 模拟移动 (time.sleep)，返回成功
3. **视觉**: 返回模拟数据 (如 "红色外套, 1号座位")
4. **操作**: 模拟执行 (time.sleep)，返回成功
5. **声纹**: 跳过录入，模拟定位结果
6. **语音**: 使用 Mock 后端 (控制台输入输出)
7. **LLM**: 使用规则匹配 (MockLLMInterface)

## 快速启动

### 1. 全自动离线测试 (推荐, 无需任何交互)

```bash
python task1_receptionist/task1_controller.py \
    --script "My name is Alice,I would like some orange juice,My name is Bob,I would like some cola" \
    --auto-doorbell --delay 0.3
```

- `--script`: 预置 ASR 回复, 按顺序返回, 跳过真实语音
- `--auto-doorbell`: 自动模拟门铃, 无需按 Enter
- `--delay 0.3`: 缩短状态间延迟, 加快测试

### 2. 交互模拟模式 (无 ROS, 手动输入)

```bash
python task1_receptionist/task1_controller.py
```

所有模块使用 Mock 后端，LLM 信息提取使用规则匹配。门铃需要按 Enter 模拟，ASR 需要手动输入。

### 3. 脚本测试模式 (预置 ASR 回复, 但门铃需手动)

```bash
python task1_receptionist/task1_controller.py --script "Alice,orange juice,Bob,coffee"
```

### 3. ROS 模式 (真实语音 + LLM)

需要先启动相关 ROS 节点：

```bash
# 终端 1: roscore
roscore

# 终端 2: ASR + TTS 节点
roslaunch asr_tts speech.launch

# 终端 3: LLM 节点
rosrun asr_tts llm_node.py

# 终端 4: 导航节点 (需实现 /navigation/command + /navigation/result)
rosrun navigation nav_server.py

# 终端 5: 视觉节点 (需实现 /vision/command + /vision/result)
rosrun vision vision_server.py

# 终端 6: 操作节点 (需实现 /manipulation/command + /manipulation/result)
rosrun manipulation manip_server.py

# 终端 7: 声纹节点 (需实现 /speaker_doa/*)
rosrun speaker_doa speaker_doa_node.py

# 终端 8: Task1 控制器
python task1_receptionist/task1_controller.py --ros --llm-ros
```

### 4. 仅 ROS 语音 + Mock LLM

```bash
python task1_receptionist/task1_controller.py --ros
```

## 对接外部节点指南

对接外部 ROS 节点时，只需：

1. **修改话题名**: 编辑 `sub_modules/topic_names.py` 中的常量
2. **实现结果话题**: 外部节点订阅 `*/command` 话题，处理指令后发布结果到 `*/result` 话题
3. **遵循结果格式**: 返回 JSON `{"status": "success", "data": {...}}` 或 `{"status": "failed", "error": "msg"}`
4. **超时处理**: 如果外部节点处理时间较长，调整 `topic_names.py` 中的超时常量

### 最小对接示例 (Python)

```python
#!/usr/bin/env python
import rospy
import json
from std_msgs.msg import String

def on_command(msg):
    command = msg.data
    rospy.loginfo(f"收到指令: {command}")

    # 处理指令...
    result = {"status": "success", "data": {}}

    pub.publish(String(data=json.dumps(result)))

rospy.init_node("nav_server")
pub = rospy.Publisher("/navigation/result", String, queue_size=10)
sub = rospy.Subscriber("/navigation/command", String, on_command)
rospy.spin()
```

## 对外建议

以下是对项目其他模块的改进建议（不在本次修改范围内）：

1. **`brain/llm_client.py`**：当前 `LLMClient` 的 `get_decision()` 方法硬编码了 `RobotDecision` 的解析逻辑，建议增加通用的 `chat_and_parse()` 方法，支持自定义 JSON schema 解析，便于 `llm_node` 复用。
2. **`brain/prompts.py`**：建议将 Task1 的信息提取 prompt 也纳入统一管理，避免 `llm_interface.py` 和 `llm_node.py` 中重复定义 prompt。
3. **`config.py`**：建议增加 LLM 节点相关配置项（如 `LLM_REQUEST_TIMEOUT`、`LLM_MAX_CONCURRENT`），方便部署时调参。
4. **`src/asr_tts/`**：建议在 `CMakeLists.txt` 中将 `llm_node.py` 注册为 ROS 节点，以便 `rosrun asr_tts llm_node.py` 可直接使用。
5. **消息类型**：当前使用 `std_msgs/String` 传递 JSON，建议未来定义自定义 ROS 消息类型（如 `LLMRequest.msg`、`LLMResponse.msg`），提升类型安全性和可读性。
