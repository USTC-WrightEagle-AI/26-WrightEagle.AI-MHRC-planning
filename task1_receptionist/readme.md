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
    ├── base_module.py           — ROSTopicBridge + DoorbellModule + NavigationModule + SpeechModule + VisionModule + ManipulationModule
    ├── speech_interaction.py    — SpeechInterface (TTS + ASR)，Mock/ROS/Scripted 三后端
    └── llm_interface.py         — LLMInterface (信息提取)，Mock/ROS 双后端
```

## 状态机流程 (15 个主流程执行状态；状态 9 保留兼容但默认跳过)

```
IDLE
  │
  ▼
[1]  WAIT_FOR_DOORBELL_1 → 等待门铃声 (guest1 到达)
[2]  GO_TO_DOOR          → 移动到门口
[3]  ASK_GUEST1_INFO     → 询问 guest1 姓名和饮料，并缓存 guest1 外貌特征
[4]  GUIDE_GUEST1        → 带 guest1 去客厅
[5]  POINT_EMPTY_SEAT    → 识别最近空座，导航到面向空座 1.5 米处，再指向空座请 guest1 入座
[6]  RETURN_TO_START     → 返回起点
[7]  WAIT_FOR_DOORBELL_2 → 等待门铃声 (guest2 到达)
[8]  PICK_UP_GUEST2      → 到门口接 guest2 (导航 + ASR + LLM 提取姓名和饮品 + 缓存 guest2 外貌)
[9]  DESCRIBE_GUEST1     → 保留兼容；主流程直接跳过
[10] SEAT_GUEST2         → 带 guest2 到客厅，启动导航后描述 guest1 外貌，再识别最近空座，导航到 1.5 米接近点并指向
[11] INTRODUCE_GUESTS    → 按衣着识别并面向对应客人，相互介绍两位客人
[12] REQUEST_GUEST2_BAG  → 请求 guest2 递包给机器人，导航到 0.7 米接近点后接包
[13] FIND_HOST           → 导航到 host 交互点，请 host 站到前方并确认人物
[14] FOLLOW_HOST         → 告知携带行李，请求指引并跟随 host
[15] PLACE_BAG           → 到达后将包交给 host
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
| ASK_GUEST1_INFO | speech → vision | 语音对话 + LLM 提取，并缓存 guest1 外貌 |
| GUIDE_GUEST1 | speech → navigation | 提示 guest1 跟随并导航到客厅 |
| POINT_EMPTY_SEAT | navigation → manipulation | 识别空座并导航到 1.5 米接近点 + 播报入座提示并手臂指向 |
| RETURN_TO_START | navigation | 返回起点 |
| WAIT_FOR_DOORBELL_2 | doorbell | 等待 guest2 门铃 |
| PICK_UP_GUEST2 | navigation → speech → vision | 导航到门口 + 语音对话 + 缓存 guest2 外貌 |
| DESCRIBE_GUEST1 | speech | 兼容旧入口；主流程从 PICK_UP_GUEST2 直接进入 SEAT_GUEST2 |
| SEAT_GUEST2 | navigation → manipulation | 启动客厅导航后播报已缓存的 guest1 外貌 + 识别空座并导航到 1.5 米接近点 + 播报入座提示并手臂指向 |
| INTRODUCE_GUESTS | speech | 按 guest1/guest2 衣着签名匹配人物，先对准并停止视觉跟随后再播报介绍 |
| REQUEST_GUEST2_BAG | speech → navigation → manipulation | 语音请求 + 识别接包目标并导航到 0.7 米接近点 + 重新识别目标后伸手接包 |
| FIND_HOST | navigation → speech → vision | 到 host 交互点 + 自然请求 host 到前方 + 视觉确认前方人物 |
| FOLLOW_HOST | speech → navigation | 告知携带行李并请求指引 + 导航跟随 |
| PLACE_BAG | speech → manipulation | 到达提示 + 将包交给 host |
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
| `/tts` | std_msgs/String | Task1 → TTS 节点 | TTS 播报文本 |
| `/asr` | std_msgs/String | ASR 节点 → Task1 | ASR 识别结果 |
| `/tts/playing` | std_msgs/String | TTS 节点 → ASR/Task1 | TTS 播放状态，取值 `playing` / `idle` |
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

`GO_TO_DOOR`、`GUIDE_GUEST1`、`POINT_EMPTY_SEAT`、`RETURN_TO_START`、
`PICK_UP_GUEST2` 和 `SEAT_GUEST2` 不再依赖外部 `/navigation/command`
服务器；这些状态会直接调用
`task1_receptionist/navigation/set_nav_goal/set_nav_goal.py`，把固定 `map`
坐标或动态计算出来的 `map` 坐标发送给 `/move_base` action。

固定点位配置方式：

```python
# 推荐方式: 写入代码，之后正常启动即可
# task1_receptionist/sub_modules/base_module.py
DEFAULT_NAV_GOALS = {
    "door": [-2.784, 0.768, 173.131],
    "start": [-1.432, -0.078, 169.610],
    "living_room": [-0.274, -1.151, -2.771],
    "host_interaction": [-1.432, -0.078, 169.610],
}
```

```bash
# 临时覆盖 1: 启动参数
python task1_receptionist/task1_controller.py --door-goal -2.784 0.768 173.131
python task1_receptionist/task1_controller.py --start-goal -1.432 -0.078 169.610
python task1_receptionist/task1_controller.py --living-room-goal -0.274 -1.151 -2.771
python task1_receptionist/task1_controller.py --host-interaction-goal -1.432 -0.078 169.610

# 临时覆盖 2: 环境变量，格式为 x y yaw_deg
export TASK1_DOOR_GOAL="-2.784 0.768 173.131"
export TASK1_START_GOAL="-1.432 -0.078 169.610"
export TASK1_LIVING_ROOM_GOAL="-0.274 -1.151 -2.771"
export TASK1_HOST_INTERACTION_GOAL="-1.432 -0.078 169.610"
python task1_receptionist/task1_controller.py
```

可选环境变量：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `TASK1_NAV_MAP_FRAME` | `map` | 目标坐标系 |
| `TASK1_NAV_BASE_FRAME` | `base_link` | 读取当前机器人位姿使用的 TF 机器人坐标系 |
| `TASK1_MOVE_BASE_ACTION` | `/move_base` | move_base action 名称 |
| `TASK1_NAV_SERVER_TIMEOUT` | `10.0` | 等待 action server 秒数 |
| `TASK1_NAV_GOAL_TIMEOUT` | `120.0` | 等待到达固定点位结果秒数；设为 `0` 表示一直等 |
| `TASK1_NAV_GOAL_NO_WAIT` | `false` | 只发送固定点位目标，不等待结果 |
| `TASK1_EMPTY_SEAT_APPROACH_DISTANCE` | `1.5` | 空座接近点距离空座的距离，单位米 |
| `TASK1_EMPTY_SEAT_LEFTBASE_FRAME` | `left_arm_base_link` | `camera_middle_to_leftbase.txt` 外参输出对应的 TF frame |
| `TASK1_EMPTY_SEAT_CAMERA_TO_LEFTBASE` | `task1_receptionist/object_search/camera_middle_to_leftbase.txt` | 相机坐标转 leftbase 坐标的 4x4 外参 |
| `TASK1_EMPTY_SEAT_TF_FRAMES` | 空 | 旧兜底路径使用的空座源 TF frame 候选；为空时按 JSON frame 自动尝试，`leftbase/left_arm_base` 会额外尝试 `left_arm_base_link` |
| `TASK1_EMPTY_SEAT_TF_TIMEOUT` | `3.0` | 等待空座坐标 TF 转换的秒数 |
| `TASK1_EMPTY_SEAT_MAX_FRAMES` | `30` | 空座识别最多处理帧数；识别到空座会提前退出 |
| `TASK1_OBJECT_SEARCH_MAX_FRAMES` | `120` | 接包手腕等通用 object_search 识别最多处理帧数 |
| `TASK1_OBJECT_SEARCH_SKIP_VISION` | `false` | 跳过相机刷新，复用已有 `latest_*.json` |
| `TASK1_OBJECT_SEARCH_PREFER_CACHE` | `true` | 优先复用 `start_all.sh` 启动的后台视觉缓存；缓存不存在时才回退单次识别脚本 |
| `TASK1_OBJECT_SEARCH_CACHE_MAX_AGE_SEC` | `3.0` | `latest_*.json` 或 `latest_vision_cache_status.json` 超过该秒数视为过期 |
| `TASK1_ENABLE_VISION_CACHE` | `true` | `start_all.sh` 是否启动后台视觉缓存；也可用 `bash start_all.sh --no-vision-cache` 跳过 |
| `TASK1_VISION_CACHE_DISPLAY` | `false` | `start_all.sh` 是否为后台视觉缓存打开 OpenCV 可视化窗口；也可用 `--vision-display` / `--no-vision-display` |
| `TASK1_VISION_CACHE_DISPLAY_DEPTH` | `true` | 是否同时显示深度伪彩色窗口 |
| `TASK1_VISION_CACHE_DISPLAY_SCALE` | `1.0` | 可视化窗口缩放比例 |
| `TASK1_VISION_CACHE_FRAME_TIMEOUT_MS` | `500` | RealSense 等帧超时；调小可避免可视化窗口在无帧时长时间不刷新 |
| `TASK1_VISION_CACHE_ARGS` | 空 | 透传给后台视觉缓存进程的额外参数，例如 `--disable-bag --empty-every 0.5` |
| `TASK1_VISION_CACHE_WATCHDOG` | `true` | `start_all.sh` 是否用 watchdog 方式启动视觉缓存；子进程退出后自动重启 |
| `TASK1_VISION_CACHE_RESTART_DELAY_SEC` | `3` | watchdog 重启视觉缓存前等待秒数 |
| `TASK1_VISION_CACHE_PEOPLE_EVERY` | `0.5` | 后台人物检测刷新间隔，门口目光跟随依赖该缓存 |
| `TASK1_VISION_CACHE_CLOTHING` | `true` | 后台人物缓存是否为 `latest_people.json` 补充衣着字段 |
| `TASK1_HAND_SEARCH_CARRY_OBJECT_CLASSES` | `umbrella,bottle,wine glass,cup,bowl,book,cell phone,laptop,remote,sports ball,teddy bear` | 接包识别第三优先级“提着东西的手”的可携带物体类别；逗号分隔可保留 `cell phone` 这类多词类别 |
| `TASK1_HAND_SEARCH_BAG_CONF` | `0.10` | 接包阶段包/可携带物体检测阈值 |
| `TASK1_HAND_SEARCH_KEYPOINT_CONF` | `0.20` | 接包阶段手腕关键点阈值 |
| `TASK1_HAND_SEARCH_DEPTH_PERCENTILE` | `20.0` | 手腕关键点附近取深度的百分位；低于 50 可减少背景深度把手腕坐标拉远 |
| `TASK1_HAND_SEARCH_MAX_HAND_BAG_PIXEL_DISTANCE` | `90.0` | 手腕和包/物体扩展框之间允许的最大像素距离 |
| `TASK1_HANDOVER_RECOGNITION_DELAY_SEC` | `5.0` | 状态 12 语音提示客人递包后，等待多少秒再开始接包识别 |
| `TASK1_HANDOVER_APPROACH_DISTANCE` | `0.7` | 状态 12 底盘先导航到距离接包目标这么远的位置，单位米 |
| `TASK1_HANDOVER_CACHE_WAIT_SEC` | `6.0` | 到接包接近点后，等待后台刷新新鲜接包目标的最长秒数 |
| `TASK1_HANDOVER_REQUIRE_APPROACH_NAV` | `true` | 未到达接包接近点时是否禁止左臂伸手接包 |
| `TASK1_HANDOVER_HAND_STANDOFF` | `0.14` | 接包动作离目标点保留的距离，单位米；调大可让手臂离客人的手更远 |
| `TASK1_HANDOVER_MAX_ARM_REACH` | `0.70` | 接包机械臂最终目标距离 `left_arm_base_link` 的最大半径，超出时自动沿同方向缩回 |
| `TASK1_CADE_VISION_SRC` | `/home/nvidia/Desktop/task3/cade_ws/src/cade_vision/src` | Task3 `cade_vision` Python 源码路径，用于复用衣着分析逻辑 |
| `TASK1_CLOTH_SEG_MODEL` | Task3 `yolov8s-seg-fashionpedia-best.pt` | 衣着分割模型路径 |
| `TASK1_CLOTH_DETECT_MODEL` | 空 | 可选衣着检测模型路径；为空时只用分割模型 |
| `TASK1_CLOTH_CONF` | `0.25` | 衣着模型置信度阈值 |
| `TASK1_CLOTH_EVERY` | `2.0` | 衣着模型最小刷新间隔，间隔内复用最近一次衣着结果 |
| `TASK1_CLOTH_CACHE_MATCH_IOU` | `0.2` | 复用衣着结果时当前人框和缓存人框的最小 IoU |
| `TASK1_VISION_CACHE_FRAME_ERROR_RESTART_THRESHOLD` | `3` | 连续相机取帧/align 异常多少次后自动重启 RealSense pipeline |
| `TASK1_GUEST_APPEARANCE_PEOPLE_JSON` | `task1_receptionist/object_search/latest_people.json` | 第 3 状态读取 guest1 外貌字段的人物缓存 JSON |
| `TASK1_GUEST_APPEARANCE_CACHE_MAX_AGE_SEC` | `60.0` | 第 3 状态接受人物缓存的最大年龄；设为 `0` 表示不检查时间戳 |
| `TASK1_GUEST_APPEARANCE_ACTIVE_QUERY` | `false` | 是否在人物缓存没有衣着字段时主动向 `/vision/command` 发送 `"识别人物外貌"`；默认关闭，避免扰动已启动的相机链路 |
| `TASK1_PEOPLE_CACHE_EVERY` | `0.3` | FIND_HOST/FOLLOW_HOST 自动拉起轻量人物缓存时的人物检测间隔 |
| `TASK1_PEOPLE_SEARCH_CONF` | `0.2` | 人物检测置信度阈值；该值同时影响后台视觉缓存和轻量人物缓存 |
| `TASK1_PEOPLE_CACHE_ARGS` | 空 | FIND_HOST/FOLLOW_HOST 自动拉起轻量人物缓存时额外传给 `run_vision_cache.sh` 的参数 |
| `TASK1_HOST_FIND_WAIT_SEC` | `60.0` | FIND_HOST 到达交互点后等待 host 站到机器人前方的最长秒数 |
| `TASK1_HOST_FIND_CENTER_RATIO` | `0.45` | FIND_HOST 判定前方人物时允许的水平偏移比例 |
| `TASK1_HOST_FIND_MIN_DISTANCE_M` | `0.3` | FIND_HOST 接受人物的最近距离；设为 `0` 表示不限制 |
| `TASK1_HOST_FIND_MAX_DISTANCE_M` | `4.5` | FIND_HOST 接受人物的最远距离；设为 `0` 表示不限制 |
| `TASK1_DOOR_GAZE_TRACK_ENABLED` | `true` | 门口询问客人姓名/饮料时是否根据 `latest_people.json` 自动转向，让最近人物保持在画面中间 |
| `TASK1_DOOR_GAZE_CENTER_DEADBAND_RATIO` | `0.12` | 人物中心距离画面中心小于该比例时不转动，避免抖动 |
| `TASK1_DOOR_GAZE_MAX_ANGULAR_SPEED_DEG` | `18.0` | 门口目光跟随最大旋转角速度，人物在左侧时为正、右侧时为负 |
| `TASK1_DOOR_GAZE_CACHE_MAX_AGE_SEC` | `2.5` | `latest_people.json` 超过该秒数未刷新时停止转动 |
| `TASK1_DOOR_GAZE_DRY_RUN` | `false` | 只打印门口目光跟随判断，不发布底盘速度 |
| `TASK1_GUEST_GAZE_MATCH_MIN_SCORE` | `3.0` | 第 11 状态按衣着识别 guest1/guest2 时接受候选人的最低匹配分 |
| `TASK1_INTRO_GAZE_SETTLE_SEC` | `2.0` | 第 11 状态每句介绍前先面向对应 guest 的等待时间；等待后会停止视觉跟随再播报 |
| `TASK1_INTRO_SPEECH_HOLD_SEC` | `4.0` | 第 11 状态每句介绍播报后保持当前朝向的额外等待时间，避免立刻转向下一位客人 |
| `TASK1_INTRO_GAZE_SEAT_FALLBACK_ENABLED` | `true` | 第 11 状态按衣着找不到目标客人时，是否按之前记录的座位 map 坐标兜底转向 |
| `TASK1_INTRO_GAZE_SEAT_DEADBAND_DEG` | `8.0` | 按座位坐标兜底转向时，小于该 yaw 误差就认为已经面向座位 |
| `TASK1_INTRO_GAZE_TF_TIMEOUT` | `0.35` | 第 11 状态读取 `map -> base_link` TF 的等待时间 |
| `TASK1_HOST_FOLLOW_MODE` | `nav_goal` | FOLLOW_HOST 跟随模式；`nav_goal` 按视觉 3D 坐标刷新 `move_base` 目标，`velocity` 回退到旧的直接速度控制 |
| `TASK1_HOST_FOLLOW_TARGET_DISTANCE_M` | `1.2` | FOLLOW_HOST 跟随 host 时保持的目标距离 |
| `TASK1_HOST_FOLLOW_NAV_UPDATE_DISTANCE_M` | `0.6` | `nav_goal` 模式下，目标站位点变化超过该距离才刷新 `move_base` goal |
| `TASK1_HOST_FOLLOW_NAV_UPDATE_INTERVAL_SEC` | `1.0` | `nav_goal` 模式下，两次刷新导航目标的最小时间间隔 |
| `TASK1_HOST_FOLLOW_INITIAL_MATCH_DISTANCE_M` | `1.0` | 没有或丢失 `track_id` 时，按最后一次视觉坐标重新锁定 host 的最大距离 |
| `TASK1_HOST_FOLLOW_MAX_LINEAR_SPEED` | `0.25` | FOLLOW_HOST 最大前进速度，单位 m/s |
| `TASK1_HOST_FOLLOW_MAX_ANGULAR_SPEED_DEG` | `22.0` | FOLLOW_HOST 最大转向角速度；人物在左侧为正，右侧为负 |
| `TASK1_HOST_FOLLOW_WAIT_START_SIGNAL_SEC` | `8.0` | 等待 host 发出开始带路信号的秒数 |
| `TASK1_HOST_FOLLOW_AUTO_START` | `false` | 等待开始信号超时后是否自动开始跟随；比赛默认应等待 host 信号 |
| `TASK1_HOST_FOLLOW_START_ON_PERSON` | `true` | FOLLOW_HOST 等待开始信号期间，看到前方人物时是否直接开始跟随 |
| `TASK1_HOST_FOLLOW_START_CENTER_RATIO` | `0.55` | 允许直接开始跟随的人物水平偏移比例 |
| `TASK1_HOST_FOLLOW_LOST_TIMEOUT_SEC` | `30.0` | 连续找不到人物时先原地等待该秒数；设为 `0` 表示只受 FOLLOW_HOST 总超时限制 |
| `TASK1_HOST_FOLLOW_TIMEOUT_SEC` | `180.0` | FOLLOW_HOST 最大跟随时间 |
| `TASK1_HOST_FOLLOW_LOG_ASR` | `true` | 打印 FOLLOW_HOST 收到的 ASR 文本及是否匹配 follow/stop，方便确认 `stop` 是否传到模块 |
| `TASK1_HOST_FOLLOW_DRY_RUN` | `false` | 只打印 FOLLOW_HOST 控制量，不发布底盘速度 |
| `TASK1_EMPTY_SEAT_SEARCH_ARGS` | 空 | 透传给 `run_object_search.sh` 的额外参数 |
| `TASK1_SEAT_PROMPT` | `Please take a seat here.` | 指向座位时同步播报的入座提示 |
| `TASK1_TTS_START_TIMEOUT` | `8.0` | `speech.say()` 等待 `/tts/playing=playing` 的秒数 |
| `TASK1_TTS_WAIT_TIMEOUT` | `20.0` | `speech.say()` 等待 `/tts/playing=idle` 的秒数；设为 `0` 表示只发布不等待 |
| `TASK1_TTS_MIN_SAY_DURATION_SEC` | `0.0` | ROS TTS 发布后的额外最短阻塞时间；默认不额外等待，现场需要慢速播报保护时再调大 |
| `TASK1_PRE_NAV_ADJUST_LINEAR_SPEED` | `0.15` | 导航前底盘微调的平移速度，单位 m/s |
| `TASK1_PRE_NAV_ADJUST_ANGULAR_SPEED_DEG` | `25` | 导航前底盘微调的旋转速度，单位 deg/s |
| `TASK1_PRE_NAV_ADJUST_WAIT_SUBSCRIBER` | `3.0` | 等待底盘速度话题订阅者的秒数 |
| `TASK1_PRE_NAV_ADJUST_DRY_RUN` | `false` | 只打印导航前微调命令，不真正发布底盘速度 |

空座导航链路：

1. `start_all.sh` 默认启动 `vision_cache_daemon.py`，常驻打开一次 D455 并持续刷新 `latest_empty_seat.json`。
2. `POINT_EMPTY_SEAT` 优先读取后台缓存中的 `best_empty_seat`；如果缓存没有运行或已过期，才回退运行 `run_object_search.sh`。
3. 导航模块优先使用 `camera_xyz_m`，按 `person_tracker` 的逻辑用 `camera_middle_to_leftbase.txt` 转到 `left_arm_base_link`。
4. 再通过 TF 转成 `base_link` 下的空座相对坐标，结合当前 `map -> base_link` 位姿手算空座 `map` 坐标。
5. 根据当前机器人和空座的 `map` 坐标，计算一个距离空座 `TASK1_EMPTY_SEAT_APPROACH_DISTANCE` 米、朝向空座的目标点。
6. 目标点通过 `set_nav_goal.py` 发送到 `/move_base`。
7. 到达后，操作模块再次读取新鲜空座缓存，然后播报 `Please take a seat here.` 并调用 `point_empty_seat_arm.py`，保证左臂指向使用的是机器人移动后的本地坐标。

导航前底盘微调表在 `task1_receptionist/sub_modules/base_module.py` 的
`DEFAULT_PRE_NAV_ADJUSTMENTS`。每个导航步骤可单独设置 `enabled`、
`forward_m`、`left_m`、`rotate_deg` 和 `order`。`forward_m` 正数前进、
负数后退；`left_m` 正数左移、负数右移；`rotate_deg` 正数逆时针、
负数顺时针。

FIND_HOST 会先导航到 `host_interaction` 固定点，再自然请求 host 站到机器人前方。
视觉模块读取 `latest_people.json`，把画面中接近中心且距离合理的最近人物确认为 host。
如果 host 交互点导航失败，状态机会记录失败但继续在当前位置请求 host 到前方并尝试确认；
即使最终没有确认到 host，也会继续进入 FOLLOW_HOST，而不是直接停止整条流程。

FOLLOW_HOST 不再依赖旧 `/navigation/command` 桥接。默认 `TASK1_HOST_FOLLOW_MODE=nav_goal`：
后台视觉缓存会在 `latest_people.json.people` 里写入 `track_id`、`position_3d`、`camera_xyz_m` 和 `leftbase_xyz_m`；
FOLLOW_HOST 先锁定 host 的 `track_id`，每轮把该人的视觉 3D 坐标转换到 `map`，计算一个距离 host
`TASK1_HOST_FOLLOW_TARGET_DISTANCE_M` 的站位点，并通过 `/move_base` 异步刷新导航目标。若 `track_id` 短暂丢失，
会按最后一次相机坐标在当前候选人中重新锁定最近的人。需要回退旧行为时设置 `TASK1_HOST_FOLLOW_MODE=velocity`。
机器人对 host 只做自然引导，不播报固定口令；内部会识别常见的开始带路和到达停止短句。
只有 `--interactive` 调试模式下，等待开始阶段按 Enter 才表示开始跟随，跟随阶段按 Enter 才表示已经到达终点并进入下一状态。
如果 ASR 没有让机器人停下，先看控制台是否打印 `收到 stop 信号`；没有打印通常表示 ASR 文本没有发布到 FOLLOW_HOST 或未匹配。

其它未落地的导航状态仍可保留旧的 command/result 话题桥接：

**指令话题**: `/navigation/command` (std_msgs/String)

| 指令常量 | 值 | 说明 |
|---------|---|------|
| `NAV_CMD_GO_TO_DOOR` | `"导航到门口"` | 旧桥接指令，状态 2/8 已改为直接调用 `/move_base` |
| `NAV_CMD_GO_TO_LIVING_ROOM` | `"导航到客厅"` | 旧桥接指令，状态 4/10 已改为直接调用 `/move_base` |
| `NAV_CMD_GO_TO_START` | `"导航到起点"` | 旧桥接指令，状态 6 已改为直接调用 `/move_base` |
| `NAV_CMD_FOLLOW_PERSON` | `"跟随人物"` | 跟随前方人物 |
| `NAV_CMD_TURN_TO_ANGLE` | `"转向角度"` | 原地转向指定角度 (指令格式: `"转向角度 45"`) |

**结果话题**: `/navigation/result` (JSON)

```json
{"status": "success", "data": {"destination": "door"}}
{"status": "failed", "error": "obstacle_blocked"}
```

### 视觉接口

第 3 状态的 guest1 外貌缓存、以及第 8 状态的 guest2 外貌缓存，默认只读取 `TASK1_GUEST_APPEARANCE_PEOPLE_JSON`
指向的 `latest_people.json`，不会发布 `/vision/command`，也不会重新启动相机。
`start_all.sh` 启动的 `vision_cache_daemon.py` 默认会复用 Task3 衣着分析逻辑，
在 `latest_people.json` 的 `best_person` 中补充 `clothing_summary`、
`cloth_items`、`cloth_color` 和 `cloth_type`。如果这些字段仍为 `unknown`，
第 10 状态会降级播报未获得可靠外貌。

第 11 状态介绍两位客人时，会把状态 3/8 保存的 `clothing_signature`
和当前 `latest_people.json.people` 中每个候选人的衣着字段做匹配；播报前先面向对应 guest，
停止视觉跟随后再说话。若缓存没有可用衣着签名，会降级使用当前 `best_person`。

随时检查当前人物衣着缓存：

```bash
task1_receptionist/object_search/check_people_clothing.py
task1_receptionist/object_search/check_people_clothing.py --watch 1
```

`start_all.sh` 默认会打开两个 OpenCV 调试窗口：

- `Task1 Vision Debug`: 彩色画面，叠加人物/衣着、空座、包、递包手腕检测框。
- `Task1 Vision Depth`: 深度伪彩色画面。

关闭窗口但保留 JSON 缓存：

```bash
bash start_all.sh --no-vision-display
# 或
TASK1_VISION_CACHE_DISPLAY=false bash start_all.sh
```

**指令话题**: `/vision/command` (std_msgs/String)

| 指令常量 | 值 | 说明 |
|---------|---|------|
| `VISION_CMD_DESCRIBE_PERSON` | `"识别人物外貌"` | 兼容旧接口；仅在 `TASK1_GUEST_APPEARANCE_ACTIVE_QUERY=true` 时用于主动请求外貌 |
| `VISION_CMD_FIND_HOST` | `"寻找host"` | 搜索 host |
| `VISION_CMD_TRACK_PERSON` | `"跟踪人物"` | 视觉跟踪人物 |

**结果话题**: `/vision/result` (JSON)

```json
{"status": "success", "data": {"clothing": "red jacket", "visual_attributes": ["red jacket"], "position": "near the door"}}
{"status": "success", "data": {"location": "living_room"}}
```

### 操作接口

Task1 操作模块直接调用 `task1_receptionist/object_search` 里的脚本，不再依赖外部操作节点：

- `POINT_EMPTY_SEAT` / `SEAT_GUEST2`: 导航模块先运行 `run_object_search.sh` 计算最近空座接近点并导航过去；操作模块到达后再次刷新空座结果，播报 `Please take a seat here.`，再运行 `point_empty_seat_arm.py` 指向空座
- `REQUEST_GUEST2_BAG`: 语音提示 guest2 递包后，导航模块等待 5 秒并识别接包目标，先把底盘导航到距离目标 0.7 米的位置；到位后操作模块重新等待新鲜接包目标，再运行 `approach_handover_arm.py` 接包；接包脚本会打开夹爪、发布机械臂目标、5 秒后关闭夹爪
- `PLACE_BAG`: 到达 host 后提示确认，输入 `y` 后打开左夹爪交包

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
| `/tts/playing` | std_msgs/String | TTS 播放状态，取值 `playing` / `idle` |

## 超时配置

| 模块 | 常量 | 默认值 (秒) | 说明 |
|------|------|-----------|------|
| 导航 | `NAV_TIMEOUT` | 120 | 一般导航超时 |
| 视觉 | `VISION_TIMEOUT` | 120 | 视觉识别超时 |
> 默认比赛入口为无人值守模式：状态失败会记录并继续，控制台 `Enter/y` 人工确认默认禁用。

## ROS 降级策略

默认启动会使用已实现的真实模块：语音优先连接 ROS TTS/ASR，操作直接调用 `object_search`。当部分外部 ROS 能力不可用时，相关模块会自动超时、补默认上下文并继续后续流程：

1. **门铃**: 等待 `/doorbell/detected`，超时后记录失败并继续
2. **导航**: 固定点位导航失败后记录失败并继续
3. **视觉**: 优先读取后台缓存，失败后写入 unknown/fallback 上下文
4. **操作**: 调用 `object_search` 空座识别、拿包手腕识别和机械臂脚本
5. **语音**: ROS 语音不可用时降级为非交互 Mock，返回默认姓名/饮品
6. **LLM**: 本地 LLM 不可用时降级为规则提取

需要人工调试时加 `--interactive`；语音等待 ASR、门铃等待和机械臂确认才会接受控制台输入。

## 快速启动

### 1. 默认实机流程

```bash
python task1_receptionist/task1_controller.py
```

默认会使用已实现的语音、空座指向、拿包手腕识别、接包和交包流程。建议先用 `bash start_all.sh` 启动后台视觉缓存；主流程会优先读取 `latest_empty_seat.json`、`latest_handover_hand.json`、`latest_bag.json` 和 `latest_people.json`。在门口询问 guest1/guest2 姓名和饮料时，机器人会持续读取 `latest_people.json`，让最近人物保持在画面中间；第 3 状态也只从该缓存读取 guest1 外貌字段，不会向视觉节点主动发命令。若代码刚更新过衣着缓存逻辑，需要重启已有 `vision_cache_daemon.py` 进程后新字段才会出现在 `latest_people.json`。不要同时另开 `run_object_search.sh`、`run_hand_search.sh` 或 `run_bag_search.sh`，同一台 D455 只能被一个进程占用。

默认入口等价于比赛无人值守模式：`TASK1_INTERACTIVE=0`、`TASK1_ASSUME_YES=1`、失败继续、超时不自停。若要恢复旧的失败即停行为，使用 `--stop-on-failure`；若要超过 `TASK1_MAX_DURATION_SEC` 后主动停机，使用 `--enforce-deadline`。

### 2. 测试模式

```bash
python task1_receptionist/task1_controller.py --test
```

每个状态开始时会提示：直接回车跳过该状态，输入 `y` 正常执行该状态。跳过时会自动补充必要上下文，方便测试某个已经实现的具体模块。

人工调试真实流程但不逐状态测试时，可以使用：

```bash
python task1_receptionist/task1_controller.py --interactive
```

### 3. 安全 dry-run 测试

```bash
python task1_receptionist/task1_controller.py \
    --test \
    --script "My name is Alice,I would like some orange juice,My name is Bob,I would like some cola" \
    --auto-doorbell --delay 0.3 \
    --object-search-dry-run --object-search-skip-vision
```

`--object-search-dry-run --object-search-skip-vision` 不打开 D455，不发布机械臂或夹爪指令，复用已有 `latest_*.json`。

### 4. 实机启动依赖

需要先启动相关 ROS 节点：

```bash
# 终端 1: roscore
roscore

# 终端 2: ASR + TTS 节点
roslaunch asr_tts speech.launch

# TTS 默认会读取 speech.launch 的 speaker_volume，当前为 100%。
# 如需调整:
roslaunch asr_tts speech.launch speaker_volume:=60

# 使用一键启动脚本时，默认同步 speech.launch；也可以用环境变量覆盖:
SPEAKER_VOLUME=60 ./start_all.sh

# 终端 3: 导航/视觉
# GO_TO_DOOR/GUIDE_GUEST1/RETURN_TO_START/PICK_UP_GUEST2/SEAT_GUEST2/FIND_HOST 直接使用 /move_base；
# POINT_EMPTY_SEAT/SEAT_GUEST2 的空座接近点会动态发送到 /move_base。
# guest1 外貌、FIND_HOST/FOLLOW_HOST 都读取 start_all.sh 启动的 latest_people.json 视觉缓存。

# 终端 5: Task1 控制器
python task1_receptionist/task1_controller.py --door-goal 1.0 2.0 90

# 调试时只计算目标、不发布机械臂或夹爪指令:
python task1_receptionist/task1_controller.py --object-search-dry-run

# 如需复用已有 latest_*.json，不重新打开 D455:
python task1_receptionist/task1_controller.py --object-search-skip-vision

# 调整空座接近点距离，默认 1.5 米:
TASK1_EMPTY_SEAT_APPROACH_DISTANCE=1.5 \
python task1_receptionist/task1_controller.py

# 空座识别最多处理 30 帧，且识别成功会提前退出；现场还嫌慢可继续降到 10-15:
python task1_receptionist/task1_controller.py --empty-seat-max-frames 15

# 接包目标优先级: 提着包的手 -> 包 -> 提着东西的手 -> 最近手腕。
# 漏检时，可给 Task1 内部 hand_search 透传参数:
TASK1_HAND_SEARCH_ARGS="--bag-conf 0.08 --keypoint-conf 0.10 --max-hand-bag-pixel-distance 120" \
python task1_receptionist/task1_controller.py

# 接包识别等待/底盘接近距离/机械臂安全距离/夹爪开合值也可以通过环境变量调整:
TASK1_HANDOVER_RECOGNITION_DELAY_SEC=5 \
TASK1_HANDOVER_APPROACH_DISTANCE=0.7 \
TASK1_HANDOVER_HAND_STANDOFF=0.14 \
TASK1_HANDOVER_MAX_ARM_REACH=0.70 \
TASK1_OPEN_GRIPPER_VALUE=1.0 TASK1_CLOSE_GRIPPER_VALUE=0.0 \
python task1_receptionist/task1_controller.py
```

### 5. object_search 单独调试

```bash
# 空座识别 (默认自动选择 RealSense D455 + YOLO, 使用 camera_middle_to_leftbase.txt 输出 leftbase 坐标)
task1_receptionist/object_search/run_object_search.sh

# 如需指定 D455 串号:
REALSENSE455_SERIAL=333422301212 task1_receptionist/object_search/run_object_search.sh

# 空座识别运行后，可让左臂指向最新空椅:
# 默认从 TF 读取 left_arm_base_link -> left_gripper_link 当前夹爪坐标。
# 单独调试时会先打印目标坐标，输入 y 才发布；加 --yes 可自动确认。
task1_receptionist/object_search/point_empty_seat_arm.py
task1_receptionist/object_search/point_empty_seat_arm.py --yes

# 只计算并打印，不发布机械臂指令:
task1_receptionist/object_search/point_empty_seat_arm.py --dry-run

# 默认让末端本地 +X 轴指向椅子；如果现场方向相反，可改成 -X:
task1_receptionist/object_search/point_empty_seat_arm.py --tool-axis -x --dry-run

# 接包：识别客人手里的包，输出 latest_bag.json
# 注意同一台 D455 不能同时跑空座识别和包识别，需要先停掉另一个 RealSense 脚本。
task1_receptionist/object_search/run_bag_search.sh

# 包识别默认置信度阈值为 0.15；还漏检时可继续降低，例如:
task1_receptionist/object_search/run_bag_search.sh --conf 0.10

# 更推荐的接包方式：同时识别包、可携带物体和手腕，按
# hand_with_bag -> bag -> hand_with_object -> nearest_wrist 选择接包目标，
# 输出 latest_handover_hand.json。
# 注意同一台 D455 同一时间只能跑一个识别脚本。
task1_receptionist/object_search/run_hand_search.sh

# 漏检时可调低包检测阈值/手腕关键点阈值，或放宽手-包像素距离:
task1_receptionist/object_search/run_hand_search.sh --bag-conf 0.08 --keypoint-conf 0.10 --max-hand-bag-pixel-distance 120

# 如果客人拿的是非包物体，可扩展第二优先级类别:
task1_receptionist/object_search/run_hand_search.sh --carry-object-classes bottle cup "cell phone" book

# 根据 latest_handover_hand.json 让左臂靠近接包目标，单独调试时输入 y 后会:
# 打开左夹爪 -> 发布机械臂目标 -> 等待 5 秒 -> 关闭左夹爪。
task1_receptionist/object_search/approach_handover_arm.py --dry-run
task1_receptionist/object_search/approach_handover_arm.py
task1_receptionist/object_search/approach_handover_arm.py --yes

# 调整离接包目标的距离：数值越大离人的手越远。
task1_receptionist/object_search/approach_handover_arm.py --hand-standoff 0.14 --dry-run

# 限制机械臂目标伸出半径，超过 70cm 会沿同方向缩回到 70cm 内。
task1_receptionist/object_search/approach_handover_arm.py --max-arm-reach 0.70 --dry-run

# 如果现场夹爪开合值相反或幅度不合适，可改这两个值；也可用 --no-gripper 只移动手臂。
task1_receptionist/object_search/approach_handover_arm.py --open-gripper-value 1.0 --close-gripper-value 0.0

# 接包：根据 latest_bag.json 让左臂靠近包旁边，输入 y 才发布机械臂目标。
task1_receptionist/object_search/approach_bag_arm.py --dry-run
task1_receptionist/object_search/approach_bag_arm.py

# 调整靠近包的距离：数值越小越贴近包中心。
task1_receptionist/object_search/approach_bag_arm.py --bag-standoff 0.08 --dry-run

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
