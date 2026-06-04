# cade_vision 项目说明

本文档描述 `cade_vision` ROS 节点的当前架构、各模块职责、文件功能与重构状态。当前代码中存在两套姿态/手势实现：`posture_gesture.py` 是入口实际使用的线上路径，`kits/posture_gesture/` 是拆分重构中的候选结构，二者功能大量重复但接入状态不同。

## 1. 项目定位

`cade_vision` 是 CADE 系统中的开放视觉感知节点，核心职责是：

- 监听 `/cade/task_cmd`，接收大脑/任务层下发的 JSON 指令。
- 从 RealSense、USB 摄像头或图片/视频文件获取图像帧。
- 使用 YOLO/YOLO-World 做目标检测，并按任务动态调整开放词表类别。
- 对人框做跨帧追踪、衣物属性、姿态和手势分析。
- 维护最新检测缓存，供任务执行器查询、过滤和汇总。
- 通过 `/vision/detections_3d`、`/vision/point_3d` 和 `/cade/task_status` 发布任务结果。

该节点按源码注释设计为“纯感知节点”，不负责底盘控制、模型状态修改或 LLM 调用。

## 2. 当前运行主链路

入口文件是 `scripts/open_vision_node.py`，当前主流程如下：

1. `main()` 解析命令行参数，创建 `OpenVisionNode`。
2. `OpenVisionNode.__init__()` 初始化 ROS publisher/subscriber、图像源、YOLO 模型、TF、MediaPipe 姿态手势分析器、属性过滤器、线程池与追踪器。
3. `run()` 循环调用 `process_frame()`。
4. `process_frame()` 读取帧，执行 YOLO 检测，生成 `new_detections`。
5. `CadeTracker.assign()` 为检测目标分配稳定 `track_id`。
6. `associate_clothing()` 将衣物框关联到 person，并写入 `cloth_color`、`cloth_type`。
7. 对 person 框并行执行 `process_pose()` 和 `process_hands()`，再调用 `analyze_from_landmarks()` 输出 `posture`、`gesture` 和 landmarks。
8. 最新检测结果写入 `self.detected_objects`，任务执行器通过 `get_latest_detections()` 获取快照。
9. 当 `/cade/task_cmd` 到来时，节点创建对应 `Task` 子类在线程中执行，成功或失败后发布状态。

注意：代码当前每帧都会更新内部检测缓存，但 `/vision/detections_3d` 并不是每帧全量发布；它主要在 `SearchTask` 找到目标后通过 `_publish_detection()` 发布一次任务结果。`/vision/point_3d` 是兼容旧订阅者的 `PointStamped` 输出，仅当任务结果包含 `position_3d` 时发布。

## 3. ROS 接口与任务动作

### 订阅

- `/cade/task_cmd` (`std_msgs/String`)：JSON 指令入口。

### 发布

- `/cade/task_status` (`std_msgs/String`)：任务状态，形如 `{"status": "SUCCESS", "result": ...}` 或 `{"status": "FAILED", "error": ...}`。
- `/vision/detections_3d` (`std_msgs/String`)：任务命中的检测结果 JSON。
- `/vision/point_3d` (`geometry_msgs/PointStamped`)：旧版兼容点位输出，frame 为 `camera_color_optical_frame`。

### 支持动作

`OpenVisionNode._task_mapping` 当前支持：

- `find_object`：查找指定物体。
- `find_person`：查找 person，可附带 `posture`、`gesture`、`cloth_color`、`cloth_type` 等属性过滤。
- `filter_by_attributes`：只对 person 做属性过滤并返回匹配列表。
- `count_objects`：统计指定类别数量。
- `count_people`：统计 person 数量。
- `get_person_info`：返回当前检测缓存中的对象信息。
- `get_nearest_person`：返回最近 person 的基础属性。

## 4. 关键 features

- 多图像源：`realsense`、`file`、`usb_cam`，文件模式支持图片/视频、循环播放和播放速度控制。
- YOLO-World 开放类别检测：基础类别常开，任务目标和衣物类型会动态加入 `set_classes()`。
- RealSense 3D 坐标：对检测框中心 ROI 取中值深度，并过滤 0.2m 到 1.0m 以外的结果。
- 跨帧 ID：优先使用 3D 距离，其次使用 2D IOU，最后用 Hungarian/贪心匹配分配 `track_id`。
- 衣物关联：用衣物框和 person 框 IOU 匹配，HSV 均值估计颜色，输出上衣/下装类型组合。
- 姿态识别：输出 `standing`、`sitting`、`lying`、`unknown`。
- 手势识别：输出 `waving`、`raising_left_arm`、`raising_right_arm`、`raising_both_arms`、`pointing_left`、`pointing_right`、`pointing_both`、`none`、`unknown`。
- 时序挥手：每个 `track_id` 维护 30 帧角度缓冲，通过前臂摆动方差和手腕旋转方差判定。
- 任务异步执行：任务线程轮询检测缓存，新任务到来时取消旧任务，避免阻塞主视觉循环。
- 属性过滤：`Analyzer` 支持标量、列表和 `/ , | ;` 分隔的多值匹配。

## 5. 文件与模块说明

### 包配置与资源

| 文件 | 作用 |
| --- | --- |
| `package.xml` | ROS/catkin 包元数据和运行依赖声明。 |
| `CMakeLists.txt` | catkin 构建配置，安装 `open_vision_node.py` 并启用 `catkin_python_setup()`。 |
| `setup.py` | Python 包配置，包根为 `src/`，声明 `rospy`、`ultralytics`、`opencv-python`、`pyrealsense2`、`numpy` 等依赖。 |
| `yolo11x-seg.pt` | 仓库内的本地模型权重文件；当前脚本默认参数仍是 `yolov8x-worldv2.pt`，实际使用哪个取决于 `--model`。 |
| `TEST_GUIDE.md` | 较完整的独立测试说明，覆盖文件模式、ROS 手动发指令和话题监听。 |
| `docs/TEST_GUIDE.md` | 面向实机/人工测试的简版流程。 |
| `docs/PROJECT.md` | 本文档。 |

### 入口脚本

#### `scripts/open_vision_node.py`

`OpenVisionNode` 是整个节点的编排层：

- 初始化 ROS 节点、publisher、subscriber。
- 初始化 RealSense、文件源或 USB 摄像头。
- 加载 YOLO 模型并维护基础类别和任务动态类别。
- 初始化 TF buffer/listener。当前代码只初始化，尚未在 3D 点转换中实际使用。
- 初始化根目录 `cade_vision.posture_gesture.PostureGestureAnalyzer`。这是当前实际接入版本。
- 初始化 `Analyzer`、`CadeTracker` 和 Pose/Hands 推理线程池。
- 保存一个 `transformation_matrix`，注释表示相机到夹爪变换；当前主流程没有使用该矩阵。
- 接收任务 JSON，按 action 创建 `SearchTask`、`CountTask` 或 `InfoTask`。
- 在 `process_frame()` 中完成检测、追踪、衣物关联、姿态/手势分析和显示窗口绘制。
- 在任务命中时发布 detection/status。

### Python 包根

#### `src/cade_vision/__init__.py`

空文件，用于声明 `cade_vision` Python 包。

#### `src/cade_vision/analyzer.py`

属性过滤管线。`Analyzer.filter_by_attributes()` 会按成本顺序依次过滤：

1. `posture`
2. `gesture`
3. `cloth_color`
4. `cloth_type`
5. `identity`

它支持 expected 或 actual 为列表/集合，也支持 actual 中用 `/`、`,`、`|`、`;` 分隔多个值。该模块不直接跑视觉模型，只基于已经写入检测对象的属性做筛选。

#### `src/cade_vision/tracker.py`

`CadeTracker` 是轻量跨帧追踪器，为每个检测框分配稳定 `track_id`：

- 若目标有 3D 坐标，优先以 0.3m 内的欧氏距离匹配。
- 否则使用 bbox IOU，阈值为 0.3。
- 有 SciPy 时使用 `linear_sum_assignment()` 做全局匹配，无 SciPy 时退化为贪心匹配。
- 未匹配轨迹最多保留 5 帧。

`track_id` 主要服务于姿态/手势时序缓冲，尤其是挥手检测。

#### `src/cade_vision/cloth.py`

无状态衣物分析模块：

- 将非 person 检测框视为衣物候选。
- 通过 person 与衣物框 IOU 大于 0.3 建立关联。
- 将衣物分成上衣类和下装类，每人每类保留 IOU 最高的一件。
- 在衣物框中心区域计算 HSV 均值，映射到 `white`、`black`、`gray`、`red`、`yellow`、`blue` 或 `unknown`。
- 最终写入 person 的 `cloth_color`、`cloth_type`，多件衣物用 `/` 拼接。

#### `src/cade_vision/posture_gesture.py`

当前实际使用的姿态/手势模块，入口导入语句是：

```python
from cade_vision.posture_gesture import PostureGestureAnalyzer
```

该文件是一个 900 多行的单文件实现，包含：

- MediaPipe 可用性探测，缺少 `mediapipe` 时不会让整个节点 import 失败。
- `RingBuffer`：每个 person/side 保存前臂角度和手腕旋转角度，支持跳变过滤和连续 None 清空。
- `PostureGestureAnalyzer`：封装 MediaPipe Pose、MediaPipe Hands、姿态分类、静态手势分类和时序挥手判定。
- `process_pose()`：对 person crop 输出 33 个 Pose landmarks。
- `process_hands()`：输出左右手 wrist、index_tip 和 21 点 hand landmarks。
- `analyze_from_landmarks()`：在不重复跑模型的情况下执行姿态/手势规则，并可接收 RealSense 提取的 3D 关键点。
- `_classify_posture()`：有 2D 兜底和 3D 规则。3D 规则重点处理站立/坐姿高度链、侧向坐姿、后靠和躺姿。
- `_classify_static_gesture()`：单帧规则识别举手、指向和 none，并返回供时序分析使用的角度。
- `_apply_temporal()`：通过前臂/手腕方差和肩部高度门控覆盖为 `waving`，并输出 debug 方差字段。

当前节点的 Pose + Hands 是在线程池中并行执行的，然后调用 `analyze_from_landmarks()` 做融合，避免同一人框重复跑模型。

### 任务模块

#### `src/cade_vision/tasks/__init__.py`

导出 `SearchTask`、`CountTask`、`InfoTask`。

#### `src/cade_vision/tasks/base_task.py`

任务执行器基类，提供：

- 统一的取消/继续标志。
- `RESERVED_COMMAND_KEYS` 和 `PERSON_ATTRIBUTE_KEYS`。
- `extract_attributes()`：合并 `attributes` 字段和顶层属性字段。
- `filter_candidates()`：优先调用节点上的 `Analyzer`，不存在时使用精确匹配 fallback。

#### `src/cade_vision/tasks/search_task.py`

搜索类任务：

- `find_object`、`find_person` 会在 timeout 内轮询最新检测缓存。
- 对候选先执行属性过滤，再按 `class_name` 包含目标名筛选。
- RealSense 模式要求命中目标必须有 `position_3d`，并选择 z 值最小的目标。
- 非 RealSense 模式选择 confidence 最高的目标，`position_3d` 为 `None`。
- 命中后发布 `/vision/detections_3d` 和 `/cade/task_status`。
- `filter_by_attributes` 只筛 person，返回 `track_id`、`bbox`、`position_3d`、衣物属性和一些尚未实现的人体属性占位。

#### `src/cade_vision/tasks/count_task.py`

计数任务：

- 从当前检测缓存取一次快照。
- 可先按属性过滤，再按 `category` 或默认 person 统计。
- 返回 `count_result`，包含类别、placement/room、数量和命中的 class 列表。

#### `src/cade_vision/tasks/info_task.py`

信息查询任务：

- `get_person_info` 返回当前缓存中符合属性过滤的对象名、置信度和 3D 坐标。
- `get_nearest_person` 选择有 3D 坐标时距离最近的 person，否则退回第一个 person，返回衣物和基础占位属性。

## 6. `kits/posture_gesture` 重构模块说明

`src/cade_vision/kits/posture_gesture/` 是把根目录 `posture_gesture.py` 拆分后的重构方向，但当前没有被 `open_vision_node.py` 导入，也不应视为线上路径。

### 目录结构

| 文件 | 设计职责 | 当前状态 |
| --- | --- | --- |
| `kits/__init__.py` | kits 命名空间标记。 | 空文件。 |
| `kits/posture_gesture/__init__.py` | 拆分版 `PostureGestureAnalyzer` 门面，负责 MediaPipe 推理、调用 rules、应用 temporal。 | 未接入入口；直接 `import mediapipe as mp`，没有根版的可用性保护。 |
| `kits/posture_gesture/temporal_buffer.py` | 拆出的 `RingBuffer`。 | 基本复制根版缓冲逻辑。 |
| `kits/posture_gesture/rules/posture.py` | 拆出的 2D/3D 姿态规则。 | 基本对应根版 `_classify_posture()`。 |
| `kits/posture_gesture/rules/gesture.py` | 拆出的静态手势和时序手势规则。 | 引入 `maybe_*` 初筛状态和 `judge_all_temporal_gestures()`，但存在调用签名不一致问题。 |
| `kits/posture_gesture/rules/utils.py` | 手势规则工具函数，如 landmarks 访问、可见性判断、角度计算。 | 当前 `compute_side_angles()` 内部调用缺少 landmarks 参数，与 `gesture.py` 的调用也不匹配。 |
| `kits/posture_gesture/rules/__init__.py` | rules 子包标记。 | 空文件。 |

### 与根版 `posture_gesture.py` 的关系

两套实现重复的原因是节点正在从“单文件大模块”向“门面 + rules + temporal buffer”重构：

- 根版 `posture_gesture.py`：当前可运行、已被入口使用，包含完整功能和 MediaPipe import fallback。
- kits 版：目标是把 `RingBuffer`、姿态规则、手势规则拆出来，便于维护和测试；当前还处于半接入状态。

迁移时需要特别注意：

- `open_vision_node.py` 目前只导入根版，不会触发 kits 版。
- kits 版 `gesture.py` 中 `is_arm_raised()`、`is_pointing()`、`compute_side_angles()` 的定义和调用参数不一致，直接切换入口会报错。
- kits 版 `classify_static_gesture()` 返回 `maybe_raising_*`、`maybe_pointing_*`，再由 `judge_all_temporal_gestures()` 统一决策；根版则在单帧阶段直接返回最终静态手势，时序只覆盖 `waving`。
- kits 版 `PostureGestureAnalyzer._unknown_result()` 仍包含 `confidence` 字段，但正常成功结果不再写入根版那样的 `confidence = min(posture_conf, gesture_conf)`。
- 若要完成重构，应先修正 kits 版函数签名，补齐 MediaPipe 可用性保护，再把入口 import 从 `cade_vision.posture_gesture` 切到 `cade_vision.kits.posture_gesture`，最后用现有 `TEST_GUIDE.md` 跑端到端验证。

## 7. 数据结构约定

单帧检测对象大致结构如下：

```json
{
  "index": 0,
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.93,
  "bbox": [x1, y1, x2, y2],
  "center": [cx, cy],
  "position_3d": [x, y, z],
  "track_id": 3,
  "cloth_color": "blue",
  "cloth_type": "shirt",
  "posture": "sitting",
  "gesture": "waving",
  "landmarks": []
}
```

其中：

- `position_3d` 只在 RealSense 模式且深度有效时存在，否则为 `None`。
- `cloth_*` 只对 person 有意义，默认 `unknown`。
- `posture`、`gesture` 只对 person 有意义，MediaPipe 不可用或关键点不足时为 `unknown`。
- `track_id` 是跨帧稳定 ID，但不是长期身份识别 ID。

## 8. 当前技术债与注意事项

- `posture_gesture.py` 与 `kits/posture_gesture/` 功能重复。当前必须以根版为准，kits 版是未完成重构。
- `tf_buffer`、`tf_listener` 和 `transformation_matrix` 已初始化但主流程未实际用于坐标变换。
- `Analyzer` 注释里说 clothing 是“后续实现”，但当前 `cloth.py` 已经实现 HSV 颜色和衣物类型关联。
- `filter_by_attributes` 的返回结果目前不包含已用于过滤的 `posture`、`gesture` 字段，只返回部分 person 属性。
- `CountTask` 和 `InfoTask` 是快照式任务，不像 `SearchTask` 那样等待 timeout 内持续轮询。
- `__pycache__/` 是 Python 运行生成缓存，不属于源码架构。
