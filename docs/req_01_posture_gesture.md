# 需求：CADE 视觉系统扩展 — 姿态+手势识别管线

## 一、背景

GPSR 指令中约 33% 涉及人物姿态/手势描述（standing/sitting/lying/waving/raising arm/pointing）。目前 vision node 只能做人框检测，无法区分这些特征。

## 二、架构变更

从"一个 vision node 干所有事"改为"管线编排"模式。

### 新增文件

```
cade_ws/src/cade_vision/
  scripts/
    open_vision_node.py        ← 修改：主循环 + 帧捕获（精简，作为管线编排者）
  src/                          ← 新增目录（Python package）
    __init__.py
    analyzer.py                ← 新增：命令解析 + 按需调度子模块
    posture_gesture.py         ← 新增：MediaPipe Pose + 分类逻辑
```

### 不变的部分

- vision node 的主循环仍然持续 30fps 跑
- 仍然发布 `/vision/detections_3d`
- 仍然接收 `/cade/task_cmd`

### 变更的部分

- `process_frame` 中，在 YOLO 推理之后、发布之前，插入 MediaPipe 后处理
- 检测结果中新增 `posture`、`gesture` 字段
- MediaPipe 每帧常开（~5ms/帧，CPU 可跑），数据进环形缓冲区

---

## 三、详细设计

### 3.1 posture_gesture.py

#### 职责

接收一张**人框裁剪图**（或全图+bbox），输出该人的姿态和手势分类结果。

#### 输入/输出

```python
# 输入
class PostureGestureAnalyzer:
    def analyze(person_crop: np.ndarray) -> dict:
        """
        Returns:
        {
            "landmarks": [(x0,y0,score0), (x1,y1,score1), ...],  # 33 keypoints (MediaPipe 定义)
            "posture": "standing" | "sitting" | "lying" | "unknown",
            "gesture": "waving" | "raising_left_arm" | "raising_right_arm"
                    | "pointing_left" | "pointing_right" | "none" | "unknown",
            "confidence": float,  # 0~1
        }
        """
```

#### 姿态分类逻辑

基于 MediaPipe Pose 33 个关键点中的下肢关键点（髋、膝、踝）：

```
关键点索引（MediaPipe Pose）:
  23: 左髋 (left_hip)
  24: 右髋 (right_hip)
  25: 左膝 (left_knee)
  26: 右膝 (right_knee)
  27: 左踝 (left_ankle)
  28: 右踝 (right_ankle)

角度计算（以左腿为例）:
  vector1 = knee - hip
  vector2 = ankle - knee
  angle = arccos(dot(vector1, vector2) / (|v1| * |v2|))

分类规则:
  angle > 150° → standing（腿基本伸直）
  80° < angle < 120° → sitting（膝盖弯曲约90°）
  angle < 30° 或 躯干与水平面夹角 < 30° → lying（躯干接近水平）

  注：左右腿分别算，取更可信的那一侧。如果下半身关键点不可见（遮挡），返回 "unknown"。
```

#### 手势分类逻辑

分为两类：**静态手势**（举手、指向）和**动态手势**（挥手）。

##### 静态手势（单帧可判断）

```
关键点索引:
  11: 左肩 (left_shoulder)
  12: 右肩 (right_shoulder)
  13: 左肘 (left_elbow)
  14: 右肘 (right_elbow)
  15: 左手腕 (left_wrist)
  16: 右手腕 (right_wrist)

举手判断:
  wrist_y < shoulder_y - threshold  → 该侧手臂"举起"

指向判断:
  hand_x - shoulder_x > arm_length * 0.8 且 wrist_y ≈ shoulder_y  → 向该侧"指向"
  指向的本质是手臂水平或接近水平向前/向侧面伸出，而不是向上举起

具体规则:
  if left_wrist_y < left_shoulder_y - T and right_wrist_y > right_shoulder_y - T:
      gesture = "raising_left_arm"
  elif right_wrist_y < right_shoulder_y - T and left_wrist_y > left_shoulder_y - T:
      gesture = "raising_right_arm"
  elif left_wrist相对肩膀水平伸出 或 right_wrist相对肩膀水平伸出:
      gesture = "pointing_left" / "pointing_right"
  else:
      gesture = "none"
```

##### 动态手势（挥手，需要时序信息）

挥手检测需要跨帧数据，通过分析**前臂相对于肘关节的旋转运动**来识别：

```
关键点:
  13(L)/14(R): 肘
  15(L)/16(R): 手腕

核心指标:
  计算每帧中 wrist → elbow 的 2D 向量角度（相对于图像平面）
  角度 = atan2(wrist.y - elbow.y, wrist.x - elbow.x)

  在滑动窗口（30帧 ≈ 1 秒）中计算角度序列的方差:
  - 方差 > T_wave → 手腕在周期性旋转 → "waving"
  - 方差 < T_wave 且手腕在举高位 → "raising_arm"（举手后静止）
  - 方差 < T_wave 且手腕在低位 → "none"

关键区别:
  waving 的特征是肘关节固定、前臂在摆 → wrist相对elbow运动
  走路摆臂的特征是肩关节摆动 → wrist和elbow一起动，wrist相对elbow角度变化小
  → 用 wrist相对elbow 的角度变化可以天然区分挥手的走路摆臂
```

#### 环形缓冲区

`posture_gesture.py` 内部维护每人最近的 N 帧关键点数据：

```python
class RingBuffer:
    """每人维护一个环形缓冲区，存储最近 30 帧的手腕-肘关节角度"""
    def __init__(self, capacity=30):
        self.capacity = capacity
        self.buffer = {}
    
    def push(self, person_id, angle_left, angle_right):
        """添加一帧的数据"""
        ...
    
    def get_angle_variance(self, person_id, side="left", window=30):
        """获取指定窗口内角度序列的方差"""
        ...
```

person_id 使用当前帧中检测框的位置哈希（或跟踪 ID）来确定是否为同一个人。简单做法：用检测框中心位置的 IOU 追踪。

### 3.2 analyzer.py

#### 职责

接收 brain 发来的指令 JSON，解析出需要检索的人物属性，按代价排序后逐层过滤检测到的人。

#### 判断管线

```python
def analyze(people: list[dict], attributes: dict) -> list[dict]:
    """
    people: 当前帧检测到的所有人（含 bbox, landmarks）
    attributes: 要找的特征，如 {"posture": "sitting", "gesture": "waving"}
    
    返回: 过滤后的人列表（符合全部特征）
    
    处理顺序:
      1. posture  → MediaPipe（~5ms/人，最先过滤）
      2. gesture  → MediaPipe + 环形缓冲区（~5ms/人）
      3. clothing → CLIP（~100ms/人，最后过滤，因为最贵）
    """
```

**第一步只实现 posture 和 gesture 的过滤**。clothing 和 identity 是后续需求。

### 3.3 open_vision_node.py 修改

#### 在 process_frame 中插入 MediaPipe

```python
def process_frame(self):
    # ... 原有的 YOLO 推理代码不变 ...
    
    # [新增] 对每个检测到的人框，跑 MediaPipe Pose
    for obj in new_detections:
        if obj["class_name"] == "person":
            x1, y1, x2, y2 = obj["bbox"]
            person_crop = color_image[y1:y2, x1:x2]
            result = self.posture_gesture.analyze(person_crop)
            # 把结果合并到 obj 中
            obj["posture"] = result["posture"]
            obj["gesture"] = result["gesture"]
            obj["landmarks"] = result["landmarks"]
            # 更新环形缓冲区（用于 waving 检测）
            self.ring_buffer.push(obj.get("track_id"), 
                                  result.get("elbow_angle_left"),
                                  result.get("elbow_angle_right"))
    
    # [新增] 如果有活跃任务，调用 analyzer 做过滤
    if self.task_active and self.task_attributes:
        new_detections = self.analyzer.filter_by_attributes(
            new_detections, self.task_attributes
        )
    
    # ... 原有的发布代码不变 ...
    # 发布的数据格式新增字段
```

#### 发布的检测结果格式扩展

```json
{
    "class_name": "person",
    "bbox": [100, 200, 300, 400],
    "confidence": 0.95,
    "position_3d": [0.5, 0.3, 1.2],
    "posture": "standing",
    "gesture": "waving",
    "landmarks": [[x0,y0,s0], [x1,y1,s1], ...]
}
```

---

## 四、依赖安装

```bash
pip install mediapipe
```

无其他依赖。MediaPipe 默认 CPU 推理，不需要 CUDA。

---

## 五、实施步骤（给 Claude Code）

1. 创建 `cade_ws/src/cade_vision/src/` 目录和 `__init__.py`
2. 实现 `posture_gesture.py`
   - MediaPipe Pose 初始化
   - `analyze(person_crop)` 方法（姿态 + 静态手势）
   - `RingBuffer` 类（时序挥手检测）
   - `analyze_with_temporal(person_id, person_crop)` 方法（包含时序手势）
3. 实现 `analyzer.py`
   - `filter_by_attributes(people, attributes)` 方法
   - 按代价排序属性并串联过滤
4. 修改 `open_vision_node.py`
   - 在 `__init__` 中初始化 PostureGestureAnalyzer 和 Analyzer
   - 在 `process_frame` 中插入 MediaPipe 后处理
   - 扩展发布数据格式
   - 处理 `/cade/task_cmd` 中的新命令格式（带 attributes 字段）

---

## 六、验收标准

1. 在单张测试图片上，对一个人框调用 `posture_gesture.analyze()` 能正确输出 posture 和 gesture
2. 在视频流上，挥手检测能在 1 秒内给出 waving/raising_arm 的区分
3. vision node 30fps 运行时，CPU 占用率增加不超过 20%（MediaPipe 的消耗）
4. 不破坏现有的 YOLO 检测和 3D 坐标发布功能
