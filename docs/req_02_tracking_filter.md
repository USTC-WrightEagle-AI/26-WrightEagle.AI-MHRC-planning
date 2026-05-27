# 需求：CADE 视觉系统修正 — 跨帧追踪 + Analyzer 过滤位置修正

## 一、背景

当前 `open_vision_node.py` 存在两个问题，导致 RingBuffer 时序挥手检测实际不工作，且 Analyzer 属性过滤污染了全局缓存。

---

## 二、问题 1：Analyzer 过滤放错位置

### 当前行为

```python
# process_frame 中，写入缓存之前就做了过滤
if self.task_active and self.task_attributes and self.analyzer is not None:
    new_detections = self.analyzer.filter_by_attributes(...)

self.detected_objects = new_detections    # ← 缓存里存的是残缺的
```

### 后果

- 任务进行期间，`self.detected_objects` 只包含符合属性条件的人
- 不符合条件的人（如 standing）被永久丢弃，即使之后改变了姿态也无法进入缓存
- 三个任务函数共用同一个缓存，一个任务的属性条件污染另一个任务的数据

### 改正方案

**职责分离原则：`self.detected_objects` 应是 vision node 的完整感知快照，不应因任何任务的偏好被截断。**

1. `process_frame` 中删除 Analyzer 过滤代码，`self.detected_objects` 始终写入所有检测结果（YOLO + MediaPipe 完整信息）
2. 将 Analyzer 的 `filter_by_attributes` 调用下放到三个任务函数内部（`_execute_search_task`、`_execute_count_task`、`_execute_info_task`）
3. 每个任务函数在从 `self.detected_objects` 读数据后，自行按自己的 `task_attributes` 做属性过滤
4. 确保即使没有 `task_attributes`，三个任务函数的行为不受影响

---

## 三、问题 2：缺乏跨帧追踪，RingBuffer 时序检测失效

### 当前行为

```python
# open_vision_node.py 第 628 行
pid = obj.get("track_id", obj["index"])
# track_id 从未被写入 → 最终用 obj["index"]
# obj["index"] = len(new_detections)  # 当前帧序数，每帧重洗牌
```

### 后果

同一个人在不同帧的 person_id 不可预测地变化：
- YOLO 检测顺序（按置信度降序）不稳定，置信度波动、新目标闯入都会导致重排
- RingBuffer 中 person_id 对应的历史数据跨越可能混合了不同的人
- 方差序列得不到足够长的连续数据 → 挥手检测实际不工作

### 改正方案

**在每帧写入 obj_info 之前，建立一个跨帧追踪机制：将当前帧的检测框与上一帧的检测框做匹配，为每个检测对象分配一个稳定的跨帧 ID（track_id）。**

核心思路（3 层约束，从前到后依次增强）：

**第 1 层：3D 位置连续性匹配（利用 CADE 的 RealSense 深度优势）**
- 如果当前帧的检测对象和上一帧的某个对象都有 3D 坐标
- 且在 3D 空间中的欧氏距离小于阈值（如 30cm）
- 则认为是同一个人，分配相同的 track_id
- 这是最可靠的约束，因为 3D 坐标在相邻帧之间变化极小

**第 2 层：2D IOU 匹配**
- 对于没有 3D 坐标的对象（yolo 类别的物体），用 2D bbox IOU 匹配
- IOU 超过阈值（如 0.3）判定为同一对象
- 注意：IOU 单独使用时，两人紧靠时会互相错分——但 CADE 场景下绝大多数情况有深度数据，所以 IOU 作为辅助即可

**第 3 层：匈牙利匹配保障全局最优**
- 上述两层为每一对（旧 track，新检测）算一个匹配分数（IOU + 3D 位置接近度）
- 用匈牙利算法做全局最优匹配，避免贪心匹配导致的冲突（一个检测被两个旧 track 抢，或一个旧 track 占了两个检测）
- 匹配失败的学习分数阈值以下的分到新 ID

**生命周期管理：**
- 新对象分配新 ID（从 0 递增或随机生成）
- 连续多帧未匹配到的旧 track 自动过期（一个人离开视野后应视作新对象）
- 匹配到的 track_id 写入 obj_info，供后续 MediaPipe 的 RingBuffer 使用

---

## 四、修改范围

| 文件 | 修改内容 |
|---|---|
| `open_vision_node.py` | `process_frame`：删除 Analyzer 过滤；新增跨帧追踪逻辑，生成 `track_id` 写入 obj_info |
| `open_vision_node.py` | 三个任务函数：各自在消费 `self.detected_objects` 后做 Analyzer 属性过滤 |
| `open_vision_node.py` | `__init__`：新增追踪器状态容器（上一帧检测记录、ID 计数器） |
| `posture_gesture.py` | 无改动（RingBuffer 的接口不变，person_id 传入正确的 track_id 后自动修复） |
| `analyzer.py` | 无改动（接口不变，只是被调用的位置变了） |

---

## 五、验收标准

1. **单人场景**：人在画面中移动，`track_id` 保持不变成同一 ID
2. **多人场景**：两个人交叉走过，各自的 `track_id` 不互换、不串数据
3. **RingBuffer 正常验证**：单人在画面中挥手，RingBuffer 能累积 30 帧同一个 person_id 的连续数据，方差序列有意义
4. **Analyzer 过滤不污染缓存**：有活跃任务时，`self.detected_objects` 仍然包含全部检测结果，不因 `task_attributes` 截断任何人或物
5. **零破坏**：现有的 YOLO 检测、RealSense 3D 坐标、MediaPipe 姿态手势识别、发布机制全部不受影响
