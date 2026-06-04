# open_vision_node 检测结果生成流程

本文说明 `open_vision_node.py` 运行时，`process_frame()` 如何把一帧图像变成 CV 窗口里每个物体/人的显示结果。重点覆盖窗口中能看到的内容：检测框、类别、置信度、追踪 ID、衣物、姿态、手势、3D 坐标。

源码入口是：

- `src/cade_vision/scripts/open_vision_node.py`

相关功能文件是：

- `src/cade_vision/src/cade_vision/tracker.py`
- `src/cade_vision/src/cade_vision/cloth.py`
- `src/cade_vision/src/cade_vision/kits/posture_gesture/__init__.py`
- `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py`
- `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/gesture.py`
- `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/utils.py`
- `src/cade_vision/src/cade_vision/kits/posture_gesture/temporal_buffer.py`

## 1. 总体链路

`main()` 创建 `OpenVisionNode`，然后 `run()` 进入循环。每轮循环调用一次 `process_frame()`：

```text
main()
  -> OpenVisionNode(args)
       -> 初始化图像源
       -> 加载 YOLO-World
       -> 初始化 PostureGestureAnalyzer
       -> 初始化 CadeTracker
  -> node.run()
       -> process_frame()
            -> _capture_frame()
            -> YOLO model.predict()
            -> 组装 obj_info
            -> get_median_depth_in_roi() / get_3d_coordinates()
            -> CadeTracker.assign()
            -> associate_clothing()
            -> PostureGestureAnalyzer.process_pose()
            -> PostureGestureAnalyzer.process_hands()
            -> PostureGestureAnalyzer.analyze_from_landmarks()
            -> 更新 self.detected_objects
            -> cv2.rectangle() / cv2.putText() / cv2.imshow()
```

CV 窗口显示的每个目标，最终都来自 `process_frame()` 内的 `new_detections` 列表。每个元素是一个 `obj` 字典，字段逐步增加。

## 2. 启动阶段准备了什么

文件：`scripts/open_vision_node.py`

### `main()`

解析命令行参数，例如：

- `--model`
- `--conf`
- `--iou`
- `--device`
- `--image-source`
- `--display`

然后创建 `OpenVisionNode(args)` 并调用 `node.run()`。

### `OpenVisionNode.__init__()`

初始化运行时依赖：

- ROS publisher/subscriber。
- 图像源：`realsense` / `file` / `usb_cam`。
- YOLO-World 模型：`self.model = YOLO(args.model)`。
- 姿态手势模块：`self.posture_gesture = PostureGestureAnalyzer()`。
- 线程池：`self._pool = ThreadPoolExecutor(max_workers=2)`，用于 person crop 的 Pose 与 Hands 并行推理。
- 跨帧追踪器：`self._tracker = CadeTracker()`。
- 当前任务状态：`self.target_class` 会影响窗口里目标框是红色还是绿色。

### `_init_image_source()`

根据 `args.image_source` 分发到：

- `_init_realsense()`
- `_init_file_source()`
- `_init_usb_cam()`

这一步只准备输入源，不直接产生检测结果。

### `_on_task_cmd()` 相关函数

如果 ROS 任务消息到达，`_on_task_cmd()` 会更新任务状态，并可能调用：

- `_extract_task_attributes()`
- `_target_for_command()`
- `_build_model_classes()`
- `_set_model_classes()`

这些函数不在 `process_frame()` 内直接生成单个检测框，但会影响两件事：

- YOLO-World 当前检测类别集合。
- `self.target_class`，进而影响 CV 窗口里目标框颜色和线宽。

## 3. `run()` 如何驱动每一帧

文件：`scripts/open_vision_node.py`

### `OpenVisionNode.run()`

循环调用：

```python
if not self.process_frame():
    time.sleep(0.01)
    continue
```

如果 `--display` 打开，还会调用 `cv2.waitKey(1)` 检查 `q` 退出。真正的检测、字段填充和窗口绘制都发生在 `process_frame()`。

## 4. 第一步：取得当前帧

文件：`scripts/open_vision_node.py`

### `OpenVisionNode.process_frame()`

一开始调用：

```python
color_image, depth_frame = self._capture_frame()
```

### `_capture_frame()`

按图像源返回 `(color_image, depth_frame)`：

- `realsense`：
  - `self.pipeline.wait_for_frames()`
  - `self.align.process(frames)` 对齐深度到彩色图
  - 返回 BGR 彩色图 `color_image` 和 RealSense `depth_frame`
- `file`：
  - 视频：`self._file_cap.read()`
  - 图片：`self._static_image.copy()`
  - `depth_frame` 为 `None`
- `usb_cam`：
  - `self._usb_cap.read()`
  - `depth_frame` 为 `None`

如果拿不到图像，`process_frame()` 返回 `False`，这一轮没有窗口结果。

## 5. 第二步：YOLO 生成基础检测框

文件：`scripts/open_vision_node.py`

### `process_frame()` 中的 YOLO 推理

```python
results = self.model.predict(
    source=color_image,
    conf=self.args.conf,
    iou=self.args.iou,
    device=self.args.device,
    verbose=False,
)
```

这里调用的是 Ultralytics YOLO-World。输出的每个 `box` 提供：

- `box.cls[0]`：类别 ID
- `model_names[class_id]`：类别名
- `box.conf[0]`：检测置信度
- `box.xyxy[0]`：边界框 `(x1, y1, x2, y2)`

随后 `process_frame()` 为每个框创建基础对象：

```python
obj_info = {
    "index": len(new_detections),
    "class_id": class_id,
    "class_name": class_name,
    "confidence": float(conf),
    "bbox": (x1, y1, x2, y2),
    "center": (center_x, center_y),
    "position_3d": point_3d,
}
```

这些字段会直接影响窗口第一行文字：

```text
#index class_name confidence
```

例如：

```text
#0 person 0.86
```

## 6. 第三步：计算目标中心点 3D 坐标

文件：`scripts/open_vision_node.py`

这一步只在 `image_source == "realsense"` 且 `depth_frame` 有效时执行。图片、视频文件、USB 摄像头模式下，`position_3d` 固定为 `None`。

### `get_median_depth_in_roi(depth_frame, x, y, roi_size=20)`

`process_frame()` 先在检测框中心点附近取一个 ROI：

- ROI 默认大小 20 像素。
- 从深度图中取有效深度。
- 过滤掉小于 `0.1m` 和大于 `2.0m` 的点。
- 返回中值深度。

这样比直接取中心一个像素更稳，因为深度图可能有空洞或噪声。

### `get_3d_coordinates(depth_frame, pixel_x, pixel_y, depth_value=None)`

把 2D 像素坐标反投影为 3D 点：

- 如果传入 `depth_value`，使用 ROI 中值深度。
- 否则使用 `depth_frame.get_distance(pixel_x, pixel_y)`。
- 调用 `rs.rs2_deproject_pixel_to_point()` 得到 `[x, y, z]`。

### 距离过滤

`process_frame()` 会计算目标中心到相机的距离：

```python
distance = math.sqrt(x**2 + y**2 + z**2)
```

只保留 `0.2m <= distance <= 1.0m` 的 3D 点。超出范围时：

```python
point_3d = None
```

如果 `position_3d` 最终不为 `None`，窗口会增加一行：

```text
XYZ: (x, y, z)m
```

## 7. 第四步：分配跨帧 track_id

文件：`src/cade_vision/src/cade_vision/tracker.py`

`process_frame()` 调用：

```python
track_ids = self._tracker.assign(new_detections)
for obj, tid in zip(new_detections, track_ids):
    obj["track_id"] = tid
```

### `CadeTracker.assign(detections)`

为当前帧每个检测对象分配稳定 ID。

如果没有旧轨迹，直接分配新 ID：

```text
0, 1, 2, ...
```

如果已有轨迹，则计算每个旧轨迹和当前检测的匹配代价。

### `CadeTracker._cost(track, det)`

匹配代价分两层：

1. 如果旧轨迹和当前检测都有 `position_3d`：
   - 计算 3D 欧氏距离。
   - 小于 `MAX_3D_DIST = 0.3m` 时认为可以匹配。
2. 否则退回 2D bbox IOU：
   - 调用 `_iou()`
   - IOU 大于 `MIN_IOU = 0.3` 时可匹配。

### 匹配算法

- 如果 `scipy` 可用，使用 `linear_sum_assignment()` 做匈牙利全局最优匹配。
- 如果不可用，使用 `_greedy_match()` 贪心回退。

### 生命周期

旧轨迹连续 `MAX_AGE = 5` 帧未匹配才会过期。

### `CadeTracker._update()` / `_new_tracks()`

将当前检测的 `bbox`、`center`、`position_3d` 写入轨迹状态，供下一帧匹配。

窗口第一行会在类别置信度后追加：

```text
[ID: track_id]
```

例如：

```text
#0 person 0.86 [ID: 3]
```

## 8. 第五步：衣物框关联到 person

文件：`src/cade_vision/src/cade_vision/cloth.py`

`process_frame()` 调用：

```python
associate_clothing(new_detections, color_image)
```

这一步只修改 `class_name == "person"` 的对象，给 person 增加：

- `cloth_color`
- `cloth_type`

### `associate_clothing(detections, color_image)`

流程：

1. 把 `person` 框分成 `persons`。
2. 把非 person 框视为衣物框：
   - `_is_clothing(class_name)`
3. 对每个衣物框，找 IOU 最大的人框：
   - `_iou(person["bbox"], cloth["bbox"])`
4. 只接受 IOU 大于 `0.3` 的匹配。
5. 对每个人，上衣和下装各保留 IOU 最高的一件。
6. 调用 `_extract_color()` 提取颜色。
7. 写回 person：
   - `person["cloth_color"] = "..."`
   - `person["cloth_type"] = "..."`

### `_extract_color(color_image, bbox)`

从衣物 bbox 裁剪图像：

- 去掉四周 15% 边缘，减少背景干扰。
- 转 HSV。
- 求 HSV 均值。
- 调用 `_hsv_to_color()` 映射到颜色名。

### `_hsv_to_color(hsv_mean)`

按经验阈值返回：

- `white`
- `black`
- `gray`
- `red`
- `yellow`
- `blue`
- `unknown`

如果没有衣物框或关联失败，person 默认：

```python
cloth_color = "unknown"
cloth_type = "unknown"
```

窗口中 person 会增加一行：

```text
Cloth: cloth_color/cloth_type
```

## 9. 第六步：person 姿态和手势分析

这一步只对 `class_name == "person"` 的检测执行。普通物体不会进入姿态手势流程。

入口文件：

- `scripts/open_vision_node.py`

规则文件：

- `kits/posture_gesture/__init__.py`
- `kits/posture_gesture/rules/posture.py`
- `kits/posture_gesture/rules/gesture.py`
- `kits/posture_gesture/rules/utils.py`
- `kits/posture_gesture/temporal_buffer.py`

### 9.1 `process_frame()` 裁剪 person 图像

对每个 person：

```python
x1, y1, x2, y2 = obj["bbox"]
person_crop = color_image[y1:y2, x1:x2]
```

如果 crop 为空，跳过该人。

### 9.2 两段式提交 MediaPipe 任务

第一遍循环只提交，不等待：

```python
obj["_pg_fut_pose"] = self._pool.submit(
    self.posture_gesture.process_pose, person_crop
)
obj["_pg_fut_hands"] = self._pool.submit(
    self.posture_gesture.process_hands, person_crop
)
```

第二遍循环统一收集：

```python
pose_data = fut_pose.result()
hands_data = fut_hands.result()
```

这样多人场景下，所有人的 Pose/Hands 任务先进入线程池，再统一等待结果，避免每个人串行阻塞。

### 9.3 `PostureGestureAnalyzer.process_pose()`

文件：`kits/posture_gesture/__init__.py`

输入：`person_crop`

流程：

1. 检查 MediaPipe 是否可用。
2. 检查 crop 大小。
3. BGR 转 RGB。
4. 调用 `self.pose.process(rgb)`。
5. 如果检测到 Pose，返回：

```python
{
    "landmarks": [(x, y, visibility), ...],
    "img_h": h,
    "img_w": w,
}
```

如果未检测到人体关键点，返回 `None`。

### 9.4 `PostureGestureAnalyzer.process_hands()`

文件：`kits/posture_gesture/__init__.py`

输入：`person_crop`

流程：

1. BGR 转 RGB。
2. 调用 `self.hands.process(rgb)`。
3. 对每只手读取 handedness：
   - `Left`
   - `Right`
4. 只保留 score 大于 `0.5` 的手。
5. 返回：

```python
{
    "Left": {
        "wrist": (x, y),
        "index_tip": (x, y),
        "landmarks": [(x, y), ...],
    } 或 None,
    "Right": ...,
}
```

### 9.5 从深度图补 person 关键点 3D

文件：`scripts/open_vision_node.py`

如果是 RealSense 且 `pose_data` 有效，`process_frame()` 会把 Pose 归一化坐标映射回原图像素：

```python
px = int(lm[idx][0] * (x2 - x1) + x1)
py = int(lm[idx][1] * (y2 - y1) + y1)
```

只采样这些关键点：

```text
11, 12: shoulders
23, 24: hips
25, 26: knees
27, 28: ankles
```

每个点再调用：

```python
self.get_3d_coordinates(depth_frame, px, py)
```

最终形成：

```python
keypoints_3d = {idx: pt, ...}
```

这个字典只用于姿态规则，不直接显示在窗口里。

### 9.6 `PostureGestureAnalyzer.analyze_from_landmarks()`

文件：`kits/posture_gesture/__init__.py`

输入：

- `pose_data`
- `person_id = obj["track_id"]`
- `hands_data`
- `with_temporal=True`
- `keypoints_3d`

如果 `pose_data is None`，返回 `_unknown_result()`：

```python
{
    "landmarks": [],
    "posture": "unknown",
    "gesture": "unknown",
    "elbow_angle_left": None,
    ...
}
```

如果 Pose 有效，调用两类规则：

```python
posture, _ = classify_posture_3d(landmarks, h, keypoints_3d)
gesture, elbow_l, elbow_r, wrist_l, wrist_r = classify_static_gesture(
    landmarks, h, w, hands_data
)
```

然后得到初始结果：

```python
{
    "landmarks": landmarks,
    "posture": posture,
    "gesture": gesture,
    "elbow_angle_left": elbow_l,
    "elbow_angle_right": elbow_r,
    "wrist_angle_left": wrist_l,
    "wrist_angle_right": wrist_r,
}
```

因为 `with_temporal=True` 且传了 `person_id`，还会调用 `_apply_temporal()`。

## 10. 姿态 posture 是如何来的

文件：`kits/posture_gesture/rules/posture.py`

入口函数：

```python
classify_posture_3d(landmarks, img_height, keypoints_3d=None)
```

返回：

```python
(posture, confidence)
```

其中 `confidence` 当前只在姿态规则内部用于表达规则命中强度，`process_frame()` 不把它显示出来。

### 无 3D 深度时：`_classify_posture_2d_fallback()`

如果 `keypoints_3d` 为空，走 2D fallback：

1. `_leg_angle_2d()` 分别计算左右腿 2D 投影角度。
2. 根据平均角度粗分：
   - 大于 140：`standing`
   - 70 到 125：`sitting`
   - 小于 30：`lying`
   - 否则 `unknown`

辅助函数：

- `_is_visible_2d()`
- `_get_2d_vec()`
- `_leg_angle_2d()`

### 有 3D 深度时：`classify_posture_3d()`

先预计算：

- `is_torso_vertical_3d(keypoints_3d)`
- `_leg_angle_2d()` 左右腿角度

然后按优先级判断：

1. `check_sitting_3d()`
   - 坐姿优先，因为坐姿容易被站立规则吞掉。
   - 使用 3D 躯干/大腿夹角，以及髋膝踝高度链。
2. `check_standing_3d()`
   - 要求躯干直立、腿部角度接近站立、3D 高度链合理。
   - 命中后还会用 `check_lying_3d()` 反查高度压缩，避免倒地误判为站立。
3. `check_lying_3d()`
   - 看肩、髋、膝在高度轴上是否压缩到较小范围。
4. 都没命中：
   - 返回 `unknown`

窗口中 person 的 `PG:` 第一部分来自这里：

```text
PG: posture/gesture
```

## 11. 手势 gesture 是如何来的

文件：`kits/posture_gesture/rules/gesture.py`

入口函数：

```python
classify_static_gesture(landmarks, h, w, hands_data=None)
```

它先输出一个单帧初筛手势，通常是：

- `maybe_raising_left_arm`
- `maybe_raising_right_arm`
- `maybe_raising_both_arms`
- `maybe_pointing_left`
- `maybe_pointing_right`
- `maybe_pointing_both`
- `none`
- `unknown`

这个 `maybe_...` 还不是窗口最终显示的手势。最终手势会由时序规则 `_apply_temporal()` 再判一次。

### `rules/utils.py`

`classify_static_gesture()` 会调用这些工具函数：

- `compute_side_angles()`
  - 计算左右前臂角度和手腕旋转角度，供时序缓冲使用。
- `is_visible()`
  - 检查关键点可见性。
- `get_landmark()`
  - 取 2D 关键点坐标。
- `get_index_pose()`
  - 取左右食指 MCP 点。

### `is_arm_raised()`

纯 2D 几何举手判断：

- 手腕接近或高过肩膀。
- 前臂方向接近竖直向上。

命中后输出 `maybe_raising_*`。

### `is_pointing()`

指向判断结合 Pose 和 Hands：

- Pose 上 elbow、wrist、index_mcp 三点接近水平。
- Hands 上食指伸出，其余手指收回或按兜底模式通过。
- 根据肘腕 x 坐标判断指向方向。

命中后输出 `maybe_pointing_*`。

## 12. 时序手势如何覆盖单帧结果

文件：

- `kits/posture_gesture/__init__.py`
- `kits/posture_gesture/temporal_buffer.py`
- `kits/posture_gesture/rules/gesture.py`

### `PostureGestureAnalyzer._apply_temporal()`

`analyze_from_landmarks()` 生成单帧结果后，调用 `_apply_temporal(result, person_id)`。

这里的 `person_id` 是 `obj["track_id"]`，所以同一个人跨帧会进入同一个时序缓冲。

### `RingBuffer.push_forearm()` / `push_wrist()`

文件：`temporal_buffer.py`

把当前帧左右手角度写入缓冲：

- `forearm`
- `wrist_rot`

底层使用 `_push_side()`：

- 非 None 角度追加到对应 person/side。
- 连续 None 超过阈值会清空该侧旧数据。

### `RingBuffer.get_forearm_variance()` / `get_wrist_variance()`

计算最近窗口内的角度方差。

如果传入 `jump_threshold`，会走 `_jump_aware_variance()`：

- 从最新帧向前查找角度跳变。
- 只保留跳变之后的数据。
- 跳变后帧数太少时返回 0，避免刚发生突变就误判为挥手。

### `judge_all_temporal_gestures()`

文件：`rules/gesture.py`

输入：

- 单帧初筛 `raw_gesture`
- 左右手 forearm/wrist 方差
- 当前 landmarks
- 阈值 `T_FOREARM`、`T_WRIST`、`SHOULDER_OFFSET`

决策顺序：

1. 如果前臂或手腕方差超过阈值，且手在肩附近：
   - 输出 `waving`
2. 如果单帧是 `maybe_raising_*` 且对应手足够静止：
   - 输出 `raising_left_arm` / `raising_right_arm` / `raising_both_arms`
3. 如果单帧是 `maybe_pointing_*` 且对应手足够静止：
   - 输出 `pointing_left` / `pointing_right` / `pointing_both`
4. 如果有 maybe 意图但时序不稳定：
   - 输出 `none`
5. 否则：
   - 输出 `none`

最终窗口中的 `gesture` 来自这个时序决策后的结果。

## 13. person 对象写回哪些姿态手势字段

文件：`scripts/open_vision_node.py`

`process_frame()` 拿到 `pg_result` 后，只把三个字段写回 `obj`：

```python
obj["posture"] = pg_result.get("posture", "unknown")
obj["gesture"] = pg_result.get("gesture", "unknown")
obj["landmarks"] = pg_result.get("landmarks", [])
```

窗口里只显示 `posture` 和 `gesture`：

```text
PG: posture/gesture
```

`landmarks` 不直接画到窗口上，但会保存在 `self.detected_objects` 中，供其他模块读取。

## 14. 第七步：更新节点内的检测缓存

文件：`scripts/open_vision_node.py`

`process_frame()` 完成所有后处理后：

```python
with self._lock:
    self.detected_objects = new_detections
```

这份缓存包含完整检测结果。任务模块通过 `get_latest_detections()` 读取它。

相关函数：

- `get_latest_detections()`
  - 返回 `copy.deepcopy(self.detected_objects)`。

任务文件例如 `tasks/search_task.py`、`tasks/info_task.py` 会消费这些检测结果，但它们不参与 CV 窗口里单帧结果的生成。

## 15. 第八步：CV 窗口如何画出每个目标

文件：`scripts/open_vision_node.py`

这一段仍在 `process_frame()` 中。

### 没有检测结果

如果 `new_detections` 为空：

```python
cv2.putText(display_image, "No objects detected", ...)
```

窗口显示一行红字：

```text
No objects detected
```

### 有检测结果

对每个 `obj`：

```python
x1, y1, x2, y2 = obj["bbox"]
c_name = obj["class_name"]
```

### 框颜色和粗细

如果当前有任务目标：

```python
is_target = target_cls is not None and target_cls in c_name.lower()
```

- 当前目标：红色 `(0, 0, 255)`，线宽 3。
- 非当前目标：绿色 `(0, 255, 0)`，线宽 1。

调用：

```python
cv2.rectangle(display_image, (x1, y1), (x2, y2), box_color, thickness)
```

### 第一行：索引、类别、置信度、追踪 ID

基础文字：

```python
lines = [
    f"#{obj['index']} {c_name} {obj['confidence']:.2f}",
]
```

如果有 `track_id`：

```python
lines[0] += f" [ID: {obj['track_id']}]"
```

显示形态：

```text
#0 person 0.86 [ID: 3]
```

### person 第二行：衣物

仅当 `c_name == "person"`：

```python
cloth_str = f"Cloth: {obj.get('cloth_color', 'unk')}/{obj.get('cloth_type', 'unk')}"
lines.append(cloth_str)
```

显示形态：

```text
Cloth: blue/jacket
```

如果没有衣物结果：

```text
Cloth: unknown/unknown
```

### person 第三行：姿态手势

仅当 `c_name == "person"`：

```python
pg_str = f"PG: {obj.get('posture', 'unk')}/{obj.get('gesture', 'unk')}"
lines.append(pg_str)
```

显示形态：

```text
PG: standing/waving
```

### 3D 坐标行

如果 `position_3d` 不为 `None`：

```python
cx, cy, cz = obj["position_3d"]
lines.append(f"XYZ: ({cx:.2f}, {cy:.2f}, {cz:.2f})m")
```

显示形态：

```text
XYZ: (0.12, -0.05, 0.85)m
```

### 多行文字绘制

文字从框上方开始：

```python
line_height = 18
curr_y = y1 - 10 - (len(lines) - 1) * line_height
```

每一行画两遍：

1. 黑色粗描边：
   - `(0, 0, 0)`
   - thickness 2
2. 前景彩色文字：
   - `box_color`
   - thickness 1

调用：

```python
cv2.putText(...)
```

最后显示窗口：

```python
cv2.imshow("CADE Vision", display_image)
```

## 16. 一个 person 检测结果的字段生命周期

一个人从 YOLO 输出到窗口显示，字段大致这样增长：

```python
{
    # YOLO 阶段
    "index": 0,
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.86,
    "bbox": (x1, y1, x2, y2),
    "center": (center_x, center_y),
    "position_3d": [x, y, z] 或 None,

    # tracker.py
    "track_id": 3,

    # cloth.py
    "cloth_color": "blue" 或 "unknown",
    "cloth_type": "jacket" 或 "unknown",

    # kits/posture_gesture/*
    "posture": "standing",
    "gesture": "waving",
    "landmarks": [...],
}
```

窗口显示对应为：

```text
#0 person 0.86 [ID: 3]
Cloth: blue/jacket
PG: standing/waving
XYZ: (0.12, -0.05, 0.85)m
```

## 17. 一个普通物体检测结果的字段生命周期

普通物体不会进入衣物 person 写回，也不会进入姿态手势分析。

字段通常是：

```python
{
    "index": 1,
    "class_id": 12,
    "class_name": "apple",
    "confidence": 0.74,
    "bbox": (x1, y1, x2, y2),
    "center": (center_x, center_y),
    "position_3d": [x, y, z] 或 None,
    "track_id": 4,
}
```

窗口显示对应为：

```text
#1 apple 0.74 [ID: 4]
XYZ: (0.20, -0.03, 0.62)m
```

如果没有 RealSense 深度，则没有 `XYZ` 行。

## 18. 文件与函数职责总表

| 文件 | 函数/类 | 在检测结果生成中的作用 |
| --- | --- | --- |
| `src/cade_vision/scripts/open_vision_node.py` | `main()` | 解析参数，创建节点，启动运行循环。 |
| `src/cade_vision/scripts/open_vision_node.py` | `OpenVisionNode.__init__()` | 初始化图像源、YOLO、姿态手势、线程池、追踪器和显示窗口。 |
| `src/cade_vision/scripts/open_vision_node.py` | `_init_image_source()` | 选择 RealSense、文件或 USB 摄像头输入。 |
| `src/cade_vision/scripts/open_vision_node.py` | `_init_realsense()` | 初始化 RealSense 彩色流、深度流、对齐器和相机内参。 |
| `src/cade_vision/scripts/open_vision_node.py` | `_init_file_source()` | 初始化图片或视频文件输入。 |
| `src/cade_vision/scripts/open_vision_node.py` | `_init_usb_cam()` | 初始化 USB 摄像头输入。 |
| `src/cade_vision/scripts/open_vision_node.py` | `run()` | 循环调用 `process_frame()`，并处理显示窗口退出。 |
| `src/cade_vision/scripts/open_vision_node.py` | `_capture_frame()` | 读取当前彩色帧和可选深度帧。 |
| `src/cade_vision/scripts/open_vision_node.py` | `process_frame()` | 单帧主流水线：YOLO、3D、追踪、衣物、姿态手势、缓存、绘制窗口。 |
| `src/cade_vision/scripts/open_vision_node.py` | `get_median_depth_in_roi()` | 在检测框中心附近取稳健中值深度。 |
| `src/cade_vision/scripts/open_vision_node.py` | `get_3d_coordinates()` | 将 2D 像素点反投影成 3D 坐标。 |
| `src/cade_vision/scripts/open_vision_node.py` | `_on_task_cmd()` | 更新当前任务目标，间接影响 YOLO 类别集合和窗口框颜色。 |
| `src/cade_vision/scripts/open_vision_node.py` | `_set_model_classes()` | 更新 YOLO-World 检测类别。 |
| `src/cade_vision/src/cade_vision/tracker.py` | `CadeTracker.assign()` | 为每个检测对象分配稳定 `track_id`。 |
| `src/cade_vision/src/cade_vision/tracker.py` | `CadeTracker._cost()` | 按 3D 距离或 2D IOU 计算匹配代价。 |
| `src/cade_vision/src/cade_vision/tracker.py` | `CadeTracker._iou()` | 计算两个 bbox 的 IOU。 |
| `src/cade_vision/src/cade_vision/tracker.py` | `CadeTracker._update()` / `_new_tracks()` | 更新或创建轨迹状态。 |
| `src/cade_vision/src/cade_vision/tracker.py` | `CadeTracker._greedy_match()` | 在无 scipy 时提供贪心匹配回退。 |
| `src/cade_vision/src/cade_vision/cloth.py` | `associate_clothing()` | 将衣物框关联到 person，并写入 `cloth_color` / `cloth_type`。 |
| `src/cade_vision/src/cade_vision/cloth.py` | `_is_clothing()` | 判断非 person 检测是否作为衣物候选。 |
| `src/cade_vision/src/cade_vision/cloth.py` | `_iou()` | 计算 person 框和衣物框的 IOU。 |
| `src/cade_vision/src/cade_vision/cloth.py` | `_extract_color()` | 从衣物框中心区域提取 HSV 均值。 |
| `src/cade_vision/src/cade_vision/cloth.py` | `_hsv_to_color()` | 把 HSV 均值映射成颜色名。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/__init__.py` | `PostureGestureAnalyzer.__init__()` | 初始化 MediaPipe Pose、Hands 和时序缓冲。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/__init__.py` | `process_pose()` | 对 person crop 运行 Pose，输出 33 个关键点。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/__init__.py` | `process_hands()` | 对 person crop 运行 Hands，输出左右手关键点。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/__init__.py` | `analyze_from_landmarks()` | 融合 Pose、Hands、3D 关键点，得到 posture/gesture。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/__init__.py` | `_apply_temporal()` | 将单帧手势放入时序缓冲，输出最终 gesture。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/__init__.py` | `_unknown_result()` | Pose 失败时提供 unknown 结果。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py` | `classify_posture_3d()` | 姿态分类主入口。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py` | `_classify_posture_2d_fallback()` | 无 3D 时的 2D 姿态兜底。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py` | `_leg_angle_2d()` | 计算腿部 2D 投影角度。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py` | `is_torso_vertical_3d()` | 判断躯干是否接近竖直。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py` | `check_sitting_3d()` | 坐姿规则。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py` | `check_standing_3d()` | 站立规则。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/posture.py` | `check_lying_3d()` | 平躺/倒地规则。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/gesture.py` | `classify_static_gesture()` | 单帧静态手势初筛。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/gesture.py` | `is_arm_raised()` | 举手几何规则。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/gesture.py` | `is_pointing()` | 指向几何和手指规则。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/gesture.py` | `check_shoulder_proximity()` | 挥手时序判定的肩部空间门控。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/gesture.py` | `judge_all_temporal_gestures()` | 将 maybe 手势和多帧方差转成最终手势。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/utils.py` | `compute_side_angles()` | 计算前臂角度和手腕旋转角度。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/utils.py` | `get_landmark()` / `get_index_pose()` | 提取 Pose 关键点坐标。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/rules/utils.py` | `is_visible()` / `is_visible_relaxed()` | 判断关键点可见性。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/temporal_buffer.py` | `RingBuffer.push_forearm()` / `push_wrist()` | 按 person track_id 记录角度序列。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/temporal_buffer.py` | `RingBuffer.get_forearm_variance()` / `get_wrist_variance()` | 计算前臂/手腕角度方差。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/temporal_buffer.py` | `RingBuffer._jump_aware_variance()` | 跳变过滤后的方差计算。 |
| `src/cade_vision/src/cade_vision/kits/posture_gesture/temporal_buffer.py` | `RingBuffer.clear()` | 视频循环重置时清空时序缓冲。 |

## 19. 注意事项

- `Analyzer` 当前在 `OpenVisionNode.__init__()` 中可能初始化，但 `process_frame()` 的检测窗口结果没有使用它。
- `tasks/*` 模块不会直接绘制窗口；它们读取 `self.detected_objects` 做任务判断。
- `position_3d` 是检测框中心点的 3D 坐标，不是人体骨架点坐标。
- `keypoints_3d` 是姿态规则内部使用的人体关键点 3D 坐标，不直接显示。
- `confidence` 是 YOLO 检测置信度；姿态/手势结果目前不向窗口输出置信度。
- 普通物体也会获得 `track_id`，但只有 person 会显示 `Cloth:` 和 `PG:`。
