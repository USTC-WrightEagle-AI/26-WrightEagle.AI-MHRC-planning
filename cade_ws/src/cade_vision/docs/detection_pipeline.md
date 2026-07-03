# cade_vision 检测流程

## 多进程数据流

1. `frame_source` 发布最新 color/depth/camera_info。
2. `person_worker` 订阅帧和 profile，输出 `/vision/person_raw_task3`。
3. `pose_gesture_worker` 订阅 color + person，在 `gesture/posture/full` 下输出 `/vision/pose_gesture_raw_task3`。
4. `cloth_worker` 订阅 color + person，在 `cloth/full` 下输出 `/vision/cloth_raw_task3`。
5. `gateway` 根据当前 profile 选择最新结果融合，更新 `detected_objects`，发布 task 状态和 people tracks。

## 检测对象结构

基础字段：

```python
{
    "index": int,
    "class_id": int,
    "class_name": str,
    "source_model": str,
    "confidence": float,
    "bbox": (x1, y1, x2, y2),
    "center": (cx, cy),
    "position_3d": [x, y, z] | None,
    "track_id": int | None,
}
```

person 可选字段：

```python
{
    "posture": str,
    "gesture": str,
    "landmarks": list,
    "cloth_items": list,
    "cloth_summary": str,
    "cloth_color": str,
    "cloth_type": str,
}
```

## 性能原则

worker 常驻并加载模型，但只有 profile 匹配时才推理。这样保留任务响应速度，同时避免 idle 时 3 个 YOLO + MediaPipe 同时抢资源。

`--perf-debug` 会显示当前 `profile` 和各 worker 的 fps/result age，用于判断是否有 worker 在错误 profile 下仍然推理。

## Task 输出

`observe_people` 返回当前 person 列表及 posture/gesture/cloth 字段。`filter_by_attributes` 会根据任务属性触发对应 profile 并等待新结果，避免读取旧缓存。
