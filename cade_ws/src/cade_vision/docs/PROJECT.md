# cade_vision 当前架构

`cade_vision` 是 task3 的视觉感知 ROS 包。当前线上路径已经拆成多进程 task-driven pipeline：

- `open_vision_node.py`：薄 launcher + gateway，负责任务、profile、融合、发布和显示。
- `open_vision_frame_source.py`：唯一相机拥有者，发布 color/depth/camera_info。
- `open_vision_person_worker.py`：只跑 person YOLO 和人物 3D/track_id。
- `open_vision_pose_gesture_worker.py`：只在 `gesture/posture/full` profile 下跑 MediaPipe + LightGBM。
- `open_vision_cloth_worker.py`：只在 `cloth/full` profile 下跑 cloth segmentation/color/association。

## 启动

默认从包内 `models/` 读取权重：

```bash
rosrun cade_vision open_vision_node.py --image-source realsense --device cuda
```

如需单独调试 worker，可先启动 gateway：

```bash
rosrun cade_vision open_vision_node.py --no-launch-workers --no-display
```

再分别启动需要的 worker 脚本。

## 内部话题

公开接口保持不变：

- 订阅 `/cade/task_cmd_task3`
- 发布 `/cade/task_status_task3`
- 发布 `/vision/detections_3d_task3`
- 发布 `/vision/people_tracks_task3`

内部 worker 话题：

- `/vision/profile_task3`
- `/vision/frame/color_task3`
- `/vision/frame/depth_task3`
- `/vision/frame/camera_info_task3`
- `/vision/person_raw_task3`
- `/vision/pose_gesture_raw_task3`
- `/vision/cloth_raw_task3`

## Profile 策略

gateway 根据任务自动切换 profile：

- `person`：只跑人物检测。
- `gesture`：跑人物检测 + pose/gesture。
- `posture`：跑人物检测 + pose/posture。
- `cloth`：跑人物检测 + cloth。
- `full`：同时跑 pose/gesture/posture/cloth。
- `idle`：默认低频 person，其他 worker 待命不推理。

任务结束后 profile 保温短时间，再自动回到 `idle`。

## 模型资源

包内模型位于 `src/cade_vision/models/`。CMake 会同步 `.pt` 和可选 `.engine` 到 devel Python 包目录。
