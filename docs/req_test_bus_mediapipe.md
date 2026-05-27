# 测试任务：单张 bus.jpg 姿态手势识别 + 可视化

## 背景

刚实现了 posture_gesture.py 管线，需要对 bus.jpg 做一次端到端测试，验证 YOLO-World + MediaPipe Pose 的联动是否正常。

## 要求

### 1. 模型路径

YOLO-World 已存在，**不要下载新模型**：

```
CADE/models/yolo_world/yolo_world.pt
```

### 2. 输入图片

```
CADE/test_images/bus.jpg
```

### 3. 测试步骤

1. 加载 YOLO-World，检测类别设为 ["person"]
2. 对每个检测到的人框，裁剪后喂给 `PostureGestureAnalyzer.analyze()`
3. 收集每人的：bbox、置信度、posture、gesture、33 个关键点坐标

### 4. 可视化输出

在 bus.jpg 原图上叠加绘制以下信息，保存到：

```
CADE/test_images_result/bus_mediapipe_test.jpg
```

绘制内容（每人的框上）：
- 人框（绿色矩形）
- 左上角标注：`#{index} posture/gesture`
- 33 个 MediaPipe Pose 关键点（红色小圆点）
- 关键点之间的连线（骨架，用淡蓝色线条）

### 5. 同时打印到终端的文本输出

```
检测到 N 个人

人 #0 (置信度 0.XX) bbox=(x1,y1,x2,y2)
  姿态: standing
  手势: none
  关键点: 33 个

人 #1 ...
```

### 6. 注意事项

- 使用 conda CADE 环境的 Python（`/home/huyanshen/miniforge3/envs/CADE/bin/python`）
- mediapipe 版本已降级到 0.10.7，使用旧版 API（`mp.solutions.pose`）
- YOLO-World 路径：`CADE/models/yolo_world/yolo_world.pt`，加载后用 `.set_classes(["person"])` 设置检测类别
- 不要走 ROS 管线，直接用 Python 脚本独立运行
- 测试脚本放在 `CADE/tests/test_bus_mediapipe.py`
- 输出图片放在 `CADE/test_images_result/bus_mediapipe_test.jpg`
