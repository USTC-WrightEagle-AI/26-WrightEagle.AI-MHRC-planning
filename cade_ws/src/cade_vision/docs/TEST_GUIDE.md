# cade_vision 测试指南

## RealSense 实机

```bash
cd ~/Desktop/task3/cade_ws
source devel/setup.bash

rosrun cade_vision open_vision_node.py \
  --image-source realsense \
  --device cuda \
  --display
```

指定 RealSense serial number：

```bash
rosrun cade_vision open_vision_node.py \
  --image-source realsense \
  --serial-number 346522071650 \
  --device cuda \
  --display
```

默认会加载包内模型：

- `models/yolo11n.pt`
- `models/yolo11s-fashionpedia-best.pt`

如需覆盖：

```bash
rosrun cade_vision open_vision_node.py \
  --image-source realsense \
  --model /path/to/yolo11n.pt \
  --cloth-model /path/to/yolo11s-fashionpedia-best.pt \
  --device cuda \
  --display
```

## 文件/视频测试

```bash
rosrun cade_vision open_vision_node.py \
  --image-source file \
  --image-path /path/to/test_video.mp4 \
  --device cpu \
  --display \
  --loop \
  --playback-speed 0.5
```

## 任务命令测试

观察人物：

```bash
rostopic pub -1 /cade/task_cmd_task3 std_msgs/String \
  "data: '{\"action\": \"observe_people\"}'"
```

观察物体/衣物：

```bash
rostopic pub -1 /cade/task_cmd_task3 std_msgs/String \
  "data: '{\"action\": \"observe_objects\"}'"
```

监听结果：

```bash
rostopic echo /vision/detections_3d_task3
rostopic echo /cade/task_status_task3
```
