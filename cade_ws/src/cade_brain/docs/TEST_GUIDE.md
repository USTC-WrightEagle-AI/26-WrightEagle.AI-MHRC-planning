# CADE Brain 集成测试指南

本文档用于在比赛任务风格下测试 `cade_brain`、`cade_vision`、`cade_navigation`、ASR/TTS 和底层导航栈的完整链路。

核心链路：

```text
/asr
  -> cade_brain
  -> /cade/task_cmd_task3 -> cade_vision -> /cade/task_status_task3
  -> /cade/task_cmd       -> cade_navigation -> move_base -> /cade/task_status
  -> /tts
```

## 1. 启动顺序

### 1.1 启动 roscore

```bash
roscore
```

### 1.2 启动底层机器人、定位和导航栈

`cade_navigation` 只是 Brain 到 `move_base` 的任务桥，不负责启动底盘、定位、costmap 或 `move_base`。先启动现有底层系统。

常用 ZHXY 启动方式：

```bash
cd /home/nvidia/ZHXY/sh
./launch_all.sh
```

打开 rviz:

```bash
source /home/nvidia/workdir/localizaion_ws/devel/setup.bash
rviz -d /home/nvidia/workdir/localizaion_ws/src/jh_localization/rviz/localization.rviz
```

如果地图选对，RViz 里当前激光/点云应能和地图结构对齐；用 `2D Pose Estimate` 给出初始位姿后，机器人在 `map` 下的位置应稳定。如果地图选错，激光/点云会明显对不上墙、走廊或房间结构，导航目标也会整体错位。

启动后检查：

```bash
cd /home/nvidia/Desktop/task3/cade_ws
source devel/setup.bash

rosnode list | grep -E 'move_base|localization|laser|pointcloud'
rostopic list | grep -E '^/tf$|^/map$|^/move_base|^/local_odom$'
rosrun cade_navigation get_current_map_pose.py --timeout 3.0
```

`get_current_map_pose.py` 应输出当前机器人在 `map` 下的 `x y yaw_deg`。如果这里失败，先修定位、TF 或 `move_base`。

### 1.3 初始化机器人位姿

如果定位启动后不知道机器人在地图中的初始位置，用 RViz 初始化：

1. 打开 RViz，确认 Fixed Frame 是 `map`。
2. 使用 `2D Pose Estimate`，在地图上点机器人当前真实位置和朝向。
3. 等 2 到 3 秒后执行：

```bash
rosrun cade_navigation get_current_map_pose.py --timeout 3.0
```

输出稳定后再开始任务测试。

也可以直接向 `/initialpose` 发布初始位姿；RViz 更安全直观，优先用 RViz。

### 1.4 启动 CADE 上层节点

```bash
cd /home/nvidia/Desktop/task3/cade_ws
source devel/setup.bash
roslaunch launch/cade_full.launch
```

当前 `cade_full.launch` 默认启动：

- `cade_brain`
- `cade_vision`
- `cade_navigation`

ASR/TTS 节点默认不启动；测试时直接向 `/asr` 发布文本、监听 `/tts` 即可。如果要启用真实语音节点，使用：

```bash
roslaunch launch/cade_full.launch enable_asr:=true enable_tts:=true
```

启用真实语音前要确保当前 Python 环境已经安装 `sherpa_onnx`、`soundfile` 等语音依赖。

其中 `cade_navigation` 会加载：

```text
src/cade_navigation/config/named_locations.json
```

这个文件定义了 `kitchen`、`sofa`、`desk` 等语义地点到地图坐标的映射。

## 2. 地点坐标文件与标定

地点文件：

```bash
src/cade_navigation/config/named_locations.json
```

当前初始内容是占位坐标，用于先跑通链路：

```json
{
  "sofa": [0.0, 0.0],
  "side_tables": [1.8, 0.0],
  "desk": [3.5, 0.8],
  "desk_lamp": [3.9, 1.1],
  "office": [5.0, 0.2],
  "bathroom": [0.4, 3.0],
  "bedroom": [2.2, 3.4],
  "kitchen": [5.2, 3.2],
  "tv_stand": [1.0, 4.8],
  "trash": [4.8, 4.6],
  "table": [1.8, 0.0]
}
```

支持格式：

```json
"kitchen": [5.2, 3.2]
```

等价于 map 坐标 `[x, y, 0]`。

如果需要固定朝向：

```json
"kitchen": {"position": [5.2, 3.2, 0.0], "yaw_deg": 90}
```

标定一个地点的推荐流程：

1. 启动底层定位和导航栈。
2. 用遥控器或 RViz 把机器人移动到地点附近，例如沙发旁。
3. 让机器人朝向你希望的到达朝向。
4. 读取当前坐标：

```bash
rosrun cade_navigation get_current_map_pose.py --timeout 3.0
```

输出示例：

```text
Current robot pose in map:
x:       1.234
y:       5.678
yaw_deg: 91.200

1.234 5.678 91.200
```

5. 修改 `named_locations.json`：

```json
"sofa": {"position": [1.234, 5.678, 0.0], "yaw_deg": 91.2}
```

6. 重启 `cade_navigation` 或重启 `cade_full.launch`。`cade_navigation` 启动时读取地点表，不会实时监听文件变化。

地点名匹配支持大小写、空格和下划线差异，例如：

- `side tables`
- `side_tables`
- `Side Tables`

都会匹配到 `side_tables`。

## 3. 直接验证导航模块

先绕过 LLM，直接测试 `cade_navigation` 是否能接收命名地点并发给 `move_base`。

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"navigation\",\"position\":\"kitchen\",\"timeout\":60}'"
```

观察：

```bash
rostopic echo /cade/task_cmd
rostopic echo /cade/task_status
rostopic echo /move_base/goal
rostopic echo /move_base/status
```

期望：

- `/cade/task_cmd` 收到 `navigation`。
- `/move_base/goal` 收到 `frame_id: map` 的目标。
- `/cade/task_status` 最终返回 `SUCCESS`、`FAILED` 或 `TIMEOUT`。

测试未知地点：

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"navigation\",\"position\":\"garage\",\"timeout\":10}'"
```

期望 `/cade/task_status` 中出现类似：

```text
Unknown named location: garage
```

## 4. 向 Brain 发送一句话

Brain 订阅 `/asr`，消息类型是 `std_msgs/String`。发布一条文本即可触发一次完整决策。

模板：

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Navigate to the kitchen'"
rostopic pub -1 /asr std_msgs/String "data: 'Search for the people with white T shirt'"
```

查看 Brain 输出：

```bash
rostopic echo /tts
```

如果只想让 TTS 播放一句机器人回复，可以直接发布到 `/tts`：

```bash
rostopic pub -1 /tts std_msgs/String "data: 'I am ready.'"
```

这只测试 TTS，不会触发 Brain 决策。

## 5. 比赛任务测试指令

下面 10 条按真实任务口吻发布到 `/asr`。不要在指令里加入工具选择、模块限制或调试提示。

### 5.1 Escort the person raising their left arm from the sofa to the side tables

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Escort the person raising their left arm from the sofa to the side tables'"
```

预期链路：

1. Brain 导航到 `sofa`。
2. Vision 查找 `raising_left_arm` 的人。
3. Navigation 跟随或陪同该人。
4. Brain 导航/确认到达 `side_tables`。

### 5.2 Give me a dice from the desk

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Give me a dice from the desk'"
```

预期链路：

1. Brain 导航到 `desk`。
2. Vision 查找 `dice`。
3. 抓取模块执行取物。
4. 回到用户附近或完成递交动作。

### 5.3 Meet jane in the office and follow them to the desk lamp

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Meet jane in the office and follow them to the desk lamp'"
```

预期链路：

1. 导航到 `office`。
2. 查找或识别 Jane。
3. 启动跟随。
4. 到达 `desk_lamp` 附近后结束。

### 5.4 Get a food from the desk and bring it to me

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Get a food from the desk and bring it to me'"
```

预期链路：

1. 导航到 `desk`。
2. 查找 `food`。
3. 抓取。
4. 返回用户或当前交互位置。

### 5.5 Look for a dish in the bathroom then fetch it and deliver it to simone in the bedroom

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Look for a dish in the bathroom then fetch it and deliver it to simone in the bedroom'"
```

预期链路：

1. 导航到 `bathroom`。
2. 查找并抓取 `dish`。
3. 导航到 `bedroom`。
4. 查找 Simone 并递交。

### 5.6 Meet angel in the office and follow them

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Meet angel in the office and follow them'"
```

预期链路：

1. 导航到 `office`。
2. 查找或识别 Angel。
3. 启动持续跟随。

### 5.7 Tell me the name of the person at the sofa

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Tell me the name of the person at the sofa'"
```

预期链路：

1. 导航到 `sofa`。
2. 观察附近的人。
3. 返回识别到的姓名；如果姓名识别模块未接入，应明确说明未能识别姓名。

### 5.8 Meet charlie in the bedroom and follow them to the kitchen

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Meet charlie in the bedroom and follow them to the kitchen'"
```

预期链路：

1. 导航到 `bedroom`。
2. 查找或识别 Charlie。
3. 跟随 Charlie。
4. 到达 `kitchen` 附近后结束。

### 5.9 Get a plate from the tv stand and throw it in the trash

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Get a plate from the tv stand and throw it in the trash'"
```

预期链路：

1. 导航到 `tv_stand`。
2. 查找并抓取 `plate`。
3. 导航到 `trash`。
4. 执行丢弃/放置动作。

### 5.10 Navigate to the side tables then meet charlie and follow them to the kitchen

```bash
rostopic pub -1 /asr std_msgs/String "data: 'Navigate to the side tables then meet charlie and follow them to the kitchen'"
```

预期链路：

1. 导航到 `side_tables`。
2. 查找或识别 Charlie。
3. 启动跟随。
4. 到达 `kitchen` 附近后结束。

## 6. 监控窗口

建议开几个独立终端观察：

```bash
rostopic echo /cade/task_cmd
rostopic echo /cade/task_status
rostopic echo /cade/task_cmd_task3
rostopic echo /cade/task_status_task3
rostopic echo /tts
```

导航相关：

```bash
rostopic echo /move_base/goal
rostopic echo /move_base/status
rostopic echo /target_marker
```

视觉相关：

```bash
rostopic echo /vision/people_tracks_task3
rostopic echo /vision/detections_3d_task3
```

跟随调试：

```bash
rostopic echo /cade/navigation/follow_debug
```

## 7. 判断测试是否通过

一次任务至少满足：

- Brain 收到 `/asr` 后有 `[ASR] Received` 日志。
- Brain 发布了合理的 vision 或 navigation action。
- 对语义地点，例如 `kitchen`，`/cade/task_cmd` 中 position 保持为地点名或 navigation 已解析到 map 目标。
- `cade_navigation` 向 `/move_base` 发出目标，或对不可执行任务返回明确失败。
- `cade_vision` 在需要找人、手势、衣物时被激活。
- `/tts` 最终有一句自然语言回复。

## 8. 常见问题

### `Navigate to kitchen` 没有移动

检查：

```bash
rostopic echo /cade/task_cmd
rostopic echo /cade/task_status
rostopic echo /move_base/goal
```

如果 `/cade/task_status` 显示 `Cannot connect to move_base action server`，说明底层 `move_base` 没启动或 action 名称不对。

如果显示 `Unknown named location`，检查 `named_locations.json` 是否有该地点，并重启 `cade_navigation`。

### 修改地点文件后没有生效

`cade_navigation` 启动时读取 `named_locations`，不会自动监听文件变化。修改后重启：

```bash
rosnode kill /cade_navigation
roslaunch launch/cade_full.launch
```

如果只单独启动导航桥：

```bash
rosrun cade_navigation navigation_node.py
```

但单独启动时要确保 `~named_locations` 已加载；更推荐用 `cade_full.launch`。

### `get_current_map_pose.py` 失败

说明 `map -> base_link` TF 不通。检查：

```bash
rostopic list | grep '^/tf$'
rosnode list | grep -E 'localization|move_base'
rostopic echo -n 1 /local_odom
```

先解决定位，再测试 Brain。

### 视觉找不到人或手势

检查：

```bash
rostopic echo /cade/task_cmd_task3
rostopic echo /cade/task_status_task3
rostopic echo /vision/people_tracks_task3
```

确认 `open_vision_node.py` 窗口中能看到人体框、pose 和 gesture/posture 结果。动作类任务需要人进入 RealSense 视野，并保持手势 2 到 3 秒。

### 跟随失败但视觉还看得到人

如果 `/cade/task_status` 中出现 `move_base_state: ABORTED` 或路径不可达，优先按导航问题处理：目标点可能落在障碍物、未知区域或 inflation layer 内。先测试命名地点导航，再测试跟随。

## 9. 建议测试顺序

1. 启动底层定位和导航，确认 `get_current_map_pose.py` 正常。
2. 直接测试 `/cade/task_cmd` 到 `kitchen`。
3. 标定并修正 `named_locations.json` 中的地点。
4. 启动 `cade_full.launch`。
5. 发布 `Navigate to the side tables`，确认语义导航通过。
6. 发布 `Tell me the name of the person at the sofa`，确认导航 + 视觉链路。
7. 发布 `Escort the person raising their left arm from the sofa to the side tables`，测试导航 + 手势 + 跟随。
8. 最后测试包含抓取/递交的物品任务。
