# Follow Person 单独测试指南

本文档用于单独验证 `cade_navigation` 的 `follow_person` 功能：从 `cade_vision` 的 OpenCV 窗口读取人物 `ID` 和 `XYZ`，然后手动向 `/cade/task_cmd` 发布跟随指令。

## 1. 前置条件

需要先启动：

1. 机器人底盘、定位、导航底层。
2. `move_base`。
3. `cade_navigation` 的 `navigation_node.py`。
4. `cade_vision` 的 `open_vision_node.py`，并打开显示窗口。

确认关键 topic 存在：

```bash
rostopic list | grep -E '^/cade/task_cmd$|^/cade/task_status$|^/vision/people_tracks_task3$|^/cade/navigation/follow_debug$'
```

确认 vision 正在持续发布人物轨迹：

```bash
rostopic echo /vision/people_tracks_task3
```

确认 TF 可用：

```bash
rosrun tf tf_echo map base_link
rosrun tf tf_echo base_link left_arm_base_link
```

如果 `map -> base_link` 不通，`move_base` 无法执行跟随目标；如果 `base_link -> left_arm_base_link` 不通，vision 的 3D 坐标无法接入机器人本体坐标系。

## 2. OpenCV 窗口里看什么

`open_vision_node.py` 的显示窗口中，人物框上会显示类似：

```text
#0 person 0.91 [ID: 36]
Cloth: blue jacket
PG: standing/unknown
XYZ: (-0.68, 0.16, 1.46)m
```

含义：

- `ID: 36` 是 vision 分配的人物 `track_id`。
- `XYZ` 是 vision 输出的人物 3D 坐标，单位是米。
- 这个 `XYZ` 不是 map 坐标，也不是导航命令里的 `[x, y, yaw_deg]`。
- `follow_person` 和 `frame_id=vision` 的 `navigation` 会把这个 `XYZ` 当作相机观测坐标，经内置相机外参和 TF 转换后交给 `move_base`。

## 3. 按 track_id 跟随

如果窗口中能看清 `ID`，推荐直接按 `track_id` 跟随：

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"follow_person\",\"track_id\":36,\"person_pos\":[-0.68,0.16,1.46],\"duration\":60.0,\"follow_distance\":0.8}'"
```

预期行为：

- navigation 在 `/vision/people_tracks_task3` 中寻找 `track_id=36` 的人物。
- 找到后持续读取该 ID 的最新 `position_3d`。
- 持续更新 `move_base` goal，使机器人保持约 `follow_distance` 米距离。

## 4. 按 person_pos 附近最近人跟随

如果你想测试“在我发的这个位置附近找最近的人，然后持续跟踪”，可以不传 `track_id`，只传窗口里看到的 `XYZ`：

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"follow_person\",\"person_pos\":[-0.68,0.16,1.46],\"duration\":60.0,\"follow_distance\":0.8}'"
```

预期行为：

- navigation 会读取 `/vision/people_tracks_task3` 当前 people 列表。
- 在所有带 `position_3d` 的人物中，找离 `person_pos` 最近的人。
- 最近距离必须小于 `initial_match_distance`，默认是 `1.0` 米。
- 匹配成功后，navigation 会锁定这个人的 `track_id`，后续持续按该 ID 跟随。

如果窗口中的 `XYZ` 抖动较大，或画面里多人靠得很近，可以临时放宽匹配半径：

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"follow_person\",\"person_pos\":[-0.68,0.16,1.46],\"duration\":60.0,\"follow_distance\":0.8,\"initial_match_distance\":1.5}'"
```

## 5. 按 vision 坐标一次性靠近人物

如果你只想让机器人走到窗口里这个人的附近，并在到达后结束任务，不需要持续跟随，可以使用普通 `navigation`，但必须设置 `frame_id` 为 `vision`：

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"navigation\",\"position\":[-0.68,0.16,1.46],\"frame_id\":\"vision\",\"follow_distance\":0.8,\"timeout\":30.0}'"
```

含义：

- `position` 是 OpenCV 窗口中的 vision `XYZ: (x, y, z)m`。
- `frame_id="vision"` 告诉 navigation 这不是 `[x, y, yaw_deg]`。
- `follow_distance=0.8` 表示目标站位点会离该视觉目标约 0.8 米。
- 该任务是一次性的，到达或失败后会结束，不会持续追踪 `track_id`。

不要这样写：

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"navigation\",\"position\":[-0.68,0.16,1.46],\"frame_id\":\"base_link\",\"timeout\":30.0}'"
```

因为这会把第三个值 `1.46` 当成 `yaw_deg`，而不是 vision 深度 `z`。

## 6. 观察调试输出

跟随过程中打开：

```bash
rostopic echo /cade/navigation/follow_debug
```

常见事件：

- `LOCKING`：开始按 `track_id` 或 `person_pos` 锁定目标。
- `LOCKED`：通过 `person_pos` 最近匹配锁定到了某个 `track_id`。
- `GOAL_UPDATED`：已向 `move_base` 发送新的跟随目标点。
- `TRACKING`：正在持续跟踪。
- `LOST`：目标暂时丢失。
- `SEARCHING`：目标丢失后，尝试原地搜索。
- `FAILED`：跟随失败。
- `STOPPED`：跟随正常结束或被停止。

也可以看终态结果：

```bash
rostopic echo /cade/task_status
```

## 7. 停止跟随

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\":\"stop_navigation\"}'"
```

## 8. 常见问题

### `/vision/people_tracks_task3` 没有输出

说明 `open_vision_node.py` 没有正常运行、没有启用 ROS，或者没有检测到人物。先确认 OpenCV 窗口中能看到人物检测框和 `XYZ`。

### `No tracked person has position_3d`

说明 vision 检测到了人，但没有可用深度。检查 RealSense 深度流、目标距离、遮挡和光照。

### `Closest person is ... over ...`

说明你发的 `person_pos` 附近没有足够近的人。重新读取窗口里的 `XYZ`，或临时增大 `initial_match_distance`。

### `move_base failed while following: ABORTED`

通常是导航目标不可达、局部代价地图阻塞、TF 不通，或目标点落在障碍物/未知区域。先确认普通 `navigation` map 目标可以执行，再测试跟随。

### 不要给 `follow_person` 传 `frame_id`

当前 `follow_person` 不接受 `frame_id` 或 `source_frame`。vision 的 `XYZ` 会由 navigation 内部用相机外参和 TF 转换，不需要手动指定 camera frame。

### 普通 `navigation` 和 `frame_id=vision` 的区别

普通 `navigation` 中，`position` 是 `[x, y, yaw_deg]`。只有当 `frame_id="vision"` 时，`position` 才是 vision 返回的 `[x, y, z]`。
