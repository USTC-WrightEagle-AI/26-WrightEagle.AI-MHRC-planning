# Task1 导航测试模块

这个目录提供两个独立测试入口，并被 `task1_controller.py` 的
`GO_TO_DOOR`、`GUIDE_GUEST1`、`RETURN_TO_START`、`PICK_UP_GUEST2`、
`POINT_EMPTY_SEAT`、`SEAT_GUEST2` 状态复用。
实现参考 `/home/nvidia/setgoal`：

- `get_current_map_pose/`：读取当前机器人在 `map` 坐标系下的位置和朝向。
- `set_nav_goal/`：发送目标坐标，让机器人导航到指定位置。

默认坐标和导航接口：

- 全局坐标系：`map`
- 机器人坐标系：`base_link`
- `move_base` action：`/move_base`
- 目标格式：`x y yaw_deg`

## 1. 读取当前机器人坐标

启动导航/定位后执行：

```bash
cd /home/nvidia/task1/26-WrightEagle.AI-MHRC-planning
./task1_receptionist/navigation/get_current_map_pose/run_get_current_map_pose.sh
```

默认读取 `map -> base_link` TF。输出最后一行就是可直接复制给发送目标脚本的格式。

常用参数：

```bash
./task1_receptionist/navigation/get_current_map_pose/run_get_current_map_pose.sh \
  --map-frame map \
  --base-frame base_link \
  --timeout 3.0
```

## 2. 发送目标坐标导航

直接发送到 `move_base` action：

```bash
./task1_receptionist/navigation/set_nav_goal/run_set_nav_goal.sh 1.0 2.0 90
```

含义是让机器人去 `map` 坐标系下 `x=1.0, y=2.0, yaw=90deg`。

只发目标，不等待结果：

```bash
./task1_receptionist/navigation/set_nav_goal/run_set_nav_goal.sh 1.0 2.0 90 --no-wait
```

## 3. 接入 Task1 固定点位状态

建图并记录门口、起点等固定坐标后，推荐统一写入代码：

```python
# task1_receptionist/sub_modules/base_module.py
DEFAULT_NAV_GOALS = {
    "door": [-2.784, 0.768, 173.131],
    "start": [-1.432, -0.078, 169.610],
    "living_room": [-0.274, -1.151, -2.771],
}
```

之后正常启动总控即可，`GO_TO_DOOR`、`GUIDE_GUEST1`、`RETURN_TO_START`、
`PICK_UP_GUEST2` 和 `SEAT_GUEST2` 的“去客厅”部分会默认使用这些固定点位：

```bash
python3 task1_receptionist/task1_controller.py
```

也可以临时把同样的 `x y yaw_deg` 传给总控，覆盖代码里的默认值：

```bash
python3 task1_receptionist/task1_controller.py --door-goal -2.784 0.768 173.131
python3 task1_receptionist/task1_controller.py --start-goal -1.432 -0.078 169.610
python3 task1_receptionist/task1_controller.py --living-room-goal -0.274 -1.151 -2.771
```

或者用环境变量覆盖：

```bash
export TASK1_DOOR_GOAL="-2.784 0.768 173.131"
export TASK1_START_GOAL="-1.432 -0.078 169.610"
export TASK1_LIVING_ROOM_GOAL="-0.274 -1.151 -2.771"
python3 task1_receptionist/task1_controller.py
```

`GO_TO_DOOR`、`GUIDE_GUEST1`、`RETURN_TO_START`、`PICK_UP_GUEST2` 和
`SEAT_GUEST2` 会调用 `set_nav_goal.py`，默认发送到 `/move_base`，等待到达结果。
可用 `TASK1_NAV_GOAL_TIMEOUT` 调整等待秒数；设为 `0` 表示一直等。

## 4. 导航前底盘微调表

每个 direct 导航目标发送前，可以先调用
`task1_receptionist/navigation/chassis_move_util.py` 做开环底盘微调。
配置集中在 `task1_receptionist/sub_modules/base_module.py` 的
`DEFAULT_PRE_NAV_ADJUSTMENTS`：

```python
DEFAULT_PRE_NAV_ADJUSTMENTS = {
    "go_to_door": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
}
```

填写规则：

- `enabled`: 是否执行该步骤前的微调
- `forward_m`: 正数前进，负数后退，单位米
- `left_m`: 正数左移，负数右移，单位米
- `rotate_deg`: 正数逆时针，负数顺时针，单位度
- `order`: 执行顺序，可填 `forward`、`left`、`rotate`

可填写的 key：

| key | 执行位置 |
|-----|----------|
| `go_to_door` | 状态 2，第一次去门口前 |
| `guide_guest1_living_room` | 状态 4，带 guest1 去客厅前 |
| `point_empty_seat_approach` | 状态 5，识别空座并导航到空座接近点前 |
| `return_to_start` | 状态 6，返回起点前 |
| `pick_up_guest2_door` | 状态 8，去门口接 guest2 前 |
| `seat_guest2_living_room` | 状态 10，带 guest2 去客厅前 |
| `seat_guest2_empty_seat_approach` | 状态 10，识别空座并导航到空座接近点前 |
| `follow_host` | 状态 14，开始跟随 host 前 |

示例：去门口前先后退 0.15m，再顺时针转 8 度：

```python
"go_to_door": {
    "enabled": True,
    "forward_m": -0.15,
    "left_m": 0.0,
    "rotate_deg": -8.0,
    "order": ["forward", "rotate"],
},
```

全局速度参数：

```bash
export TASK1_PRE_NAV_ADJUST_LINEAR_SPEED=0.15
export TASK1_PRE_NAV_ADJUST_ANGULAR_SPEED_DEG=25
export TASK1_PRE_NAV_ADJUST_WAIT_SUBSCRIBER=3.0

# 只打印微调命令，不真正发布底盘速度
export TASK1_PRE_NAV_ADJUST_DRY_RUN=1
```

## 5. 空座接近点导航

`POINT_EMPTY_SEAT` 和 `SEAT_GUEST2` 会先使用
`task1_receptionist/object_search/run_object_search.sh` 识别最近空座，读取
`latest_empty_seat.json` 中的 `best_empty_seat`。导航模块默认按
`/home/nvidia/Desktop/ultralytics-main/person_tracker` 的逻辑处理坐标：
先把 `camera_xyz_m` 通过 `camera_middle_to_leftbase.txt` 转到
`left_arm_base_link`，再通过 TF 得到 `base_link` 下的空座相对坐标，最后结合
当前 `map -> base_link` 位姿手算空座 `map` 坐标。接着计算一个距离空座 1.5 米、
朝向空座的导航目标，并通过 `set_nav_goal.py` 发送到 `/move_base`。

默认使用的坐标字段顺序：

1. `camera_xyz_m` + `camera_middle_to_leftbase.txt`
2. `leftbase_xyz_m`
3. `left_arm_base_xyz_m`
4. `target_frame_xyz_m`
5. `calibrated_xyz_m`

默认认为外参输出对应 TF frame 是 `left_arm_base_link`，可用
`TASK1_EMPTY_SEAT_LEFTBASE_FRAME` 临时覆盖。旧的“JSON frame 直接转 map”路径
仍保留为兜底。

常用参数：

```bash
# 默认 1.5 米，可按现场通道宽度临时调整
TASK1_EMPTY_SEAT_APPROACH_DISTANCE=1.5 \
python3 task1_receptionist/task1_controller.py

# 跳过相机刷新，复用 latest_empty_seat.json 做导航测试
python3 task1_receptionist/task1_controller.py --object-search-skip-vision

# 空座识别默认最多 30 帧，且识别成功会提前退出；现场还嫌慢可继续降到 10-15
python3 task1_receptionist/task1_controller.py --empty-seat-max-frames 15

# 如果现场 TF frame 名不同，可手动指定候选
TASK1_EMPTY_SEAT_LEFTBASE_FRAME="left_arm_base_link" \
python3 task1_receptionist/task1_controller.py
```

到达接近点后，操作模块会重新刷新一次空座识别，再调用
`point_empty_seat_arm.py` 指向空座。这样左臂使用的是移动后的本地坐标。

## 测试前检查

先确认以下条件：

```bash
rostopic list | grep /tf
rosnode list | grep move_base
```

如果读取坐标失败，优先检查是否存在 `map -> base_link` TF。

如果发送目标失败，优先检查 `/move_base` action server 是否已经启动。
