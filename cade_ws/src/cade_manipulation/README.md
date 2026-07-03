# CADE Manipulation — 物体抓取与放置模块

## 功能概述

`cade_manipulation` 是 CADE 机器人系统的**物体操作执行层**，负责将大脑（cade_brain）发出的抓取/放置指令转化为实际的机械臂动作。

核心流程：

```
大脑 LLM 决策 "抓取杯子"
  → ObjectGraspAction(type="bringMeObj", object_name="cup")
    → 发布到 /cade/task_cmd
      → ManipulationNode 接收
        → ① 相机拍照（RGB + 深度）
        → ② TCP 发送到 5090 GraspNet 服务器
        ← ③ 返回 6-DOF 抓取姿态
        → ④ 机械臂执行抓取
        → 发布结果到 /cade/task_status
```

## 目录结构

```
cade_manipulation/
├── config/
│   └── graspnet.yaml          # GraspNet 服务器地址、相机内参、机械臂配置
├── launch/
│   └── manipulation.launch    # 独立启动文件
├── scripts/
│   └── manipulation_node.py   # rosrun 入口
└── src/cade_manipulation/
    ├── manipulation_node.py   # 主节点：订阅指令 → 任务分发 → 发布结果
    ├── graspnet_client.py     # TCP 客户端（与 5090 服务器的通信协议）
    ├── camera_capture.py      # RealSense 相机采集（RGB + 16位深度图）
    ├── arm_controller.py      # 机械臂控制（当前为桩代码，需替换实际驱动）
    └── tasks/
        ├── base_task.py       # 任务基类
        ├── grasp_task.py      # 抓取任务（bringMeObj）
        └── dump_task.py       # 放置任务（object_dump）
```

## 支持的动作

| 动作类型 | 说明 | 输入 | 输出 |
|---------|------|------|------|
| `bringMeObj` | 抓取指定物体 | `object_name`, `grasp_force`(可选) | 抓取姿态 + 执行结果 |
| `object_dump` | 释放当前物体 | `target_position`, `release_safe` | 执行结果 |

## 依赖

- **ROS**：`rospy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`
- **Python**：`numpy`, `opencv-python`, `pyrealsense2`
- **外部服务**：5090 机器上的 GraspNet TCP 服务器（端口 9090）

## 启动方式

### 方式一：作为完整系统的一部分

```bash
roslaunch cade_full.launch enable_manipulation:=true graspnet_host:=192.168.31.37
```

### 方式二：独立启动

```bash
roslaunch cade_manipulation manipulation.launch graspnet_host:=192.168.31.37 arm_enabled:=false
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `graspnet_host` | `192.168.31.37` | 5090 服务器 IP 地址 |
| `graspnet_port` | `9090` | 5090 服务器端口 |
| `graspnet_timeout` | `10.0` | TCP 超时时间（秒） |
| `arm_enabled` | `false` | 是否启用物理机械臂（否→桩模式） |
| `use_realsense` | `true` | 使用 pyrealsense2 直连相机 |
| `cmd_topic` | `/cade/task_cmd` | 接收指令的 ROS topic |
| `status_topic` | `/cade/task_status` | 发布结果的 ROS topic |

## 与 5090 GraspNet 服务器的通信协议

通过 TCP 端口 9090，使用 **长度前缀二进制协议**：

### 请求（机器人 → 5090）

每条消息前有 8 字节大端序长度头：

1. **JSON 文本**：`{"text_prompt": "cup"}`
2. **RGB 图像**：PNG 编码的彩色帧（uint8）
3. **深度图像**：PNG 编码的 16 位深度帧（uint16，**必须 16 位**）

### 响应（5090 → 机器人）

```json
{
  "success": true,
  "error_code": "OK",
  "message": "成功检测到 1 个最佳抓取姿态.",
  "grasp_poses": {
    "position": [0.123, -0.045, 0.678],
    "orientation": [0.0, 0.0, 0.707, 0.707]
  }
}
```

## 机械臂接入指南

当前 `arm_controller.py` 为**桩实现**（`arm_enabled=false` 时直接返回 SUCCESS）。

接入真实机械臂时：

1. 修改 `config/graspnet.yaml`，设置 `arm.enabled: true`
2. 替换 `arm_controller.py` 中的以下方法：
   - `_move_to_pregrasp()` — 移动到抓取点上方约 10cm
   - `_move_to_grasp()` — 下降到抓取点
   - `_close_gripper(force)` — 闭合夹爪
   - `_open_gripper()` — 打开夹爪

抓取姿态坐标系说明：
- `position`: `[x, y, z]` — 单位：米，相对于相机坐标系
- `orientation`: `[x, y, z, w]` — 四元数

## 注意事项

1. **深度图必须是 16 位**：8 位深度图会无声地破坏 GraspNet 结果，`graspnet_client.py` 会自动校验
2. **相机内参必须准确**：`config/graspnet.yaml` 中的 fx/fy/cx/cy 需要与实际相机标定结果一致
3. **桩模式下抓取不真实执行**：测试时设置 `arm_enabled:=false`，大脑会收到 SUCCESS 响应但机械臂不动作
4. **多节点共用同一 topic**：`cade_navigation` 和 `cade_manipulation` 都监听 `/cade/task_cmd`，各自只处理自己认识的动作类型，互不干扰
