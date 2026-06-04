## 🛠️ 第一步：启动核心基础（Terminal 1）

首先，任何 ROS 系统运行前都必须启动 Master 节点。
在第一个终端运行：

```bash
roscore
cd cade_ws
source devel/setup.bash 
```

---

## 📷 第二步：启动视觉感知节点（Terminal 2）

因为我们要测试手势和姿态，强烈建议你**开启 OpenCV 窗口显示**（默认开启），这样你可以直观地看到 MediaPipe 有没有成功画出你的骨骼点，以及画面上有没有打上 `waving` 或 `sitting` 的标签。

* **实机测试（带 RealSense 相机）**：
```bash
rosrun cade_vision ./scripts/open_vision_node.py --image-source realsense --device cuda --serial-number 346522071650

```
---

## 🧠 第三步：人工扮演大脑，下发任务（Terminal 3）

现在，视觉节点正在“盲目地”进行全量单帧检测，但它还没有启动任何任务执行器。

我们在第三个终端，使用 `rostopic pub` 向 `/cade/task_cmd` 发送一条单次（`-1`）的标准 JSON 消息，要求它**寻找一个正在挥手、且坐着、穿蓝色上衣的人**：

```bash
rostopic pub -1 /cade/task_cmd std_msgs/String "data: '{\"action\": \"find_person\", \"target\": \"person\", \"timeout\": 45.0, \"attributes\": {\"gesture\": \"waving\", \"posture\": \"sitting\", \"cloth_color\": \"blue\"}}'"

```

> 💡 **检查点**：此时注意看 **Terminal 2（视觉节点）**，它应该会瞬间打印出换轨日志：
> `[Vision Task] find_person: target='person' attrs={'gesture': 'waving', 'posture': 'sitting', 'cloth_color': 'blue'}`
> 证明我们的重构路由表完全正确，`SearchTask` 已经在后台线程里玩命轮询了！

---

## 📡 第四步：观察接收结果（Terminal 4 或新建分屏）

在第三个终端发完命令后，立刻在这个新终端里挂上监听，盯着视觉节点的结果输出话题：

```bash
rostopic echo /vision/detections_3d

```

### 🎬 现场测试动作：

此时，你（或者让同学）来到相机面前，**换上一件蓝色的衣服，坐下（posture: sitting），并且对着相机持续高频挥手（gesture: waving）**。

### 🎉 预期的成功结果：

只要你在相机前坐着挥手，MediaPipe 累积满时序滑窗的瞬间（约 1 秒内），`rostopic echo` 的窗口就会**啪地弹出一模如下格式的标准结构化数据**：

```json
data: "{\"type\": \"object_detection\", \"name\": \"person\", \"confidence\": 0.92, \"position_3d\": [0.12, -0.05, 0.85], \"bbox\": [120, 80, 280, 450]}"

```

同时，你还可以在另一个窗口输入 `rostopic echo /cade/task_status`，它会吐出：

```json
data: "{\"status\": \"SUCCESS\", \"result\": {\"type\": \"object_detection\", \"name\": \"person\", ...}}"

```

一旦看到这个，证明**老任务安全交接 $\rightarrow$ 后台线程异步轮询 $\rightarrow$ 属性多层 AND 过滤 $\rightarrow$ 触发原子校验发布**这条长达数千行的逻辑链路，已经被我们完美打通！
