# 集成与系统测试指南（中文）

本指南补充 `TEST_GUIDE.md`（主要面向 `cade_brain` 单元测试），扩展说明如何对整个 CADE 工作区进行集成与系统测试，包括 `cade_brain`、`cade_vision`、`cade_voice`、以及 ROS Pub/Sub、硬件/仿真接口和 CI 流程的验证。

## 目标读者
- 开发工程师：希望在开发流程中验证各模块协同工作。
- 测试工程师：负责搭建自动化测试、回归测试与系统验收测试。
- 机器人集成工程师：在实际机器人或仿真环境中验证端到端功能。

## 测试范围概述
1. 单元测试（各模块内部）
2. 模块间接口测试（RPC/ROS topic / service）
3. 子系统集成测试（vision + brain、voice + brain 等）
4. 端到端（E2E）测试（从语音输入到动作执行与视觉反馈）
5. 硬件在环（HIL）与仿真测试（Gazebo / ROS仿真）
6. CI/CD 管线中的自动化测试与质量门禁

---

## 1. 单元测试（各模块）
每个子包应包含自己的测试目录（已有 `cade_voice/test` 示例）。建议使用 `pytest`。

测试目标：
- 验证模块公共函数、数据模型（Pydantic schemas）和关键逻辑。
- 用 Mock/Stub 隔离外部依赖（例如网络请求、硬件驱动、ROS）。

建议目录结构：
```
cade_ws/src/cade_vision/src/cade_vision/
  tests/
    test_analyzer.py
    test_tracker.py
```

运行方法（工作区根或模块根）：
```bash
# 进入虚拟环境后
pytest -q src/cade_vision/src/cade_vision/tests
```

---

## 2. 模块间接口测试（ROS topics / services）
测试目标：确保模块之间通过 ROS topic/service 的数据契约正确。

方法：
- 使用 `rostest` 或直接用 ROS Python 脚本模拟发布/订阅。
- 用 Mock publisher 发布典型消息（例如 `/vision/detections_3d`），并断言 `cade_brain` 正确解析并更新 world model 或调用相应 Skill。

示例：测试 brain 接收到 vision 消息后的行为：
1. 启动 `cade_brain`（或引入其核心类而不运行全节点）
2. 发布一个 person_detection JSON 到 `/vision/detections_3d`
3. 验证 `Controller.observe()` 已接收到观测并更新 `episodic` 或 world state

---

## 3. 子系统集成测试（vision + brain, voice + brain）
目标：在受控环境中运行两到三个模块，验证端到端子工作流。

示例场景：
- 场景 A：用户说“去厨房找 Alice” -> `cade_voice` 识别后发布 `/asr` -> `cade_brain` 处理并发布导航命令 `/cade/task_cmd` -> `cade_vision` 反馈检测 -> `cade_brain` 完成任务。

测试步骤（自动化）：
1. 启动所需 ROS 节点（可用 launch 文件），或用容器分别启动 `cade_voice`, `cade_brain`, `cade_vision` 的最小进程。
2. 模拟 ASR：向 `/asr` 发布识别结果（字符串）。
3. 监听 `/cade/task_cmd` 与 `/tts` 并验证预期命令/回复。
4. 模拟视觉反馈：在合适时刻发布 `/vision/detections_3d` 检测结果，验证脑对反馈的回应。

自动化实现建议：用 Python 脚本配合 ROS（或 rostest）驱动上述流程，并在每一步断言消息内容。

---

## 4. 端到端测试（E2E）
目标：在尽可能真实的环境中验证完整任务链（ASR -> Brain -> Skill -> Actuator/Sim）。

两种可行方案：
- 软硬结合：用真实 ASR/视觉组件（或其近似）运行系统，并人工或自动化验证输出。
- 完全仿真：用 Gazebo 或自定义仿真程序模拟环境（含相机、物体与人物），并把视觉输出转为 `/vision/detections_3d` 消息。

示例 E2E 自动化流程：
1. 启动 Gazebo 场景（含机器人模型与人物/物体）。
2. 启动所有节点或其仿真替代（vision 可能由仿真脚本替代）。
3. 发布 /asr 指令或触发真实语音输入路径。
4. 等待并断言机器人到达目标或完成任务（可从 `/cade/task_status` / world model 状态判断）。

---

## 5. 硬件在环（HIL）与仿真要点
- 确保安全措施（紧急停止、软限位）。
- 对移动平台测试，使用低速/安全环境或仅在仿真中执行定位/导航动作。
- 记录并回放真实传感器数据以便重复复现问题。

---

## 6. CI/CD 集成建议
- 单元测试（pytest）与 linters（flake8/black/isort）在每次 PR 时运行。
- 对难以在 CI 中运行的集成测试，采用两类策略：
  - 模拟化测试套件（在 CI 中可执行的）——用 Mock/仿真替代真实硬件。
  - 长时/硬件测试套件（手动或专门 Runner）——在带硬件的 runner 上周期执行，并把结果回传。
- 报告与失败告警：把测试日志和关键 topic 消息打包到 artifacts，便于 debug。

---

## 7. 日志、监控与可观测性建议
- 在关键节点记录 structured logs（JSON）并包含 message id、timestamps。
- 将 ROS topic 的关键消息（/asr, /tts, /cade/task_cmd, /cade/task_status, /vision/detections_3d）写入回放包（rosbag），便于问题回放。
- 在 controller 的 observe() 中保留完整 action-result 对，便于 LL M 决策回溯。

---

## 8. 常见问题与排查步骤
- 节点无法通信：检查 ROS_MASTER_URI、topic 列表、node 名称冲突。
- LLM 无响应或格式错误：开启 LLMClient 的调试打印并保存原始输出用于排查。
- 视觉检测延迟/丢帧：检查相机频率、消息队列大小和处理瓶颈。
- 硬件资源不足（OOM/无GPU）：降低模型精度、增加 swap 或改为 CPU 测试。

---

## 9. 模板测试脚本（示例）
下面给出一个简单的 Python 脚本骨架，用于在不启动完整 ROS 系统的情况下，测试 `cade_brain` 对 `/vision/detections_3d` 的观测处理：

```python
# minimal_integration_test.py
import json
import time
from cade_brain.controller import RobotController

controller = RobotController()

# 1. 模拟视觉检测消息
msg = {
    "type": "person_detection",
    "name": "test_person",
    "position_3d": [1.0, 0.5, 0.0],
    "location": "kitchen",
    "cloth_color": "red",
}

# 2. 注入为 observation
controller.observe(json.dumps(msg))

# 3. 验证 episodic / conversation history
print(controller.episodic)
print(controller.conversation_history[-1])
```

---

## 10. 最佳实践与小结
- 先保证单元测试覆盖关键逻辑，再逐步推进接口与集成测试。优先把接口契约（消息格式）写成测试用例。
- 使用 rosbag 回放与结构化日志能显著提高调试效率。
- 在 CI 中保留一套轻量级的集成测试（用 Mock）和一套硬件 Runner 的完整测试计划。

如果你想，我可以：
- 把本指南合并到仓库 README 或 `docs/` 下的索引页面；
- 为某个具体场景生成完整的测试脚本（包含 ros publishers/subscribers 的实现）；
- 在 CI（GitHub Actions）中添加示例 workflow 来运行单元测试与模拟集成测试。

请告诉我你接下来想让我做哪一项。