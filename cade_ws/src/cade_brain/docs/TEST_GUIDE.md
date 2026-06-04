# CADE Brain 测试指南

本指南帮助你单独测试 `cade_brain` 的完整流程，无需真实硬件。
核心思路：手动向 ROS 话题发布消息，模拟 ASR 输入和硬件响应。

---

## 1. 架构概览

```
┌─────────────┐    /asr     ┌──────────────┐   /cade/task_cmd    ┌─────────────┐
│  ASR 节点   │ ──────────→ │  Brain Node  │ ──────────────────→ │  硬件节点   │
│  (模拟)     │             │  (LLM+ReAct) │                     │  (模拟)     │
└─────────────┘             └──────────────┘                     └─────────────┘
                                │    ↑                                    │
                                │    │ /cade/task_status                  │
                                │    └────────────────────────────────────┘
                                ↓
                           /tts (最终回复)
```

**关键 ROS 话题：**
- `/asr` (std_msgs/String) — 语音识别输入，触发 Brain 处理
- `/cade/task_cmd` (std_msgs/String) — Brain 发布的动作指令
- `/cade/task_status` (std_msgs/String) — 硬件返回的执行结果
- `/tts` (std_msgs/String) — Brain 发布的语音回复

---

## 2. 环境准备

### 2.1 启动 ROS Master

```bash
# 终端 1：启动 roscore
roscore
```

### 2.2 配置 Python 环境

```bash
# 终端 2：进入工作空间
cd /home/nvidia/Desktop/task3/cade_ws
source devel/setup.bash

# 设置环境变量（根据你的 LLM 配置）
export LLM_MODE=cloud          
export LLM_API_KEY=your_key
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-chat

# 切换为本地模式
export CADE_MODE=LOCAL
export CADE_LOCAL_BASE_URL="http://localhost:11434/v1"
# 本地模型名（根据你本地运行的模型改）
export CADE_LOCAL_MODEL="qwen3:8b"
# 本地 API key（如果你的本地服务需要，可填；Ollama 默认可用 'ollama'）
export CADE_LOCAL_API_KEY="ollama"


# 可选：把 Cloud key 取消/清空以避免误用
unset CADE_CLOUD_API_KEY

```

### 2.3 验证环境

```bash
# 检查 ROS 话题是否就绪
rostopic list | grep -E "asr|task_cmd|task_status|tts"
```

---

## 3. 测试流程

### 3.1 启动 Brain Node

```bash
# 终端 2：启动大脑节点
cd /home/nvidia/Desktop/task3/cade_ws
source devel/setup.bash
rosrun cade_brain brain_node.py --env "You are sitting in a lab."
```

启动后应看到：
```
╔═══════════════════════════════════════════════════════════╗
║   CADE - Cognitive Agent for Domestic Environment        ║
║   Architecture: Function-as-Tool (Refactored v2)         ║
╚═══════════════════════════════════════════════════════════╝

Robot Controller initialized (Tool Registry + ReAct)
  Tools Registry: 14 functions loaded
  Memory: conversation_history only
```

### 3.2 启动监控终端

```bash
# 终端 3：监控 Brain 发出的动作指令
rostopic echo /cade/task_cmd

# 终端 4：监控 Brain 的语音回复
rostopic echo /tts
```

### 3.3 发送测试指令

```bash
# 终端 5：模拟 ASR 输入
rostopic pub -1 /asr std_msgs/String "data: 'Go to the kitchen'"
```

### 3.4 手动响应硬件反馈

当 Brain 发布 `/cade/task_cmd` 后，**立即**在另一个终端发布假的成功响应：

```bash
# 终端 6：模拟硬件成功响应
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":\"arrived at kitchen\"}'"
```

**核心原则：每看到一次 `/cade/task_cmd`，就手动往 `/cade/task_status` 发一次 SUCCESS。**

---

## 4. 完整测试场景

### 场景 1：简单导航

**目标：** 测试单步导航动作

**步骤：**
1. 发送指令：
   ```bash
   rostopic pub -1 /asr std_msgs/String "data: 'Go to the kitchen'"
   ```

2. 观察 Brain 输出，等待 `/cade/task_cmd` 出现：
   ```json
   {"action": "goToLoc", "target": "kitchen"}
   ```

3. 手动响应：
   ```bash
   rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":\"arrived at kitchen\"}'"
   ```

4. 观察 Brain 是否在 `/tts` 发布回复

---

### 场景 2：多步 ReAct 循环

**目标：** 测试 LLM 的多步推理和动作链

**指令：**
```
Go to the kitchen, find a waving person, then follow them
```

**预期流程：**

| 步骤 | Brain 发布的 /cade/task_cmd | 你应发布的 /cade/task_status |
|------|----------------------------|----------------------------|
| 1 | `{"action":"goToLoc","target":"kitchen"}` | `{"status":"SUCCESS","result":"arrived at kitchen"}` |
| 2 | `{"action":"find_person","gesture":"waving","room":"kitchen"}` | `{"status":"SUCCESS","result":{"person_pos":"near table","gesture":"waving"}}` |
| 3 | `{"action":"follow_person","person_name":"waving_person"}` | `{"status":"SUCCESS","result":"following the waving person"}` |

**详细步骤：**

1. 发送指令：
   ```bash
   rostopic pub -1 /asr std_msgs/String "data: 'Go to the kitchen, find a waving person, then follow them'"
   ```

2. 等待第一步指令，手动响应：
   ```bash
   rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":\"arrived at kitchen\"}'"
   ```

3. 等待第二步指令，手动响应：
   ```bash
   rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":{\"person_pos\":\"near table\",\"gesture\":\"waving\"}}'"
   ```

4. 等待第三步指令，手动响应：
   ```bash
   rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":\"following the waving person\"}'"
   ```

5. 观察 Brain 的最终回复

---

### 场景 3：物体抓取

**目标：** 测试物体搜索和抓取流程

**指令：**
```
Find the red cup and bring it to me
```

**预期流程：**

| 步骤 | /cade/task_cmd | /cade/task_status |
|------|----------------|-------------------|
| 1 | `{"action":"find_object","target":"red cup"}` | `{"status":"SUCCESS","result":{"position":[1.2,0.5,0.8]}}` |
| 2 | `{"action":"bringMeObj","object_name":"red cup","placement":"near user"}` | `{"status":"SUCCESS","result":"brought the red cup"}` |

**手动响应：**

```bash
# 步骤 1：找到物体
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":{\"position\":[1.2,0.5,0.8]}}'"

# 步骤 2：抓取成功
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":\"brought the red cup\"}'"
```

---

### 场景 4：人数统计

**目标：** 测试视觉计数功能

**指令：**
```
How many people are sitting in the classroom?
```

**预期流程：**

| 步骤 | /cade/task_cmd | /cade/task_status |
|------|----------------|-------------------|
| 1 | `{"action":"count_people","room":"classroom","gesture":"sitting"}` | `{"status":"SUCCESS","result":{"count":5}}` |

**手动响应：**

```bash
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":{\"count\":5}}'"
```

---

### 场景 5：衣服识别

**目标：** 测试按衣服颜色识别人物

**指令：**
```
Find the person wearing a red shirt in the living room
```

**预期流程：**

| 步骤 | /cade/task_cmd | /cade/task_status |
|------|----------------|-------------------|
| 1 | `{"action":"find_person","cloth_color":"red","room":"living_room"}` | `{"status":"SUCCESS","result":{"person_pos":"near sofa"}}` |

**手动响应：**

```bash
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":{\"person_pos\":\"near sofa\"}}'"
```

---

### 场景 6：失败响应测试

**目标：** 测试 Brain 对失败的处理

**指令：**
```
Go to the bedroom
```

**手动响应（模拟失败）：**

```bash
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"FAILED\",\"error\":\"Path blocked\"}'"
```

**预期行为：** Brain 应该在下一轮 LLM 调用中处理失败，并可能重试或报告错误。

---

## 5. 动作类型速查表

### 导航类动作 (nav_skills)

| 动作类型 | 参数 | 说明 |
|---------|------|------|
| `goToLoc` | `target` | 导航到指定位置 |
| `follow_person` | `person_name`, `gesture`, `location` | 跟随人物 |
| `guide_person` | `person_name`, `from_beacon`, `to_beacon` | 引导人物 |
| `bringMeObj` | `object_name`, `placement` | 取物体并带回 |
| `takeObjFromPlcmt` | `object_name`, `placement` | 从指定位置拿物体 |
| `wait` | `duration`, `interruptible` | 等待指定时间 |
| `listening` | `continuous`, `vad_enabled`, `wake_word` | 进入监听模式 |

### 视觉类动作 (vision_skills)

| 动作类型 | 参数 | 说明 |
|---------|------|------|
| `find_person` | `room`, `gesture`, `cloth_color`, `category` | 寻找人物 |
| `find_object` | `target`, `room` | 寻找物体 |
| `count_people` | `room`, `gesture`, `category` | 统计人数 |
| `count_objects` | `category`, `placement` | 统计物体数量 |
| `get_nearest_person` | 无 | 获取最近人物属性 |

---

## 6. 响应格式模板

### 成功响应

```json
{
  "status": "SUCCESS",
  "result": "简单文本结果"
}
```

```json
{
  "status": "SUCCESS",
  "result": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

### 失败响应

```json
{
  "status": "FAILED",
  "error": "错误描述"
}
```

### 超时响应

```json
{
  "status": "TIMEOUT",
  "error": "No response within 30s"
}
```

---

## 7. 调试技巧

### 7.1 查看已注册的工具

Brain 启动时会打印：
```
Tools Registry: 14 functions loaded
```

### 7.2 查看 LLM 输出

Brain 会打印 LLM 的原始输出：
```
📄 LLM 原始输出:
------------------------------------------------------------
{
  "thought": "用户想去厨房...",
  "reply": "好的，我正在前往厨房",
  "action": {
    "type": "navigation",
    "position": "kitchen"
  }
}
------------------------------------------------------------
```

### 7.3 查看动作执行

Brain 会打印每个动作的执行结果：
```
[Action] navigation succeeded: {'status': 'SUCCESS', 'result': 'arrived at kitchen'}
```

### 7.4 查看 ReAct 循环

Brain 会打印循环状态：
```
[Brain thinking... (loop 1)]
[Thought #1]: 用户想去厨房，我需要导航过去
[Reply #1]: 好的，我正在前往厨房
[Action #1]: navigation

[ReAct] Loop ended after 2 round(s)
```

### 7.5 无 ROS 模式

如果没有 ROS，Brain 会自动进入交互模式：
```bash
python scripts/brain_node.py
# 输出: Running in interactive mode (no ROS)
# 然后可以直接在终端输入命令
You: Go to the kitchen
```

在无 ROS 模式下，所有技能调用会返回模拟的成功响应：
```
[Skill] (no ROS) Simulating task status: SUCCESS
```

---

## 8. 常见问题

### Q: Brain 收不到 /asr 消息

**检查：**
```bash
# 确认 /asr 话题存在
rostopic list | grep asr

# 确认消息格式正确
rostopic pub -1 /asr std_msgs/String "data: 'test'"
```

### Q: Brain 发布了 /cade/task_cmd 但没有收到响应

**检查：**
```bash
# 确认 /cade/task_status 话题存在
rostopic list | grep task_status

# 确认响应格式正确（必须是 JSON）
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":\"ok\"}'"
```

### Q: Brain 卡在等待响应

**原因：** 你没有及时发布 `/cade/task_status`

**解决：** Brain 默认超时 30-120 秒（取决于动作类型），超时后会返回 TIMEOUT 错误。

### Q: LLM 输出格式错误

**检查：**
- 查看 Brain 终端的 LLM 原始输出
- 确认 LLM API 配置正确
- 检查网络连接

### Q: 工具函数找不到

**检查：**
- 确认 `skills/__init__.py` 正确导入了 `nav_skills` 和 `vision_skills`
- 确认装饰器 `@register_nav_tool` / `@register_vision_tool` 正确应用

---

## 9. 自动化测试脚本

如果需要自动化测试，可以创建一个简单的 shell 脚本：

```bash
#!/bin/bash
# test_brain.sh - 自动化测试 Brain 流程

# 启动 Brain（后台）
rosrun cade_brain brain_node.py &
BRAIN_PID=$!
sleep 3

# 发送测试指令
rostopic pub -1 /asr std_msgs/String "data: 'Go to the kitchen'"

# 等待 Brain 发布指令
sleep 5

# 模拟硬件响应
rostopic pub -1 /cade/task_status std_msgs/String "data: '{\"status\":\"SUCCESS\",\"result\":\"arrived at kitchen\"}'"

# 等待 Brain 处理
sleep 3

# 清理
kill $BRAIN_PID
echo "Test complete"
```

---

## 10. 测试检查清单

- [ ] Brain 成功启动，显示 14 个工具注册
- [ ] 发送简单指令后，Brain 在 /cade/task_cmd 发布动作
- [ ] 手动响应后，Brain 继续处理或回复
- [ ] 多步指令能正确触发 ReAct 循环
- [ ] 失败响应能被 Brain 正确处理
- [ ] 最终回复正确发布到 /tts
- [ ] 无 ROS 模式下可以交互测试

---

**核心原则：每看到一次 `/cade/task_cmd`，就手动往 `/cade/task_status` 发一次 SUCCESS。**
