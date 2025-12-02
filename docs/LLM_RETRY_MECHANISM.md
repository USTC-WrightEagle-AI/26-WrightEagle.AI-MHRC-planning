# LLM 重试机制详解

## ❓ 你的问题

为什么看到"解析失败"的警告，但最后还是执行成功了？

## 📖 答案

这是 **自动重试机制** 在工作！

## 🔄 完整流程

### 场景：用户说 "去桌子那里"

#### 第1次尝试（失败）

**LLM 可能的输出：**
```
好的，我这就去桌子那里。我会立即执行导航动作。
```

**问题：** 这不是 JSON 格式！

**系统反应：**
```
⚠ 解析失败 (尝试 1/3): 无法从文本中提取JSON
```

系统会自动把这个错误信息反馈给 LLM：
```
你的输出格式有误，错误信息：无法从文本中提取JSON
请严格按照JSON格式重新输出。
```

---

#### 第2次尝试（成功）

**LLM 的输出：**
```json
{
  "thought": "用户明确指示我去桌子那里。这是一个直接的导航指令。我应该执行导航动作。",
  "reply": "好的，我这就去桌子那里。",
  "action": {
    "type": "navigate",
    "target": "table"
  }
}
```

**系统反应：**
```
✅ 解析成功！
💭 思考过程: 用户明确指示我去桌子那里...
💬 回复: 好的，我这就去桌子那里。
⚡ 计划动作: navigate
```

执行动作 ✅

---

## 🎯 核心代码逻辑

在 `brain/llm_client.py` 的 `get_decision()` 方法中：

```python
for attempt in range(max_retries):  # 最多重试 3 次
    try:
        # 1. 调用 LLM
        response = self.chat(messages)

        # 2. 尝试解析 JSON
        decision_dict = self._extract_json(response)

        # 3. 验证格式
        decision = RobotDecision(**decision_dict)

        # ✅ 成功！返回结果
        return decision

    except Exception as e:
        # ❌ 失败！打印警告
        print(f"⚠ 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")

        # 把错误反馈给 LLM，让它重新生成
        messages.append({
            "role": "assistant",
            "content": response
        })
        messages.append({
            "role": "user",
            "content": f"你的输出格式有误，错误信息：{e}\n请严格按照JSON格式重新输出。"
        })
        # 继续下一次循环...
```

---

## 🧩 JSON 输出和机器人动作的关系

### 数据流图

```
用户输入 "去桌子那里"
    ↓
┌─────────────────────┐
│  LLM 大脑思考       │
│  (System Prompt)    │
└─────────────────────┘
    ↓
生成 JSON 决策
{
  "thought": "...",    ← 📝 LLM 的思考过程（供调试）
  "reply": "...",      ← 💬 说给用户听的话
  "action": {          ← ⚡ 关键！机器人要执行的动作
    "type": "navigate",
    "target": "table"
  }
}
    ↓
解析 action 字段
    ↓
┌─────────────────────┐
│  根据 type 调用     │
│  对应的函数：       │
│  robot.navigate()   │
└─────────────────────┘
    ↓
🤖 机器人移动到桌子
```

### 关键点

1. **JSON 是协议**
   - LLM（大脑）和 Robot（躯体）之间的"通信协议"
   - 就像人的"大脑发出神经信号"，必须是特定格式

2. **action 字段是核心**
   - 决定了调用哪个函数：`navigate`, `pick`, `search`...
   - 包含了函数参数：去哪里？拿什么？

3. **Pydantic 保证安全**
   - 严格校验 JSON 结构
   - 防止 LLM 输出奇怪的格式导致程序崩溃

---

## 🐛 为什么会解析失败？

### 常见原因

1. **LLM 忘记了 JSON 格式**
   ```
   # ❌ 错误输出
   好的，我去桌子！
   ```

2. **JSON 语法错误**
   ```json
   # ❌ 缺少引号
   {
     thought: "...",
     action: {...}
   }
   ```

3. **使用了未定义的动作类型**
   ```json
   # ❌ 没有 "fly" 这个动作
   {
     "thought": "...",
     "action": {"type": "fly", "target": "sky"}
   }
   ```

4. **参数格式错误**
   ```json
   # ❌ navigate 需要 target 参数
   {
     "thought": "...",
     "action": {"type": "navigate"}
   }
   ```

---

## 🛠️ 如何减少解析失败？

### 方法 1：优化 System Prompt

在 `brain/prompts.py` 中：
- ✅ 给出**大量正确示例**
- ✅ 强调"必须严格按照 JSON 格式"
- ✅ 明确列出所有可用动作

### 方法 2：增加重试次数

在调用时：
```python
decision = llm_client.get_decision(
    user_input=user_input,
    system_prompt=system_prompt,
    max_retries=5  # 从 3 次改为 5 次
)
```

### 方法 3：使用更强大的模型

```python
# config_local.py
CLOUD_MODEL = "deepseek-chat"  # 当前
# 或
CLOUD_MODEL = "gpt-4"  # 更准确，但更贵
```

### 方法 4：添加 JSON Schema

使用 OpenAI 的 Function Calling：
```python
# 未来改进：使用结构化输出
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    response_format={"type": "json_object"}  # 强制 JSON 输出
)
```

---

## 📊 统计数据建议

你可以在 `RobotController` 中添加失败统计：

```python
class RobotController:
    def __init__(self):
        # ...
        self.parse_failures = 0  # 记录解析失败次数
        self.parse_retries = 0   # 记录重试次数
```

然后在 `print_statistics()` 中显示：
```
📊 统计信息:
   总交互次数: 10
   解析失败次数: 2
   平均重试次数: 1.2
   成功率: 80%
```

---

## ✅ 结论

**"解析失败但最终成功"是正常现象！**

- 🎯 这是**容错机制**，不是 bug
- 🔄 系统会自动重试，把错误反馈给 LLM
- 🧠 LLM 从错误中学习，第二次通常会成功
- ⚡ 只要最终成功，动作就会正常执行

这就像人说话一样：
- 第一次表达不清楚 → 对方听不懂
- 重新组织语言 → 对方理解了
- 最终完成沟通 ✅

---

## 🔧 调试工具

如果你想看 LLM 的原始输出，可以：

1. 运行修改后的代码（已添加调试输出）
2. 查看 `📄 LLM 原始输出` 部分
3. 对比第1次（失败）和第2次（成功）的输出

示例：
```bash
python main.py --mode debug  # 使用调试模式的 prompt
```
