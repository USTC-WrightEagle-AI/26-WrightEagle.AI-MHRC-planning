# 机器人如何"理解"指令？

## 以 navigate 为例的完整流程

### 🎯 核心问题

用户说："去厨房"
机器人怎么知道要执行 `robot.navigate("kitchen")` ？

---

## 📖 完整流程拆解

### 阶段1：用户输入（自然语言）

```
用户: "去厨房"
```

这只是普通的人类语言，机器人不能直接理解。

---

### 阶段2：System Prompt 的作用（关键！）

在 `brain/prompts.py` 中，我们给 LLM 提供了一个"说明书"：

```python
ROBOT_SYSTEM_PROMPT = """
你是 LARA，一个智能服务机器人。

## 核心能力

你拥有以下物理动作能力：

1. **navigate** - 导航到指定位置
   - 参数: target (语义位置如"kitchen"或坐标[x,y,z])
   - 示例: {"type": "navigate", "target": "kitchen"}

2. **search** - 搜索物体
   ...

## 输出格式

你必须严格按照以下JSON格式输出：

{
  "thought": "你的思考过程",
  "reply": "给用户的回复",
  "action": {
    "type": "navigate",  ← 这里指定动作类型
    "target": "kitchen"  ← 这里指定参数
  }
}

## 示例

用户："去厨房"
输出：
{
  "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
  "reply": "好的，我这就去厨房",
  "action": {"type": "navigate", "target": "kitchen"}
}
"""
```

**关键点：**
- ✅ 告诉 LLM 有哪些"动作菜单"（navigate, search, pick...）
- ✅ 告诉 LLM 每个动作需要什么参数
- ✅ 给出大量示例，教 LLM 如何把自然语言转换为 JSON

---

### 阶段3：LLM 的推理过程

LLM 看到用户说 "去厨房"，会这样思考：

```
LLM 的内心独白：

1. "用户说'去厨房'"
2. "这明显是一个位置移动的需求"
3. "查看我的动作列表... 有 navigate 可以用！"
4. "navigate 需要一个 target 参数"
5. "用户说的'厨房'就是 target"
6. "我应该输出：
   {
     "action": {
       "type": "navigate",
       "target": "厨房"  或 "kitchen"
     }
   }
```

输出结果：
```json
{
  "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
  "reply": "好的，我这就去厨房",
  "action": {
    "type": "navigate",
    "target": "kitchen"
  }
}
```

---

### 阶段4：Pydantic Schema 的验证

在 `brain/schemas.py` 中定义的 `NavigateAction`：

```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"  # ← 强制 type 必须是 "navigate"
    target: Union[str, List[float]]         # ← target 可以是字符串或坐标数组

    @field_validator('target')
    def validate_target(cls, v):
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError("坐标必须是 [x, y, z] 格式")
        return v
```

**作用：**
```python
# ✅ 合法输入
NavigateAction(type="navigate", target="kitchen")
NavigateAction(type="navigate", target=[1.0, 2.0, 0.0])

# ❌ 非法输入 - 会抛出错误
NavigateAction(type="fly", target="sky")       # type 错误
NavigateAction(type="navigate", target=[1, 2]) # 坐标不是3个
NavigateAction(type="navigate")                # 缺少 target
```

这是**防御性编程**：
- 即使 LLM 输出错了，也能立即捕获
- 不会让错误的指令传递到机器人

---

### 阶段5：动作派发（Dispatch）

在 `robot_controller.py` 中：

```python
def _execute_action(self, action: RobotAction) -> bool:
    action_type = action.type  # 读取 "navigate"

    if action_type == "navigate":
        return self.robot.navigate(action.target)  # ← 调用真实函数
                                   ↑
                        把 "kitchen" 传进去

    elif action_type == "search":
        return self.robot.search(action.object_name)

    elif action_type == "pick":
        return self.robot.pick(action.object_name)
    # ...
```

**这就是"理解"的本质：**
```
JSON 中的 "type": "navigate"
    ↓
通过 if-elif 链条匹配
    ↓
调用对应的 Python 函数 robot.navigate()
```

---

### 阶段6：机器人执行

在 `body/mock_robot.py` 中：

```python
def navigate(self, target) -> bool:
    """模拟导航"""

    # target 就是 LLM 传过来的 "kitchen"

    if isinstance(target, str):
        if target in self.known_locations:  # 检查是否认识这个地方
            coords = self.known_locations[target]  # [5.0, 2.0, 0.0]

            # 模拟移动（真实机器人会调用 ROS 导航）
            self.current_position = target
            print(f"✓ 已到达 {target} (坐标: {coords})")
            return True
        else:
            print(f"✗ 未知位置: {target}")
            return False
```

**最终效果：**
```
🚗 [导航] 从 home 前往 kitchen
✓ 已到达 kitchen (坐标: [5.0, 2.0, 0.0])
```

---

## 🧠 为什么"简单的 Schema"就够用？

### 1. 你看到的"简单"，其实不简单

```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]
```

这短短3行代码，实际做了：

✅ **类型约束**
- `type` 必须是字符串 "navigate"
- `target` 可以是字符串或3个浮点数的列表

✅ **自动验证**
- Pydantic 自动检查类型
- 自动转换兼容类型（如 `1` → `1.0`）

✅ **错误提示**
- 如果格式错误，立即抛出详细的错误信息

✅ **文档生成**
- 可以自动生成 JSON Schema
- 可以自动生成 API 文档

### 2. LLM 负责"智能"，Schema 负责"安全"

**分工明确：**

| 模块 | 职责 | 举例 |
|------|------|------|
| **System Prompt** | 教 LLM 如何把自然语言转为 JSON | "去厨房" → `{"type": "navigate", "target": "kitchen"}` |
| **LLM** | 理解用户意图，生成指令 | 判断用户是要移动、抓取还是搜索 |
| **Pydantic Schema** | 验证格式，防止错误 | 确保 target 不是 `null` 或奇怪的值 |
| **Robot 函数** | 真实执行动作 | 调用硬件、ROS、传感器 |

### 3. Schema 是"合同"，不是"智能"

Schema 的作用类似于：
```
你：去厨房
合同：确保"厨房"是一个有效的地点名称
执行者：按照合同去厨房
```

不需要 Schema "理解"什么是厨房，只需要：
- ✅ 检查格式对不对
- ✅ 检查参数齐不齐
- ✅ 检查类型符不符合

---

## 🔍 深入 navigate 的两种模式

### 模式1：语义导航（推荐）

```json
{
  "action": {
    "type": "navigate",
    "target": "kitchen"  ← 语义标签
  }
}
```

**优点：**
- ✅ LLM 容易理解（厨房、客厅、卧室）
- ✅ 不需要精确坐标
- ✅ 可以动态调整（厨房的位置改了，只需更新地图）

**机器人的处理：**
```python
if target == "kitchen":
    coords = self.known_locations["kitchen"]  # [5.0, 2.0, 0.0]
    # 调用底层导航系统
    self.ros_nav.navigate_to(coords)
```

### 模式2：坐标导航

```json
{
  "action": {
    "type": "navigate",
    "target": [1.5, 2.3, 0.0]  ← 直接坐标
  }
}
```

**缺点：**
- ❌ LLM 很难凭空生成准确的数字
- ❌ 用户不知道坐标

**适用场景：**
- 从视觉系统获得物体位置
- 从地图上计算出的中间点

---

## 🚀 如果需要更复杂的功能？

### 示例：导航时避开障碍物

#### 当前 Schema（简单）
```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]
```

#### 扩展后的 Schema（复杂）
```python
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]

    # 新增参数
    speed: Optional[float] = 1.0          # 移动速度
    avoid_obstacles: bool = True          # 是否避障
    max_retries: int = 3                  # 最大重试次数
    path_planning: Literal["shortest", "safest"] = "shortest"
```

#### 更新 System Prompt
```python
1. **navigate** - 导航到指定位置
   - 参数:
     - target (必需): 目标位置
     - speed (可选): 移动速度，默认1.0
     - avoid_obstacles (可选): 是否避障，默认true
   - 示例:
     {"type": "navigate", "target": "kitchen", "speed": 0.5}
```

#### 更新机器人函数
```python
def navigate(self, target, speed=1.0, avoid_obstacles=True, **kwargs):
    # 根据参数调整导航策略
    if avoid_obstacles:
        self.enable_obstacle_avoidance()

    self.set_speed(speed)
    # ...
```

---

## 💡 关键洞察

### 机器人"理解"的本质

机器人并不是真的"理解"你说的话，而是：

1. **LLM 做翻译**
   - 自然语言 → 结构化指令（JSON）

2. **Schema 做校验**
   - 确保指令格式正确

3. **代码做映射**
   - JSON 的 `type` 字段 → 对应的函数名

4. **函数做执行**
   - 调用真实的硬件/算法

### 类比：餐厅点餐

```
你：我要一份宫保鸡丁        ← 用户输入（自然语言）
    ↓
服务员：好的，点了一份      ← LLM（理解意图）
         宫保鸡丁（辣度中等）
    ↓
订单系统：                  ← Schema（验证格式）
  {
    "dish": "宫保鸡丁",
    "spicy_level": "medium",
    "quantity": 1
  }
    ↓
厨房：收到订单，开始做菜     ← Robot 函数（执行）
```

厨房不需要"理解"你为什么想吃宫保鸡丁，只需要：
- ✅ 知道要做什么菜
- ✅ 知道用什么食材
- ✅ 按照菜谱执行

---

## 📚 总结

| 问题 | 答案 |
|------|------|
| 机器人怎么理解指令？ | LLM 把自然语言翻译成 JSON，代码根据 JSON 调用函数 |
| Schema 为什么这么简单？ | 因为"智能"在 LLM，Schema 只负责格式验证 |
| navigate 怎么工作的？ | `{"type": "navigate", "target": "kitchen"}` → `if type == "navigate": robot.navigate(target)` |
| 能扩展更复杂的功能吗？ | 可以！在 Schema 中添加参数，更新 Prompt 和函数 |

**核心思想：**
> 分层设计 = LLM（理解） + Schema（验证） + 代码（执行）

每一层只做自己擅长的事，整体就能处理复杂任务！
