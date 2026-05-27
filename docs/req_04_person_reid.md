# Req 04: 人名→外观绑定（Episodic Memory + 新 Action）

## 设计概述

将人名（如 "Charlie"）绑定到文字形式的外观描述（"蓝T恤+眼镜"），后续通过外观属性在视觉模块中查找这个人。LLM 通过主动查询（action）获取信息，不依赖预加载的 WorldModel 上下文注入。

## 核心原则

1. EpisodicMemory 生命周期 = Controller 生命周期 = brain 节点生命周期
2. 启动时自然为空，关闭时自然销毁，不设显式 clear()
3. 记忆层只存外观文字，不存位置，不存 track_id
4. LLM 通过 action 主动查询，不走缓存捷径
5. WorldModel 的上下文注入功能砍掉（_enrich_prompt_with_world_state 删除）

---

## 改动清单

| 文件 | 改动 |
|------|------|
| `memory.py` | 新增 EpisodicMemory 类 |
| `controller.py` | 删 WorldModel 注入 + 加 episodic + 3 个新 action |
| `vision_skill.py` | 新增 get_nearest_person_attributes, filter_by_attributes |
| `schemas.py` | 新增 action 类型（bind_person / recall / remember / find_person_by_attributes / speak） |
| `prompts.py` | system_prompt 加规则教会 LLM 使用这些 action |

---

## 1. EpisodicMemory 类（memory.py 新增）

在 `memory.py` 中新增 `EpisodicMemory` 类，与 `WorldModel` 并列：

```
class EpisodicMemory:
    """任务级情景记忆 — 在整个 brain 节点生命周期内持久"""

    # 内部数据结构：
    # people: {
    #   "Charlie": {
    #     "appearance": {cloth_color, cloth_type, hair_color, has_glasses, height},
    #     "info": {drink: "orange juice", ...}
    #   }
    # }

    方法：
    bind_person(name, appearance_dict)
        → people[name]["appearance"] = appearance_dict

    remember(name, key, value)
        → people[name]["info"][key] = value
        → 如果该人不存在，自动创建空条目

    recall_appearance(name) → dict | None
        → 返回 people[name]["appearance"]，不存在返回 None

    recall_info(name) → dict
        → 返回 people[name]["info"]，不存在返回 {}
```

`bind_person` 的 `appearance_dict` 字段：
```
{
    "cloth_color": "blue",
    "cloth_type":  "t-shirt",
    "hair_color":  "black",      ← 当前管线可能没有，填 "unknown"
    "has_glasses": false,        ← 当前管线可能没有，填 false
    "height":      "medium",     ← 当前管线可能没有，填 "unknown"
}
```

---

## 2. Controller 改动（controller.py）

### 2.1 导入

```python
from cade_brain.memory import WorldModel, EpisodicMemory  # 新增 EpisodicMemory
```

### 2.2 __init__

```python
# 新增 episodic memory（生命周期 = Controller 生命周期）
self.episodic = EpisodicMemory()

# 保留 WorldModel（仅用于 current_position 和 holding_object）
self.world = WorldModel()
```

### 2.3 process_input

```python
# 以前：
self.world = WorldModel()                          # 删除
enriched_prompt = self._enrich_prompt_with_world_state()  # 删除
decision = self.llm_client.get_decision(
    user_input=user_input,
    system_prompt=enriched_prompt,                  # 改成 self.system_prompt
    ...
)

# 改成：
decision = self.llm_client.get_decision(
    user_input=user_input,
    system_prompt=self.system_prompt,               # 纯静态 prompt
    conversation_history=self.conversation_history
)
```

ReAct 循环内的 LLM 调用也改成 `self.system_prompt`（不加 world_state）。

### 2.4 _execute_action 新增 3 个分支

在 `_execute_action` 的 action 分发中，紧接现有分支后面新增：

```
elif action_type == "bind_person":
    # LLM: {"action": "bind_person", "name": "Charlie"}
    # Controller 调 VisionSkill 取离机器人最近的人的视觉属性，
    # 然后绑到 episodic memory
    attributes = self.vision_skill.get_nearest_person_attributes()
    if attributes:
        self.episodic.bind_person(action.name, attributes)
        result = {"status": "SUCCESS",
                  "result": f"Bound {action.name} to appearance: {attributes}"}
    else:
        result = {"status": "FAILED", "error": "No person detected nearby"}

elif action_type == "recall":
    # LLM: {"action": "recall", "what": "appearance", "name": "Charlie"}
    #   或 {"action": "recall", "what": "info", "name": "Charlie"}
    if getattr(action, 'what', None) == "appearance":
        data = self.episodic.recall_appearance(action.name)
    else:
        data = self.episodic.recall_info(action.name)
    if data:
        result = {"status": "SUCCESS", "result": data}
    else:
        result = {"status": "FAILED",
                  "error": f"No memory of {action.name}"}

elif action_type == "remember":
    # LLM: {"action": "remember", "name": "Charlie",
    #        "key": "drink", "value": "orange juice"}
    self.episodic.remember(action.name,
                           getattr(action, 'key', ''),
                           getattr(action, 'value', ''))
    result = {"status": "SUCCESS",
              "result": f"Remembered {action.name}.{getattr(action,'key','')}"}

elif action_type == "find_person_by_attributes":
    # LLM 先 recall 外观，再把属性传给这个 action
    # {"action": "find_person_by_attributes",
    #   "attributes": {"cloth_color": "blue", "cloth_type": "t-shirt"}}
    attrs = getattr(action, 'attributes', {})
    result = self.vision_skill.filter_by_attributes(attrs)
```

### 2.5 observe() 简化

```python
def observe(self, observation: str, update_world: bool = True) -> str:
    """注入外部观测到对话上下文"""
    self.conversation_history.append({
        "role": "system",
        "content": f"[OBSERVATION] {observation}"
    })
    print(f"\n[OBSERVE] {observation}")
    return observation
```

不再调用 `self.world.get_world_state()` 和 `_enrich_prompt_with_world_state()`。

### 2.6 _enrich_prompt_with_world_state 方法

删除或标记为废弃。

### 2.7 砍掉死掉的 WorldModel 查询 action

以下旧 action 依赖 WorldModel 查询，但 WorldModel 在 process_input 开头被清空，里面永远是空的——这些 action 现在是死的，全部删除：

```
❌ tellPrsInfoInLoc    → _execute_action 中删除此分支
❌ tellObjPropOnPlcmt   → _execute_action 中删除此分支
❌ tellCatPropOnPlcmt   → _execute_action 中删除此分支
❌ greetNameInRm        → _execute_action 中删除此分支
```

同时从 schemas.py 的 action 列表中移除。

### 2.8 WorldModel 降级

WorldModel 类降级为 Controller 的两个简单状态变量，不再作为一个完整的类：

```python
# controller.py __init__:
self.current_position: str = "home"
self.holding_object: Optional[str] = None

# 导航 action 执行后更新:
if action_type == "goToLoc" and success:
    self.current_position = action.target

# 抓取 action 执行后更新:
if action_type in ("pickObject", "bringMeObjFromPlcmt") and success:
    self.holding_object = action.object_name
```

WorldModel 类本身（memory.py 中整个 WorldModel class）不再需要，可以删除。如果 controller.py 中有其他引用 WorldModel 的地方（如 `self.world.current_position`, `self.world.get_world_state()`, `_enrich_prompt_with_world_state`），全部替换或删除。

---

## 3. VisionSkill 新增（vision_skill.py）

### 3.1 get_nearest_person_attributes

```
def get_nearest_person_attributes(self, timeout: float = 5.0):
    """
    返回离机器人最近的人的视觉属性。
    向视觉节点发指令，视觉节点从当前帧 detected_objects 中
    取 3D 距离最近的人，返回其属性 dict。

    Returns:
        dict 或 None:
        {
            "cloth_color": "blue",
            "cloth_type": "t-shirt",
            "hair_color": "black",
            "has_glasses": false,
            "height": "medium",
            "position_3d": [x, y, z],
        }
    """
    实现：调用 self.execute("get_nearest_person", timeout=timeout)
```

视觉节点（open_vision_node.py）需要在收到 `get_nearest_person` 指令时：
- 遍历 `self.detected_objects` 中 class_name=="person" 的所有条目
- 取 3D 坐标距离相机原点最近的那个人
- 返回其 `cloth_color`, `cloth_type` 等属性
- 如果当前没有检测到任何人，返回 status=FAILED

### 3.2 filter_by_attributes

```
def filter_by_attributes(self, attributes: dict, timeout: float = 5.0):
    """
    从当前帧中找出匹配所有指定属性的人。
    向视觉节点发指令，视觉节点遍历 detected_objects 逐人比对。

    匹配规则：
      每个属性独立判断 match/mismatch。
      cloth_color: 精确匹配（字符串相等）
      cloth_type:  精确匹配
      has_glasses: 精确匹配
      hair_color:  精确匹配（如果已知）
      height:      精确匹配（如果已知）

    Returns:
        dict: {"status": "SUCCESS", "result": {"persons": [...], "count": N}}
        或 {"status": "FAILED", ...}
        每个 person 包含: bbox, position_3d, 以及所有属性字段
    """
    实现：调用 self.execute("filter_by_attributes", timeout=timeout, **attributes)
```

视觉节点（open_vision_node.py）需要在收到 `filter_by_attributes` 指令时：
- 遍历 `self.detected_objects` 中 class_name=="person" 的所有条目
- 对每个人，比对指令中每个属性字段
- 返回所有匹配的人的 3D 位置和属性
- 匹配逻辑：对每个非空属性字段，person 对象的对应字段必须完全相等
  - 如果 attribute 值为 "unknown" 或 None → 该属性不参与匹配（跳过）
  - 如果 person 对象缺少该字段 → 该属性不匹配

---

## 4. Schema 改动（schemas.py）

在 `RobotAction` 的 type 字段中，确保 LLM 可以输出以下 action type：

```
旧 action（保留）:
  goToLoc, findPrsInRoom, countPrsInRoom, findObjInRoom, countObjOnPlcmt,
  followNameFromBeacToRoom, guideNameFromBeacToBeac,
  bringMeObjFromPlcmt, takeObjFromPlcmt,
  tellPrsInfoInLoc, tellObjPropOnPlcmt, tellCatPropOnPlcmt,
  meetPrsAtBeac, greetNameInRm, greetClothDscInRm, countClothPrsInRoom

新 action（新增）:
  bind_person, recall, remember, find_person_by_attributes, speak
```

`bind_person`:
```
type: "bind_person"
name: str   ← 人名，如 "Charlie"
```

`recall`:
```
type: "recall"
what: str   ← "appearance" 或 "info"
name: str   ← 人名
```

`remember`:
```
type: "remember"
name: str   ← 人名
key: str    ← 属性名，如 "drink"
value: str  ← 属性值，如 "orange juice"
```

`find_person_by_attributes`:
```
type: "find_person_by_attributes"
attributes: dict  ← {"cloth_color": "blue", "cloth_type": "t-shirt", ...}
```

`speak`:
```
type: "speak"
text: str   ← TTS 要说的话
```

---

## 5. System Prompt 改动（prompts.py）

在 system_prompt 中新增以下规则（加入到 "可用 action" 描述区）：

```
## Person Memory Actions

When you meet a new person (e.g., a guest says their name):
  1. Call `bind_person` with their name to record their visual appearance.
  2. Call `remember` to store any extra info they tell you
     (e.g., drink preference).

When you need to describe a person you've met:
  - Call `recall(what="appearance")` to get their stored attributes,
    then describe them in natural language using `speak`.

When you need to find a person you've met:
  - Call `recall(what="appearance")` to get their attributes.
  - Then call `find_person_by_attributes` with those attributes.

When you need to greet or address a specific person by name:
  - Call `recall(what="appearance")` to know what they look like.
  - Then call `find_person_by_attributes` to locate them.
  - Then speak to them.

Attributes stored per person:
  cloth_color: "blue", "red", "white", "black", "gray", "yellow", "orange"
  cloth_type:  "t-shirt", "shirt", "sweater", "jacket", "coat", "blouse"
  hair_color:  "black", "brown", "blond", "gray", "white"
  has_glasses: true or false
  height:      "short", "medium", "tall"
```

---

## 6. 视觉节点改动（open_vision_node.py）

在任务指令处理函数中新增对以下两个指令的处理：

### get_nearest_person
```
收到: {"action": "get_nearest_person"}
处理: 遍历 self.detected_objects 中的 person 条目
      找到 3D 位置距离相机原点最近的人
返回: {
  "status": "SUCCESS",
  "result": {
    "cloth_color": "blue",
    "cloth_type": "t-shirt",
    "hair_color": "unknown",     ← 当前管线没有就填 "unknown"
    "has_glasses": false,        ← 当前管线没有就填 false
    "height": "unknown",         ← 当前管线没有就填 "unknown"
    "position_3d": [x, y, z]
  }
}
如果没有人 → {"status": "FAILED", "error": "No person detected"}
```

### filter_by_attributes
```
收到: {"action": "filter_by_attributes", "cloth_color": "blue", "cloth_type": "t-shirt"}
处理: 遍历 self.detected_objects 中的 person 条目
      对每个人逐属性比对（"unknown" 值跳过比对）
返回: {
  "status": "SUCCESS",
  "result": {
    "persons": [
      {
        "track_id": 3,
        "bbox": [x1, y1, x2, y2],
        "position_3d": [x, y, z],
        "cloth_color": "blue",
        "cloth_type": "t-shirt",
        ...
      }
    ],
    "count": 1
  }
}
如果没人匹配 → {"status": "SUCCESS", "result": {"persons": [], "count": 0}}
```

---

## 7. 删除 WorldModel 类

WorldModel 类整体删除（`memory.py` 中整个 `class WorldModel` 及其所有方法）。保留 `memory.py` 文件，只留 `EpisodicMemory`。

Controller 中：
- 删除 `from cade_brain.memory import WorldModel`，改为 `from cade_brain.memory import EpisodicMemory`
- 删除 `self.world = WorldModel()`
- 删除 `_enrich_prompt_with_world_state` 方法
- 删除 `observe()` 中对 `self.world.get_world_state()` 和 `_enrich_prompt_with_world_state()` 的调用
- 删除 `reset()` 中 `self.world = WorldModel()`
- 以 `self.current_position` 和 `self.holding_object` 两个简单变量替代 WorldModel 的状态存储
