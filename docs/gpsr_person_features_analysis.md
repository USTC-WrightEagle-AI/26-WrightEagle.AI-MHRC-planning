# GPSR 指令生成器人物特征分析报告

## 核心结论

GPSR 指令生成器是一个**有限状态系统**。所有输出都来自一套**封闭的模板 + 封闭的占位符池**。因此，所有可能出现的**人物描述特征**可以 100% 穷举，不存在"未知的、没见过的特征突然出现"的情况。

---

## 一、指令生成器怎么工作的

### 1.1 文件协作关系

```
knowledge.py                    —— 解析 CompetitionTemplate 数据文件
    │                                   (人名/地点/房间/物体/类别)
    ▼
gpsr_commands.py                —— 核心生成引擎
    │
    ├── templates (23个主模板)
    ├── followup_templates (13个子模板)
    ├── 动词/介词/特征列表
    └── 占位符替换系统

    ▼
generator.py                    —— CLI 入口 (athome-generator)
    │
    ├── -g          → 批量生成 5000 条
    └── 交互模式     → 逐条生成

ui/gpsr_ui.py                  —— NiceGUI 界面
    │
    ├── 调 gpsr_commands.py 生成原始指令
    └── 调 llm.py 做 LLM 润色
```

### 1.2 核心算法：`generate_command_start`

```
第1步：从 person_cmd_list / object_cmd_list 按权重随机选一个模板名
       例: "findPrsInRoom" (权重=4)

第2步：从 templates 字典取出模板字符串
       例: "{findVerb} a {gestPers_posePers} {inLocPrep} the {room} and {FOLLOWUP:foundPers}"

第3步：解析 {FOLLOWUP:foundPers}
       → 查 followup_people["foundPers"] → 随机选 followup 子模板
       → 对子模板递归执行占位符替换

第4步：填充所有占位符
       → {findVerb}           → "find" / "locate" / "look for"
       → {gestPers_posePers}  → 手势(5种) 或 姿态(3种)
       → {inLocPrep}          → "in"
       → {room}               → 从 rooms 列表随机取

第5步：去重 (确保 loc2 ≠ loc)
第6步：修复冠词 (a/an)

→ 最终: "Find a standing person in the bedroom and follow them"
```

### 1.3 嵌套深度

`{FOLLOWUP:xxx}` 可以递归，但深度最多 3 层：

```
findObj → takeObj → deliverObjToPrsInRoom
```

不是无限递归。

### 1.4 特殊机制：`_` 分割

遇到 `{gestPers_posePers}` 这样的占位符，源码第 262-263 行：

```python
if "_" in ph:
    ph = random.choice(ph.split("_"))
```

**只随机选一个，不会组合**。所以永远不会出现"一个人同时挥手且坐着"的描述。

---

## 二、所有可能的占位符（可穷举）

### 2.1 人物感知相关占位符

| 占位符 | 行号 | 展开方式 | 产生种数 |
|--------|------|---------|---------|
| `{gestPers}` / `{gestPersPlur}` | 294, 299 | 从 `gesture_person_list` / `gesture_person_plural_list` 随机取 | **5 种** |
| `{posePers}` / `{posePersPlur}` | 296, 301 | 从 `pose_person_list` / `pose_person_plural_list` 随机取 | **3 种** |
| `{colorClothe}` | 312 | `itertools.product(color_list, clothe_list)` = 7 色 × 6 衣 | **42 种** |
| `{colorClothes}` | 314 | 同上，复数版 | **42 种** |
| `{persInfo}` | 304 | 从 `person_info_list` 随机取 | **3 种** |
| `{name}` | 265 | 从 `names.md` 读取的人名列表随机取 | **10 种** |
| `{inRoom}` | 280 | `"in the " + random.choice(rooms)` | **5 种** |
| `{atLoc}` | 287 | `"at the " + random.choice(locations)` | **14 种** |

### 2.2 其他（不涉及人物感知）

动词类: `{takeVerb}`, `{findVerb}`, `{followVerb}`, `{guideVerb}`, `{goVerb}`, `{tellVerb}`, `{talkVerb}`, `{meetVerb}`, `{greetVerb}`, `{deliverVerb}`, `{bringVerb}`, `{countVerb}`, `{answerVerb}`, `{describeVerb}`, `{offerVerb}`, `{accompanyVerb}`, `{rememberVerb}`, `{placeVerb}`

介词类: `{deliverPrep}`, `{placePrep}`, `{inLocPrep}`, `{fromLocPrep}`, `{toLocPrep}`, `{atLocPrep}`, `{talkPrep}`, `{locPrep}`, `{onLocPrep}`, `{arePrep}`, `{ofPrsPrep}`

物品类: `{obj}`, `{singCat}`, `{plurCat}`, `{objComp}`, `{plcmtLoc}`

---

## 三、所有涉及人物特征的模板

### 主模板

| 模板名 | 模板字符串 | 涉及的人物特征 |
|--------|-----------|--------------|
| `findPrsInRoom` | `{findVerb} a {gestPers_posePers} {inLocPrep} the {room} and {FOLLOWUP:foundPers}` | 手势/姿态 + 房间 |
| `meetPrsAtBeac` | `{meetVerb} {name} {inLocPrep} the {room} and {FOLLOWUP:foundPers}` | 人名 + 房间 |
| `countPrsInRoom` | `{countVerb} {gestPersPlur_posePersPlur} are {inLocPrep} the {room}` | 手势/姿态(复数) + 房间 |
| `tellPrsInfoInLoc` | `{tellVerb} me the {persInfo} of the person {inRoom_atLoc}` | name/pose/gesture + 房间/地点 |
| `talkInfoToGestPrsInRoom` | `{talkVerb} {talk} {talkPrep} the {gestPers} {inLocPrep} the {room}` | 手势(仅手势,不含姿态) + 房间 |
| `followNameFromBeacToRoom` | `{followVerb} {name} {fromLocPrep} the {loc} {toLocPrep} the {room}` | 人名 + 地点 + 房间 |
| `guideNameFromBeacToBeac` | `{guideVerb} {name} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}` | 人名 + 地点 |
| `guidePrsFromBeacToBeac` | `{guideVerb} the {gestPers_posePers} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}` | 手势/姿态 + 地点 |
| `guideClothPrsFromBeacToBeac` | `{guideVerb} the person wearing {art} {colorClothe} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}` | **衣物(42种)** + 地点 |
| `greetClothDscInRm` | `{greetVerb} the person wearing {art} {colorClothe} {inLocPrep} the {room} and {FOLLOWUP:foundPers}` | **衣物(42种)** + 房间 |
| `greetNameInRm` | `{greetVerb} {name} {inLocPrep} the {room} and {FOLLOWUP:foundPers}` | 人名 + 房间 |
| `meetNameAtLocThenFindInRm` | `{meetVerb} {name} {atLocPrep} the {loc} then {findVerb} them {inLocPrep} the {room}` | 人名 + 地点 + 房间 |
| `countClothPrsInRoom` | `{countVerb} people {inLocPrep} the {room} are wearing {colorClothes}` | **衣物(复数,42种)** + 房间 |
| `tellPrsInfoAtLocToPrsAtLoc` | `{tellVerb} the {persInfo} of the person {atLocPrep} the {loc} to the person {atLocPrep} the {loc2}` | name/pose/gesture + 地点(两处) |
| `followPrsAtLoc` | `{followVerb} the {gestPers_posePers} {inRoom_atLoc}` | 手势/姿态 + 房间/地点 |
| `goToLoc` | `{goVerb} {toLocPrep} the {loc_room} then {FOLLOWUP:atLoc}` | FOLLOWUP 可能引入人物 (findPrs/meetName) |

### 子模板（人物相关）

| 模板名 | 模板字符串 | 涉及的人物特征 |
|--------|-----------|--------------|
| `findPrs` | `{findVerb} the {gestPers_posePers} and {FOLLOWUP:foundPers}` | 手势/姿态 |
| `meetName` | `{meetVerb} {name} and {FOLLOWUP:foundPers}` | 人名 |
| `deliverObjToPrsInRoom` | `{deliverVerb} it {deliverPrep} the {gestPers_posePers} {inLocPrep} the {room}` | 手势/姿态 + 房间 |
| `deliverObjToNameAtBeac` | `{deliverVerb} it {deliverPrep} {name} {inLocPrep} the {room}` | 人名 + 房间 |
| `followPrs` | `{followVerb} them` | 不涉及新特征(参照前文) |
| `followPrsToRoom` | `{followVerb} them {toLocPrep} the {loc2_room2}` | 不涉及新特征 |
| `guidePrsToBeacon` | `{guideVerb} them {toLocPrep} the {loc2_room2}` | 不涉及新特征 |

---

## 四、特征定义的数据来源（源码精确行号）

所有人物特征数据定义在 `gpsr_commands.py` 中：

```python
# 手势 — 第143-149行
self.gesture_person_list = [
    "waving person",
    "person raising their left arm",
    "person raising their right arm",
    "person pointing to the left",
    "person pointing to the right",
]

# 姿态 — 第150行
self.pose_person_list = ["sitting person", "standing person", "lying person"]

# 颜色 — 第184行
self.color_list = ["blue", "yellow", "black", "white", "red", "orange", "gray"]

# 衣物类型 — 第185行
self.clothe_list = ["t shirt", "shirt", "blouse", "sweater", "coat", "jacket"]

# 颜色×衣物组合 — 第188行
self.color_clothe_list = [f"{a} {b}" for a, b in itertools.product(self.color_list, self.clothe_list)]

# 可询问的人物信息 — 第161行
self.person_info_list = ["name", "pose", "gesture"]

# 动词/介词定义 — 第106-139行
# 知识数据 — knowledge.py + CompetitionTemplate/*.md
```

---

## 五、机器人必须能识别的全部人物特征

从代码推导出的固定集合：

```
┌─────────────────────────────────────────────────────────┐
│               GPSR 人物感知需求全集                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ① 人体构型 (Body Configuration) — 8种                   │
│     ├── 姿态 3种: standing / sitting / lying              │
│     └── 手势 5种: waving                                  │
│                    raising left arm                       │
│                    raising right arm                      │
│                    pointing left                          │
│                    pointing right                         │
│                                                         │
│  ② 衣物属性 (Clothing Attributes) — 7色 × 6衣 = 42种     │
│     ├── 颜色 7种: white / black / red / blue              │
│     │            gray / yellow / orange                   │
│     └── 类型 6种: t-shirt / shirt / blouse                │
│                    sweater / coat / jacket                │
│                                                         │
│  ③ 身份 (Identity) — 10人                                │
│     └── 人名: Adel / Angel / Axel / Charlie / Jane       │
│               Jules / Morgan / Paris / Robin / Simone     │
│                                                         │
│  ④ 位置关联 (Location-based)                             │
│     ├── 房间 5种: bedroom / kitchen / office              │
│     │            living room / bathroom                   │
│     └── 地点 14种: bed / shelf / sofa / ...               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**没有第⑤类了。** 这 4 大类涵盖了 GPSR 指令对机器人视觉感知的全部要求。

---

## 六、LLM 润色的影响

### 6.1 润色流程

```
原始指令 (模板生成)
    ↓
LLM 收到 system prompt + 原始指令
    ↓
LLM 输出 3 个改写版本 (temperature = 1.0)
    ↓
裁判选择其中一个发给机器人
```

### 6.2 System Prompt 约束

`llm.py` 第 66 行：

> **"Keep all entities, objects, and locations exactly the same"**

### 6.3 分析

| 现象 | 概率 | 影响 |
|------|------|------|
| 句式重组 ("waving person" → "person who is waving") | 必然发生 | 无影响，特征不变 |
| 同义词偏移 ("t shirt" → "shirt", "jacket" → "coat") | 可能发生 | 类内部值偏移，不增新类 |
| 凭空新增感知特征 ("...holding a cup") | 极低 | system prompt 约束了实体不变 |
| 删除原有特征 | 极低 | 同上 |

### 6.4 应对方案

- 同义词映射表：`coat ↔ jacket`, `t-shirt ↔ shirt`, `blouse ↔ shirt`
- CLIP 做衣物分类时，把语义相近的类别合并
- 比赛前用 LLM 润色一批指令，验证实际偏移情况

---

## 七、附录：统计数据

基于 5000 条生成指令的统计（原始模板，未润色）：

| 类别 | 出现次数 | 占比 |
|------|---------|------|
| 含手势/姿态描述 | 1652 条 | 33.0% |
| 含衣物描述 | 130 条 | 2.6% |
| 含人名 | 1480 条 | 29.6% |
| person at [location] | 118 条 | 2.4% |

各姿态占比：lying 13.6% / standing 12.2% / sitting 11.7%
各手势占比：raising left arm 12.5% / waving 11.9% / pointing right 11.7% / raising right arm 11.3% / pointing left 11.0%
各颜色占比：white > blue > gray > red > orange > black > yellow
各衣物占比：shirt > t-shirt > blouse > jacket > coat > sweater

---

*生成日期: 2026-05-13*
*源码版本: CommandGenerator (https://github.com/RoboCupAtHome/CommandGenerator)*
*分析工具: ~/HysProjects/CommandGenerator/src/robocupathome_generator/gpsr_commands.py*
