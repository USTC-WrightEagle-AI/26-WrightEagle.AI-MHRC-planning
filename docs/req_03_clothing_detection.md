# 需求：CADE 视觉系统扩展 — 衣物检测 + 动态类别扩展

## 一、背景

GPSR 指令中包含衣物描述，如 "the person wearing a red t-shirt"。需要识别画面中人物的衣物颜色和类型。

**已知技术限制**（已验证）：
- CLIP 衣物颜色判别的分差仅 ~0.015，勉强可用
- CLIP 不适合姿态/手势/手持物判断（已交给 MediaPipe）
- YOLO-World 的 `set_classes` 不能做属性绑定（"person wearing white clothes" ≈ "person"）

**因此采用混合方案**：HSV 提取颜色（确定性） + CLIP 验证类型（语义）。

## 二、架构：基础类常开 + 动态注入

### 2.1 设计原则

- **不硬编码**：裁判可能说出不在预定义列表里的衣物词
- **不降级**：一旦注入新类别就保留，任务结束不还原（避免竞态）
- **LLM 解耦**：LLM 不需要知道某个词在不在检测列表里，它只管输出 `cloth_type`

### 2.2 基础类列表

```python
BASE_CLASSES = [
    # 人
    "person",
    # 上装
    "t-shirt", "shirt", "sweater", "jacket", "coat", "blouse",
    # 下装
    "pants", "skirt", "shorts",
]
```

这些类从节点启动时设为 YOLO-World 的检测目标，**始终检测，不因任务变化而移除**。

### 2.3 动态注入机制

从 `/cade/task_cmd` 的 `attributes` 中提取可能的 YOLO 类别。初始动态键：

```python
DYNAMIC_KEYS = ["cloth_type"]  # 可扩展
```

处理逻辑：

```python
def _build_classes(attributes: dict | None) -> list[str]:
    """合并基础类 + 从 attributes 中提取的动态词"""
    merged = list(BASE_CLASSES)  # 复制
    if attributes:
        for key in DYNAMIC_KEYS:
            val = attributes.get(key, "")
            if val and val not in merged:
                merged.append(val)
    return merged
```

**调用时机**：节点初始化时首次设置，每次收到 `/cade/task_cmd` 时重新合并设置。


## 三、衣物识别算法

### 3.1 数据流概览

```text
process_frame:
  YOLO 推理 → boxes (person + clothes)
                    │
     ┌──────────────┴──────────────┐
     ▼                             ▼
  person boxes                clothes boxes
  (走 MediaPipe 管线)          (走衣物分析)
     │                             │
     │                      ① IOU 关联到 person
     │                      ② 裁剪衣服区域
     │                      ③ HSV → 颜色
     │                      ④ 写入 person 的 obj_info
     │                             │
     └──────────────┬──────────────┘
                    ▼
           detected_objects（每人含 cloth_color, cloth_type）
```

### 3.2 Person-Clothes 关联（IOU 匹配）

对每个衣服框，找它和哪个 person 框重叠最大：


**多件衣服叠加**（如 jacket 套在 t-shirt 外面）：一个 person 可能关联多件衣服。这时取 IOU 最高的上装和最高的下装分别作为该人的上衣/裤子属性。也可以只用第一件匹配到的（简化处理）。

### 3.3 HSV 颜色提取（7 色）

从衣服裁剪图中提取主色。逻辑分两步：

**第一步：取衣服裁剪区域，转 HSV，统计 V 通道和 S 通道**

```text
取衣服框对应的像素区域 → cv2.cvtColor(BGR→HSV)
计算: mean_H, mean_S, mean_V
```

**第二步：映射规则**

```text
无彩色（白/灰/黑）：S < 50
  V > 200 → white
  V < 80  → black
  其他    → gray

有彩色（S >= 50）：按 H 通道划分
  H < 20 或 H >= 160 → red
  20 <= H < 40      → orange  (或 yellow，可合并)
  40 <= H < 80      → yellow
  80 <= H < 140     → blue
  其余              → unknown

映射表（OpenCV HSV: H∈[0,180], S,V∈[0,255]）:
  "red":    (0 <= H < 15) or (165 <= H <= 180), S >= 50
  "orange": 15 <= H < 25, S >= 50
  "yellow": 25 <= H < 40, S >= 50  (可与 orange 合并为 yellow)
  "blue":   100 <= H < 140, S >= 50
  "white":  S < 50, V > 200
  "black":  S < 50, V < 80
  "gray":   S < 50, 80 <= V <= 200
```

**颜色提取健壮性**：取衣服框中心区域（框的 50%~80% 宽高）的 HSV 均值，避免衣服边缘混入背景像素。

**处理多个衣服框**：
- 上装和下装分别提取颜色
- `cloth_color` 默认为上衣颜色（因为 GPSR 指令 90% 描述的是上衣）
- 也可返回结构化数据 `cloth_top_color` / `cloth_bottom_color`

### 3.4 写入检测结果

将结果写入 person obj_info：

```python
obj["cloth_color"] = detected_color      # "red" | "blue" | ... | "unknown"
obj["cloth_type"] = detected_type        # "t-shirt" | "hoodie" | ... | "unknown"
```

**不需要修改 `detected_objects` 的结构** — 只是给 person 的 dict 加了两个新 key。

## 四、与现有系统的集成

### 4.1 不需要改动的地方

- **`Analyzer.filter_by_attributes`**：已经在 `_cost_order` 中预留了 `"cloth_color"`，按 `str(p.get("cloth_color", "")).lower() == expected_lower` 过滤。无需修改。
- **`_execute_search_task`** / **`_execute_count_task`**：只读 `detected_objects` 缓存，不感知内部字段。无需修改。
- **`VisionSkill`**：已支持 `cloth_color` 参数传递。无需修改。
- **MediaPipe 管线**：不受影响。

### 4.2 需要改动的地方

| 文件 | 改动 |
|------|------|
| `open_vision_node.py` | 1. 新增 `BASE_CLASSES` 常量<br>2. 启动时 `set_classes(BASE_CLASSES)`<br>3. 任务回调改为 `set_classes(合并列表)`<br>4. `process_frame` 中插入衣物分析（在 tracker 之后、MediaPipe 之前）<br>5. 将 `cloth_color` / `cloth_type` 写入 person obj_info |

### 4.3 插入点（process_frame 中的顺序）

```text
1. YOLO 推理（已有）
2. 构建 new_detections 列表（已有）
3. 跨帧追踪分配 track_id（已有）
4. [NEW] 衣物分析：关联 clothes ↔ person → HSV + CLIP
5. MediaPipe Pose/Hands 后处理（已有）
6. 写入 detected_objects（已有）
```

衣物分析放在 MediaPipe 之前，因为衣物分析只需要 YOLO 框，不依赖关键点。

## 五、实现步骤（给 Claude Code）

1. **在 `open_vision_node.py` 顶部定义 `BASE_CLASSES` 和颜色映射表**
2. **实现 `_build_classes(attributes)` 方法**
3. **修改 `__init__`**：YOLO 加载后立即 `set_classes(BASE_CLASSES)`
4. **修改 `_on_task_cmd`**：把现有的 `set_classes([target_class])` 替换为 `set_classes(合并列表)`
5. **实现 `_analyze_clothing(detections, color_image)` 方法**：
   - 分离 person boxes 和 clothes boxes
   - IOU 匹配 clothes → person
   - HSV 颜色提取
   - CLIP 类型验证（如果可用）
   - 回写 `cloth_color` / `cloth_type` 到 person obj_info
6. **在 `process_frame` 中调用 `_analyze_clothing`**（在 tracker 之后、MediaPipe 之前）
7. **CLIP 加载**：在 `__init__` 或首次使用时加载（如果环境可用）

## 六、验收标准

1. 在测试图片上，穿蓝色 t-shirt 的人，`cloth_color` 正确输出 `"blue"`，`cloth_type` 正确输出 `"t-shirt"`
2. 穿红色 jacket 的人能被 `filter_by_attributes({"cloth_color": "red"})` 过滤出来
3. 裁判说一个不在基础列表里的词（如 `"hoodie"`），系统自动注入该类并开始检测
4. 不破坏现有的 YOLO 检测、3D 坐标发布、MediaPipe 姿态手势功能
5. 每种颜色的 HSV 阈值调整后，在室内光照下（比赛环境）7 色准确率 > 80%

## 七、注意事项

1. **CLIP 模型大小**：确保加载后不超过 GTX 1660 SUPER 的 6GB VRAM。现有 YOLO-World 已占用一部分，CLIP ViT-B/32（~350MB）应可共存
2. **HSV 阈值需要标定**：在 CADE 的实际摄像头上采样几张不同颜色衣服的图做阈值验证，不要直接拍脑袋设
3. **颜色歧义**：red/orange 边界（H ~15-25）可能误判，允许在实际测试中合并为 "red" 或调整
4. **多件衣服**：如果一个人同时穿了 t-shirt 和 jacket（两件上装），以 IOU 最高的为准，还是取最外层的？**建议取 IOU 最高的**（通常就是最外层可见的）
