# 修复：test_bus_mediapipe.py 中的骨架连线 Bug

## Bug 描述

`draw_skeleton()` 函数中，骨架连线乱掉的根因是第 68-69 行：

```python
# ❌ 当前错误代码
px2 = int(x2 + lm2[0] * crop_w)  # 用了 x2（框的右边缘）
py2 = int(y2 + lm2[1] * crop_h)  # 用了 y2（框的下边缘）

# ✅ 应该改为
px2 = int(x1 + lm2[0] * crop_w)  # 用 x1（框的左边缘）
py2 = int(y1 + lm2[1] * crop_h)  # 用 y1（框的上边缘）
```

landmarks 的坐标是**相对于人框左上角 (x1, y1) 的归一化值（0~1）**，所以还原到原图时两个端点都应该以 (x1, y1) 为基准。用了 (x2, y2) 导致第二个端点被向右下偏移了 `(crop_w, crop_h)` 像素，连线全乱。

## 次要修复

第 64 行可见性判断逻辑反了：

```python
# ❌ 当前：两个都不可见才跳过（错误）
if lm1[2] < 0.5 and lm2[2] < 0.5:
    continue

# ✅ 应该：有一个不可见就跳过
if lm1[2] < 0.5 or lm2[2] < 0.5:
    continue
```

## 修复后

重新运行测试，结果图保存到 `CADE/test_images_result/bus_mediapipe_test.jpg`。
