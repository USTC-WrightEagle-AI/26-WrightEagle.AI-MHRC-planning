# 修复：gesture "pointing" 方向判断逻辑

## Bug 描述

当前 `posture_gesture.py` 第 270-273 行判断 pointing 方向时，用的是"哪条手臂在伸"：

```python
elif left_pointing and not right_pointing:
    return "pointing_left"    # 左臂伸出 → "pointing_left"
elif right_pointing and not left_pointing:
    return "pointing_right"   # 右臂伸出 → "pointing_right"
```

这不对。GPSR 指令中的 `person pointing to the left / right` 是从**观察者/摄像头视角**看的，不是从人自身的左右看的。

例如：人面对镜头时，他的左手在图像的右侧。如果他左手水平伸出指向图像右侧，当前代码返回 `"pointing_left"`（因为是左手），但实际从摄像头看，**他指向了右边**。

## 修复方案

改成**用手在身体中线的哪一侧**来判断方向：

```python
# 先算身体中线
body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2

# 如果只有一只手在伸出，看这只手在中线的哪一侧
if left_pointing and not right_pointing:
    if left_wrist[0] > body_center_x:
        gesture = "pointing_right"   # 手在中线右侧 → 指向右
    else:
        gesture = "pointing_left"    # 手在中线左侧 → 指向左
elif right_pointing and not left_pointing:
    if right_wrist[0] > body_center_x:
        gesture = "pointing_right"
    else:
        gesture = "pointing_left"
elif left_pointing and right_pointing:
    gesture = "pointing_both"        # 双手伸出
```

## 验证

修完后重新跑一遍 bus.jpg 测试验证结果。
