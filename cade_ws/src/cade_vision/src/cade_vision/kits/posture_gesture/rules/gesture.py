import numpy as np

from .utils import (
    compute_side_angles,
    get_index_pose,
    get_landmark,
    is_visible,
)

# MediaPipe Pose 关键点索引恒量定义
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20


def is_arm_raised(wrist, elbow, shoulder, shoulder_dist):
    """
    静态举手检测（纯空间几何计算）
    """
    # 规则 1: 手腕接近或超过肩膀高度
    MARGIN = shoulder_dist * 0.2
    rule1 = wrist[1] <= shoulder[1] + MARGIN
    if not rule1:
        return False

    # 规则 2: 前臂竖直（肘→腕方向接近向上）
    horiz_ok = abs(wrist[0] - elbow[0]) < shoulder_dist * 0.3
    vert_ok = wrist[1] < elbow[1]
    if not (horiz_ok and vert_ok):
        return False

    return True


def is_pointing(wrist, elbow, index_pose, side, hands_data, shoulder_dist):
    """
    静态指向检测（结合 Pose 箭身与 Hands 精细手指模式）
    """
    # 规则 1: 箭身水平（elbow, wrist, index_mcp 三点 y 对齐容差）
    T_ARROW_Y = shoulder_dist * 0.25
    rule1 = (
        abs(elbow[1] - wrist[1]) < T_ARROW_Y
        and abs(wrist[1] - index_pose[1]) < T_ARROW_Y
    )
    if not rule1:
        return False, None

    # 规则 2: 手指姿势（必须依赖精细 Hands 数据，无数据直接拒判）
    if not hands_data:
        return False, None

    hd = hands_data.get("Left" if side == "left" else "Right")
    if not hd or not hd.get("landmarks") or len(hd["landmarks"]) < 21:
        return False, None

    lm = hd["landmarks"]

    # 内部辅助闭包：计算 TIP 到 MCP 的欧氏距离
    def tip_mcp_dist(mcp_idx):
        tip = np.array(lm[mcp_idx + 3])  # TIP = MCP + 3
        mcp = np.array(lm[mcp_idx])
        return np.linalg.norm(tip - mcp)

    # 参考长度基准：食指 PIP-MCP 长度 × 2.5
    ref_len = np.linalg.norm(np.array(lm[6]) - np.array(lm[5])) * 2.5
    if ref_len < 1e-6:
        ref_len = 0.1  # Fallback 兜底

    index_ext = tip_mcp_dist(5) > ref_len * 0.75
    middle_ret = tip_mcp_dist(9) < ref_len * 0.50
    ring_ret = tip_mcp_dist(13) < ref_len * 0.50
    pinky_ret = tip_mcp_dist(17) < ref_len * 0.50
    thumb_ret = np.linalg.norm(np.array(lm[4]) - np.array(lm[2])) < ref_len * 0.50

    # 模式 A：食指伸出，其余缩回； 模式 B：食指中指双伸出（剪刀差兜底）
    pattern_a = index_ext and middle_ret and ring_ret and pinky_ret and thumb_ret
    pattern_b = index_ext and ring_ret and pinky_ret and thumb_ret

    if not (pattern_a or pattern_b):
        return False, None

    # 规则 3: 建立方向（以相机视角为准：肘.x < 腕.x 则是向右指）
    direction = "right" if elbow[0] < wrist[0] else "left"
    return True, direction


def classify_static_gesture(landmarks, h, w, hands_data: dict = None):
    """
    【单帧核心】基于上肢关键点分类静态手势拓扑。
    修复了入参缺失 h, w 的问题，移除了越权持有的 person_id。
    """
    # 1. 初始化精细手腕与食指坐标映射
    fine_wrist = {"left": None, "right": None}
    fine_index = {"left": None, "right": None}
    if hands_data:
        for side_key, side_lower in [("Left", "left"), ("Right", "right")]:
            hd = hands_data.get(side_key)
            if hd and hd.get("wrist") and hd.get("index_tip"):
                fine_wrist[side_lower] = np.array(hd["wrist"])
                fine_index[side_lower] = np.array(hd["index_tip"])

    # 2. 调用 utils 独立计算每侧关节角（用于时序 Buffer 压入）
    elbow_l, wrist_l = compute_side_angles(
        landmarks, "left", LEFT_ELBOW, LEFT_WRIST, fine_wrist, fine_index
    )
    elbow_r, wrist_r = compute_side_angles(
        landmarks, "right", RIGHT_ELBOW, RIGHT_WRIST, fine_wrist, fine_index
    )

    # 3. 静态拓扑分类基础可见性拦截
    key_points = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]
    if not all(
        is_visible(landmarks, i) for i in key_points
    ):  # 确保 utils 中的 is_visible 接收了 landmarks
        return "unknown", elbow_l, elbow_r, wrist_l, wrist_r

    # 4. 提取关键点空间向量
    left_shoulder = get_landmark(landmarks, LEFT_SHOULDER)
    right_shoulder = get_landmark(landmarks, RIGHT_SHOULDER)
    left_elbow = get_landmark(landmarks, LEFT_ELBOW)
    right_elbow = get_landmark(landmarks, RIGHT_ELBOW)
    left_wrist = get_landmark(landmarks, LEFT_WRIST)
    right_wrist = get_landmark(landmarks, RIGHT_WRIST)

    # 以肩宽作为自适应动态尺度因子
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist < 1e-6:
        return "unknown", elbow_l, elbow_r, wrist_l, wrist_r

    # 5. 执行“举手”解耦校验
    left_raised = is_arm_raised(left_wrist, left_elbow, left_shoulder, shoulder_dist)
    right_raised = is_arm_raised(
        right_wrist, right_elbow, right_shoulder, shoulder_dist
    )

    # 6. 执行“指向”解耦校验
    left_index_mcp = get_index_pose(landmarks, "left")
    right_index_mcp = get_index_pose(landmarks, "right")

    left_pointing, l_dir = (
        is_pointing(
            left_wrist, left_elbow, left_index_mcp, "left", hands_data, shoulder_dist
        )
        if left_index_mcp is not None
        else (False, None)
    )
    right_pointing, r_dir = (
        is_pointing(
            right_wrist,
            right_elbow,
            right_index_mcp,
            "right",
            hands_data,
            shoulder_dist,
        )
        if right_index_mcp is not None
        else (False, None)
    )

    # 7. 静态优先级决策树（严格规范输出字符串，带上标准的 _arm 后缀）
    if left_raised and right_raised:
        return "maybe_raising_both_arms", elbow_l, elbow_r, wrist_l, wrist_r
    if left_raised and not right_raised:
        return "maybe_raising_left_arm", elbow_l, elbow_r, wrist_l, wrist_r
    if right_raised and not left_raised:
        return "maybe_raising_right_arm", elbow_l, elbow_r, wrist_l, wrist_r

    if left_pointing and right_pointing:
        return "maybe_pointing_both", elbow_l, elbow_r, wrist_l, wrist_r
    if left_pointing and not right_pointing:
        return f"maybe_pointing_{l_dir}", elbow_l, elbow_r, wrist_l, wrist_r
    if right_pointing and not left_pointing:
        return f"maybe_pointing_{r_dir}", elbow_l, elbow_r, wrist_l, wrist_r

    return "none", elbow_l, elbow_r, wrist_l, wrist_r


def check_shoulder_proximity(landmarks, SHOULDER_OFFSET) -> bool:
    """
    时序辅助空间门控：校验手关节是否高过肩膀或在附近，防止下垂手势误触 waved
    """
    if len(landmarks) <= RIGHT_INDEX:
        return False

    def visible(idx):
        return landmarks[idx][2] > 0.5

    ok_left = False
    if visible(LEFT_SHOULDER) and visible(LEFT_INDEX):
        ok_left = (
            landmarks[LEFT_INDEX][1] <= landmarks[LEFT_SHOULDER][1] + SHOULDER_OFFSET
        )

    ok_right = False
    if visible(RIGHT_SHOULDER) and visible(RIGHT_INDEX):
        ok_right = (
            landmarks[RIGHT_INDEX][1] <= landmarks[RIGHT_SHOULDER][1] + SHOULDER_OFFSET
        )

    return ok_left or ok_right


def judge_all_temporal_gestures(
    raw_gesture: str, variances: dict, landmarks: list, thresholds: dict
) -> tuple:
    """
    【时序最高法院】结合单帧“maybe_xxx”状态与多帧时序方差决策最终互斥动作
    """
    max_fa = max(variances["fa_l"], variances["fa_r"])
    max_wr = max(variances["wr_l"], variances["wr_r"])

    # 肢体静止度校验门槛
    T_STATIC = 50.0
    left_is_static = variances["fa_l"] < T_STATIC and variances["wr_l"] < T_STATIC
    right_is_static = variances["fa_r"] < T_STATIC and variances["wr_r"] < T_STATIC

    # 1. 高频周期晃动判定 —— 强行强占为挥手（Waving 优先级最高）
    rule_a_waving = max_fa > thresholds["T_FOREARM"]
    rule_b_waving = max_wr > thresholds["T_WRIST"]
    shoulder_ok = check_shoulder_proximity(landmarks, thresholds["SHOULDER_OFFSET"])

    if (rule_a_waving or rule_b_waving) and shoulder_ok:
        return "waving", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "waving_override",
        }

    # 2. 举手时序状态决策（精确匹配单帧 maybe 状态）
    if raw_gesture == "maybe_raising_left_arm" and left_is_static:
        return "raising_left_arm", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "stable_raising",
        }

    if raw_gesture == "maybe_raising_right_arm" and right_is_static:
        return "raising_right_arm", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "stable_raising",
        }

    if raw_gesture == "maybe_raising_both_arms" and left_is_static and right_is_static:
        return "raising_both_arms", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "stable_raising",
        }

    # 3. 指向时序状态决策
    if raw_gesture == "maybe_pointing_left" and left_is_static:
        return "pointing_left", {"decision": "stable_pointing"}
    if raw_gesture == "maybe_pointing_right" and right_is_static:
        return "pointing_right", {"decision": "stable_pointing"}
    if raw_gesture == "maybe_pointing_both" and left_is_static and right_is_static:
        return "pointing_both", {"decision": "stable_pointing"}

    # 4. 晃动抑制：如果单帧有动作意图，但时序方差过大（未到挥手门槛），直接打回 none 拒绝误触
    maybe_gestures = {
        "maybe_raising_left_arm",
        "maybe_raising_right_arm",
        "maybe_raising_both_arms",
        "maybe_pointing_left",
        "maybe_pointing_right",
        "maybe_pointing_both",
    }
    if raw_gesture in maybe_gestures:
        return "none", {"decision": "filtered_by_shaking"}

    return "none", {"decision": "static_none"}
