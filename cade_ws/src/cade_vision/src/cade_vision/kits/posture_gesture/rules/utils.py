import math

import numpy as np


def get_index_pose(landmarks, side):
    """获取食指的坐标"""
    LEFT_INDEX = 19  # 左手食指 MCP（掌指关节）
    RIGHT_INDEX = 20  # 右手食指 MCP
    idx = LEFT_INDEX if side == "left" else RIGHT_INDEX
    if idx < len(landmarks) and landmarks[idx][2] > 0.3:
        return get_landmark(landmarks, idx)
    return None


def get_landmark(landmarks, idx):
    return np.array([landmarks[idx][0], landmarks[idx][1]])


def is_visible(landmarks, idx):
    return idx < len(landmarks) and landmarks[idx][2] > 0.5


def is_visible_relaxed(landmarks, idx):
    """松弛可见性：visibility > 0.3 即可用于角度计算"""
    return idx < len(landmarks) and landmarks[idx][2] > 0.3


def compute_side_angles(landmarks, side, elbow_idx, wrist_idx, fine_wrist, fine_index):
    """返回 (elbow_angle, wrist_angle)。
    Rule A: 始终用 Pose wrist。
    Rule B: 用 Hands wrist + index_tip，两者缺一则该侧返回 None。"""
    if not is_visible_relaxed(landmarks, elbow_idx) or not is_visible_relaxed(
        landmarks, wrist_idx
    ):
        return None, None

    elbow = get_landmark(landmarks, elbow_idx)
    wrist = get_landmark(landmarks, wrist_idx)

    # Rule A: wrist->elbow atan2（始终 Pose）
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    ea = math.degrees(math.atan2(dy, dx))

    # Rule B: palm direction atan2（Hands wrist + index_tip，缺一不可）
    wa = None
    if fine_wrist[side] is not None and fine_index[side] is not None:
        dx = fine_index[side][0] - fine_wrist[side][0]
        dy = fine_index[side][1] - fine_wrist[side][1]
        wa = math.degrees(math.atan2(dy, dx))

    return ea, wa
