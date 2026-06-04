from collections import defaultdict

import numpy as np


class RingBuffer:
    """每人维护一个环形缓冲区，存储最近 N 帧的 forearm + wrist 角度"""

    MIN_POST_JUMP = 15  # 跳变后最少保留帧数，不够则方差返回 0

    MAX_CONSECUTIVE_NONE = 5  # 连续 None 超过此次数 → 清空缓冲区

    def __init__(self, capacity=30):
        self.capacity = capacity
        self.forearm = defaultdict(lambda: {"left": [], "right": []})
        self.wrist_rot = defaultdict(lambda: {"left": [], "right": []})
        # 跟踪连续 None 次数: _none[person_id][source_name][side] = count
        self._none = defaultdict(lambda: defaultdict(lambda: {"left": 0, "right": 0}))

    def _push_side(self, source_name, store, person_id, side, angle):
        """单侧推送：非 None 追加并重置计数；None 累计计数，连续超阈值则清空旧数据"""
        if angle is not None:
            store[person_id][side].append(angle)
            if len(store[person_id][side]) > self.capacity:
                store[person_id][side].pop(0)
            self._none[person_id][source_name][side] = 0
        else:
            self._none[person_id][source_name][side] += 1
            if self._none[person_id][source_name][side] >= self.MAX_CONSECUTIVE_NONE:
                store[person_id][side].clear()

    def push_forearm(self, person_id, angle_left=None, angle_right=None):
        """Rule A: 前臂角度（弧度）"""
        self._push_side("forearm", self.forearm, person_id, "left", angle_left)
        self._push_side("forearm", self.forearm, person_id, "right", angle_right)

    def push_wrist(self, person_id, angle_left=None, angle_right=None):
        """Rule B: 手腕旋转角度（度）"""
        self._push_side("wrist", self.wrist_rot, person_id, "left", angle_left)
        self._push_side("wrist", self.wrist_rot, person_id, "right", angle_right)

    @staticmethod
    def _jump_aware_variance(angles, jump_threshold, window=30):
        """
        从最新帧往前扫描跳变点，只取跳变之后的数据计算方差。

        ① 计算 diff_n = |θₙ − θₙ₋₁|，从最新帧往前扫
        ② 找到第一个 > jump_threshold 的跳变点，切掉跳变前数据
        ③ 跳变后数据 < MIN_POST_JUMP → 返回 0
        ④ 无跳变 → 照常算

        Returns:
            (variance, post_jump_count) 或 (0.0, 0)
        """
        if len(angles) < 2:
            return 0.0, 0

        recent = angles[-min(window, len(angles)) :]

        # 从最新帧往前扫描跳变
        cut_idx = 0  # 切掉 [0:cut_idx]，保留 [cut_idx:]
        n = len(recent)
        for i in range(n - 1, 0, -1):
            diff = abs(recent[i] - recent[i - 1])
            if diff > jump_threshold:
                cut_idx = i
                break

        post_jump = recent[cut_idx:]
        if len(post_jump) < RingBuffer.MIN_POST_JUMP:
            return 0.0, len(post_jump)

        return float(np.var(post_jump)), len(post_jump)

    def get_forearm_variance(
        self, person_id, side="left", window=30, jump_threshold=None
    ):
        """Rule A 方差（弧度²），带跳变过滤"""
        angles = self.forearm[person_id][side]
        if not angles:
            return 0.0
        if jump_threshold is not None:
            var, _ = self._jump_aware_variance(angles, jump_threshold, window)
            return var
        # 无跳变阈值 → 照常
        if len(angles) < max(5, window // 2):
            return 0.0
        recent = angles[-min(window, len(angles)) :]
        return float(np.var(recent))

    def get_wrist_variance(
        self, person_id, side="left", window=30, jump_threshold=None
    ):
        """Rule B 方差（度数²），带跳变过滤"""
        angles = self.wrist_rot[person_id][side]
        if not angles:
            return 0.0
        if jump_threshold is not None:
            var, _ = self._jump_aware_variance(angles, jump_threshold, window)
            return var
        # 无跳变阈值 → 照常
        if len(angles) < max(5, window // 2):
            return 0.0
        recent = angles[-min(window, len(angles)) :]
        return float(np.var(recent))

    def clear(self):
        self.forearm.clear()
        self.wrist_rot.clear()
        self._none.clear()
