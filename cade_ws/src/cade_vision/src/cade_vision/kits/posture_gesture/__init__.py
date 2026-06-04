"""
Posture & Gesture Analyzer - 基于 MediaPipe Pose 的姿态和手势识别

职责：
- 接收人框裁剪图，输出 posture（standing/sitting/lying/unknown）
  和 gesture（waving/raising_left_arm/raising_right_arm/pointing_left/pointing_right/none/unknown）
- 维护每人 30 帧环形缓冲区，支持跨帧挥手检测（两路 OR：前臂摆动 + 手腕旋转）
"""

import cv2
import numpy as np

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_IMPORT_ERROR = None
except Exception as exc:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_IMPORT_ERROR = exc

from .rules.gesture import classify_static_gesture, judge_all_temporal_gestures
from .rules.posture import classify_posture_3d
from .temporal_buffer import RingBuffer


class PostureGestureAnalyzer:
    """基于 MediaPipe Pose 的姿态 + 手势分析器"""

    def __init__(
        self, T_FOREARM=300, T_WRIST=500, SHOULDER_OFFSET=0.15, JUMP_THRESHOLD=50
    ):
        """
        Args:
            T_FOREARM: 前臂角度方差阈值（度²），默认 300
            T_WRIST:  手掌方向角度方差阈值（度²），默认 200
            SHOULDER_OFFSET: IMCP 低于肩部的最大容忍值（归一化坐标），默认 0.15
            JUMP_THRESHOLD: 角度跳变检测阈值，默认 20 度
        """
        self.pose = None
        self.hands = None
        self.available = MEDIAPIPE_AVAILABLE
        self.mp_pose = None
        self.mp_hands = None
        self.ring_buffer = RingBuffer(capacity=30)

        # 挥手检测阈值（可调参数）
        self.T_FOREARM = T_FOREARM
        self.T_WRIST = T_WRIST
        self.SHOULDER_OFFSET = SHOULDER_OFFSET
        self.JUMP_THRESHOLD = JUMP_THRESHOLD

        if self.available:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def process_pose(self, person_crop: np.ndarray) -> dict:
        """
        对单个人框裁剪图跑 MediaPipe Pose，返回原始关键点（不做分类）。

        Returns:
            {"landmarks": [(x,y,score), ... 33 keypoints], "img_h": int, "img_w": int}
            或 None（如果 Pose 不可用或未检测到人）。
        """
        if not self.available or self.pose is None or person_crop.size == 0:
            return None
        h, w = person_crop.shape[:2]
        if h < 30 or w < 30:
            return None
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks is None:
            return None
        landmarks = [
            (lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark
        ]
        return {"landmarks": landmarks, "img_h": h, "img_w": w}

    def process_hands(self, person_crop: np.ndarray) -> dict:
        """
        对单个人框裁剪图跑 MediaPipe Hands，返回手腕/食指指尖的精细坐标 + 全部 21 点。

        Returns:
            {
                "Left":  {"wrist": (x,y), "index_tip": (x,y), "landmarks": [(x,y),...]} | None,
                "Right": ...,
            }
            仅当 hand detection confidence > 0.5 时才返回该侧数据。
        """
        result = {"Left": None, "Right": None}
        if not self.available or self.hands is None or person_crop.size == 0:
            return result

        h, w = person_crop.shape[:2]
        if h < 30 or w < 30:
            return result

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        hands_results = self.hands.process(rgb)

        if hands_results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx]
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score
                if score < 0.5:
                    continue
                wrist = (hand_lms.landmark[0].x, hand_lms.landmark[0].y)
                index_tip = (hand_lms.landmark[8].x, hand_lms.landmark[8].y)
                landmarks = [(lm.x, lm.y) for lm in hand_lms.landmark]
                result[label] = {
                    "wrist": wrist,
                    "index_tip": index_tip,
                    "landmarks": landmarks,
                }

        return result

    def analyze_from_landmarks(
        self,
        pose_data: dict,
        person_id=None,
        hands_data: dict = None,
        with_temporal: bool = True,
        keypoints_3d: dict = None,
    ) -> dict:
        """
        从预计算的 landmarks 做分类 + 角度计算（不再跑模型推理）。

        Args:
            pose_data: process_pose() 的输出 {"landmarks": [...], "img_h": h, "img_w": w}
            person_id:  时序挥手检测用的人物 ID（with_temporal=True 时必传）
            hands_data: process_hands() 的输出
            with_temporal: 是否跑时序挥手检测
            keypoints_3d: {23: (x,y,z), 24: (x,y,z), ...} 深度图提取的 3D 关键点

        Returns:
            同 analyze() 或 analyze_with_temporal()
        """
        if pose_data is None:
            return self._unknown_result()

        landmarks = pose_data["landmarks"]
        h, w = pose_data["img_h"], pose_data["img_w"]

        posture, _ = classify_posture_3d(landmarks, h, keypoints_3d)
        gesture, elbow_l, elbow_r, wrist_l, wrist_r = classify_static_gesture(
            landmarks, h, w, hands_data
        )

        result = {
            "landmarks": landmarks,
            "posture": posture,
            "gesture": gesture,
            "elbow_angle_left": elbow_l,
            "elbow_angle_right": elbow_r,
            "wrist_angle_left": wrist_l,
            "wrist_angle_right": wrist_r,
        }

        if with_temporal and person_id is not None:
            result = self._apply_temporal(result, person_id)

        return result

    def _apply_temporal(self, result: dict, person_id) -> dict:
        """
        全动作时序大一统整合总线
        """
        # 1. 始终推入缓冲区更新时序
        self.ring_buffer.push_forearm(
            person_id, result["elbow_angle_left"], result["elbow_angle_right"]
        )
        self.ring_buffer.push_wrist(
            person_id, result["wrist_angle_left"], result["wrist_angle_right"]
        )

        # 2. 统一提取两手肢体的最新方差特征
        variances = {
            "fa_l": self.ring_buffer.get_forearm_variance(
                person_id, "left", jump_threshold=self.JUMP_THRESHOLD
            ),
            "fa_r": self.ring_buffer.get_forearm_variance(
                person_id, "right", jump_threshold=self.JUMP_THRESHOLD
            ),
            "wr_l": self.ring_buffer.get_wrist_variance(
                person_id, "left", jump_threshold=self.JUMP_THRESHOLD
            ),
            "wr_r": self.ring_buffer.get_wrist_variance(
                person_id, "right", jump_threshold=self.JUMP_THRESHOLD
            ),
        }
        thresholds = {
            "T_FOREARM": self.T_FOREARM,
            "T_WRIST": self.T_WRIST,
            "SHOULDER_OFFSET": self.SHOULDER_OFFSET,
        }

        final_gesture, debug_info = judge_all_temporal_gestures(
            raw_gesture=result["gesture"],  # 这里传入的是刚刚初筛出来的 maybe_xxx 状态
            variances=variances,
            landmarks=result["landmarks"],
            thresholds=thresholds,
        )

        # 4. 用时序判定的最终手势，覆盖掉单帧初筛的不稳定状态
        result["gesture"] = final_gesture

        # 挂载调试方差
        result.update(debug_info)
        return result

    def analyze(self, person_crop: np.ndarray, hands_data: dict = None) -> dict:
        """
        对单个人框裁剪图做姿态+静态手势分析（便捷方法，内部跑 Pose 模型）。

        如需并行 Pose + Hands，请使用 process_pose() + process_hands() +
        analyze_from_landmarks() 替代。
        """
        return self.analyze_from_landmarks(
            self.process_pose(person_crop), hands_data=hands_data, with_temporal=False
        )

    def analyze_with_temporal(
        self, person_id, person_crop: np.ndarray, hands_data: dict = None
    ) -> dict:
        """
        包含时序挥手检测的完整分析（便捷方法，内部跑 Pose 模型）。

        如需并行 Pose + Hands，请使用 process_pose() + process_hands() +
        analyze_from_landmarks(with_temporal=True) 替代。
        """
        return self.analyze_from_landmarks(
            self.process_pose(person_crop),
            person_id=person_id,
            hands_data=hands_data,
            with_temporal=True,
        )

    @staticmethod
    def _unknown_result():
        return {
            "landmarks": [],
            "posture": "unknown",
            "gesture": "unknown",
            "elbow_angle_left": None,
            "elbow_angle_right": None,
            "wrist_angle_left": None,
            "wrist_angle_right": None,
        }
