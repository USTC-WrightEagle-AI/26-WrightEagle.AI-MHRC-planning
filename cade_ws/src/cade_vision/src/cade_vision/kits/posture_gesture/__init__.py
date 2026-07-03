"""
Posture & Gesture Analyzer - 基于 MediaPipe Pose + LightGBM 的姿态和手势识别

职责：
- 接收人框裁剪图，输出 posture（standing/sitting/lying/unknown）
  和 gesture（waving/raising_left_arm/raising_right_arm/pointing_left/pointing_right/none/unknown）
- 维护每人独立的 LightGBM gesture 投票/窗口状态，支持动态 waving 识别
"""

import os
import threading

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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

from .gesture_model import LightGBMGestureClassifier
from .posture_model import LightGBMPostureClassifier


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
        self._pose_lock = threading.Lock()
        self._hands_lock = threading.Lock()
        self.posture_classifier = LightGBMPostureClassifier()
        if not self.posture_classifier.available:
            print(
                "LightGBM posture classifier unavailable: "
                f"{self.posture_classifier.load_error}"
            )
        self.gesture_classifier = LightGBMGestureClassifier()
        if not self.gesture_classifier.available:
            print(
                "LightGBM gesture classifier unavailable: "
                f"{self.gesture_classifier.load_error}"
            )

        # 挥手检测阈值（可调参数）
        self.T_FOREARM = T_FOREARM
        self.T_WRIST = T_WRIST
        self.SHOULDER_OFFSET = SHOULDER_OFFSET
        self.JUMP_THRESHOLD = JUMP_THRESHOLD

        if self.available:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                enable_segmentation=False,
                smooth_landmarks=False,
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
            {
                "landmarks": [(x,y,visibility), ... 33 keypoints],
                "landmark_z": [z, ... 33 keypoints],
                "img_h": int,
                "img_w": int,
            }
            或 None（如果 Pose 不可用或未检测到人）。
        """
        if not self.available or self.pose is None or person_crop.size == 0:
            return None
        h, w = person_crop.shape[:2]
        if h < 30 or w < 30:
            return None
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        with self._pose_lock:
            results = self.pose.process(rgb)
        if results.pose_landmarks is None:
            return None
        landmarks = [
            (lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark
        ]
        landmark_z = [lm.z for lm in results.pose_landmarks.landmark]
        return {
            "landmarks": landmarks,
            "landmark_z": landmark_z,
            "img_h": h,
            "img_w": w,
        }

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
        with self._hands_lock:
            hands_results = self.hands.process(rgb)

        if (hands_results.multi_hand_landmarks is not None) and (
            hands_results.multi_handedness is not None
        ):
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
        else:
            return None

        return result

    def analyze_from_landmarks(
        self,
        pose_data: dict,
        person_id=None,
        hands_data: dict = None,
        with_temporal: bool = True,
        keypoints_3d: dict = None,
        timestamp: float = None,
    ) -> dict:
        """
        从预计算的 landmarks 做分类 + 角度计算（不再跑模型推理）。

        Args:
            pose_data: process_pose() 的输出，包含 landmarks、landmark_z、img_h/img_w。
            person_id:  时序挥手检测用的人物 ID（with_temporal=True 时必传）
            hands_data: 保留旧调用签名；LightGBM 手势模型不使用 MediaPipe Hands。
            with_temporal: 是否跑时序挥手检测
            keypoints_3d: 保留旧调用签名；LightGBM 姿态模型不使用该参数。
            timestamp: 当前帧时间戳，用于 waving 窗口特征。

        Returns:
            同 analyze() 或 analyze_with_temporal()
        """
        if pose_data is None:
            result = self._unknown_result()
            gesture_result = self.gesture_classifier.predict(
                None,
                person_id=person_id,
                timestamp=timestamp,
                with_temporal=with_temporal,
            )
            result.update(gesture_result)
            return result

        landmarks = pose_data["landmarks"]

        posture_result = self.posture_classifier.predict(pose_data)
        posture = posture_result.get("posture", "unknown")
        gesture_result = self.gesture_classifier.predict(
            pose_data,
            person_id=person_id,
            timestamp=timestamp,
            with_temporal=with_temporal,
        )

        result = {
            "landmarks": landmarks,
            "posture": posture,
            "posture_confidence": posture_result.get("confidence", 0.0),
            "posture_probabilities": posture_result.get("probabilities", {}),
            "gesture": gesture_result.get("gesture", "unknown"),
            "elbow_angle_left": None,
            "elbow_angle_right": None,
            "wrist_angle_left": None,
            "wrist_angle_right": None,
        }
        result.update(gesture_result)
        return result

    def clear_temporal(self):
        """清空每个 track 的手势投票和 waving 窗口状态。"""
        self.gesture_classifier.clear()

    def analyze(self, person_crop: np.ndarray, hands_data: dict = None) -> dict:
        """
        对单个人框裁剪图做姿态+静态手势分析（便捷方法，内部跑 Pose 模型）。

        如需并行 Pose，请使用 process_pose() + analyze_from_landmarks() 替代。
        """
        return self.analyze_from_landmarks(
            self.process_pose(person_crop), hands_data=hands_data, with_temporal=False
        )

    def analyze_with_temporal(
        self, person_id, person_crop: np.ndarray, hands_data: dict = None
    ) -> dict:
        """
        包含时序挥手检测的完整分析（便捷方法，内部跑 Pose 模型）。

        如需并行 Pose，请使用 process_pose() +
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
            "posture_confidence": 0.0,
            "posture_probabilities": {},
            "gesture": "unknown",
            "elbow_angle_left": None,
            "elbow_angle_right": None,
            "wrist_angle_left": None,
            "wrist_angle_right": None,
        }
