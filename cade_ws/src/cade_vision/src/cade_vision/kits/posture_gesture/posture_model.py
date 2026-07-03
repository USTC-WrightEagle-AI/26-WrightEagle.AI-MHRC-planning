"""LightGBM-based posture classifier for MediaPipe Pose landmarks."""

import json
import math
import os
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:
    lgb = None
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = exc


MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "lightgbm_pose_nan_pose_only_with_p03_far_phone_lying"
)
DEFAULT_MODEL_PATH = MODEL_DIR / "model.txt"
DEFAULT_FEATURES_PATH = MODEL_DIR / "feature_columns.json"

LANDMARKS = (
    ("left_hip", 23),
    ("right_hip", 24),
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_knee", 25),
    ("right_knee", 26),
    ("left_ankle", 27),
    ("right_ankle", 28),
    ("left_heel", 29),
    ("right_heel", 30),
    ("left_foot_index", 31),
    ("right_foot_index", 32),
)


class LightGBMPostureClassifier:
    """Run the trained sitting/lying/standing LightGBM posture model."""

    def __init__(self, model_path=None, features_path=None):
        self.model_path = Path(
            model_path
            or os.environ.get("CADE_POSTURE_LGBM_MODEL")
            or DEFAULT_MODEL_PATH
        )
        self.features_path = Path(
            features_path
            or os.environ.get("CADE_POSTURE_LGBM_FEATURES")
            or DEFAULT_FEATURES_PATH
        )
        self.model = None
        self.feature_columns = []
        self.id_to_label = {}
        self.visibility_threshold = 0.5
        self.bbox_feature_mode = None
        self.available = False
        self.load_error = None

        self._load()

    def _load(self):
        if not LIGHTGBM_AVAILABLE:
            self.load_error = f"lightgbm import failed: {LIGHTGBM_IMPORT_ERROR}"
            return

        if not self.model_path.exists():
            self.load_error = f"model file not found: {self.model_path}"
            return
        if not self.features_path.exists():
            self.load_error = f"feature config not found: {self.features_path}"
            return

        try:
            with self.features_path.open() as f:
                payload = json.load(f)
            self.feature_columns = list(payload["feature_columns"])
            label_ids = payload.get("label_ids", {})
            self.id_to_label = {
                int(cls_id): label for label, cls_id in label_ids.items()
            }
            self.visibility_threshold = float(
                payload.get("nan_visibility_threshold", self.visibility_threshold)
            )
            self.bbox_feature_mode = payload.get("bbox_feature_mode")
            self.model = lgb.Booster(model_file=str(self.model_path))
            self.available = True
        except Exception as exc:
            self.load_error = str(exc)
            self.model = None
            self.available = False

    @staticmethod
    def _visibility(landmark):
        if landmark is None or len(landmark) < 3:
            return 0.0
        return float(landmark[2])

    @staticmethod
    def _value_or_nan(values, index):
        if values is None or index >= len(values):
            return math.nan
        value = values[index]
        if value is None:
            return math.nan
        return float(value)

    def _feature_map(self, pose_data):
        landmarks = pose_data.get("landmarks") or []
        landmark_z = pose_data.get("landmark_z") or []
        features = {}

        for name, index in LANDMARKS:
            visible = (
                index < len(landmarks)
                and self._visibility(landmarks[index]) >= self.visibility_threshold
            )
            if not visible:
                features[f"{name}_x"] = math.nan
                features[f"{name}_y"] = math.nan
                features[f"{name}_z"] = math.nan
                continue

            landmark = landmarks[index]
            features[f"{name}_x"] = self._value_or_nan(landmark, 0)
            features[f"{name}_y"] = self._value_or_nan(landmark, 1)
            features[f"{name}_z"] = self._value_or_nan(landmark_z, index)

        if "bbox_w" in self.feature_columns:
            features["bbox_w"] = float(pose_data.get("bbox_w", math.nan))
        if "bbox_h" in self.feature_columns:
            features["bbox_h"] = float(pose_data.get("bbox_h", math.nan))
        return features

    def predict(self, pose_data):
        if pose_data is None or not self.available:
            return {
                "posture": "unknown",
                "confidence": 0.0,
                "probabilities": {},
            }

        feature_map = self._feature_map(pose_data)
        row = [feature_map.get(column, math.nan) for column in self.feature_columns]
        x = np.asarray([row], dtype=np.float32)

        try:
            probabilities = self.model.predict(x)
        except Exception as exc:
            self.load_error = str(exc)
            return {
                "posture": "unknown",
                "confidence": 0.0,
                "probabilities": {},
            }

        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 2:
            probabilities = probabilities[0]
        if probabilities.size == 0 or np.all(np.isnan(probabilities)):
            return {
                "posture": "unknown",
                "confidence": 0.0,
                "probabilities": {},
            }

        class_id = int(np.nanargmax(probabilities))
        posture = self.id_to_label.get(class_id, "unknown")
        confidence = float(probabilities[class_id])
        return {
            "posture": posture,
            "confidence": confidence,
            "probabilities": {
                self.id_to_label.get(i, str(i)): float(prob)
                for i, prob in enumerate(probabilities)
            },
        }
