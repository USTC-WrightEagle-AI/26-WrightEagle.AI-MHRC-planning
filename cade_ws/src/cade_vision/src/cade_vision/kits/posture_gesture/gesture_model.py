"""LightGBM-based gesture classifiers for MediaPipe Pose landmarks."""

import json
import math
import os
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:
    lgb = None
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = exc


STATIC_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "lightgbm_gesture_static_yolo_pose"
)
WAVING_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "lightgbm_gesture_waving_motion_v2"
)
DEFAULT_STATIC_MODEL_PATH = STATIC_MODEL_DIR / "model.txt"
DEFAULT_STATIC_FEATURES_PATH = STATIC_MODEL_DIR / "feature_columns.json"
DEFAULT_WAVING_MODEL_PATH = WAVING_MODEL_DIR / "model.txt"
DEFAULT_WAVING_FEATURES_PATH = WAVING_MODEL_DIR / "window_feature_columns.json"

DEFAULT_POSE_LANDMARKS = (
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_index", 19),
    ("right_index", 20),
    ("left_pinky", 17),
    ("right_pinky", 18),
    ("left_thumb", 21),
    ("right_thumb", 22),
)
WAVING_LABELS = ("no_waving", "waving")
WINDOW_STATS = (
    "mean",
    "std",
    "min",
    "max",
    "range",
    "delta",
    "mean_abs_diff",
    "std_diff",
    "max_abs_diff",
)
SIDES = ("left", "right")
HAND_POINTS = ("wrist", "index", "pinky", "thumb")
OUTPUT_LABEL_MAP = {
    "raising_left": "raising_left_arm",
    "raising_right": "raising_right_arm",
}


def _is_finite(value):
    return isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))


def _mean(values):
    return sum(values) / len(values) if values else math.nan


def _std(values):
    if not values:
        return math.nan
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _finite_values(values):
    return [float(value) for value in values if _is_finite(value)]


def _sequence_diffs(values):
    diffs = []
    previous = None
    for value in values:
        if not _is_finite(value):
            previous = None
            continue
        value = float(value)
        if previous is not None:
            diffs.append(value - previous)
        previous = value
    return diffs


def _sequence_stats(values):
    finite = _finite_values(values)
    if not finite:
        return {stat: math.nan for stat in WINDOW_STATS}

    diffs = _sequence_diffs(values)
    abs_diffs = [abs(value) for value in diffs]
    return {
        "mean": _mean(finite),
        "std": _std(finite),
        "min": min(finite),
        "max": max(finite),
        "range": max(finite) - min(finite),
        "delta": finite[-1] - finite[0],
        "mean_abs_diff": _mean(abs_diffs) if abs_diffs else 0.0,
        "std_diff": _std(diffs) if diffs else 0.0,
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
    }


def _direction_changes(values, epsilon):
    last_sign = None
    changes = 0
    saw_step = False
    for diff in _sequence_diffs(values):
        if abs(diff) < epsilon:
            continue
        sign = 1 if diff > 0 else -1
        if last_sign is not None and sign != last_sign:
            changes += 1
        last_sign = sign
        saw_step = True
    return float(changes) if saw_step else math.nan


def _max_finite(values):
    finite = _finite_values(values)
    return max(finite) if finite else math.nan


def _safe_ratio(numerator, denominator):
    if not _is_finite(numerator) or not _is_finite(denominator):
        return math.nan
    if abs(float(denominator)) < 1e-6:
        return 100.0 if float(numerator) > 0.0 else 0.0
    return min(100.0, float(numerator) / float(denominator))


def _longest_valid_run(values):
    longest = 0
    current = 0
    for value in values:
        if _is_finite(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest)


def _finite_value_time_pairs(values, times):
    return [
        (float(value), float(time_sec))
        for value, time_sec in zip(values, times)
        if _is_finite(value) and _is_finite(time_sec)
    ]


def _timed_velocities(values, times):
    velocities = []
    previous_value = None
    previous_time = None
    for value, time_sec in zip(values, times):
        if not _is_finite(value) or not _is_finite(time_sec):
            previous_value = None
            previous_time = None
            continue
        value = float(value)
        time_sec = float(time_sec)
        if previous_value is not None and previous_time is not None:
            dt = time_sec - previous_time
            if dt > 1e-6:
                velocities.append((value - previous_value) / dt)
        previous_value = value
        previous_time = time_sec
    return velocities


def _get_axis(feature_name):
    return feature_name.rsplit("_", 1)[1]


def _normalize_frame_features(raw_features, frame_feature_columns):
    left = {
        axis: raw_features.get("left_shoulder_%s" % axis, math.nan)
        for axis in "xyz"
    }
    right = {
        axis: raw_features.get("right_shoulder_%s" % axis, math.nan)
        for axis in "xyz"
    }

    if (
        _is_finite(left["x"])
        and _is_finite(right["x"])
        and _is_finite(left["y"])
        and _is_finite(right["y"])
    ):
        shoulder_width = math.hypot(
            left["x"] - right["x"], left["y"] - right["y"]
        )
    else:
        shoulder_width = math.nan

    center = {}
    for axis in "xyz":
        if _is_finite(left[axis]) and _is_finite(right[axis]):
            center[axis] = (left[axis] + right[axis]) * 0.5
        else:
            center[axis] = math.nan

    normalized = {}
    valid_scale = _is_finite(shoulder_width) and shoulder_width > 1e-6
    for column in frame_feature_columns:
        axis = _get_axis(column)
        value = raw_features.get(column, math.nan)
        if valid_scale and _is_finite(value) and _is_finite(center[axis]):
            normalized[column] = (value - center[axis]) / shoulder_width
        else:
            normalized[column] = math.nan
    return normalized


def _hand_coordinate_values(window_rows, side, axis):
    values = []
    for row in window_rows:
        features = row["norm_features"]
        parts = [
            features.get("%s_%s_%s" % (side, point, axis), math.nan)
            for point in HAND_POINTS
        ]
        finite = _finite_values(parts)
        values.append(_mean(finite) if finite else math.nan)
    return values


def _hand_points(window_rows, side):
    x_values = _hand_coordinate_values(window_rows, side, "x")
    y_values = _hand_coordinate_values(window_rows, side, "y")
    points = []
    for x_value, y_value in zip(x_values, y_values):
        if _is_finite(x_value) and _is_finite(y_value):
            points.append((float(x_value), float(y_value)))
        else:
            points.append(None)
    return x_values, y_values, points


def _path_stats(points, times):
    path_length = 0.0
    speeds = []
    previous_point = None
    previous_time = None
    first_point = None
    last_point = None
    first_time = None
    last_time = None

    for point, time_sec in zip(points, times):
        if point is None or not _is_finite(time_sec):
            previous_point = None
            previous_time = None
            continue
        x_value, y_value = point
        time_sec = float(time_sec)
        if first_point is None:
            first_point = point
            first_time = time_sec
        last_point = point
        last_time = time_sec

        if previous_point is not None and previous_time is not None:
            dt = time_sec - previous_time
            distance = math.hypot(
                x_value - previous_point[0], y_value - previous_point[1]
            )
            path_length += distance
            if dt > 1e-6:
                speeds.append(distance / dt)
        previous_point = point
        previous_time = time_sec

    if first_point is None or last_point is None:
        displacement = math.nan
        sample_span = 0.0
    else:
        displacement = math.hypot(
            last_point[0] - first_point[0], last_point[1] - first_point[1]
        )
        sample_span = max(0.0, float(last_time - first_time))

    return {
        "path_length": path_length if first_point is not None else math.nan,
        "displacement": displacement,
        "path_to_displacement_ratio": _safe_ratio(path_length, displacement),
        "mean_speed": _mean(speeds) if speeds else 0.0,
        "max_speed": max(speeds) if speeds else 0.0,
        "speed_std": _std(speeds) if speeds else 0.0,
        "sample_span_sec": sample_span,
    }


def _unwrap_finite_angles(values):
    finite = _finite_values(values)
    if not finite:
        return []
    return [float(value) for value in np.unwrap(np.asarray(finite))]


def _forearm_angles(window_rows, side):
    values = []
    for row in window_rows:
        features = row["norm_features"]
        elbow_x = features.get("%s_elbow_x" % side, math.nan)
        elbow_y = features.get("%s_elbow_y" % side, math.nan)
        wrist_x = features.get("%s_wrist_x" % side, math.nan)
        wrist_y = features.get("%s_wrist_y" % side, math.nan)
        if all(_is_finite(value) for value in (elbow_x, elbow_y, wrist_x, wrist_y)):
            values.append(math.atan2(wrist_y - elbow_y, wrist_x - elbow_x))
        else:
            values.append(math.nan)
    return values


def _timed_angle_summary(values, times, epsilon):
    pairs = _finite_value_time_pairs(values, times)
    if not pairs:
        return {
            "range": math.nan,
            "std": math.nan,
            "mean_abs_velocity": math.nan,
            "direction_changes": math.nan,
        }
    unwrapped = [
        float(value)
        for value in np.unwrap(np.asarray([value for value, _ in pairs]))
    ]
    pair_times = [time_sec for _, time_sec in pairs]
    velocities = _timed_velocities(unwrapped, pair_times)
    return {
        "range": max(unwrapped) - min(unwrapped),
        "std": _std(unwrapped),
        "mean_abs_velocity": (
            _mean([abs(value) for value in velocities]) if velocities else 0.0
        ),
        "direction_changes": _direction_changes(unwrapped, epsilon),
    }


def _make_waving_window_features(
    window_rows, frame_feature_columns, direction_epsilon, angle_direction_epsilon
):
    _ = frame_feature_columns
    times = [row["time_sec"] for row in window_rows]
    features = {}

    for side in SIDES:
        hand_x, hand_y, hand_xy = _hand_points(window_rows, side)
        hand_x_stats = _sequence_stats(hand_x)
        hand_y_stats = _sequence_stats(hand_y)
        path = _path_stats(hand_xy, times)
        x_velocities = _timed_velocities(hand_x, times)
        y_velocities = _timed_velocities(hand_y, times)
        valid_frames = sum(1 for point in hand_xy if point is not None)
        valid_ratio = valid_frames / len(window_rows) if window_rows else 0.0
        angle_stats = _timed_angle_summary(
            _forearm_angles(window_rows, side),
            times,
            angle_direction_epsilon,
        )

        features["%s_hand_x_range" % side] = hand_x_stats["range"]
        features["%s_hand_y_range" % side] = hand_y_stats["range"]
        features["%s_hand_x_std" % side] = hand_x_stats["std"]
        features["%s_hand_y_std" % side] = hand_y_stats["std"]
        features["%s_hand_x_delta" % side] = hand_x_stats["delta"]
        features["%s_hand_y_delta" % side] = hand_y_stats["delta"]
        features["%s_hand_path_length" % side] = path["path_length"]
        features["%s_hand_displacement" % side] = path["displacement"]
        features["%s_hand_path_to_displacement_ratio" % side] = path[
            "path_to_displacement_ratio"
        ]
        features["%s_hand_mean_speed" % side] = path["mean_speed"]
        features["%s_hand_max_speed" % side] = path["max_speed"]
        features["%s_hand_speed_std" % side] = path["speed_std"]
        features["%s_hand_x_mean_abs_velocity" % side] = (
            _mean([abs(value) for value in x_velocities])
            if x_velocities
            else 0.0
        )
        features["%s_hand_y_mean_abs_velocity" % side] = (
            _mean([abs(value) for value in y_velocities])
            if y_velocities
            else 0.0
        )
        features["%s_hand_x_max_abs_velocity" % side] = (
            max([abs(value) for value in x_velocities])
            if x_velocities
            else 0.0
        )
        features["%s_hand_y_max_abs_velocity" % side] = (
            max([abs(value) for value in y_velocities])
            if y_velocities
            else 0.0
        )
        features["%s_hand_x_direction_changes" % side] = _direction_changes(
            hand_x,
            direction_epsilon,
        )
        features["%s_hand_y_direction_changes" % side] = _direction_changes(
            hand_y,
            direction_epsilon,
        )
        features["%s_hand_valid_frame_ratio" % side] = valid_ratio
        features["%s_hand_longest_valid_run" % side] = _longest_valid_run(hand_x)
        features["%s_hand_sample_span_sec" % side] = path["sample_span_sec"]
        features["%s_forearm_angle_range" % side] = angle_stats["range"]
        features["%s_forearm_angle_std" % side] = angle_stats["std"]
        features["%s_forearm_angle_mean_abs_velocity" % side] = angle_stats[
            "mean_abs_velocity"
        ]
        features["%s_forearm_angle_direction_changes" % side] = angle_stats[
            "direction_changes"
        ]

    features["max_hand_x_range"] = _max_finite(
        [features["%s_hand_x_range" % side] for side in SIDES]
    )
    features["max_hand_y_range"] = _max_finite(
        [features["%s_hand_y_range" % side] for side in SIDES]
    )
    features["max_hand_path_length"] = _max_finite(
        [features["%s_hand_path_length" % side] for side in SIDES]
    )
    features["max_hand_displacement"] = _max_finite(
        [features["%s_hand_displacement" % side] for side in SIDES]
    )
    features["max_hand_path_to_displacement_ratio"] = _max_finite(
        [features["%s_hand_path_to_displacement_ratio" % side] for side in SIDES]
    )
    features["max_hand_mean_speed"] = _max_finite(
        [features["%s_hand_mean_speed" % side] for side in SIDES]
    )
    features["max_hand_max_speed"] = _max_finite(
        [features["%s_hand_max_speed" % side] for side in SIDES]
    )
    features["max_hand_x_mean_abs_velocity"] = _max_finite(
        [features["%s_hand_x_mean_abs_velocity" % side] for side in SIDES]
    )
    features["max_hand_y_mean_abs_velocity"] = _max_finite(
        [features["%s_hand_y_mean_abs_velocity" % side] for side in SIDES]
    )
    features["max_hand_x_direction_changes"] = _max_finite(
        [features["%s_hand_x_direction_changes" % side] for side in SIDES]
    )
    features["max_hand_y_direction_changes"] = _max_finite(
        [features["%s_hand_y_direction_changes" % side] for side in SIDES]
    )
    features["max_hand_valid_frame_ratio"] = _max_finite(
        [features["%s_hand_valid_frame_ratio" % side] for side in SIDES]
    )
    features["max_forearm_angle_range"] = _max_finite(
        [features["%s_forearm_angle_range" % side] for side in SIDES]
    )
    features["max_forearm_angle_mean_abs_velocity"] = _max_finite(
        [features["%s_forearm_angle_mean_abs_velocity" % side] for side in SIDES]
    )
    features["max_forearm_angle_direction_changes"] = _max_finite(
        [features["%s_forearm_angle_direction_changes" % side] for side in SIDES]
    )
    return features


class PredictionRingBuffer:
    def __init__(self, capacity, min_votes, min_majority_ratio, missing_reset_frames):
        self.capacity = int(capacity)
        self.min_votes = int(min_votes)
        self.min_majority_ratio = float(min_majority_ratio)
        self.missing_reset_frames = int(missing_reset_frames)
        self.items = deque(maxlen=self.capacity)
        self.missing_frames = 0

    def push(self, label, confidence, probabilities):
        self.items.append(
            {
                "label": label,
                "confidence": float(confidence),
                "probabilities": probabilities,
            }
        )
        self.missing_frames = 0

    def mark_missing(self):
        self.missing_frames += 1
        if self.missing_frames >= self.missing_reset_frames:
            self.clear()
            return True
        return False

    def clear(self):
        self.items.clear()
        self.missing_frames = 0

    def vote(self, labels):
        if len(self.items) < self.min_votes:
            return {
                "label": "unknown",
                "stable": False,
                "count": 0,
                "total": len(self.items),
                "ratio": 0.0,
                "confidence": 0.0,
                "counts": {name: 0 for name in labels},
            }

        counts = Counter(item["label"] for item in self.items)
        label, count = counts.most_common(1)[0]
        total = len(self.items)
        ratio = count / total if total else 0.0
        confidence_values = [
            item["confidence"] for item in self.items if item["label"] == label
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        stable = ratio >= self.min_majority_ratio
        return {
            "label": label if stable else "unknown",
            "stable": stable,
            "count": count,
            "total": total,
            "ratio": ratio,
            "confidence": confidence,
            "counts": {name: counts.get(name, 0) for name in labels},
        }


class WavingWindowState:
    def __init__(
        self,
        spec,
        threshold,
        sample_interval,
        direction_epsilon,
        angle_direction_epsilon,
    ):
        self.frame_feature_columns = spec["frame_feature_columns"]
        self.window_feature_columns = spec["feature_columns"]
        self.window_sec = spec["window_sec"]
        self.min_window_frames = spec["min_window_frames"]
        self.threshold = float(threshold)
        self.sample_interval = float(sample_interval)
        self.direction_epsilon = float(direction_epsilon)
        self.angle_direction_epsilon = float(angle_direction_epsilon)
        self.history = deque()
        self.last_sample_time = None

    def clear(self):
        self.history.clear()
        self.last_sample_time = None

    def prune(self, timestamp):
        while self.history and timestamp - self.history[0]["time_sec"] > self.window_sec:
            self.history.popleft()

    def add_frame(self, timestamp, feature_map):
        self.prune(timestamp)
        if (
            self.last_sample_time is not None
            and timestamp - self.last_sample_time < self.sample_interval
        ):
            return False

        raw_features = {
            column: float(feature_map.get(column, math.nan))
            for column in self.frame_feature_columns
        }
        self.history.append(
            {
                "time_sec": timestamp,
                "raw_features": raw_features,
                "norm_features": _normalize_frame_features(
                    raw_features,
                    self.frame_feature_columns,
                ),
            }
        )
        self.last_sample_time = timestamp
        self.prune(timestamp)
        return True

    def predict(self, booster, timestamp):
        self.prune(timestamp)
        if len(self.history) < self.min_window_frames:
            return {
                "status": "warming_up",
                "label": "no_waving",
                "confidence": 0.0,
                "probabilities": np.asarray([], dtype=np.float32),
                "history_frames": len(self.history),
                "sampled": False,
            }

        feature_map = _make_waving_window_features(
            list(self.history),
            self.frame_feature_columns,
            self.direction_epsilon,
            self.angle_direction_epsilon,
        )
        x = np.asarray(
            [
                [
                    float(feature_map.get(column, math.nan))
                    for column in self.window_feature_columns
                ]
            ],
            dtype=np.float32,
        )
        positive_probability = float(np.asarray(booster.predict(x)).reshape(-1)[0])
        probabilities = np.asarray(
            [1.0 - positive_probability, positive_probability],
            dtype=np.float32,
        )
        label = "waving" if positive_probability >= self.threshold else "no_waving"
        confidence = (
            positive_probability if label == "waving" else 1.0 - positive_probability
        )
        return {
            "status": "ok",
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "history_frames": len(self.history),
            "sampled": True,
        }


class GestureTrackState:
    def __init__(self, classifier):
        self.static_ring = PredictionRingBuffer(
            classifier.vote_window,
            classifier.min_votes,
            classifier.min_majority_ratio,
            classifier.missing_reset_frames,
        )
        self.waving_state = WavingWindowState(
            classifier.waving_spec,
            threshold=classifier.waving_threshold,
            sample_interval=classifier.waving_sample_interval,
            direction_epsilon=classifier.waving_direction_epsilon,
            angle_direction_epsilon=classifier.waving_angle_direction_epsilon,
        )
        self.waving_ring = PredictionRingBuffer(
            classifier.waving_vote_window,
            classifier.waving_min_votes,
            classifier.waving_min_majority_ratio,
            classifier.missing_reset_frames,
        )

    def clear(self):
        self.static_ring.clear()
        self.waving_ring.clear()
        self.waving_state.clear()

    def mark_missing(self):
        static_reset = self.static_ring.mark_missing()
        waving_reset = self.waving_ring.mark_missing()
        if static_reset or waving_reset:
            self.waving_state.clear()


class LightGBMGestureClassifier:
    """Run static raising/pointing and dynamic waving LightGBM gesture models."""

    def __init__(
        self,
        static_model_path=None,
        static_features_path=None,
        waving_model_path=None,
        waving_features_path=None,
        vote_window=15,
        min_votes=5,
        min_majority_ratio=0.55,
        missing_reset_frames=8,
        waving_threshold=0.65,
        waving_sample_interval=0.10,
        waving_direction_epsilon=0.02,
        waving_angle_direction_epsilon=0.15,
        waving_vote_window=9,
        waving_min_votes=3,
        waving_min_majority_ratio=0.55,
    ):
        self.static_model_path = Path(
            static_model_path
            or os.environ.get("CADE_GESTURE_STATIC_LGBM_MODEL")
            or DEFAULT_STATIC_MODEL_PATH
        )
        self.static_features_path = Path(
            static_features_path
            or os.environ.get("CADE_GESTURE_STATIC_LGBM_FEATURES")
            or DEFAULT_STATIC_FEATURES_PATH
        )
        self.waving_model_path = Path(
            waving_model_path
            or os.environ.get("CADE_GESTURE_WAVING_LGBM_MODEL")
            or DEFAULT_WAVING_MODEL_PATH
        )
        self.waving_features_path = Path(
            waving_features_path
            or os.environ.get("CADE_GESTURE_WAVING_FEATURES")
            or DEFAULT_WAVING_FEATURES_PATH
        )

        self.vote_window = vote_window
        self.min_votes = min_votes
        self.min_majority_ratio = min_majority_ratio
        self.missing_reset_frames = missing_reset_frames
        self.waving_threshold = waving_threshold
        self.waving_sample_interval = waving_sample_interval
        self.waving_direction_epsilon = waving_direction_epsilon
        self.waving_angle_direction_epsilon = waving_angle_direction_epsilon
        self.waving_vote_window = waving_vote_window
        self.waving_min_votes = waving_min_votes
        self.waving_min_majority_ratio = waving_min_majority_ratio

        self.static_model = None
        self.waving_model = None
        self.feature_columns = []
        self.labels = []
        self.pose_landmarks = list(DEFAULT_POSE_LANDMARKS)
        self.visibility_threshold = 0.5
        self.box_margin = 0.12
        self.yolo_conf = 0.25
        self.imgsz = 320
        self.waving_spec = None
        self.available = False
        self.load_error = None
        self._states = {}

        self._load()

    def _load(self):
        if not LIGHTGBM_AVAILABLE:
            self.load_error = "lightgbm import failed: %s" % LIGHTGBM_IMPORT_ERROR
            return
        if not self.static_model_path.exists():
            self.load_error = "static gesture model not found: %s" % (
                self.static_model_path,
            )
            return
        if not self.static_features_path.exists():
            self.load_error = "static gesture feature config not found: %s" % (
                self.static_features_path,
            )
            return
        if not self.waving_model_path.exists():
            self.load_error = "waving gesture model not found: %s" % (
                self.waving_model_path,
            )
            return
        if not self.waving_features_path.exists():
            self.load_error = "waving gesture feature config not found: %s" % (
                self.waving_features_path,
            )
            return

        try:
            with self.static_features_path.open() as handle:
                static_payload = json.load(handle)
            self.feature_columns = list(static_payload["feature_columns"])
            self.labels = self._labels_from_payload(static_payload)
            preprocess = static_payload.get("preprocess", {})
            self.visibility_threshold = float(
                preprocess.get("nan_visibility_threshold", self.visibility_threshold)
            )
            self.box_margin = float(preprocess.get("box_margin", self.box_margin))
            self.yolo_conf = float(preprocess.get("yolo_conf", self.yolo_conf))
            self.imgsz = int(preprocess.get("imgsz", self.imgsz))
            self.pose_landmarks = self._pose_landmarks_from_payload(static_payload)

            self.waving_spec = self._load_waving_spec(
                self.waving_features_path,
                self.feature_columns,
            )
            self.static_model = lgb.Booster(model_file=str(self.static_model_path))
            self.waving_model = lgb.Booster(model_file=str(self.waving_model_path))
            self.available = True
        except Exception as exc:
            self.load_error = str(exc)
            self.static_model = None
            self.waving_model = None
            self.available = False

    @staticmethod
    def _labels_from_payload(payload):
        if "labels" in payload:
            return list(payload["labels"])
        label_ids = {str(label): int(index) for label, index in payload["label_ids"].items()}
        labels = [None] * len(label_ids)
        for label, index in label_ids.items():
            labels[index] = label
        if any(label is None for label in labels):
            raise RuntimeError("non-contiguous gesture label ids")
        return labels

    @staticmethod
    def _pose_landmarks_from_payload(payload):
        landmarks = []
        for item in payload.get("pose_landmarks", []):
            landmarks.append((str(item["name"]), int(item["mediapipe_index"])))
        return landmarks or list(DEFAULT_POSE_LANDMARKS)

    @staticmethod
    def _load_waving_spec(path, expected_frame_feature_columns):
        with path.open() as handle:
            payload = json.load(handle)
        labels = list(payload["labels"])
        if labels != list(WAVING_LABELS):
            raise RuntimeError("unexpected waving labels in %s: %s" % (path, labels))
        feature_version = payload.get("feature_version")
        if feature_version != "motion_v2":
            raise RuntimeError(
                "unsupported waving feature_version in %s: %r" % (
                    path,
                    feature_version,
                )
            )
        frame_feature_columns = list(payload["frame_feature_columns"])
        if frame_feature_columns != expected_frame_feature_columns:
            raise RuntimeError("waving frame features do not match static feature order")
        return {
            "labels": labels,
            "frame_feature_columns": frame_feature_columns,
            "feature_columns": list(payload["feature_columns"]),
            "window_sec": float(payload.get("window_sec", 1.2)),
            "min_window_frames": int(payload.get("min_window_frames", 8)),
        }

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

    @staticmethod
    def _probability_dict(labels, probabilities):
        return {
            label: float(probability)
            for label, probability in zip(labels, probabilities)
        }

    @staticmethod
    def _output_label(label):
        return OUTPUT_LABEL_MAP.get(label, label)

    def _track_state(self, person_id):
        if person_id not in self._states:
            self._states[person_id] = GestureTrackState(self)
        return self._states[person_id]

    def clear(self):
        for state in self._states.values():
            state.clear()
        self._states.clear()

    def mark_missing(self, person_id):
        if person_id is None:
            return
        state = self._states.get(person_id)
        if state is not None:
            state.mark_missing()

    def _feature_map(self, pose_data):
        landmarks = pose_data.get("landmarks") or []
        landmark_z = pose_data.get("landmark_z") or []
        features = {}
        missing = []
        visibility_values = []

        for name, index in self.pose_landmarks:
            visible = (
                index < len(landmarks)
                and self._visibility(landmarks[index]) >= self.visibility_threshold
            )
            visibility = self._visibility(landmarks[index]) if index < len(landmarks) else 0.0
            visibility_values.append(visibility)

            if not visible:
                features["%s_x" % name] = math.nan
                features["%s_y" % name] = math.nan
                features["%s_z" % name] = math.nan
                missing.append(name)
                continue

            landmark = landmarks[index]
            features["%s_x" % name] = self._value_or_nan(landmark, 0)
            features["%s_y" % name] = self._value_or_nan(landmark, 1)
            features["%s_z" % name] = self._value_or_nan(landmark_z, index)

        mean_visibility = (
            sum(visibility_values) / len(visibility_values)
            if visibility_values
            else 0.0
        )
        return features, missing, mean_visibility

    def _predict_static(self, feature_map):
        row = [
            float(feature_map.get(column, math.nan))
            for column in self.feature_columns
        ]
        x = np.asarray([row], dtype=np.float32)
        probabilities = np.asarray(self.static_model.predict(x), dtype=float)
        if probabilities.ndim == 2:
            probabilities = probabilities[0]
        if probabilities.size == 0 or np.all(np.isnan(probabilities)):
            return "unknown", 0.0, np.asarray([], dtype=np.float32)

        pred_id = int(np.nanargmax(probabilities))
        label = self.labels[pred_id] if pred_id < len(self.labels) else str(pred_id)
        confidence = float(probabilities[pred_id])
        return label, confidence, probabilities

    def _empty_result(self, status="unknown"):
        return {
            "gesture": "unknown",
            "gesture_model_label": "unknown",
            "gesture_raw": "unknown",
            "gesture_raw_model_label": "unknown",
            "gesture_static": "unknown",
            "gesture_static_model_label": "unknown",
            "gesture_static_voted": "unknown",
            "gesture_waving": "no_waving",
            "gesture_waving_voted": "unknown",
            "gesture_confidence": 0.0,
            "gesture_raw_confidence": 0.0,
            "gesture_static_confidence": 0.0,
            "gesture_waving_confidence": 0.0,
            "gesture_probabilities": {},
            "gesture_waving_probabilities": {},
            "gesture_vote_count": 0,
            "gesture_vote_total": 0,
            "gesture_vote_ratio": 0.0,
            "gesture_vote_stable": False,
            "gesture_status": status,
            "gesture_missing_landmarks": [],
            "gesture_missing_landmark_count": 0,
            "gesture_mean_visibility": 0.0,
            "waving_probability": 0.0,
            "waving_status": "no_window",
            "waving_history_frames": 0,
            "waving_history_span_sec": 0.0,
            "waving_sampled": False,
        }

    def _vote_result(
        self,
        state,
        raw_label="unknown",
        raw_confidence=0.0,
        static_label="unknown",
        static_confidence=0.0,
        static_probabilities=None,
        waving_label="no_waving",
        waving_confidence=0.0,
        waving_probabilities=None,
        waving_status="no_window",
        waving_sampled=False,
        status="ok",
        missing=None,
        mean_visibility=0.0,
    ):
        static_vote = state.static_ring.vote(self.labels)
        waving_vote = state.waving_ring.vote(list(WAVING_LABELS))
        if raw_label == "waving" or waving_label == "waving":
            final_vote = dict(waving_vote)
            final_vote["label"] = "waving"
            final_vote["stable"] = True
            final_vote["confidence"] = float(waving_confidence)
            final_label = "waving"
        else:
            final_vote = static_vote
            final_label = static_vote["label"]

        history = state.waving_state.history
        if len(history) >= 2:
            history_span = history[-1]["time_sec"] - history[0]["time_sec"]
        else:
            history_span = 0.0

        static_probabilities = (
            np.asarray(static_probabilities, dtype=float)
            if static_probabilities is not None
            else np.asarray([], dtype=float)
        )
        waving_probabilities = (
            np.asarray(waving_probabilities, dtype=float)
            if waving_probabilities is not None
            else np.asarray([], dtype=float)
        )
        waving_probability = (
            float(waving_probabilities[1])
            if len(waving_probabilities) >= 2
            else 0.0
        )

        return {
            "gesture": self._output_label(final_label),
            "gesture_model_label": final_label,
            "gesture_raw": self._output_label(raw_label),
            "gesture_raw_model_label": raw_label,
            "gesture_static": self._output_label(static_label),
            "gesture_static_model_label": static_label,
            "gesture_static_voted": self._output_label(static_vote["label"]),
            "gesture_waving": waving_label,
            "gesture_waving_voted": waving_vote["label"],
            "gesture_confidence": float(final_vote["confidence"]),
            "gesture_raw_confidence": float(raw_confidence),
            "gesture_static_confidence": float(static_confidence),
            "gesture_waving_confidence": float(waving_confidence),
            "gesture_probabilities": self._probability_dict(
                self.labels,
                static_probabilities,
            ),
            "gesture_waving_probabilities": self._probability_dict(
                list(WAVING_LABELS),
                waving_probabilities,
            ),
            "gesture_vote_count": int(final_vote["count"]),
            "gesture_vote_total": int(final_vote["total"]),
            "gesture_vote_ratio": float(final_vote["ratio"]),
            "gesture_vote_stable": bool(final_vote["stable"]),
            "gesture_status": status,
            "gesture_missing_landmarks": list(missing or []),
            "gesture_missing_landmark_count": len(missing or []),
            "gesture_mean_visibility": float(mean_visibility),
            "waving_probability": waving_probability,
            "waving_status": waving_status,
            "waving_history_frames": len(history),
            "waving_history_span_sec": float(history_span),
            "waving_sampled": bool(waving_sampled),
        }

    def _raw_result(
        self,
        raw_label,
        raw_confidence,
        static_label,
        static_confidence,
        static_probabilities,
        missing,
        mean_visibility,
        status="ok",
    ):
        return {
            "gesture": self._output_label(raw_label),
            "gesture_model_label": raw_label,
            "gesture_raw": self._output_label(raw_label),
            "gesture_raw_model_label": raw_label,
            "gesture_static": self._output_label(static_label),
            "gesture_static_model_label": static_label,
            "gesture_static_voted": "unknown",
            "gesture_waving": "no_waving",
            "gesture_waving_voted": "unknown",
            "gesture_confidence": float(raw_confidence),
            "gesture_raw_confidence": float(raw_confidence),
            "gesture_static_confidence": float(static_confidence),
            "gesture_waving_confidence": 0.0,
            "gesture_probabilities": self._probability_dict(
                self.labels,
                static_probabilities,
            ),
            "gesture_waving_probabilities": {},
            "gesture_vote_count": 0,
            "gesture_vote_total": 0,
            "gesture_vote_ratio": 0.0,
            "gesture_vote_stable": False,
            "gesture_status": status,
            "gesture_missing_landmarks": list(missing),
            "gesture_missing_landmark_count": len(missing),
            "gesture_mean_visibility": float(mean_visibility),
            "waving_probability": 0.0,
            "waving_status": "disabled_without_temporal",
            "waving_history_frames": 0,
            "waving_history_span_sec": 0.0,
            "waving_sampled": False,
        }

    def predict(self, pose_data, person_id=None, timestamp=None, with_temporal=True):
        if not self.available:
            return self._empty_result(status="model_unavailable")

        if timestamp is None:
            timestamp = time.time()

        if pose_data is None:
            self.mark_missing(person_id)
            state = self._states.get(person_id)
            if with_temporal and state is not None:
                return self._vote_result(state, status="no_pose")
            return self._empty_result(status="no_pose")

        try:
            feature_map, missing, mean_visibility = self._feature_map(pose_data)
            static_label, static_confidence, static_probabilities = self._predict_static(
                feature_map
            )
        except Exception as exc:
            self.load_error = str(exc)
            self.mark_missing(person_id)
            return self._empty_result(status="predict_error")

        raw_label = static_label
        raw_confidence = static_confidence

        if not with_temporal or person_id is None:
            return self._raw_result(
                raw_label,
                raw_confidence,
                static_label,
                static_confidence,
                static_probabilities,
                missing,
                mean_visibility,
            )

        state = self._track_state(person_id)
        waving_label = "no_waving"
        waving_confidence = 0.0
        waving_probabilities = np.asarray([], dtype=np.float32)
        waving_status = "no_window"
        waving_sampled = False

        try:
            waving_sampled = state.waving_state.add_frame(timestamp, feature_map)
            waving_result = state.waving_state.predict(self.waving_model, timestamp)
            waving_label = waving_result["label"]
            waving_status = waving_result["status"]
            waving_confidence = float(waving_result["confidence"])
            waving_probabilities = waving_result["probabilities"]

            if waving_label == "waving":
                raw_label = "waving"
                raw_confidence = waving_confidence
            if waving_status == "ok":
                state.waving_ring.push(
                    waving_label,
                    waving_confidence,
                    waving_probabilities,
                )
        except Exception as exc:
            self.load_error = str(exc)
            waving_status = "predict_error"

        state.static_ring.push(static_label, static_confidence, static_probabilities)
        return self._vote_result(
            state,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            static_label=static_label,
            static_confidence=static_confidence,
            static_probabilities=static_probabilities,
            waving_label=waving_label,
            waving_confidence=waving_confidence,
            waving_probabilities=waving_probabilities,
            waving_status=waving_status,
            waving_sampled=waving_sampled,
            status="ok",
            missing=missing,
            mean_visibility=mean_visibility,
        )
