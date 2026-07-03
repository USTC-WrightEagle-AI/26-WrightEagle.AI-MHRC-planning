"""MediaPipe pose + LightGBM posture/gesture worker."""

import copy
import json
import math
import threading
import time

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cade_vision.kits.posture_gesture import (
    MEDIAPIPE_AVAILABLE,
    MEDIAPIPE_IMPORT_ERROR,
    PostureGestureAnalyzer,
)
from cade_vision.runtime.json_utils import json_clean
from cade_vision.runtime.profiles import profile_has_pose
from cade_vision.runtime.ros_frames import image_to_numpy


def parse_profile(data):
    try:
        return json.loads(data).get("profile", "idle")
    except Exception:
        return str(data or "idle")


def expand_bbox(bbox, image_w, image_h, margin=0.12):
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    dx = width * margin
    dy = height * margin
    return (
        max(0, int(math.floor(x1 - dx))),
        max(0, int(math.floor(y1 - dy))),
        min(image_w, int(math.ceil(x2 + dx))),
        min(image_h, int(math.ceil(y2 + dy))),
    )


class PoseGestureWorker:
    def __init__(self, args):
        rospy.init_node("cade_vision_pose_gesture_worker", anonymous=True)
        self.args = args
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(f"MediaPipe unavailable: {MEDIAPIPE_IMPORT_ERROR}")
        self.analyzer = PostureGestureAnalyzer()
        if not getattr(self.analyzer, "available", False):
            raise RuntimeError("PostureGestureAnalyzer unavailable")
        classifier = getattr(self.analyzer, "gesture_classifier", None)
        self.box_margin = float(getattr(classifier, "box_margin", 0.12) or 0.12)
        self.min_crop_size = int(getattr(args, "gesture_min_crop_size", 64) or 64)
        self.profile = "idle"
        self.lock = threading.Lock()
        self.color_msg = None
        self.person_msg = None
        self.last_sequence = 0
        self.pub = rospy.Publisher("/vision/pose_gesture_raw_task3", String, queue_size=1)
        rospy.Subscriber("/vision/profile_task3", String, self._on_profile, queue_size=1)
        rospy.Subscriber("/vision/frame/color_task3", Image, self._on_color, queue_size=1)
        rospy.Subscriber("/vision/person_raw_task3", String, self._on_person, queue_size=1)

    def _on_profile(self, msg):
        profile = parse_profile(msg.data)
        with self.lock:
            previous = self.profile
            self.profile = profile
        if profile_has_pose(previous) != profile_has_pose(profile):
            self.analyzer.clear_temporal()
            self.last_sequence = 0

    def _on_color(self, msg):
        with self.lock:
            self.color_msg = msg

    def _on_person(self, msg):
        with self.lock:
            self.person_msg = msg

    def _snapshot(self):
        with self.lock:
            return self.profile, self.color_msg, self.person_msg

    def _mark_missing(self, person_id):
        classifier = getattr(self.analyzer, "gesture_classifier", None)
        if classifier is not None and person_id is not None:
            classifier.mark_missing(person_id)

    def _apply_result(self, obj, result, pose_bbox):
        obj["posture"] = result.get("posture", "unknown")
        obj["gesture"] = result.get("gesture", "unknown")
        obj["landmarks"] = result.get("landmarks", [])
        obj["pose_bbox"] = pose_bbox
        for key in (
            "gesture_raw",
            "gesture_static",
            "gesture_static_voted",
            "gesture_waving",
            "gesture_waving_voted",
            "gesture_confidence",
            "gesture_raw_confidence",
            "gesture_static_confidence",
            "gesture_waving_confidence",
            "waving_probability",
            "waving_status",
            "waving_history_frames",
            "waving_history_span_sec",
            "waving_sampled",
            "gesture_vote_count",
            "gesture_vote_total",
            "gesture_vote_ratio",
            "gesture_vote_stable",
            "gesture_status",
            "gesture_missing_landmark_count",
            "gesture_mean_visibility",
        ):
            if key in result:
                obj[key] = result[key]

    def run(self):
        print("CADE pose/gesture worker running")
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            profile, color_msg, person_msg = self._snapshot()
            if not profile_has_pose(profile) or color_msg is None or person_msg is None:
                rate.sleep()
                continue
            payload = json.loads(person_msg.data)
            sequence = int(payload.get("sequence", 0) or 0)
            if sequence == self.last_sequence:
                rate.sleep()
                continue
            self.last_sequence = sequence
            start = time.perf_counter()
            image = image_to_numpy(color_msg)
            persons = copy.deepcopy(payload.get("detections", []) or [])
            frame_h, frame_w = image.shape[:2]
            active_ids = set()
            for obj in persons:
                track_id = obj.get("track_id")
                if track_id is not None:
                    active_ids.add(track_id)
                pose_bbox = expand_bbox(obj["bbox"], frame_w, frame_h, margin=self.box_margin)
                px1, py1, px2, py2 = pose_bbox
                person_crop = image[py1:py2, px1:px2]
                if (
                    person_crop.size <= 0
                    or px2 - px1 < self.min_crop_size
                    or py2 - py1 < self.min_crop_size
                ):
                    result = self.analyzer.analyze_from_landmarks(
                        None,
                        person_id=track_id,
                        with_temporal=True,
                        timestamp=float(payload.get("stamp", time.time())),
                    )
                    result["gesture_status"] = "crop_too_small"
                    result["waving_status"] = "crop_too_small"
                    self._apply_result(obj, result, pose_bbox)
                    continue
                pose_data = self.analyzer.process_pose(person_crop)
                if pose_data is not None and frame_w > 0 and frame_h > 0:
                    x1, y1, x2, y2 = obj["bbox"]
                    pose_data["bbox_w"] = max(0, x2 - x1) / frame_w
                    pose_data["bbox_h"] = max(0, y2 - y1) / frame_h
                result = self.analyzer.analyze_from_landmarks(
                    pose_data,
                    person_id=track_id,
                    with_temporal=True,
                    timestamp=float(payload.get("stamp", time.time())),
                )
                self._apply_result(obj, result, pose_bbox)
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.pub.publish(
                json.dumps(
                    {
                        "sequence": sequence,
                        "stamp": payload.get("stamp", time.time()),
                        "profile": profile,
                        "detections": json_clean(persons),
                        "duration_ms": duration_ms,
                    },
                    ensure_ascii=False,
                )
            )

