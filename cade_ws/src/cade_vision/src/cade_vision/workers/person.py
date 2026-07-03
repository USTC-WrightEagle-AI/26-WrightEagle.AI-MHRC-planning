"""Person YOLO worker."""

import json
import threading
import time

import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from cade_vision.runtime.detections import append_yolo_detections
from cade_vision.runtime.json_utils import json_clean
from cade_vision.runtime.model_paths import load_yolo_model
from cade_vision.runtime.profiles import VISION_PROFILES
from cade_vision.runtime.ros_frames import camera_info_dict, image_to_numpy
from cade_vision.runtime.tracking import PersonIoUTracker


def parse_profile(data):
    try:
        payload = json.loads(data)
        return payload.get("profile", "idle")
    except Exception:
        return str(data or "idle")


class PersonWorker:
    def __init__(self, args):
        rospy.init_node("cade_vision_person_worker", anonymous=True)
        self.args = args
        self.profile = "idle"
        self.lock = threading.Lock()
        self.color_msg = None
        self.depth_msg = None
        self.camera_info_msg = None
        self.last_sequence = 0
        self.last_idle_time = 0.0
        self.model = load_yolo_model(args.model, args.device, "person/general YOLO")
        self.tracker = PersonIoUTracker(
            getattr(args, "gesture_track_iou_threshold", 0.25),
            getattr(args, "gesture_max_track_missing_frames", 15),
        )
        self.pub = rospy.Publisher("/vision/person_raw_task3", String, queue_size=1)
        rospy.Subscriber("/vision/profile_task3", String, self._on_profile, queue_size=1)
        rospy.Subscriber("/vision/frame/color_task3", Image, self._on_color, queue_size=1)
        rospy.Subscriber("/vision/frame/depth_task3", Image, self._on_depth, queue_size=1)
        rospy.Subscriber("/vision/frame/camera_info_task3", CameraInfo, self._on_camera_info, queue_size=1)

    def _on_profile(self, msg):
        profile = parse_profile(msg.data)
        if profile in VISION_PROFILES:
            with self.lock:
                previous = self.profile
                self.profile = profile
            if profile == "idle" and previous != "idle":
                self.tracker.clear()

    def _on_color(self, msg):
        with self.lock:
            self.color_msg = msg

    def _on_depth(self, msg):
        with self.lock:
            self.depth_msg = msg

    def _on_camera_info(self, msg):
        with self.lock:
            self.camera_info_msg = msg

    def _snapshot(self):
        with self.lock:
            return self.profile, self.color_msg, self.depth_msg, self.camera_info_msg

    def _should_run(self, profile):
        if profile not in {"idle", "person", "gesture", "posture", "cloth", "full"}:
            return False
        if profile == "idle":
            idle_rate = float(getattr(self.args, "idle_person_rate", 0.0) or 0.0)
            if idle_rate <= 0.0:
                return False
            now = time.time()
            if now - self.last_idle_time < 1.0 / max(idle_rate, 1e-6):
                return False
            self.last_idle_time = now
        return True

    def run(self):
        print("CADE person worker running")
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            profile, color_msg, depth_msg, camera_info_msg = self._snapshot()
            if color_msg is None or color_msg.header.seq == self.last_sequence or not self._should_run(profile):
                rate.sleep()
                continue
            self.last_sequence = color_msg.header.seq
            start = time.perf_counter()
            image = image_to_numpy(color_msg)
            depth_m = image_to_numpy(depth_msg) if depth_msg is not None else None
            camera_info = camera_info_dict(camera_info_msg)
            detections = []
            results = self.model.predict(
                source=image,
                conf=self.args.conf,
                iou=self.args.iou,
                device=self.args.device,
                verbose=False,
                classes=[0],
                imgsz=self.args.person_imgsz,
            )
            names = self.model.names.copy() if hasattr(self.model.names, "copy") else self.model.names
            append_yolo_detections(
                detections,
                results,
                names,
                source_model="person",
                depth_m=depth_m,
                camera_info=camera_info,
                allowed_names={"person"},
                min_area_ratio=getattr(self.args, "person_min_box_area_ratio", 0.015),
            )
            track_ids = self.tracker.assign(detections)
            for obj, track_id in zip(detections, track_ids):
                obj["track_id"] = track_id
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.pub.publish(
                json.dumps(
                    {
                        "sequence": int(color_msg.header.seq),
                        "stamp": color_msg.header.stamp.to_sec(),
                        "profile": profile,
                        "detections": json_clean(detections),
                        "duration_ms": duration_ms,
                    },
                    ensure_ascii=False,
                )
            )
