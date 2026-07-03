"""Cloth segmentation/color worker."""

import copy
import json
import threading
import time
from collections import OrderedDict

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cade_vision.kits.cloth import associate_clothing, fuse_fashion_detections, renumber_detections
from cade_vision.runtime.detections import append_yolo_detections
from cade_vision.runtime.json_utils import json_clean
from cade_vision.runtime.model_paths import load_yolo_model
from cade_vision.runtime.profiles import profile_has_cloth
from cade_vision.runtime.ros_frames import image_to_numpy


def parse_profile(data):
    try:
        return json.loads(data).get("profile", "idle")
    except Exception:
        return str(data or "idle")


class ClothWorker:
    def __init__(self, args):
        rospy.init_node("cade_vision_cloth_worker", anonymous=True)
        self.args = args
        self.profile = "idle"
        self.lock = threading.Lock()
        self.color_msg = None
        self.person_msg = None
        self.last_sequence = 0
        self.color_cache = OrderedDict()
        self.color_cache_size = 256
        self.seg_model = load_yolo_model(args.cloth_seg_model, args.device, "cloth segmentation YOLO") if args.cloth_seg_model else None
        detect_path = getattr(args, "cloth_model", None) or getattr(args, "cloth_detect_model", None)
        self.detect_model = load_yolo_model(detect_path, args.device, "cloth detect YOLO") if detect_path else None
        self.pub = rospy.Publisher("/vision/cloth_raw_task3", String, queue_size=1)
        rospy.Subscriber("/vision/profile_task3", String, self._on_profile, queue_size=1)
        rospy.Subscriber("/vision/frame/color_task3", Image, self._on_color, queue_size=1)
        rospy.Subscriber("/vision/person_raw_task3", String, self._on_person, queue_size=1)

    def _on_profile(self, msg):
        with self.lock:
            self.profile = parse_profile(msg.data)

    def _on_color(self, msg):
        with self.lock:
            self.color_msg = msg

    def _on_person(self, msg):
        with self.lock:
            self.person_msg = msg

    def _snapshot(self):
        with self.lock:
            return self.profile, self.color_msg, self.person_msg

    def _run_model(self, model, image, source_model):
        if model is None:
            return []
        detections = []
        results = model.predict(
            source=image,
            conf=self.args.conf,
            iou=self.args.iou,
            device=self.args.device,
            verbose=False,
        )
        names = model.names.copy() if hasattr(model.names, "copy") else model.names
        append_yolo_detections(detections, results, names, source_model=source_model)
        return detections

    def run(self):
        print("CADE cloth worker running")
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            profile, color_msg, person_msg = self._snapshot()
            if not profile_has_cloth(profile) or color_msg is None:
                rate.sleep()
                continue
            sequence = int(color_msg.header.seq)
            if sequence == self.last_sequence:
                rate.sleep()
                continue
            self.last_sequence = sequence
            start = time.perf_counter()
            image = image_to_numpy(color_msg)
            persons = []
            if person_msg is not None:
                try:
                    person_payload = json.loads(person_msg.data)
                    persons = copy.deepcopy(person_payload.get("detections", []) or [])
                except Exception:
                    persons = []
            stage_ms = {"cloth_fuse_ms": 0.0, "cloth_association_ms": 0.0, "cloth_color_ms": 0.0}
            stage_start = time.perf_counter()
            seg_detections = self._run_model(self.seg_model, image, "cloth_seg")
            detect_detections = self._run_model(self.detect_model, image, "cloth_detect")
            cloth_detections = fuse_fashion_detections(
                seg_detections,
                detect_detections,
                iou_threshold=self.args.cloth_fusion_iou,
            )
            stage_ms["cloth_fuse_ms"] = (time.perf_counter() - stage_start) * 1000.0
            detections = renumber_detections(persons + cloth_detections)
            stage_start = time.perf_counter()
            associate_clothing(
                detections,
                image,
                color_cache=self.color_cache,
                max_color_cache_size=self.color_cache_size,
                perf=stage_ms,
            )
            stage_ms["cloth_association_ms"] = (time.perf_counter() - stage_start) * 1000.0
            processed_persons = [obj for obj in detections if obj.get("class_name") == "person"]
            processed_cloths = [obj for obj in detections if obj.get("class_name") != "person"]
            person_cloth = [
                {
                    "track_id": person.get("track_id"),
                    "bbox": person.get("bbox"),
                    "cloth_color": person.get("cloth_color", "unknown"),
                    "cloth_type": person.get("cloth_type", "unknown"),
                    "cloth_items": person.get("cloth_items", []),
                    "cloth_summary": person.get("cloth_summary", "unknown"),
                }
                for person in processed_persons
            ]
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.pub.publish(
                json.dumps(
                    {
                        "sequence": sequence,
                        "stamp": color_msg.header.stamp.to_sec(),
                        "profile": profile,
                        "cloth_detections": json_clean(processed_cloths),
                        "person_cloth": json_clean(person_cloth),
                        "stage_ms": json_clean(stage_ms),
                        "cache_size": len(self.color_cache),
                        "duration_ms": duration_ms,
                    },
                    ensure_ascii=False,
                )
            )
