#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep one RealSense pipeline open and continuously refresh Task1 vision JSON.

The task controller consumes JSON coordinates. When display is enabled, this
daemon also opens OpenCV debug windows and draws the latest detections on the
shared RealSense stream.
"""

import argparse
import ast
import json
import os
import re
import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_OBJECT_MODEL = SCRIPT_DIR / "yolo11n.pt"
LOCAL_POSE_MODEL = SCRIPT_DIR / "yolo11n-pose.pt"
DEFAULT_CADE_VISION_SRC = Path("/home/nvidia/Desktop/task3/cade_ws/src/cade_vision/src")
DEFAULT_CLOTH_SEG_MODEL = (
    DEFAULT_CADE_VISION_SRC
    / "cade_vision"
    / "models"
    / "yolov8s-seg-fashionpedia-best.pt"
)
DEFAULT_CLOTH_DETECT_MODEL = (
    DEFAULT_CADE_VISION_SRC
    / "cade_vision"
    / "models"
    / "yolo11s-fashionpedia-best.pt"
)
DEFAULT_OBJECT_MODEL = str(LOCAL_OBJECT_MODEL if LOCAL_OBJECT_MODEL.exists() else "yolo11n.pt")
DEFAULT_POSE_MODEL = str(LOCAL_POSE_MODEL if LOCAL_POSE_MODEL.exists() else "yolo11n-pose.pt")
DEFAULT_EMPTY_OUTPUT_JSON = str(SCRIPT_DIR / "latest_empty_seat.json")
DEFAULT_HAND_OUTPUT_JSON = str(SCRIPT_DIR / "latest_handover_hand.json")
DEFAULT_BAG_OUTPUT_JSON = str(SCRIPT_DIR / "latest_bag.json")
DEFAULT_PEOPLE_OUTPUT_JSON = str(SCRIPT_DIR / "latest_people.json")
DEFAULT_STATUS_JSON = str(SCRIPT_DIR / "latest_vision_cache_status.json")
DEFAULT_LEFTBASE_EXTRINSIC = SCRIPT_DIR / "camera_middle_to_leftbase.txt"
DEFAULT_EXTRINSIC = (
    str(DEFAULT_LEFTBASE_EXTRINSIC) if DEFAULT_LEFTBASE_EXTRINSIC.exists() else ""
)
DEFAULT_COORDINATE_FRAME = "leftbase" if DEFAULT_EXTRINSIC else "camera_color_optical"
DEFAULT_CAMERA_MODEL = os.environ.get("REALSENSE_CAMERA_MODEL", "D455")
DEFAULT_REALSENSE_SERIAL = (
    os.environ.get("REALSENSE455_SERIAL")
    or os.environ.get("REALSENSE_SERIAL")
    or os.environ.get("REALSENSE515_SERIAL")
    or os.environ.get("CADE_REALSENSE_SERIAL")
    or ""
)
DEFAULT_CARRY_OBJECT_CLASSES = [
    "umbrella",
    "bottle",
    "wine glass",
    "cup",
    "bowl",
    "book",
    "cell phone",
    "laptop",
    "remote",
    "sports ball",
    "teddy bear",
]
WRIST_KEYPOINTS = {"left": 9, "right": 10}


class PersonIoUTracker:
    """Small person-only tracker, kept aligned with the Task3 tracking contract."""

    def __init__(self, iou_threshold=0.25, max_missing_frames=15):
        self.iou_threshold = float(iou_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks = []
        self._next_id = 1
        self.missing_track_ids = []

    def assign(self, detections):
        self.missing_track_ids = []
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        candidates = []

        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                iou = self._iou(track.get("bbox"), detection.get("bbox"))
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_index, detection_index))

        matches = []
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if (
                track_index not in unmatched_tracks
                or detection_index not in unmatched_detections
            ):
                continue
            matches.append((track_index, detection_index))
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        track_ids = [None] * len(detections)
        new_tracks = []
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            track_ids[detection_index] = track["id"]
            new_tracks.append(
                {
                    "id": track["id"],
                    "bbox": detections[detection_index].get("bbox"),
                    "missing": 0,
                }
            )

        for detection_index in sorted(unmatched_detections):
            track_id = self._next_id
            self._next_id += 1
            track_ids[detection_index] = track_id
            new_tracks.append(
                {
                    "id": track_id,
                    "bbox": detections[detection_index].get("bbox"),
                    "missing": 0,
                }
            )

        for track_index in sorted(unmatched_tracks):
            track = dict(self._tracks[track_index])
            track["missing"] = int(track.get("missing", 0)) + 1
            self.missing_track_ids.append(track["id"])
            if track["missing"] <= self.max_missing_frames:
                new_tracks.append(track)

        self._tracks = new_tracks
        return track_ids

    def clear(self):
        self._tracks.clear()
        self._next_id = 1
        self.missing_track_ids = []

    @staticmethod
    def _iou(box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
        bx1, by1, bx2, by2 = [float(value) for value in box_b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_words(name, default):
    value = os.environ.get(name, "").strip()
    return value.split() if value else list(default)


def env_classes(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return list(default)
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return value.split()


def load_matrix(path, expected_shape):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                row = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                row = [
                    float(value)
                    for value in re.split(r"[,\s]+", line.strip("[]"))
                    if value
                ]
            if isinstance(row, (int, float)):
                row = [row]
            rows.append(list(row))
    matrix = np.array(rows, dtype=np.float64)
    if matrix.shape != expected_shape:
        raise ValueError(f"{path} 应为 {expected_shape} 矩阵，实际为 {matrix.shape}")
    return matrix


def get_device_info(device, info_key):
    try:
        return device.get_info(info_key)
    except RuntimeError:
        return ""


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(val) for key, val in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(json_clean(payload), file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


class VisionCacheDaemon:
    def __init__(self, args):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("缺少 ultralytics 依赖，请先运行: pip install ultralytics") from exc

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("缺少 pyrealsense2 依赖，请先运行: pip install pyrealsense2") from exc

        self.args = args
        self.rs = rs
        self.running = True
        self.frame_count = 0
        self.last_status_time = 0.0
        self.last_output_time = {
            "empty_seat": 0.0,
            "handover_hand": 0.0,
            "bag": 0.0,
            "people": 0.0,
        }
        self.last_run_time = {
            "empty_seat": 0.0,
            "handover_hand": 0.0,
            "bag": 0.0,
            "people": 0.0,
        }
        self.latest_payloads = {
            "empty_seat": {},
            "handover_hand": {},
            "bag": {},
            "people": {},
        }
        self.status_path = Path(args.status_json)
        self.outputs = {
            "empty_seat": str(args.empty_output_json),
            "handover_hand": str(args.hand_output_json),
            "bag": str(args.bag_output_json),
            "people": str(args.people_output_json),
        }

        self.object_model = YOLO(args.object_model)
        self.pose_model = YOLO(args.pose_model) if args.enable_handover_hand else None
        self.person_class_ids = self.class_ids(["person"]) or None
        self.empty_predict_class_ids = self.class_ids(["person", *args.seat_classes]) or None
        self.cloth_color_cache = OrderedDict()
        self.person_tracker = PersonIoUTracker(
            args.people_track_iou_threshold,
            args.people_track_max_missing_frames,
        )
        self.cloth_model = None
        self.cloth_detect_model = None
        self.cloth_available = False
        self.cloth_error = ""
        self._append_yolo_detections = None
        self._fuse_fashion_detections = None
        self._renumber_detections = None
        self._associate_clothing = None
        self.last_clothing_time = 0.0
        self.last_clothing_people = []
        self.consecutive_frame_errors = 0
        self.pipeline = None
        self.profile = None
        self.align = None
        self.device_name = ""
        self.serial_number = ""
        self.depth_scale = 0.0
        self.cv2 = None
        self.display_enabled = not bool(args.no_display)
        self._init_clothing(YOLO)

        empty_extrinsic = self.load_optional_matrix(args.empty_extrinsic, (4, 4))
        hand_extrinsic = self.load_optional_matrix(args.hand_extrinsic, (4, 4))
        bag_extrinsic = self.load_optional_matrix(args.bag_extrinsic, (4, 4))
        self.empty_extrinsic_matrix = (
            empty_extrinsic if empty_extrinsic is not None else np.eye(4, dtype=np.float64)
        )
        self.hand_extrinsic_matrix = (
            hand_extrinsic if hand_extrinsic is not None else np.eye(4, dtype=np.float64)
        )
        self.bag_extrinsic_matrix = (
            bag_extrinsic if bag_extrinsic is not None else np.eye(4, dtype=np.float64)
        )
        self.empty_intrinsic_matrix = self.load_optional_matrix(args.empty_intrinsic, (3, 3))
        self.hand_intrinsic_matrix = self.load_optional_matrix(args.hand_intrinsic, (3, 3))
        self.bag_intrinsic_matrix = self.load_optional_matrix(args.bag_intrinsic, (3, 3))
        self._start_camera_pipeline()
        self._init_display()

    def _init_display(self):
        if not self.display_enabled:
            return
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            print("[VisionCache] 未检测到 DISPLAY/WAYLAND_DISPLAY，关闭 OpenCV 可视化窗口", flush=True)
            self.display_enabled = False
            return
        try:
            import cv2

            self.cv2 = cv2
            cv2.namedWindow(self.args.display_window, cv2.WINDOW_NORMAL)
            if self.args.display_depth:
                cv2.namedWindow(self.args.depth_window, cv2.WINDOW_NORMAL)
        except Exception as exc:
            print(f"[VisionCache] OpenCV 可视化不可用，继续只写 JSON: {exc}", flush=True)
            self.cv2 = None
            self.display_enabled = False

    def _start_camera_pipeline(self):
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        self.serial_number = self.resolve_serial_number(self.args.serial_number, self.args.camera_model)
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(self.rs.stream.color, self.args.width, self.args.height, self.rs.format.bgr8, self.args.fps)
        config.enable_stream(self.rs.stream.depth, self.args.width, self.args.height, self.rs.format.z16, self.args.fps)
        self.profile = self.pipeline.start(config)
        self.align = self.rs.align(self.rs.stream.color)

        device = self.profile.get_device()
        self.device_name = get_device_info(device, self.rs.camera_info.name)
        self.serial_number = get_device_info(device, self.rs.camera_info.serial_number) or self.serial_number
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        stream_intrinsic = self.load_stream_intrinsic_matrix()
        if self.empty_intrinsic_matrix is None:
            self.empty_intrinsic_matrix = stream_intrinsic
        if self.hand_intrinsic_matrix is None:
            self.hand_intrinsic_matrix = stream_intrinsic
        if self.bag_intrinsic_matrix is None:
            self.bag_intrinsic_matrix = stream_intrinsic

    def _restart_camera_pipeline(self):
        print("[VisionCache] 连续相机取帧失败，重启 RealSense pipeline...", flush=True)
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as exc:
            print(f"[VisionCache] 停止旧 RealSense pipeline 时出错: {exc}", flush=True)
        time.sleep(max(0.1, self.args.frame_error_restart_sleep))
        try:
            self._start_camera_pipeline()
        except Exception as exc:
            self.pipeline = None
            self.profile = None
            self.align = None
            self.consecutive_frame_errors = 0
            print(f"[VisionCache] 重启 RealSense pipeline 失败，将继续重试: {exc}", flush=True)
            self.write_status(force=True, error=exc, status="camera_error")
            return False
        self.consecutive_frame_errors = 0
        self.write_status(force=True)
        return True

    def class_ids(self, names):
        wanted = set(names)
        return [
            class_id
            for class_id, class_name in self.object_model.names.items()
            if class_name in wanted
        ]

    def _load_optional_yolo_model(self, YOLO, path, label):
        path = str(path or "").strip()
        if not path:
            return None
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"{label} 模型不存在: {model_path}")
        print(f"Loading {label}: {model_path}", flush=True)
        model = YOLO(str(model_path))
        if self.args.device and model_path.suffix.lower() != ".engine":
            model.to(self.args.device)
        return model

    def _init_clothing(self, YOLO):
        if not self.args.enable_clothing:
            return
        cade_src = Path(self.args.cade_vision_src).expanduser()
        try:
            if cade_src.exists() and str(cade_src) not in sys.path:
                sys.path.insert(0, str(cade_src))
            from cade_vision.kits.cloth import (
                associate_clothing,
                fuse_fashion_detections,
                renumber_detections,
            )
            from cade_vision.runtime.detections import append_yolo_detections

            self._append_yolo_detections = append_yolo_detections
            self._fuse_fashion_detections = fuse_fashion_detections
            self._renumber_detections = renumber_detections
            self._associate_clothing = associate_clothing
            self.cloth_model = self._load_optional_yolo_model(
                YOLO,
                self.args.cloth_seg_model,
                "cloth segmentation YOLO",
            )
            self.cloth_detect_model = self._load_optional_yolo_model(
                YOLO,
                self.args.cloth_detect_model,
                "cloth detect YOLO",
            )
            if self.cloth_model is None and self.cloth_detect_model is None:
                raise RuntimeError("未配置可用的衣着模型")
            self.cloth_available = True
        except Exception as exc:
            self.cloth_available = False
            self.cloth_error = str(exc)
            print(f"[Cloth] 衣着识别不可用，people 缓存将只写 unknown: {exc}", flush=True)

    @staticmethod
    def load_optional_matrix(path, shape):
        return load_matrix(path, shape) if path else None

    def resolve_serial_number(self, serial_number, camera_model):
        serial_number = (serial_number or "").strip()
        if serial_number and serial_number.lower() not in {"auto", "none"}:
            return serial_number

        camera_model = (camera_model or "").strip()
        if not camera_model:
            return None

        devices = []
        device_list = self.rs.context().query_devices()
        for device_index in range(len(device_list)):
            try:
                device = device_list[device_index]
            except RuntimeError:
                continue
            name = get_device_info(device, self.rs.camera_info.name)
            serial = get_device_info(device, self.rs.camera_info.serial_number)
            if not serial:
                continue
            devices.append(f"{name}({serial})")
            if camera_model.lower() in name.lower():
                return serial

        detected = ", ".join(devices) if devices else "无"
        raise RuntimeError(f"未找到型号包含 {camera_model!r} 的 RealSense 设备，已检测到: {detected}")

    def load_stream_intrinsic_matrix(self):
        color_profile = self.profile.get_stream(
            self.rs.stream.color
        ).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        return np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def should_run(self, key, interval):
        now = time.time()
        if now - self.last_run_time[key] < interval:
            return False
        self.last_run_time[key] = now
        return True

    def output_allowed(self, key, interval):
        now = time.time()
        if now - self.last_output_time[key] < interval:
            return False
        self.last_output_time[key] = now
        return True

    def median_depth(self, depth_image, bbox, min_samples, crop_ratio=0.0, excluded_boxes=()):
        height, width = depth_image.shape
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        box_w = x2 - x1
        box_h = y2 - y1
        if crop_ratio > 0.0:
            x1 = int(x1 + box_w * crop_ratio)
            x2 = int(x2 - box_w * crop_ratio)
            y1 = int(y1 + box_h * crop_ratio)
            y2 = int(y2 - box_h * crop_ratio)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return None

        roi = depth_image[y1:y2, x1:x2]
        mask = np.ones(roi.shape, dtype=bool)
        for ex1, ey1, ex2, ey2 in excluded_boxes:
            ex1, ex2 = max(x1, int(ex1)) - x1, min(x2, int(ex2)) - x1
            ey1, ey2 = max(y1, int(ey1)) - y1, min(y2, int(ey2)) - y1
            if ex1 < ex2 and ey1 < ey2:
                mask[ey1:ey2, ex1:ex2] = False

        depths = roi[mask].astype(np.float32) * self.depth_scale
        depths = depths[
            (depths >= self.args.min_depth) & (depths <= self.args.max_depth)
        ]
        if depths.size < min_samples:
            return None
        return float(np.median(depths))

    def median_depth_around_pixel(self, depth_image, pixel_x, pixel_y):
        height, width = depth_image.shape
        percentile = min(100.0, max(0.0, float(self.args.hand_depth_percentile)))
        for radius in (
            self.args.hand_depth_radius,
            self.args.hand_depth_radius * 2,
            self.args.hand_depth_radius * 3,
        ):
            x1 = max(0, int(round(pixel_x - radius)))
            x2 = min(width, int(round(pixel_x + radius + 1)))
            y1 = max(0, int(round(pixel_y - radius)))
            y2 = min(height, int(round(pixel_y + radius + 1)))
            if x1 >= x2 or y1 >= y2:
                continue
            depths = depth_image[y1:y2, x1:x2].astype(np.float32) * self.depth_scale
            depths = depths[
                (depths >= self.args.min_depth) & (depths <= self.args.max_depth)
            ]
            if depths.size >= self.args.hand_min_depth_samples:
                return float(np.percentile(depths, percentile))
        return None

    def person_depth(self, depth_image, person_box):
        x1, y1, x2, y2 = person_box
        width = x2 - x1
        height = y2 - y1
        center_box = (
            int(x1 + width * 0.25),
            int(y1 + height * 0.2),
            int(x2 - width * 0.25),
            int(y2 - height * 0.15),
        )
        return self.median_depth(
            depth_image,
            center_box,
            self.args.people_min_depth_samples,
        )

    @staticmethod
    def intersection_over_seat(person_box, seat_box):
        px1, py1, px2, py2 = person_box
        sx1, sy1, sx2, sy2 = seat_box
        intersection_width = max(0, min(px2, sx2) - max(px1, sx1))
        intersection_height = max(0, min(py2, sy2) - max(py1, sy1))
        intersection_area = intersection_width * intersection_height
        seat_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        return intersection_area / seat_area

    def seat_occupancy(self, depth_image, seat_box, person_boxes):
        seat_depth = self.median_depth(
            depth_image,
            seat_box,
            self.args.empty_min_depth_samples,
            excluded_boxes=person_boxes,
        )
        if seat_depth is None:
            seat_depth = self.median_depth(
                depth_image,
                seat_box,
                self.args.empty_min_depth_samples,
            )
        best_match = None
        for person_box in person_boxes:
            overlap = self.intersection_over_seat(person_box, seat_box)
            if overlap < self.args.occupancy_threshold:
                continue
            person_depth = self.person_depth(depth_image, person_box)
            if seat_depth is None or person_depth is None:
                continue
            depth_gap = abs(person_depth - seat_depth)
            if best_match is None or depth_gap < best_match["depth_gap"]:
                best_match = {
                    "overlap": overlap,
                    "person_depth": person_depth,
                    "seat_depth": seat_depth,
                    "depth_gap": depth_gap,
                }
        occupied = (
            best_match is not None
            and best_match["depth_gap"] <= self.args.max_occupancy_depth_gap
        )
        return occupied, seat_depth, best_match

    def point_position(self, pixel_x, pixel_y, depth_m, intrinsic_matrix, extrinsic_matrix, coordinate_frame):
        if depth_m is None:
            return None
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        point_camera = np.array(
            [
                (pixel_x - cx) * depth_m / fx,
                (pixel_y - cy) * depth_m / fy,
                depth_m,
                1.0,
            ]
        )
        point_target = extrinsic_matrix @ point_camera
        target_xyz = [round(float(value), 3) for value in point_target[:3]]
        position = {
            "pixel": [round(float(pixel_x), 1), round(float(pixel_y), 1)],
            "camera_xyz_m": [round(float(value), 3) for value in point_camera[:3]],
            "target_frame_xyz_m": target_xyz,
            "calibrated_xyz_m": target_xyz,
        }
        if coordinate_frame in {"leftbase", "left_arm_base"}:
            position["leftbase_xyz_m"] = target_xyz
            position["left_arm_base_xyz_m"] = target_xyz
        return position

    @staticmethod
    def expand_bbox(bbox, ratio):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = x2 - x1
        height = y2 - y1
        return [
            x1 - width * ratio,
            y1 - height * ratio,
            x2 + width * ratio,
            y2 + height * ratio,
        ]

    @staticmethod
    def point_to_bbox_distance(pixel_x, pixel_y, bbox):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        dx = max(x1 - pixel_x, 0.0, pixel_x - x2)
        dy = max(y1 - pixel_y, 0.0, pixel_y - y2)
        return float((dx * dx + dy * dy) ** 0.5)

    @staticmethod
    def point_in_bbox(pixel_x, pixel_y, bbox):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return x1 <= pixel_x <= x2 and y1 <= pixel_y <= y2

    def associate_hand_with_object(self, pixel_x, pixel_y, objects):
        best_match = None
        for obj in objects:
            expanded_bbox = self.expand_bbox(obj["bbox"], self.args.bag_box_expand_ratio)
            distance_px = self.point_to_bbox_distance(pixel_x, pixel_y, expanded_bbox)
            inside_expanded = self.point_in_bbox(pixel_x, pixel_y, expanded_bbox)
            if (
                not inside_expanded
                and distance_px > self.args.max_hand_bag_pixel_distance
            ):
                continue
            center_x = (obj["bbox"][0] + obj["bbox"][2]) / 2.0
            center_y = (obj["bbox"][1] + obj["bbox"][3]) / 2.0
            center_distance_px = float(
                ((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2) ** 0.5
            )
            score = distance_px + center_distance_px * 0.01
            if best_match is None or score < best_match["score"]:
                best_match = {
                    "score": score,
                    "object": obj,
                    "distance_px": round(distance_px, 1),
                    "center_distance_px": round(center_distance_px, 1),
                    "inside_expanded_bbox": bool(inside_expanded),
                    "expanded_bbox": [round(float(value), 1) for value in expanded_bbox],
                }
        return best_match

    def handover_object_role(self, class_name):
        if class_name in self.args.bag_classes:
            return "bag"
        if class_name in self.args.carry_object_classes:
            return "object"
        return None

    def detect_handover_objects_for_hand(self, frame, depth_image):
        results = self.object_model.predict(
            source=frame,
            conf=self.args.hand_bag_conf,
            iou=self.args.iou,
            device=self.args.device,
            classes=None,
            verbose=False,
        )
        objects = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.object_model.names[class_id]
            object_role = self.handover_object_role(class_name)
            if object_role is None:
                continue
            bbox = [float(value) for value in box.xyxy[0].cpu().tolist()]
            depth_m = self.median_depth(
                depth_image,
                bbox,
                self.args.hand_object_min_depth_samples,
                crop_ratio=self.args.hand_object_depth_crop_ratio,
            )
            x1, y1, x2, y2 = bbox
            pixel_x = x1 + (x2 - x1) * self.args.hand_object_point_x_ratio
            pixel_y = y1 + (y2 - y1) * self.args.hand_object_point_y_ratio
            objects.append(
                {
                    "class_name": class_name,
                    "object_role": object_role,
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(value, 1) for value in bbox],
                    "depth_m": None if depth_m is None else round(depth_m, 3),
                    "position": self.point_position(
                        pixel_x,
                        pixel_y,
                        depth_m,
                        self.hand_intrinsic_matrix,
                        self.hand_extrinsic_matrix,
                        self.args.hand_coordinate_frame,
                    ),
                }
            )
        return objects

    def make_handover_hand_record(
        self,
        person_index,
        side,
        confidence,
        person_confidence,
        person_bbox,
        pixel_x,
        pixel_y,
        depth_m,
        object_match=None,
    ):
        wrist_position = self.point_position(
            pixel_x,
            pixel_y,
            depth_m,
            self.hand_intrinsic_matrix,
            self.hand_extrinsic_matrix,
            self.args.hand_coordinate_frame,
        )
        record = {
            "target_type": "nearest_wrist",
            "selection_label": "nearest wrist",
            "selection_priority": 4,
            "person_index": person_index,
            "hand_side": side,
            "confidence": round(confidence, 3),
            "person_confidence": person_confidence,
            "person_bbox": person_bbox,
            "depth_m": None if depth_m is None else round(depth_m, 3),
            "position": wrist_position,
            "wrist_position": wrist_position,
            "handover_position_source": "wrist",
            "handover_depth_m": None if depth_m is None else round(depth_m, 3),
        }
        if object_match is None:
            return record

        obj = object_match["object"]
        role = obj.get("object_role")
        object_position = obj.get("position")
        if role == "bag":
            record["target_type"] = "hand_with_bag"
            record["selection_label"] = "hand holding bag"
            record["selection_priority"] = 1
        else:
            record["target_type"] = "hand_with_object"
            record["selection_label"] = "hand holding object"
            record["selection_priority"] = 3
        record.update(
            {
                "matched_object_role": role,
                "object_class_name": obj["class_name"],
                "object_confidence": obj["confidence"],
                "object_bbox": obj["bbox"],
                "matched_object_position": object_position,
                "hand_object_distance_px": object_match["distance_px"],
                "hand_object_center_distance_px": object_match["center_distance_px"],
                "hand_inside_expanded_object_bbox": object_match["inside_expanded_bbox"],
                "expanded_object_bbox": object_match["expanded_bbox"],
            }
        )
        if object_position is not None:
            record["position"] = object_position
            record["handover_position_source"] = role or "matched_object"
            record["handover_depth_m"] = obj.get("depth_m")
        if role == "bag":
            record.update(
                {
                    "bag_class_name": obj["class_name"],
                    "bag_confidence": obj["confidence"],
                    "bag_bbox": obj["bbox"],
                    "hand_bag_distance_px": object_match["distance_px"],
                    "hand_bag_center_distance_px": object_match["center_distance_px"],
                    "hand_inside_expanded_bag_bbox": object_match["inside_expanded_bbox"],
                    "expanded_bag_bbox": object_match["expanded_bbox"],
                }
            )
        return record

    @staticmethod
    def make_handover_bag_target(bag):
        return {
            "target_type": "bag",
            "selection_label": "bag fallback",
            "selection_priority": 2,
            "hand_side": None,
            "confidence": bag.get("confidence"),
            "object_class_name": bag.get("class_name"),
            "object_confidence": bag.get("confidence"),
            "object_bbox": bag.get("bbox"),
            "bag_class_name": bag.get("class_name"),
            "bag_confidence": bag.get("confidence"),
            "bag_bbox": bag.get("bbox"),
            "depth_m": bag.get("depth_m"),
            "handover_depth_m": bag.get("depth_m"),
            "handover_position_source": "bag",
            "position": bag.get("position"),
        }

    def handover_sort_key(self, target):
        depth = target.get("handover_depth_m")
        if depth is None:
            depth = target.get("depth_m")
        if depth is None:
            depth = float("inf")
        distance = (
            target.get("hand_bag_distance_px")
            if target.get("hand_bag_distance_px") is not None
            else target.get("hand_object_distance_px")
        )
        if distance is None:
            distance = float("inf")
        confidence = target.get("confidence")
        if confidence is None:
            confidence = target.get("object_confidence") or target.get("bag_confidence") or 0.0
        priority = target.get("selection_priority", 99)
        if self.args.hand_select == "confidence":
            return (priority, -float(confidence), depth, distance)
        return (priority, depth, distance, -float(confidence))

    def select_handover_target(self, hands, objects):
        valid_hands = [hand for hand in hands if hand.get("position") is not None]
        bag_targets = [
            self.make_handover_bag_target(obj)
            for obj in objects
            if obj.get("object_role") == "bag" and obj.get("position") is not None
        ]
        candidates = valid_hands + bag_targets
        return min(candidates, key=self.handover_sort_key, default=None)

    def _run_cloth_model(self, model, frame, source_model):
        if model is None or self._append_yolo_detections is None:
            return []
        detections = []
        results = model.predict(
            source=frame,
            conf=self.args.cloth_conf,
            iou=self.args.cloth_iou,
            device=self.args.device,
            verbose=False,
        )
        names = model.names.copy() if hasattr(model.names, "copy") else model.names
        self._append_yolo_detections(
            detections,
            results,
            names,
            source_model=source_model,
        )
        return detections

    @staticmethod
    def _cloth_summary_from_items(items):
        parts = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            color = item.get("color", "unknown")
            ctype = item.get("type") or item.get("category") or "unknown"
            if color and color != "unknown" and ctype and ctype != "unknown":
                parts.append(f"{color} {ctype}")
            elif ctype and ctype != "unknown":
                parts.append(str(ctype))
        return ", ".join(parts) if parts else "unknown"

    @staticmethod
    def _public_cloth_items(items):
        public_items = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            public_items.append(
                {
                    "region": item.get("region"),
                    "type": item.get("type"),
                    "color": item.get("color"),
                    "confidence": item.get("confidence"),
                    "bbox": item.get("bbox"),
                    "source_model": item.get("source_model"),
                    "score": item.get("score"),
                }
            )
        return public_items

    @staticmethod
    def _bbox_iou(box_a, box_b):
        if not box_a or not box_b:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
        bx1, by1, bx2, by2 = [float(value) for value in box_b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    def _cache_current_clothing(self, people):
        cached = []
        for person in people:
            if not self._person_has_known_clothing(person):
                continue
            cached.append(
                {
                    "bbox": list(person.get("bbox") or []),
                    "cloth_color": person.get("cloth_color", "unknown"),
                    "cloth_type": person.get("cloth_type", "unknown"),
                    "cloth_items": person.get("cloth_items", []),
                    "cloth_summary": person.get("cloth_summary", "unknown"),
                    "clothing_summary": person.get("clothing_summary"),
                }
            )
        if not cached:
            return
        self.last_clothing_people = cached
        self.last_clothing_time = time.time()

    def _apply_cached_clothing(self, people):
        if not self.last_clothing_people:
            return False
        applied = False
        for person in people:
            best = None
            for cached in self.last_clothing_people:
                if not self._person_has_known_clothing(cached):
                    continue
                iou = self._bbox_iou(person.get("bbox"), cached.get("bbox"))
                if best is None or iou > best[0]:
                    best = (iou, cached)
            if best is None or best[0] < self.args.cloth_cache_match_iou:
                continue
            cached = best[1]
            person["cloth_color"] = cached.get("cloth_color", "unknown")
            person["cloth_type"] = cached.get("cloth_type", "unknown")
            person["cloth_items"] = cached.get("cloth_items", [])
            person["cloth_summary"] = cached.get("cloth_summary", "unknown")
            if cached.get("clothing_summary"):
                person["clothing_summary"] = cached["clothing_summary"]
            applied = True
        return applied

    def _set_unknown_clothing(self, people):
        for person in people:
            person.setdefault("cloth_color", "unknown")
            person.setdefault("cloth_type", "unknown")
            person.setdefault("cloth_items", [])
            person.setdefault("cloth_summary", "unknown")

    @staticmethod
    def _known_cloth_text(value):
        if value is None:
            return False
        text = str(value).strip().lower()
        return bool(text) and text not in {"unknown", "unknown clothing", "none", "null", "n/a"}

    def _person_has_known_clothing(self, person):
        for key in ("clothing_summary", "cloth_summary", "clothing"):
            if self._known_cloth_text(person.get(key)):
                return True
        color = person.get("cloth_color")
        cloth_type = person.get("cloth_type")
        if self._known_cloth_text(color) and self._known_cloth_text(cloth_type):
            return True
        for item in person.get("cloth_items") or []:
            if not isinstance(item, dict):
                continue
            if self._known_cloth_text(item.get("type") or item.get("category")):
                return True
        return False

    def _select_best_person(self, people):
        if not people:
            return None, "not_found"
        depth_people = [person for person in people if person.get("depth_m") is not None]
        clothing_people = [person for person in people if self._person_has_known_clothing(person)]
        if clothing_people:
            depth_clothing_people = [
                person for person in depth_people if self._person_has_known_clothing(person)
            ]
            if depth_clothing_people:
                return (
                    min(
                        depth_clothing_people,
                        key=lambda person: (person["depth_m"], -person.get("bbox_area_px", 0)),
                    ),
                    "nearest_depth_with_clothing",
                )
            return (
                max(clothing_people, key=lambda person: person.get("bbox_area_px", 0)),
                "largest_bbox_with_clothing",
            )
        if depth_people:
            return (
                min(depth_people, key=lambda person: person["depth_m"]),
                "nearest_depth",
            )
        return (
            max(people, key=lambda person: person.get("bbox_area_px", 0), default=None),
            "largest_bbox_fallback",
        )

    def _attach_clothing_to_people(self, frame, people):
        if not people:
            return {"enabled": bool(self.args.enable_clothing), "available": self.cloth_available}
        if not self.args.enable_clothing:
            self._set_unknown_clothing(people)
            return {"enabled": False, "available": False}
        if not self.cloth_available:
            self._set_unknown_clothing(people)
            return {"enabled": True, "available": False, "error": self.cloth_error}

        now = time.time()
        if (
            self.args.cloth_every > 0.0
            and self.last_clothing_time > 0.0
            and now - self.last_clothing_time < self.args.cloth_every
            and self._apply_cached_clothing(people)
        ):
            self._set_unknown_clothing(people)
            return {
                "enabled": True,
                "available": True,
                "cached": True,
                "cache_age_sec": round(now - self.last_clothing_time, 2),
            }

        stage_ms = {}
        start = time.perf_counter()
        seg_detections = self._run_cloth_model(self.cloth_model, frame, "cloth_seg")
        detect_detections = self._run_cloth_model(self.cloth_detect_model, frame, "cloth_detect")
        cloth_detections = self._fuse_fashion_detections(
            seg_detections,
            detect_detections,
            iou_threshold=self.args.cloth_fusion_iou,
        )
        stage_ms["cloth_detect_ms"] = round((time.perf_counter() - start) * 1000.0, 1)

        associate_start = time.perf_counter()
        detections = self._renumber_detections(people + cloth_detections)
        self._associate_clothing(
            detections,
            frame,
            color_cache=self.cloth_color_cache,
            max_color_cache_size=self.args.cloth_color_cache_size,
            perf=stage_ms,
        )
        stage_ms["cloth_association_ms"] = round(
            (time.perf_counter() - associate_start) * 1000.0,
            1,
        )

        for person in people:
            person.setdefault("cloth_color", "unknown")
            person.setdefault("cloth_type", "unknown")
            person["cloth_items"] = self._public_cloth_items(person.get("cloth_items"))
            person.setdefault("cloth_summary", self._cloth_summary_from_items(person.get("cloth_items")))
            if person.get("cloth_summary") and person.get("cloth_summary") != "unknown":
                person["clothing_summary"] = person["cloth_summary"]
        self._cache_current_clothing(people)
        return {
            "enabled": True,
            "available": True,
            "cloth_detection_count": len(cloth_detections),
            "stage_ms": stage_ms,
        }

    def refresh_people(self, frame, depth_image):
        if not self.output_allowed("people", self.args.people_output_interval):
            return
        now = time.time()
        results = self.object_model.predict(
            source=frame,
            conf=self.args.people_conf,
            iou=self.args.iou,
            device=self.args.device,
            classes=self.person_class_ids,
            verbose=False,
        )
        image_height, image_width = depth_image.shape[:2]
        people = []
        for person_index, box in enumerate(results[0].boxes, start=1):
            bbox = tuple(map(int, box.xyxy[0].cpu().tolist()))
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            depth_m = self.person_depth(depth_image, bbox)
            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
            position = self.point_position(
                center_x,
                center_y,
                depth_m,
                self.empty_intrinsic_matrix,
                self.empty_extrinsic_matrix,
                self.args.people_coordinate_frame,
            )
            camera_xyz = position.get("camera_xyz_m") if position else None
            people.append(
                {
                    "person_index": person_index,
                    "class_name": "person",
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": list(bbox),
                    "bbox_center_px": [round(center_x, 1), round(center_y, 1)],
                    "horizontal_offset_px": round(center_x - image_width / 2.0, 1),
                    "horizontal_offset_ratio": round(
                        (center_x - image_width / 2.0) / max(image_width / 2.0, 1.0),
                        3,
                    ),
                    "bbox_area_px": int(bbox_area),
                    "depth_m": None if depth_m is None else round(depth_m, 3),
                    "position": position,
                    "position_3d": camera_xyz,
                    "position_3d_frame": "camera_color_optical",
                    "frame_id": self.args.people_coordinate_frame,
                    "xyz_m": (
                        position.get("target_frame_xyz_m")
                        if position
                        else None
                    ),
                    "camera_xyz_m": camera_xyz,
                    "target_frame_xyz_m": (
                        position.get("target_frame_xyz_m")
                        if position
                        else None
                    ),
                    "calibrated_xyz_m": (
                        position.get("calibrated_xyz_m")
                        if position
                        else None
                    ),
                    "leftbase_xyz_m": (
                        position.get("leftbase_xyz_m")
                        if position
                        else None
                    ),
                    "left_arm_base_xyz_m": (
                        position.get("left_arm_base_xyz_m")
                        if position
                        else None
                    ),
                }
            )

        track_ids = self.person_tracker.assign(people)
        for person, track_id in zip(people, track_ids):
            person["track_id"] = track_id

        clothing_status = self._attach_clothing_to_people(frame, people)
        best_person, selection_method = self._select_best_person(people)
        payload = {
            "status": "success" if best_person else "not_found",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "coordinate_frame": self.args.people_coordinate_frame,
            "best_person": best_person,
            "people": people,
            "person_count": len(people),
            "selection_method": selection_method,
            "tracking": {
                "method": "iou",
                "iou_threshold": float(self.args.people_track_iou_threshold),
                "max_missing_frames": int(self.args.people_track_max_missing_frames),
                "missing_track_ids": list(self.person_tracker.missing_track_ids),
            },
            "clothing": clothing_status,
            "timestamp": round(now, 3),
        }
        self.latest_payloads["people"] = payload
        atomic_write_json(self.args.people_output_json, payload)
        print("[People] " + json.dumps(payload, ensure_ascii=False), flush=True)

    def refresh_empty_seat(self, frame, depth_image):
        if not self.output_allowed("empty_seat", self.args.empty_output_interval):
            return
        now = time.time()
        results = self.object_model.predict(
            source=frame,
            conf=self.args.empty_conf,
            iou=self.args.iou,
            device=self.args.device,
            classes=self.empty_predict_class_ids,
            verbose=False,
        )
        person_boxes = []
        seats = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.object_model.names[class_id]
            confidence = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0].cpu().tolist()))
            if class_name == "person":
                person_boxes.append(bbox)
            elif class_name in self.args.seat_classes:
                seats.append((class_name, confidence, bbox))

        empty_seats = []
        for class_name, confidence, seat_box in seats:
            occupied, seat_depth, _depth_match = self.seat_occupancy(
                depth_image, seat_box, person_boxes
            )
            if occupied:
                continue
            x1, y1, x2, y2 = seat_box
            pixel_x = (x1 + x2) / 2.0
            pixel_y = y1 + (y2 - y1) * self.args.seat_point_height_ratio
            empty_seats.append(
                {
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": list(seat_box),
                    "depth_m": None if seat_depth is None else round(seat_depth, 3),
                    "position": self.point_position(
                        pixel_x,
                        pixel_y,
                        seat_depth,
                        self.empty_intrinsic_matrix,
                        self.empty_extrinsic_matrix,
                        self.args.empty_coordinate_frame,
                    ),
                }
            )

        valid_seats = [seat for seat in empty_seats if seat["position"] is not None]
        best_seat = min(valid_seats, key=lambda seat: seat["depth_m"], default=None)
        payload = {
            "status": "success" if best_seat else "not_found",
            "coordinate_frame": self.args.empty_coordinate_frame,
            "best_empty_seat": best_seat,
            "empty_seats": valid_seats,
            "empty_seat_count": len(valid_seats),
            "timestamp": round(now, 3),
        }
        self.latest_payloads["empty_seat"] = payload
        atomic_write_json(self.args.empty_output_json, payload)
        print("[EmptySeat] " + json.dumps(payload, ensure_ascii=False), flush=True)

    def refresh_bag(self, frame, depth_image):
        if not self.output_allowed("bag", self.args.bag_output_interval):
            return
        now = time.time()
        results = self.object_model.predict(
            source=frame,
            conf=self.args.bag_conf,
            iou=self.args.iou,
            device=self.args.device,
            classes=None,
            verbose=False,
        )
        bags = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.object_model.names[class_id]
            if class_name not in self.args.bag_classes:
                continue
            confidence = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0].cpu().tolist()))
            depth_m = self.median_depth(
                depth_image,
                bbox,
                self.args.bag_min_depth_samples,
                crop_ratio=self.args.bag_depth_crop_ratio,
            )
            x1, y1, x2, y2 = bbox
            pixel_x = x1 + (x2 - x1) * self.args.bag_point_x_ratio
            pixel_y = y1 + (y2 - y1) * self.args.bag_point_y_ratio
            bags.append(
                {
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": list(bbox),
                    "depth_m": None if depth_m is None else round(depth_m, 3),
                    "position": self.point_position(
                        pixel_x,
                        pixel_y,
                        depth_m,
                        self.bag_intrinsic_matrix,
                        self.bag_extrinsic_matrix,
                        self.args.bag_coordinate_frame,
                    ),
                }
            )
        valid_bags = [bag for bag in bags if bag["position"] is not None]
        if self.args.bag_select == "confidence":
            best_bag = max(valid_bags, key=lambda bag: bag["confidence"], default=None)
        else:
            best_bag = min(valid_bags, key=lambda bag: bag["depth_m"], default=None)
        payload = {
            "status": "success" if best_bag else "not_found",
            "coordinate_frame": self.args.bag_coordinate_frame,
            "best_bag": best_bag,
            "bags": valid_bags,
            "bag_count": len(valid_bags),
            "timestamp": round(now, 3),
        }
        self.latest_payloads["bag"] = payload
        atomic_write_json(self.args.bag_output_json, payload)
        print("[Bag] " + json.dumps(payload, ensure_ascii=False), flush=True)

    def refresh_handover_hand(self, frame, depth_image):
        if self.pose_model is None:
            return
        if not self.output_allowed("handover_hand", self.args.hand_output_interval):
            return
        now = time.time()
        objects = self.detect_handover_objects_for_hand(frame, depth_image)
        results = self.pose_model.predict(
            source=frame,
            conf=self.args.hand_conf,
            iou=self.args.iou,
            device=self.args.device,
            verbose=False,
        )
        result = results[0]
        hands = []
        boxes = result.boxes
        keypoints = result.keypoints
        if keypoints is not None and keypoints.data is not None:
            keypoint_data = keypoints.data.cpu().numpy()
            box_data = (
                boxes.xyxy.cpu().numpy()
                if boxes is not None and boxes.xyxy is not None
                else []
            )
            box_conf = (
                boxes.conf.cpu().numpy()
                if boxes is not None and boxes.conf is not None
                else []
            )
            allowed_sides = (
                {"left", "right"}
                if self.args.hand_side == "both"
                else {self.args.hand_side}
            )
            for person_index, person_keypoints in enumerate(keypoint_data):
                person_bbox = None
                person_confidence = None
                if person_index < len(box_data):
                    person_bbox = [round(float(value), 1) for value in box_data[person_index].tolist()]
                    person_confidence = (
                        round(float(box_conf[person_index]), 3)
                        if person_index < len(box_conf)
                        else None
                    )
                for side, keypoint_index in WRIST_KEYPOINTS.items():
                    if side not in allowed_sides or keypoint_index >= len(person_keypoints):
                        continue
                    keypoint = person_keypoints[keypoint_index]
                    pixel_x, pixel_y = float(keypoint[0]), float(keypoint[1])
                    confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
                    if confidence < self.args.keypoint_conf:
                        continue
                    if pixel_x <= 0.0 or pixel_y <= 0.0:
                        continue
                    bag_objects = [obj for obj in objects if obj.get("object_role") == "bag"]
                    carry_objects = [obj for obj in objects if obj.get("object_role") == "object"]
                    object_match = self.associate_hand_with_object(pixel_x, pixel_y, bag_objects)
                    if object_match is None:
                        object_match = self.associate_hand_with_object(pixel_x, pixel_y, carry_objects)
                    depth_m = self.median_depth_around_pixel(depth_image, pixel_x, pixel_y)
                    hands.append(
                        self.make_handover_hand_record(
                            person_index,
                            side,
                            confidence,
                            person_confidence,
                            person_bbox,
                            pixel_x,
                            pixel_y,
                            depth_m,
                            object_match,
                        )
                    )
        valid_hands = [hand for hand in hands if hand["position"] is not None]
        valid_objects = [obj for obj in objects if obj.get("position") is not None]
        best_target = self.select_handover_target(valid_hands, valid_objects)
        payload = {
            "status": "success" if best_target else "not_found",
            "coordinate_frame": self.args.hand_coordinate_frame,
            "best_hand": best_target,
            "handover_target": best_target,
            "hands": valid_hands,
            "objects": valid_objects,
            "bags": [obj for obj in valid_objects if obj.get("object_role") == "bag"],
            "handover_hand_count": len(valid_hands),
            "handover_target_count": len(valid_hands)
            + len([obj for obj in valid_objects if obj.get("object_role") == "bag"]),
            "selection_order": [
                "hand_with_bag",
                "bag",
                "hand_with_object",
                "nearest_wrist",
            ],
            "timestamp": round(now, 3),
        }
        self.latest_payloads["handover_hand"] = payload
        atomic_write_json(self.args.hand_output_json, payload)
        print("[Hand] " + json.dumps(payload, ensure_ascii=False), flush=True)

    @staticmethod
    def _bbox_from_item(item, key="bbox"):
        if not isinstance(item, dict):
            return None
        bbox = item.get(key)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            return [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _short_text(text, limit=80):
        text = str(text or "").strip()
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _draw_label(self, image, text, origin, color, scale=0.5):
        cv2 = self.cv2
        if cv2 is None or not text:
            return
        x, y = int(origin[0]), int(origin[1])
        y = max(16, y)
        text = self._short_text(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            image,
            (max(0, x), max(0, y - height - baseline - 4)),
            (max(0, x + width + 4), max(0, y + baseline)),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (max(0, x + 2), max(height + 2, y - 3)),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_box(self, image, bbox, color, label="", thickness=2):
        cv2 = self.cv2
        if cv2 is None or bbox is None:
            return
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if label:
            self._draw_label(image, label, (x1, y1 - 4), color)

    def _draw_people(self, image):
        payload = self.latest_payloads.get("people") or {}
        people = payload.get("people") or []
        best = payload.get("best_person") or {}
        best_bbox = best.get("bbox")
        for index, person in enumerate(people, start=1):
            bbox = self._bbox_from_item(person)
            if bbox is None:
                continue
            is_best = person.get("bbox") == best_bbox
            summary = (
                person.get("clothing_summary")
                or person.get("cloth_summary")
                or person.get("cloth_type")
                or "person"
            )
            depth = person.get("depth_m")
            depth_text = f" {depth:.2f}m" if isinstance(depth, (int, float)) else ""
            track_id = person.get("track_id")
            id_text = f" ID:{track_id}" if track_id is not None else ""
            label = f"P{person.get('person_index') or index}{id_text}{depth_text}: {summary}"
            color = (0, 255, 0) if is_best else (80, 220, 80)
            self._draw_box(image, bbox, color, label, thickness=3 if is_best else 2)
            for item in person.get("cloth_items") or []:
                item_bbox = self._bbox_from_item(item)
                if item_bbox is None:
                    continue
                item_label = f"{item.get('color', '')} {item.get('type', '')}".strip()
                self._draw_box(image, item_bbox, (255, 180, 0), item_label, thickness=1)

    def _draw_empty_seats(self, image):
        payload = self.latest_payloads.get("empty_seat") or {}
        seats = payload.get("empty_seats") or []
        best = payload.get("best_empty_seat") or {}
        best_bbox = best.get("bbox")
        for index, seat in enumerate(seats, start=1):
            bbox = self._bbox_from_item(seat)
            if bbox is None:
                continue
            is_best = seat.get("bbox") == best_bbox
            depth = seat.get("depth_m")
            depth_text = f" {depth:.2f}m" if isinstance(depth, (int, float)) else ""
            label = f"empty {seat.get('class_name', 'seat')}#{index}{depth_text}"
            color = (255, 255, 0) if is_best else (200, 200, 0)
            self._draw_box(image, bbox, color, label, thickness=3 if is_best else 2)

    def _draw_bags(self, image):
        payload = self.latest_payloads.get("bag") or {}
        bags = payload.get("bags") or []
        best = payload.get("best_bag") or {}
        best_bbox = best.get("bbox")
        for index, bag in enumerate(bags, start=1):
            bbox = self._bbox_from_item(bag)
            if bbox is None:
                continue
            is_best = bag.get("bbox") == best_bbox
            depth = bag.get("depth_m")
            depth_text = f" {depth:.2f}m" if isinstance(depth, (int, float)) else ""
            label = f"bag#{index} {bag.get('class_name', '')}{depth_text}"
            color = (0, 160, 255) if is_best else (0, 120, 220)
            self._draw_box(image, bbox, color, label, thickness=3 if is_best else 2)

    def _draw_hands(self, image):
        cv2 = self.cv2
        if cv2 is None:
            return
        payload = self.latest_payloads.get("handover_hand") or {}
        hands = payload.get("hands") or []
        best = payload.get("best_hand") or {}
        best_bbox = best.get("person_bbox")
        best_object_bbox = best.get("object_bbox") or best.get("bag_bbox")
        for index, hand in enumerate(hands, start=1):
            bbox = self._bbox_from_item(hand, key="person_bbox")
            if bbox is not None:
                is_best = hand.get("person_bbox") == best_bbox
                label = f"{hand.get('target_type', 'hand')}#{index} {hand.get('hand_side', '')}"
                self._draw_box(
                    image,
                    bbox,
                    (255, 0, 255) if is_best else (220, 0, 220),
                    label,
                    thickness=2,
                )
            object_bbox = self._bbox_from_item(hand, key="object_bbox")
            if object_bbox is None:
                object_bbox = self._bbox_from_item(hand, key="bag_bbox")
            if object_bbox is not None:
                role = hand.get("matched_object_role") or "object"
                color = (0, 0, 255) if role == "bag" else (0, 180, 255)
                self._draw_box(image, object_bbox, color, f"handover {role}", thickness=1)
            position = hand.get("position") or {}
            pixel = position.get("pixel")
            if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
                px, py = int(round(float(pixel[0]))), int(round(float(pixel[1])))
                cv2.circle(image, (px, py), 6, (255, 0, 255), -1)
        if best.get("target_type") == "bag" and best_object_bbox is not None:
            self._draw_box(image, best_object_bbox, (0, 0, 255), "best bag fallback", thickness=3)

    def _draw_status_overlay(self, image):
        cv2 = self.cv2
        if cv2 is None:
            return
        lines = [
            f"frame={self.frame_count} camera={self.device_name or 'unknown'} serial={self.serial_number or 'auto'}",
            f"people={self.latest_payloads.get('people', {}).get('person_count', 0)} "
            f"empty={self.latest_payloads.get('empty_seat', {}).get('empty_seat_count', 0)} "
            f"bags={self.latest_payloads.get('bag', {}).get('bag_count', 0)} "
            f"hands={self.latest_payloads.get('handover_hand', {}).get('handover_hand_count', 0)}",
            "green=person, blue/yellow=clothes/empty seat, orange=bag, magenta=handover hand",
        ]
        y = 20
        for line in lines:
            cv2.putText(
                image,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 20

    def render_debug_windows(self, frame, depth_image):
        if not self.display_enabled or self.cv2 is None:
            return
        cv2 = self.cv2
        try:
            view = frame.copy()
            self._draw_people(view)
            self._draw_empty_seats(view)
            self._draw_bags(view)
            self._draw_hands(view)
            self._draw_status_overlay(view)
            if self.args.display_scale and self.args.display_scale != 1.0:
                view = cv2.resize(
                    view,
                    None,
                    fx=self.args.display_scale,
                    fy=self.args.display_scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(self.args.display_window, view)

            if self.args.display_depth:
                depth_view = cv2.convertScaleAbs(depth_image, alpha=self.args.depth_alpha)
                depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_JET)
                if self.args.display_scale and self.args.display_scale != 1.0:
                    depth_view = cv2.resize(
                        depth_view,
                        None,
                        fx=self.args.display_scale,
                        fy=self.args.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(self.args.depth_window, depth_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[VisionCache] 收到 q/Esc，关闭可视化窗口但继续刷新 JSON", flush=True)
                self.display_enabled = False
                cv2.destroyWindow(self.args.display_window)
                if self.args.display_depth:
                    cv2.destroyWindow(self.args.depth_window)
        except Exception as exc:
            print(f"[VisionCache] 可视化绘制失败，关闭窗口但继续刷新 JSON: {exc}", flush=True)
            self.display_enabled = False

    def render_status_window(self, title, detail=""):
        if not self.display_enabled or self.cv2 is None:
            return
        cv2 = self.cv2
        try:
            height = max(240, int(self.args.height or 480))
            width = max(320, int(self.args.width or 640))
            image = np.zeros((height, width, 3), dtype=np.uint8)
            lines = [
                "Task1 Vision Debug",
                str(title),
            ]
            if detail:
                lines.extend(str(detail).splitlines())
            lines.extend(
                [
                    f"frame_count={self.frame_count}",
                    f"camera={self.device_name or 'unknown'} serial={self.serial_number or 'auto'}",
                    "If this stays here, check camera ownership or RealSense errors.",
                ]
            )
            y = 36
            for line in lines:
                cv2.putText(
                    image,
                    self._short_text(line, limit=95),
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += 28
            cv2.imshow(self.args.display_window, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[VisionCache] 收到 q/Esc，关闭可视化窗口但继续刷新 JSON", flush=True)
                self.display_enabled = False
                cv2.destroyWindow(self.args.display_window)
                if self.args.display_depth:
                    cv2.destroyWindow(self.args.depth_window)
        except Exception as exc:
            print(f"[VisionCache] 状态窗口刷新失败，关闭窗口但继续刷新 JSON: {exc}", flush=True)
            self.display_enabled = False

    def write_status(self, force=False, error=None, status=None):
        now = time.time()
        if not force and now - self.last_status_time < self.args.status_interval:
            return
        self.last_status_time = now
        payload = {
            "status": status or ("error" if error else "running"),
            "pid": os.getpid(),
            "timestamp": round(now, 3),
            "frame_count": self.frame_count,
            "consecutive_frame_errors": self.consecutive_frame_errors,
            "camera": {
                "device_name": self.device_name or "unknown",
                "serial_number": self.serial_number or "",
                "width": self.args.width,
                "height": self.args.height,
                "fps": self.args.fps,
                "frame_timeout_ms": self.args.frame_timeout_ms,
            },
            "enabled": {
                "empty_seat": bool(self.args.enable_empty_seat),
                "handover_hand": bool(self.args.enable_handover_hand),
                "bag": bool(self.args.enable_bag),
                "people": bool(self.args.enable_people),
                "clothing": bool(self.args.enable_clothing),
            },
            "display": {
                "requested": not bool(self.args.no_display),
                "enabled": bool(self.display_enabled),
                "window": self.args.display_window,
                "depth_window": self.args.depth_window if self.args.display_depth else "",
            },
            "outputs": self.outputs,
        }
        payload["clothing"] = {
            "available": bool(self.cloth_available),
            "seg_model": str(self.args.cloth_seg_model or ""),
            "detect_model": str(self.args.cloth_detect_model or ""),
        }
        if self.cloth_error:
            payload["clothing"]["error"] = self.cloth_error
        if error:
            payload["error"] = str(error)
        atomic_write_json(self.status_path, payload)

    def stop(self, *_args):
        self.running = False

    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

        print("=" * 60, flush=True)
        print("Task1 视觉缓存进程已启动", flush=True)
        print(f"  RealSense 设备: {self.device_name or 'unknown'}", flush=True)
        print(f"  RealSense 序列号: {self.serial_number or 'auto'}", flush=True)
        print(f"  请求格式: {self.args.width}x{self.args.height} @ {self.args.fps} FPS", flush=True)
        print(f"  物体模型: {self.args.object_model}", flush=True)
        print(f"  姿态模型: {self.args.pose_model}", flush=True)
        if self.args.enable_clothing:
            print(f"  衣着识别: {'available' if self.cloth_available else 'unavailable'}", flush=True)
            print(f"  衣着分割模型: {self.args.cloth_seg_model}", flush=True)
            if self.args.cloth_detect_model:
                print(f"  衣着检测模型: {self.args.cloth_detect_model}", flush=True)
            if self.cloth_error:
                print(f"  衣着识别错误: {self.cloth_error}", flush=True)
        print(f"  状态文件: {self.status_path}", flush=True)
        print("=" * 60, flush=True)
        self.write_status(force=True)

        failed = False
        try:
            while self.running:
                if self.pipeline is None or self.align is None:
                    self.render_status_window(
                        "Camera pipeline is not available",
                        "Retrying RealSense pipeline start...",
                    )
                    if not self._restart_camera_pipeline():
                        time.sleep(max(0.1, self.args.frame_error_restart_sleep))
                    continue
                try:
                    frames = self.pipeline.wait_for_frames(self.args.frame_timeout_ms)
                    aligned_frames = self.align.process(frames)
                except RuntimeError as exc:
                    self.consecutive_frame_errors += 1
                    print(f"[VisionCache] 跳过异常相机帧: {exc}", flush=True)
                    self.render_status_window(
                        "No RealSense frame",
                        f"{exc}\nconsecutive_errors={self.consecutive_frame_errors}",
                    )
                    if (
                        self.args.frame_error_restart_threshold > 0
                        and self.consecutive_frame_errors >= self.args.frame_error_restart_threshold
                    ):
                        self._restart_camera_pipeline()
                    time.sleep(0.05)
                    continue
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    self.render_status_window(
                        "Missing color/depth frame",
                        f"color={bool(color_frame)} depth={bool(depth_frame)}",
                    )
                    time.sleep(0.05)
                    continue
                frame = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                self.frame_count += 1
                self.consecutive_frame_errors = 0
                now = time.time()

                self.render_debug_windows(frame, depth_image)
                if self.args.enable_empty_seat and self.should_run("empty_seat", self.args.empty_every):
                    self.refresh_empty_seat(frame, depth_image)
                    self.render_debug_windows(frame, depth_image)
                if self.args.enable_handover_hand and self.should_run("handover_hand", self.args.hand_every):
                    self.refresh_handover_hand(frame, depth_image)
                    self.render_debug_windows(frame, depth_image)
                if self.args.enable_bag and self.should_run("bag", self.args.bag_every):
                    self.refresh_bag(frame, depth_image)
                    self.render_debug_windows(frame, depth_image)
                if self.args.enable_people and self.should_run("people", self.args.people_every):
                    self.refresh_people(frame, depth_image)
                    self.render_debug_windows(frame, depth_image)
                self.write_status()
                if self.args.log_interval > 0 and self.frame_count % self.args.log_interval == 0:
                    print(
                        f"[VisionCache] frame={self.frame_count} timestamp={now:.3f}",
                        flush=True,
                    )
        except Exception as exc:
            failed = True
            self.write_status(force=True, error=exc)
            raise
        finally:
            if not failed:
                self.write_status(force=True, status="stopped")
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception as exc:
                print(f"[VisionCache] 停止 RealSense pipeline 时出错: {exc}", flush=True)
            if self.cv2 is not None and not self.args.no_display:
                try:
                    self.cv2.destroyAllWindows()
                except Exception:
                    pass


def build_parser():
    parser = argparse.ArgumentParser(description="Task1 RealSense vision cache daemon")
    parser.add_argument("--camera-model", default=DEFAULT_CAMERA_MODEL)
    parser.add_argument("--serial-number", default=DEFAULT_REALSENSE_SERIAL)
    parser.add_argument("--width", type=int, default=env_int("TASK1_VISION_CACHE_WIDTH", 640))
    parser.add_argument("--height", type=int, default=env_int("TASK1_VISION_CACHE_HEIGHT", 480))
    parser.add_argument("--fps", type=int, default=env_int("TASK1_VISION_CACHE_FPS", 30))
    parser.add_argument("--device", default=os.environ.get("TASK1_VISION_CACHE_DEVICE") or None)
    parser.add_argument("--iou", type=float, default=env_float("TASK1_VISION_CACHE_IOU", 0.45))
    parser.add_argument("--object-model", default=os.environ.get("TASK1_VISION_CACHE_OBJECT_MODEL", DEFAULT_OBJECT_MODEL))
    parser.add_argument("--pose-model", default=os.environ.get("TASK1_VISION_CACHE_POSE_MODEL", DEFAULT_POSE_MODEL))
    parser.add_argument(
        "--display",
        dest="no_display",
        action="store_false",
        help="Enable OpenCV debug windows with camera and detection overlays",
    )
    parser.add_argument(
        "--no-display",
        dest="no_display",
        action="store_true",
        default=not env_flag("TASK1_VISION_CACHE_DISPLAY", False),
        help="Disable OpenCV debug windows",
    )
    parser.add_argument(
        "--display-window",
        default=os.environ.get("TASK1_VISION_CACHE_DISPLAY_WINDOW", "Task1 Vision Debug"),
    )
    parser.add_argument(
        "--depth-window",
        default=os.environ.get("TASK1_VISION_CACHE_DEPTH_WINDOW", "Task1 Vision Depth"),
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=env_float("TASK1_VISION_CACHE_DISPLAY_SCALE", 1.0),
    )
    parser.add_argument(
        "--display-depth",
        action="store_true",
        default=env_flag("TASK1_VISION_CACHE_DISPLAY_DEPTH", True),
        help="Show a depth debug window when display is enabled",
    )
    parser.add_argument(
        "--no-display-depth",
        dest="display_depth",
        action="store_false",
        help="Disable the depth debug window",
    )
    parser.add_argument(
        "--depth-alpha",
        type=float,
        default=env_float("TASK1_VISION_CACHE_DEPTH_ALPHA", 0.03),
    )
    parser.add_argument("--status-json", default=os.environ.get("TASK1_VISION_CACHE_STATUS_JSON", DEFAULT_STATUS_JSON))
    parser.add_argument("--status-interval", type=float, default=env_float("TASK1_VISION_CACHE_STATUS_INTERVAL", 1.0))
    parser.add_argument("--log-interval", type=int, default=env_int("TASK1_VISION_CACHE_LOG_INTERVAL", 90))
    parser.add_argument(
        "--frame-error-restart-threshold",
        type=int,
        default=env_int("TASK1_VISION_CACHE_FRAME_ERROR_RESTART_THRESHOLD", 3),
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=env_int("TASK1_VISION_CACHE_FRAME_TIMEOUT_MS", 500),
        help="RealSense wait_for_frames timeout; lower values keep debug windows responsive",
    )
    parser.add_argument(
        "--frame-error-restart-sleep",
        type=float,
        default=env_float("TASK1_VISION_CACHE_FRAME_ERROR_RESTART_SLEEP", 1.0),
    )

    parser.set_defaults(
        enable_empty_seat=env_flag("TASK1_VISION_CACHE_EMPTY_SEAT", True),
        enable_handover_hand=env_flag("TASK1_VISION_CACHE_HANDOVER_HAND", True),
        enable_bag=env_flag("TASK1_VISION_CACHE_BAG", True),
        enable_people=env_flag("TASK1_VISION_CACHE_PEOPLE", True),
        enable_clothing=env_flag("TASK1_VISION_CACHE_CLOTHING", True),
    )
    parser.add_argument("--disable-empty-seat", dest="enable_empty_seat", action="store_false")
    parser.add_argument("--disable-handover-hand", dest="enable_handover_hand", action="store_false")
    parser.add_argument("--disable-bag", dest="enable_bag", action="store_false")
    parser.add_argument("--disable-people", dest="enable_people", action="store_false")
    parser.add_argument("--disable-clothing", dest="enable_clothing", action="store_false")

    parser.add_argument("--empty-every", type=float, default=env_float("TASK1_VISION_CACHE_EMPTY_EVERY", 0.6))
    parser.add_argument("--hand-every", type=float, default=env_float("TASK1_VISION_CACHE_HAND_EVERY", 1.0))
    parser.add_argument("--bag-every", type=float, default=env_float("TASK1_VISION_CACHE_BAG_EVERY", 1.0))
    parser.add_argument("--people-every", type=float, default=env_float("TASK1_VISION_CACHE_PEOPLE_EVERY", 0.5))
    parser.add_argument(
        "--cade-vision-src",
        default=os.environ.get("TASK1_CADE_VISION_SRC", str(DEFAULT_CADE_VISION_SRC)),
    )
    parser.add_argument(
        "--cloth-seg-model",
        default=os.environ.get("TASK1_CLOTH_SEG_MODEL", str(DEFAULT_CLOTH_SEG_MODEL)),
    )
    parser.add_argument(
        "--cloth-detect-model",
        default=os.environ.get("TASK1_CLOTH_DETECT_MODEL", ""),
    )
    parser.add_argument("--cloth-conf", type=float, default=env_float("TASK1_CLOTH_CONF", 0.25))
    parser.add_argument("--cloth-iou", type=float, default=env_float("TASK1_CLOTH_IOU", 0.45))
    parser.add_argument("--cloth-fusion-iou", type=float, default=env_float("TASK1_CLOTH_FUSION_IOU", 0.5))
    parser.add_argument("--cloth-every", type=float, default=env_float("TASK1_CLOTH_EVERY", 2.0))
    parser.add_argument("--cloth-cache-match-iou", type=float, default=env_float("TASK1_CLOTH_CACHE_MATCH_IOU", 0.2))
    parser.add_argument("--cloth-color-cache-size", type=int, default=env_int("TASK1_CLOTH_COLOR_CACHE_SIZE", 256))

    parser.add_argument("--empty-output-json", default=os.environ.get("TASK1_EMPTY_SEAT_OUTPUT_JSON", DEFAULT_EMPTY_OUTPUT_JSON))
    parser.add_argument("--hand-output-json", default=os.environ.get("TASK1_HANDOVER_HAND_OUTPUT_JSON", DEFAULT_HAND_OUTPUT_JSON))
    parser.add_argument("--bag-output-json", default=os.environ.get("TASK1_BAG_OUTPUT_JSON", DEFAULT_BAG_OUTPUT_JSON))
    parser.add_argument("--people-output-json", default=os.environ.get("TASK1_PEOPLE_OUTPUT_JSON", DEFAULT_PEOPLE_OUTPUT_JSON))
    parser.add_argument("--people-coordinate-frame", default=os.environ.get("TASK1_PEOPLE_COORDINATE_FRAME", DEFAULT_COORDINATE_FRAME))

    parser.add_argument("--seat-classes", nargs="+", default=env_words("TASK1_EMPTY_SEAT_CLASSES", ["chair", "couch", "bench"]))
    parser.add_argument("--bag-classes", nargs="+", default=env_words("TASK1_BAG_CLASSES", ["backpack", "handbag", "suitcase"]))
    parser.add_argument(
        "--carry-object-classes",
        nargs="+",
        default=env_classes("TASK1_HAND_SEARCH_CARRY_OBJECT_CLASSES", DEFAULT_CARRY_OBJECT_CLASSES),
        help="Non-bag YOLO classes that can still indicate a hand holding something",
    )
    parser.add_argument("--min-depth", type=float, default=env_float("TASK1_VISION_CACHE_MIN_DEPTH", 0.1))
    parser.add_argument("--max-depth", type=float, default=env_float("TASK1_VISION_CACHE_MAX_DEPTH", 5.0))

    parser.add_argument("--empty-conf", type=float, default=env_float("TASK1_EMPTY_SEAT_CONF", 0.5))
    parser.add_argument("--empty-intrinsic", default=os.environ.get("TASK1_EMPTY_SEAT_INTRINSIC", ""))
    parser.add_argument("--empty-extrinsic", default=os.environ.get("TASK1_EMPTY_SEAT_EXTRINSIC", DEFAULT_EXTRINSIC))
    parser.add_argument("--empty-coordinate-frame", default=os.environ.get("TASK1_EMPTY_SEAT_COORDINATE_FRAME", DEFAULT_COORDINATE_FRAME))
    parser.add_argument("--empty-output-interval", type=float, default=env_float("TASK1_EMPTY_SEAT_OUTPUT_INTERVAL", 0.5))
    parser.add_argument("--seat-point-height-ratio", type=float, default=env_float("TASK1_EMPTY_SEAT_POINT_HEIGHT_RATIO", 0.6))
    parser.add_argument("--occupancy-threshold", type=float, default=env_float("TASK1_EMPTY_SEAT_OCCUPANCY_THRESHOLD", 0.2))
    parser.add_argument("--max-occupancy-depth-gap", type=float, default=env_float("TASK1_EMPTY_SEAT_MAX_OCCUPANCY_DEPTH_GAP", 0.5))
    parser.add_argument("--empty-min-depth-samples", type=int, default=env_int("TASK1_EMPTY_SEAT_MIN_DEPTH_SAMPLES", 30))
    parser.add_argument("--people-track-iou-threshold", type=float, default=env_float("TASK1_PEOPLE_TRACK_IOU_THRESHOLD", 0.25))
    parser.add_argument("--people-track-max-missing-frames", type=int, default=env_int("TASK1_PEOPLE_TRACK_MAX_MISSING_FRAMES", 15))

    parser.add_argument("--hand-conf", type=float, default=env_float("TASK1_HAND_SEARCH_CONF", 0.25))
    parser.add_argument("--hand-intrinsic", default=os.environ.get("TASK1_HAND_SEARCH_INTRINSIC", ""))
    parser.add_argument("--hand-extrinsic", default=os.environ.get("TASK1_HAND_SEARCH_EXTRINSIC", DEFAULT_EXTRINSIC))
    parser.add_argument("--hand-coordinate-frame", default=os.environ.get("TASK1_HAND_SEARCH_COORDINATE_FRAME", DEFAULT_COORDINATE_FRAME))
    parser.add_argument("--hand-output-interval", type=float, default=env_float("TASK1_HAND_SEARCH_OUTPUT_INTERVAL", 0.5))
    parser.add_argument("--hand-side", choices=("left", "right", "both"), default=os.environ.get("TASK1_HAND_SEARCH_SIDE", "both"))
    parser.add_argument("--hand-select", choices=("nearest", "confidence"), default=os.environ.get("TASK1_HAND_SEARCH_SELECT", "nearest"))
    parser.add_argument("--hand-depth-radius", type=int, default=env_int("TASK1_HAND_SEARCH_DEPTH_RADIUS", 10))
    parser.add_argument("--hand-depth-percentile", type=float, default=env_float("TASK1_HAND_SEARCH_DEPTH_PERCENTILE", 20.0))
    parser.add_argument("--keypoint-conf", type=float, default=env_float("TASK1_HAND_SEARCH_KEYPOINT_CONF", 0.20))
    parser.add_argument("--hand-bag-conf", type=float, default=env_float("TASK1_HAND_SEARCH_BAG_CONF", 0.10))
    parser.add_argument("--bag-box-expand-ratio", type=float, default=env_float("TASK1_HAND_SEARCH_BAG_BOX_EXPAND_RATIO", 0.45))
    parser.add_argument("--max-hand-bag-pixel-distance", type=float, default=env_float("TASK1_HAND_SEARCH_MAX_HAND_BAG_PIXEL_DISTANCE", 90.0))
    parser.add_argument("--hand-min-depth-samples", type=int, default=env_int("TASK1_HAND_SEARCH_MIN_DEPTH_SAMPLES", 10))
    parser.add_argument("--hand-object-point-x-ratio", type=float, default=env_float("TASK1_HAND_SEARCH_OBJECT_POINT_X_RATIO", 0.5))
    parser.add_argument("--hand-object-point-y-ratio", type=float, default=env_float("TASK1_HAND_SEARCH_OBJECT_POINT_Y_RATIO", 0.5))
    parser.add_argument("--hand-object-depth-crop-ratio", type=float, default=env_float("TASK1_HAND_SEARCH_OBJECT_DEPTH_CROP_RATIO", 0.2))
    parser.add_argument("--hand-object-min-depth-samples", type=int, default=env_int("TASK1_HAND_SEARCH_OBJECT_MIN_DEPTH_SAMPLES", 20))

    parser.add_argument("--bag-conf", type=float, default=env_float("TASK1_BAG_SEARCH_CONF", 0.15))
    parser.add_argument("--bag-intrinsic", default=os.environ.get("TASK1_BAG_SEARCH_INTRINSIC", ""))
    parser.add_argument("--bag-extrinsic", default=os.environ.get("TASK1_BAG_SEARCH_EXTRINSIC", DEFAULT_EXTRINSIC))
    parser.add_argument("--bag-coordinate-frame", default=os.environ.get("TASK1_BAG_SEARCH_COORDINATE_FRAME", DEFAULT_COORDINATE_FRAME))
    parser.add_argument("--bag-output-interval", type=float, default=env_float("TASK1_BAG_SEARCH_OUTPUT_INTERVAL", 0.5))
    parser.add_argument("--bag-select", choices=("nearest", "confidence"), default=os.environ.get("TASK1_BAG_SEARCH_SELECT", "nearest"))
    parser.add_argument("--bag-point-x-ratio", type=float, default=env_float("TASK1_BAG_SEARCH_POINT_X_RATIO", 0.5))
    parser.add_argument("--bag-point-y-ratio", type=float, default=env_float("TASK1_BAG_SEARCH_POINT_Y_RATIO", 0.5))
    parser.add_argument("--bag-depth-crop-ratio", type=float, default=env_float("TASK1_BAG_SEARCH_DEPTH_CROP_RATIO", 0.2))
    parser.add_argument("--bag-min-depth-samples", type=int, default=env_int("TASK1_BAG_SEARCH_MIN_DEPTH_SAMPLES", 20))

    parser.add_argument("--people-conf", type=float, default=env_float("TASK1_PEOPLE_SEARCH_CONF", 0.35))
    parser.add_argument("--people-output-interval", type=float, default=env_float("TASK1_PEOPLE_SEARCH_OUTPUT_INTERVAL", 0.5))
    parser.add_argument("--people-min-depth-samples", type=int, default=env_int("TASK1_PEOPLE_SEARCH_MIN_DEPTH_SAMPLES", 20))
    return parser


def main():
    args = build_parser().parse_args()
    daemon = VisionCacheDaemon(args)
    daemon.run()


if __name__ == "__main__":
    main()
