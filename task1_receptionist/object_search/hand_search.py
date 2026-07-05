#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find a guest hand/wrist associated with a detected bag for handover.
"""

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_POSE_MODEL = SCRIPT_DIR / "yolo11n-pose.pt"
LOCAL_BAG_MODEL = SCRIPT_DIR / "yolo11n.pt"
DEFAULT_MODEL = str(LOCAL_POSE_MODEL if LOCAL_POSE_MODEL.exists() else "yolo11n-pose.pt")
DEFAULT_BAG_MODEL = str(LOCAL_BAG_MODEL if LOCAL_BAG_MODEL.exists() else "yolo11n.pt")
DEFAULT_OUTPUT_JSON = str(SCRIPT_DIR / "latest_handover_hand.json")
DEFAULT_LEFTBASE_EXTRINSIC = SCRIPT_DIR / "camera_middle_to_leftbase.txt"
DEFAULT_CAMERA_MODEL = os.environ.get("REALSENSE_CAMERA_MODEL", "D455")
DEFAULT_REALSENSE_SERIAL = (
    os.environ.get("REALSENSE455_SERIAL")
    or os.environ.get("REALSENSE_SERIAL")
    or os.environ.get("REALSENSE515_SERIAL")
    or os.environ.get("CADE_REALSENSE_SERIAL")
    or ""
)
DEFAULT_INTRINSIC = os.environ.get("HAND_SEARCH_INTRINSIC", "")
DEFAULT_EXTRINSIC = os.environ.get(
    "HAND_SEARCH_EXTRINSIC",
    str(DEFAULT_LEFTBASE_EXTRINSIC) if DEFAULT_LEFTBASE_EXTRINSIC.exists() else "",
)
DEFAULT_COORDINATE_FRAME = os.environ.get(
    "HAND_SEARCH_COORDINATE_FRAME",
    "leftbase" if DEFAULT_EXTRINSIC else "camera_color_optical",
)
DEFAULT_CONF = float(os.environ.get("HAND_SEARCH_CONF", "0.25"))
DEFAULT_BAG_CONF = float(os.environ.get("HAND_SEARCH_BAG_CONF", "0.10"))
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

WRIST_KEYPOINTS = {
    "left": 9,
    "right": 10,
}


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


class RealSenseHandDetector:
    def __init__(self, args: argparse.Namespace):
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
        self.model = YOLO(args.model)
        self.bag_model = YOLO(args.bag_model)
        self.extrinsic_matrix = (
            load_matrix(args.extrinsic, (4, 4)) if args.extrinsic else np.eye(4)
        )
        self.last_output_time = 0.0
        self.last_annotated_frame = None

        self.pipeline = rs.pipeline()
        config = rs.config()
        self.serial_number = self.resolve_serial_number(args.serial_number, args.camera_model)
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        profile = self.pipeline.start(config)

        device = profile.get_device()
        self.device_name = get_device_info(device, rs.camera_info.name)
        self.serial_number = get_device_info(device, rs.camera_info.serial_number) or self.serial_number
        self.intrinsic_matrix = self.load_intrinsic_matrix(profile)
        self.align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        if not args.no_display:
            cv2.namedWindow("Hand Search", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth View", cv2.WINDOW_NORMAL)

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

    def load_intrinsic_matrix(self, profile):
        if self.args.intrinsic:
            return load_matrix(self.args.intrinsic, (3, 3))

        color_profile = profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        return np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

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
            depths = depths[(depths >= self.args.min_depth) & (depths <= self.args.max_depth)]
            if depths.size >= self.args.min_depth_samples:
                return float(np.percentile(depths, percentile))
        return None

    def median_depth_in_bbox(self, depth_image, bbox):
        height, width = depth_image.shape
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        box_w = x2 - x1
        box_h = y2 - y1
        crop_ratio = self.args.object_depth_crop_ratio
        if crop_ratio > 0.0:
            x1 = int(x1 + box_w * crop_ratio)
            x2 = int(x2 - box_w * crop_ratio)
            y1 = int(y1 + box_h * crop_ratio)
            y2 = int(y2 - box_h * crop_ratio)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return None

        depths = depth_image[y1:y2, x1:x2].astype(np.float32) * self.depth_scale
        depths = depths[(depths >= self.args.min_depth) & (depths <= self.args.max_depth)]
        if depths.size < self.args.object_min_depth_samples:
            return None
        return float(np.median(depths))

    def hand_position(self, pixel_x, pixel_y, depth_m):
        if depth_m is None:
            return None

        fx, fy = self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]
        cx, cy = self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]
        point_camera = np.array(
            [
                (pixel_x - cx) * depth_m / fx,
                (pixel_y - cy) * depth_m / fy,
                depth_m,
                1.0,
            ]
        )
        point_target = self.extrinsic_matrix @ point_camera
        target_xyz = [round(float(value), 3) for value in point_target[:3]]
        position = {
            "pixel": [round(float(pixel_x), 1), round(float(pixel_y), 1)],
            "camera_xyz_m": [round(float(value), 3) for value in point_camera[:3]],
            "target_frame_xyz_m": target_xyz,
            "calibrated_xyz_m": target_xyz,
        }
        if self.args.coordinate_frame in {"leftbase", "left_arm_base"}:
            position["leftbase_xyz_m"] = target_xyz
            position["left_arm_base_xyz_m"] = target_xyz
        return position

    def object_position(self, bbox, depth_m):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        pixel_x = x1 + (x2 - x1) * self.args.object_point_x_ratio
        pixel_y = y1 + (y2 - y1) * self.args.object_point_y_ratio
        return self.hand_position(pixel_x, pixel_y, depth_m)

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
            expanded_bbox = self.expand_bbox(
                obj["bbox"], self.args.bag_box_expand_ratio
            )
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

    def detect_handover_objects(self, frame, depth_image):
        results = self.bag_model.predict(
            source=frame,
            conf=self.args.bag_conf,
            iou=self.args.bag_iou,
            device=self.args.device,
            classes=None,
            verbose=False,
        )
        objects = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.bag_model.names[class_id]
            object_role = self.handover_object_role(class_name)
            if object_role is None:
                continue
            bbox = [float(value) for value in box.xyxy[0].cpu().tolist()]
            depth_m = self.median_depth_in_bbox(depth_image, bbox)
            objects.append(
                {
                    "class_name": class_name,
                    "object_role": object_role,
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(value, 1) for value in bbox],
                    "depth_m": None if depth_m is None else round(depth_m, 3),
                    "position": self.object_position(bbox, depth_m),
                }
            )
        return objects

    def make_hand_record(
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
        target_type = "nearest_wrist"
        selection_label = "nearest wrist"
        selection_priority = 4
        wrist_position = self.hand_position(pixel_x, pixel_y, depth_m)
        record = {
            "target_type": target_type,
            "selection_label": selection_label,
            "selection_priority": selection_priority,
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

    def make_bag_fallback_target(self, bag):
        target = {
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
        return target

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
        if self.args.select == "confidence":
            return (priority, -float(confidence), depth, distance)
        return (priority, depth, distance, -float(confidence))

    def select_handover_target(self, hands, objects):
        valid_hands = [hand for hand in hands if hand.get("position") is not None]
        bag_targets = [
            self.make_bag_fallback_target(obj)
            for obj in objects
            if obj.get("object_role") == "bag" and obj.get("position") is not None
        ]
        candidates = valid_hands + bag_targets
        return min(candidates, key=self.handover_sort_key, default=None)

    def output_best_hand(self, hands, objects):
        now = time.time()
        if now - self.last_output_time < self.args.output_interval:
            return None
        self.last_output_time = now

        valid_hands = [hand for hand in hands if hand.get("position") is not None]
        valid_objects = [obj for obj in objects if obj.get("position") is not None]
        best_target = self.select_handover_target(valid_hands, valid_objects)

        payload = {
            "status": "success" if best_target else "not_found",
            "coordinate_frame": self.args.coordinate_frame,
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
        with open(self.args.output_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("[Hand] " + json.dumps(payload, ensure_ascii=False), flush=True)
        return payload

    def process_frame(self, frame, depth_image):
        results = self.model.predict(
            source=frame,
            conf=self.args.conf,
            iou=self.args.iou,
            device=self.args.device,
            verbose=False,
        )

        result = results[0]
        annotated_frame = frame.copy()
        hands = []
        objects = self.detect_handover_objects(frame, depth_image)
        for obj in objects:
            x1, y1, x2, y2 = [int(round(value)) for value in obj["bbox"]]
            color = (255, 0, 255) if obj.get("object_role") == "bag" else (0, 180, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                f"{obj.get('object_role', 'object').upper()} {obj['class_name']} {obj['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        boxes = result.boxes
        keypoints = result.keypoints
        if keypoints is None or keypoints.data is None:
            self.output_best_hand(hands, objects)
            self.last_annotated_frame = annotated_frame.copy()
            return annotated_frame, len(
                [
                    obj
                    for obj in objects
                    if obj.get("object_role") == "bag" and obj.get("position") is not None
                ]
            )

        keypoint_data = keypoints.data.cpu().numpy()
        box_data = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else []
        box_conf = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else []
        allowed_sides = {"left", "right"} if self.args.hand_side == "both" else {self.args.hand_side}

        for person_index, person_keypoints in enumerate(keypoint_data):
            person_bbox = None
            person_confidence = None
            if person_index < len(box_data):
                person_bbox = [round(float(value), 1) for value in box_data[person_index].tolist()]
                x1, y1, x2, y2 = [int(round(value)) for value in person_bbox]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
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
                position = self.hand_position(pixel_x, pixel_y, depth_m)
                hand_record = self.make_hand_record(
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
                hand_record["position"] = position
                hands.append(hand_record)

                if hand_record["target_type"] == "hand_with_bag":
                    color = (0, 128, 255) if side == "left" else (255, 128, 0)
                    label = f"{side.upper()} WRIST+BAG {confidence:.2f}"
                elif hand_record["target_type"] == "hand_with_object":
                    color = (0, 180, 255)
                    label = f"{side.upper()} WRIST+OBJ {confidence:.2f}"
                else:
                    color = (128, 128, 128)
                    label = f"{side.upper()} WRIST {confidence:.2f}"
                cv2.circle(annotated_frame, (int(round(pixel_x)), int(round(pixel_y))), 6, color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(round(pixel_x)) + 8, max(20, int(round(pixel_y)) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        self.output_best_hand(hands, objects)
        self.last_annotated_frame = annotated_frame.copy()
        best_target = self.select_handover_target(hands, objects)
        if best_target:
            position = best_target["position"]
            x, y, z = (
                position.get("target_frame_xyz_m")
                or position.get("left_arm_base_xyz_m")
                or position["calibrated_xyz_m"]
            )
            cv2.putText(
                annotated_frame,
                (
                    f"BEST {best_target.get('target_type', 'target')} "
                    f"{self.args.coordinate_frame}: ({x:.2f}, {y:.2f}, {z:.2f})m"
                ),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 128, 255),
                2,
            )

        valid_target_count = len([hand for hand in hands if hand.get("position") is not None])
        valid_target_count += len(
            [
                obj
                for obj in objects
                if obj.get("object_role") == "bag" and obj.get("position") is not None
            ]
        )
        return annotated_frame, valid_target_count

    def run(self):
        print("=" * 60)
        print("拿包手腕识别已启动")
        print(f"  RealSense 设备: {self.device_name or 'unknown'}")
        print(f"  RealSense 序列号: {self.serial_number or 'auto'}")
        print(f"  请求格式: {self.args.width}x{self.args.height} @ {self.args.fps} FPS")
        print(f"  姿态模型: {self.args.model}")
        print(f"  包检测模型: {self.args.bag_model}")
        print(f"  手侧: {self.args.hand_side}")
        print(f"  包类别: {', '.join(self.args.bag_classes)}")
        print(f"  可携带物体类别: {', '.join(self.args.carry_object_classes)}")
        print(f"  包检测阈值: {self.args.bag_conf}")
        print(f"  手-包最大像素距离: {self.args.max_hand_bag_pixel_distance}")
        print(f"  关键点阈值: {self.args.keypoint_conf}")
        print(f"  内参来源: {self.args.intrinsic or 'active color stream'}")
        print(f"  外参来源: {self.args.extrinsic or 'identity(camera frame)'}")
        print(f"  外参坐标系: {self.args.coordinate_frame}")
        print(f"  坐标输出文件: {self.args.output_json}")
        if self.args.max_frames > 0:
            print(f"  最大处理帧数: {self.args.max_frames}")
        if self.args.no_display:
            print("  显示窗口: disabled")
        else:
            print("  按 q 退出")
        print("=" * 60)

        frame_count = 0
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    print("无法读取图像帧，程序退出")
                    break

                frame = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                start_time = time.time()
                annotated_frame, hand_count = self.process_frame(frame, depth_image)
                fps = 1.0 / max(time.time() - start_time, 1e-9)
                cv2.putText(
                    annotated_frame,
                    f"Hands+bags: {hand_count}  FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                depth_view = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                if not self.args.no_display:
                    cv2.imshow("Hand Search", annotated_frame)
                    cv2.imshow("Depth View", depth_view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if self.args.max_frames > 0 and frame_count >= self.args.max_frames:
                    break
        finally:
            if self.args.snapshot and self.last_annotated_frame is not None:
                cv2.imwrite(self.args.snapshot, self.last_annotated_frame)
                print(f"  已保存最后一帧截图: {self.args.snapshot}")
            self.pipeline.stop()
            if not self.args.no_display:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Find guest hand/wrist holding a bag with RealSense depth")
    parser.add_argument("--camera-model", default=DEFAULT_CAMERA_MODEL)
    parser.add_argument("--serial-number", default=DEFAULT_REALSENSE_SERIAL)
    parser.add_argument("--intrinsic", default=DEFAULT_INTRINSIC)
    parser.add_argument("--extrinsic", default=DEFAULT_EXTRINSIC)
    parser.add_argument("--coordinate-frame", default=DEFAULT_COORDINATE_FRAME)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-interval", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bag-model", default=DEFAULT_BAG_MODEL)
    parser.add_argument(
        "--bag-classes",
        nargs="+",
        default=["backpack", "handbag", "suitcase"],
        help="YOLO classes considered bags",
    )
    parser.add_argument(
        "--carry-object-classes",
        nargs="+",
        default=DEFAULT_CARRY_OBJECT_CLASSES,
        help="Non-bag YOLO classes that can still indicate a hand holding something",
    )
    parser.add_argument("--hand-side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--select", choices=("nearest", "confidence"), default="nearest")
    parser.add_argument("--hand-depth-radius", type=int, default=10)
    parser.add_argument(
        "--hand-depth-percentile",
        type=float,
        default=float(os.environ.get("TASK1_HAND_SEARCH_DEPTH_PERCENTILE", os.environ.get("HAND_SEARCH_DEPTH_PERCENTILE", "20"))),
        help="Depth percentile used around wrist keypoints; lower values reduce background depth leakage",
    )
    parser.add_argument("--keypoint-conf", type=float, default=0.20)
    parser.add_argument("--bag-conf", type=float, default=DEFAULT_BAG_CONF)
    parser.add_argument("--bag-iou", type=float, default=0.45)
    parser.add_argument("--bag-box-expand-ratio", type=float, default=0.45)
    parser.add_argument("--max-hand-bag-pixel-distance", type=float, default=90.0)
    parser.add_argument("--object-point-x-ratio", type=float, default=0.5)
    parser.add_argument("--object-point-y-ratio", type=float, default=0.5)
    parser.add_argument("--object-depth-crop-ratio", type=float, default=0.2)
    parser.add_argument("--object-min-depth-samples", type=int, default=20)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--min-depth-samples", type=int, default=10)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--snapshot", default="")
    args = parser.parse_args()

    detector = RealSenseHandDetector(args)
    detector.run()


if __name__ == "__main__":
    main()
