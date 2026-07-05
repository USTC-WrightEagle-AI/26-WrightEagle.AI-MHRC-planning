#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read aligned color and depth frames from a RealSense camera and find empty seats.

Examples:
    python object_search.py
    python object_search.py --occupancy-threshold 0.25
    python object_search.py --camera-model D455
    python object_search.py --serial-number 333422301212
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


DEFAULT_MODEL = str(Path(__file__).with_name("yolo11n.pt"))
DEFAULT_OUTPUT_JSON = str(Path(__file__).with_name("latest_empty_seat.json"))
DEFAULT_LEFTBASE_EXTRINSIC = Path(__file__).with_name("camera_middle_to_leftbase.txt")
DEFAULT_CAMERA_MODEL = os.environ.get("REALSENSE_CAMERA_MODEL", "D455")
DEFAULT_REALSENSE_SERIAL = (
    os.environ.get("REALSENSE455_SERIAL")
    or os.environ.get("REALSENSE_SERIAL")
    or os.environ.get("REALSENSE515_SERIAL")
    or os.environ.get("CADE_REALSENSE_SERIAL")
    or ""
)
DEFAULT_INTRINSIC = os.environ.get("OBJECT_SEARCH_INTRINSIC", "")
DEFAULT_EXTRINSIC = os.environ.get(
    "OBJECT_SEARCH_EXTRINSIC",
    str(DEFAULT_LEFTBASE_EXTRINSIC) if DEFAULT_LEFTBASE_EXTRINSIC.exists() else "",
)
DEFAULT_COORDINATE_FRAME = os.environ.get(
    "OBJECT_SEARCH_COORDINATE_FRAME",
    "leftbase" if DEFAULT_EXTRINSIC else "camera_color_optical",
)


def load_matrix(path, expected_shape):
    """Load a matrix from one-row-per-line calibration text formats."""
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


class RealSenseEmptySeatDetector:
    def __init__(self, args: argparse.Namespace):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics 依赖，请先运行: pip install ultralytics"
            ) from exc

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "缺少 pyrealsense2 依赖，请先运行: pip install pyrealsense2"
            ) from exc

        self.args = args
        self.rs = rs
        self.model = YOLO(args.model)
        self.predict_classes = self.resolve_predict_classes()
        self.extrinsic_matrix = (
            load_matrix(args.extrinsic, (4, 4)) if args.extrinsic else np.eye(4)
        )
        self.last_output_time = 0.0
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.serial_number = self.resolve_serial_number(
            args.serial_number, args.camera_model
        )
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(
            rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
        )
        config.enable_stream(
            rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
        )
        profile = self.pipeline.start(config)
        device = profile.get_device()
        self.device_name = get_device_info(device, rs.camera_info.name)
        self.serial_number = (
            get_device_info(device, rs.camera_info.serial_number) or self.serial_number
        )
        self.intrinsic_matrix = self.load_intrinsic_matrix(profile)
        self.align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        if not args.no_display:
            cv2.namedWindow("Empty Seat Search", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth View", cv2.WINDOW_NORMAL)

    def resolve_predict_classes(self):
        wanted = {"person", *self.args.seat_classes}
        return [
            class_id
            for class_id, class_name in self.model.names.items()
            if class_name in wanted
        ] or None

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
        raise RuntimeError(
            f"未找到型号包含 {camera_model!r} 的 RealSense 设备，已检测到: {detected}"
        )

    def load_intrinsic_matrix(self, profile):
        if self.args.intrinsic:
            return load_matrix(self.args.intrinsic, (3, 3))

        color_profile = profile.get_stream(
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

    @staticmethod
    def intersection_over_seat(person_box, seat_box):
        """Return the fraction of the seat box covered by a person box."""
        px1, py1, px2, py2 = person_box
        sx1, sy1, sx2, sy2 = seat_box
        intersection_width = max(0, min(px2, sx2) - max(px1, sx1))
        intersection_height = max(0, min(py2, sy2) - max(py1, sy1))
        intersection_area = intersection_width * intersection_height
        seat_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        return intersection_area / seat_area

    def median_depth(self, depth_image, bbox, excluded_boxes=()):
        """Estimate ROI depth in meters, optionally ignoring overlapping boxes."""
        height, width = depth_image.shape
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return None

        roi = depth_image[y1:y2, x1:x2]
        mask = np.ones(roi.shape, dtype=bool)
        for ex1, ey1, ex2, ey2 in excluded_boxes:
            ex1, ex2 = max(x1, ex1) - x1, min(x2, ex2) - x1
            ey1, ey2 = max(y1, ey1) - y1, min(y2, ey2) - y1
            if ex1 < ex2 and ey1 < ey2:
                mask[ey1:ey2, ex1:ex2] = False

        depths = roi[mask].astype(np.float32) * self.depth_scale
        depths = depths[
            (depths >= self.args.min_depth) & (depths <= self.args.max_depth)
        ]
        if depths.size < self.args.min_depth_samples:
            return None
        return float(np.median(depths))

    def person_depth(self, depth_image, person_box):
        """Use the center of a person box to reduce background contamination."""
        x1, y1, x2, y2 = person_box
        width = x2 - x1
        height = y2 - y1
        center_box = (
            int(x1 + width * 0.25),
            int(y1 + height * 0.2),
            int(x2 - width * 0.25),
            int(y2 - height * 0.15),
        )
        return self.median_depth(depth_image, center_box)

    def seat_occupancy(self, depth_image, seat_box, person_boxes):
        """Return occupancy state and the closest matching depth information."""
        seat_depth = self.median_depth(depth_image, seat_box, person_boxes)
        if seat_depth is None:
            seat_depth = self.median_depth(depth_image, seat_box)
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

    def seat_position(self, seat_box, seat_depth):
        """Estimate the seat center and transform it into the configured target frame."""
        if seat_depth is None:
            return None

        x1, y1, x2, y2 = seat_box
        pixel_x = (x1 + x2) / 2.0
        pixel_y = y1 + (y2 - y1) * self.args.seat_point_height_ratio
        fx, fy = self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]
        cx, cy = self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]

        point_camera = np.array(
            [
                (pixel_x - cx) * seat_depth / fx,
                (pixel_y - cy) * seat_depth / fy,
                seat_depth,
                1.0,
            ]
        )
        point_target = self.extrinsic_matrix @ point_camera
        target_xyz = [round(float(value), 3) for value in point_target[:3]]
        position = {
            "pixel": [round(pixel_x, 1), round(pixel_y, 1)],
            "camera_xyz_m": [round(float(value), 3) for value in point_camera[:3]],
            "target_frame_xyz_m": target_xyz,
            "calibrated_xyz_m": target_xyz,
        }
        if self.args.coordinate_frame in {"leftbase", "left_arm_base"}:
            position["leftbase_xyz_m"] = target_xyz
            position["left_arm_base_xyz_m"] = target_xyz
        return position

    def output_best_empty_seat(self, empty_seats):
        """Write the nearest empty seat for downstream pointing logic."""
        now = time.time()
        if now - self.last_output_time < self.args.output_interval:
            return
        self.last_output_time = now

        valid_seats = [seat for seat in empty_seats if seat["position"] is not None]
        best_seat = min(valid_seats, key=lambda seat: seat["depth_m"], default=None)
        payload = {
            "status": "success" if best_seat else "not_found",
            "coordinate_frame": self.args.coordinate_frame,
            "best_empty_seat": best_seat,
            "empty_seat_count": len(valid_seats),
            "timestamp": round(now, 3),
        }
        with open(self.args.output_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("[EmptySeat] " + json.dumps(payload, ensure_ascii=False), flush=True)

    def process_frame(self, frame, depth_image):
        results = self.model.predict(
            source=frame,
            conf=self.args.conf,
            iou=self.args.iou,
            device=self.args.device,
            classes=self.predict_classes,
            verbose=False,
        )

        result = results[0]
        annotated_frame = frame.copy()
        seats = []
        person_boxes = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0].cpu().tolist()))

            if class_name == "person":
                person_boxes.append(bbox)
            elif class_name in self.args.seat_classes:
                seats.append((class_name, confidence, bbox))

        for person_box in person_boxes:
            x1, y1, x2, y2 = person_box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated_frame,
                "PERSON",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        empty_seats = []
        for class_name, confidence, seat_box in seats:
            x1, y1, x2, y2 = seat_box
            occupied, seat_depth, depth_match = self.seat_occupancy(
                depth_image, seat_box, person_boxes
            )
            color = (0, 0, 255) if occupied else (0, 255, 0)
            state = "OCCUPIED" if occupied else "EMPTY SEAT"
            if not occupied:
                empty_seats.append(
                    {
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": list(seat_box),
                        "depth_m": None if seat_depth is None else round(seat_depth, 3),
                        "position": self.seat_position(seat_box, seat_depth),
                    }
                )

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            depth_text = "depth unknown" if seat_depth is None else f"{seat_depth:.2f}m"
            if depth_match is not None:
                depth_text += f" gap {depth_match['depth_gap']:.2f}m"
            cv2.putText(
                annotated_frame,
                f"{state}: {class_name} {confidence:.2f} {depth_text}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        self.output_best_empty_seat(empty_seats)
        best_seat = min(
            [seat for seat in empty_seats if seat["position"] is not None],
            key=lambda seat: seat["depth_m"],
            default=None,
        )
        if best_seat:
            position = best_seat["position"]
            x, y, z = (
                position.get("target_frame_xyz_m")
                or position.get("left_arm_base_xyz_m")
                or position["calibrated_xyz_m"]
            )
            cv2.putText(
                annotated_frame,
                f"BEST EMPTY SEAT {self.args.coordinate_frame}: ({x:.2f}, {y:.2f}, {z:.2f})m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        return annotated_frame, len(empty_seats), len(seats)

    def run(self):
        print("=" * 60)
        print("空座识别已启动")
        print(f"  RealSense 设备: {self.device_name or 'unknown'}")
        print(f"  RealSense 序列号: {self.serial_number or 'auto'}")
        print(f"  请求格式: {self.args.width}x{self.args.height} @ {self.args.fps} FPS")
        print(f"  模型: {self.args.model}")
        intrinsic_source = self.args.intrinsic or "active color stream"
        extrinsic_source = self.args.extrinsic or "identity(camera frame)"
        print(f"  内参来源: {intrinsic_source}")
        print(f"  外参来源: {extrinsic_source}")
        print(f"  座位类别: {', '.join(self.args.seat_classes)}")
        print(f"  占用阈值: {self.args.occupancy_threshold}")
        print(f"  最大占用深度差: {self.args.max_occupancy_depth_gap}m")
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
        success_count = 0
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
                annotated_frame, empty_count, seat_count = self.process_frame(
                    frame, depth_image
                )
                if empty_count > 0:
                    success_count += 1
                else:
                    success_count = 0
                fps = 1.0 / max(time.time() - start_time, 1e-9)

                cv2.putText(
                    annotated_frame,
                    f"Empty seats: {empty_count}/{seat_count}  FPS: {fps:.1f}",
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
                    cv2.imshow("Empty Seat Search", annotated_frame)
                    cv2.imshow("Depth View", depth_view)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if self.args.max_frames > 0 and frame_count >= self.args.max_frames:
                    break
                if (
                    self.args.stop_on_success
                    and success_count >= self.args.success_frames
                ):
                    print(
                        f"已连续 {success_count} 帧识别到空座，提前退出",
                        flush=True,
                    )
                    break
        finally:
            self.pipeline.stop()
            if not self.args.no_display:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Find empty seats with RealSense depth and YOLO")
    parser.add_argument(
        "--camera-model",
        default=DEFAULT_CAMERA_MODEL,
        help=(
            "Auto-select a connected RealSense device whose name contains this value "
            "when --serial-number is empty/auto. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--serial-number",
        default=DEFAULT_REALSENSE_SERIAL,
        help=(
            "Explicit RealSense serial number. "
            "Default comes from REALSENSE455_SERIAL, REALSENSE_SERIAL, "
            "REALSENSE515_SERIAL, or CADE_REALSENSE_SERIAL; leave empty to use "
            "--camera-model auto-selection."
        ),
    )
    parser.add_argument(
        "--intrinsic",
        default=DEFAULT_INTRINSIC,
        help=(
            "Camera intrinsic matrix file. If omitted, intrinsics are read from "
            "the active color stream."
        ),
    )
    parser.add_argument(
        "--extrinsic",
        default=DEFAULT_EXTRINSIC,
        help=(
            "Camera-to-target-frame extrinsic matrix file. If omitted, output "
            "coordinates stay in the camera color optical frame."
        ),
    )
    parser.add_argument(
        "--coordinate-frame",
        default=DEFAULT_COORDINATE_FRAME,
        help="Name of the frame represented by the extrinsic matrix",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="File receiving the latest best empty seat coordinate",
    )
    parser.add_argument(
        "--output-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between coordinate outputs",
    )
    parser.add_argument(
        "--seat-point-height-ratio",
        type=float,
        default=0.6,
        help="Vertical position inside a seat box used as the pointing target",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--seat-classes",
        nargs="+",
        default=["chair", "couch", "bench"],
        help="YOLO classes considered seats",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.2,
        help="Minimum person overlap ratio over a seat box to mark it occupied",
    )
    parser.add_argument(
        "--max-occupancy-depth-gap",
        type=float,
        default=0.5,
        help="Maximum person-seat depth gap in meters to mark a seat occupied",
    )
    parser.add_argument("--min-depth", type=float, default=0.1, help="Minimum valid depth in meters")
    parser.add_argument("--max-depth", type=float, default=5.0, help="Maximum valid depth in meters")
    parser.add_argument(
        "--min-depth-samples",
        type=int,
        default=30,
        help="Minimum valid depth pixels required for a depth estimate",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", default=None, help="For example: cpu, cuda, or 0")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV windows for headless testing",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 means run until q/Ctrl-C",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop early once enough frames contain at least one empty seat",
    )
    parser.add_argument(
        "--success-frames",
        type=int,
        default=1,
        help="Consecutive successful frames required by --stop-on-success",
    )
    args = parser.parse_args()

    detector = RealSenseEmptySeatDetector(args)
    detector.run()


if __name__ == "__main__":
    main()
