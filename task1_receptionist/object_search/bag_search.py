#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read aligned RealSense color/depth frames and find a guest bag.
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
DEFAULT_MODEL = str(SCRIPT_DIR / "yolo11n.pt")
DEFAULT_OUTPUT_JSON = str(SCRIPT_DIR / "latest_bag.json")
DEFAULT_LEFTBASE_EXTRINSIC = SCRIPT_DIR / "camera_middle_to_leftbase.txt"
DEFAULT_CAMERA_MODEL = os.environ.get("REALSENSE_CAMERA_MODEL", "D455")
DEFAULT_REALSENSE_SERIAL = (
    os.environ.get("REALSENSE455_SERIAL")
    or os.environ.get("REALSENSE_SERIAL")
    or os.environ.get("REALSENSE515_SERIAL")
    or os.environ.get("CADE_REALSENSE_SERIAL")
    or ""
)
DEFAULT_INTRINSIC = os.environ.get("BAG_SEARCH_INTRINSIC", "")
DEFAULT_EXTRINSIC = os.environ.get(
    "BAG_SEARCH_EXTRINSIC",
    str(DEFAULT_LEFTBASE_EXTRINSIC) if DEFAULT_LEFTBASE_EXTRINSIC.exists() else "",
)
DEFAULT_COORDINATE_FRAME = os.environ.get(
    "BAG_SEARCH_COORDINATE_FRAME",
    "leftbase" if DEFAULT_EXTRINSIC else "camera_color_optical",
)
DEFAULT_CONF = float(os.environ.get("BAG_SEARCH_CONF", "0.15"))


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


class RealSenseBagDetector:
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
            cv2.namedWindow("Bag Search", cv2.WINDOW_NORMAL)
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

    def median_depth(self, depth_image, bbox):
        height, width = depth_image.shape
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        x1 = int(x1 + box_w * self.args.depth_crop_ratio)
        x2 = int(x2 - box_w * self.args.depth_crop_ratio)
        y1 = int(y1 + box_h * self.args.depth_crop_ratio)
        y2 = int(y2 - box_h * self.args.depth_crop_ratio)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return None

        depths = depth_image[y1:y2, x1:x2].astype(np.float32) * self.depth_scale
        depths = depths[(depths >= self.args.min_depth) & (depths <= self.args.max_depth)]
        if depths.size < self.args.min_depth_samples:
            return None
        return float(np.median(depths))

    def bag_position(self, bbox, depth_m):
        if depth_m is None:
            return None

        x1, y1, x2, y2 = bbox
        pixel_x = x1 + (x2 - x1) * self.args.bag_point_x_ratio
        pixel_y = y1 + (y2 - y1) * self.args.bag_point_y_ratio
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
            "pixel": [round(pixel_x, 1), round(pixel_y, 1)],
            "camera_xyz_m": [round(float(value), 3) for value in point_camera[:3]],
            "target_frame_xyz_m": target_xyz,
            "calibrated_xyz_m": target_xyz,
        }
        if self.args.coordinate_frame in {"leftbase", "left_arm_base"}:
            position["leftbase_xyz_m"] = target_xyz
            position["left_arm_base_xyz_m"] = target_xyz
        return position

    def output_best_bag(self, bags):
        now = time.time()
        if now - self.last_output_time < self.args.output_interval:
            return
        self.last_output_time = now

        valid_bags = [bag for bag in bags if bag["position"] is not None]
        if self.args.select == "confidence":
            best_bag = max(valid_bags, key=lambda bag: bag["confidence"], default=None)
        else:
            best_bag = min(valid_bags, key=lambda bag: bag["depth_m"], default=None)

        payload = {
            "status": "success" if best_bag else "not_found",
            "coordinate_frame": self.args.coordinate_frame,
            "best_bag": best_bag,
            "bag_count": len(valid_bags),
            "timestamp": round(now, 3),
        }
        with open(self.args.output_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("[Bag] " + json.dumps(payload, ensure_ascii=False), flush=True)

    def process_frame(self, frame, depth_image):
        results = self.model.predict(
            source=frame,
            conf=self.args.conf,
            iou=self.args.iou,
            device=self.args.device,
            classes=None,
            verbose=False,
        )

        result = results[0]
        annotated_frame = frame.copy()
        bags = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if class_name not in self.args.bag_classes:
                continue

            confidence = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0].cpu().tolist()))
            depth_m = self.median_depth(depth_image, bbox)
            bags.append(
                {
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": list(bbox),
                    "depth_m": None if depth_m is None else round(depth_m, 3),
                    "position": self.bag_position(bbox, depth_m),
                }
            )

            x1, y1, x2, y2 = bbox
            color = (255, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            depth_text = "depth unknown" if depth_m is None else f"{depth_m:.2f}m"
            cv2.putText(
                annotated_frame,
                f"BAG: {class_name} {confidence:.2f} {depth_text}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        self.output_best_bag(bags)
        self.last_annotated_frame = annotated_frame.copy()
        best_bag = min(
            [bag for bag in bags if bag["position"] is not None],
            key=lambda bag: bag["depth_m"],
            default=None,
        )
        if best_bag:
            position = best_bag["position"]
            x, y, z = (
                position.get("target_frame_xyz_m")
                or position.get("left_arm_base_xyz_m")
                or position["calibrated_xyz_m"]
            )
            cv2.putText(
                annotated_frame,
                f"BEST BAG {self.args.coordinate_frame}: ({x:.2f}, {y:.2f}, {z:.2f})m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
            )

        return annotated_frame, len([bag for bag in bags if bag["position"] is not None])

    def run(self):
        print("=" * 60)
        print("包识别已启动")
        print(f"  RealSense 设备: {self.device_name or 'unknown'}")
        print(f"  RealSense 序列号: {self.serial_number or 'auto'}")
        print(f"  请求格式: {self.args.width}x{self.args.height} @ {self.args.fps} FPS")
        print(f"  模型: {self.args.model}")
        print(f"  包类别: {', '.join(self.args.bag_classes)}")
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
                annotated_frame, bag_count = self.process_frame(frame, depth_image)
                fps = 1.0 / max(time.time() - start_time, 1e-9)

                cv2.putText(
                    annotated_frame,
                    f"Bags: {bag_count}  FPS: {fps:.1f}",
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
                    cv2.imshow("Bag Search", annotated_frame)
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
    parser = argparse.ArgumentParser(description="Find guest bags with RealSense depth and YOLO")
    parser.add_argument("--camera-model", default=DEFAULT_CAMERA_MODEL)
    parser.add_argument("--serial-number", default=DEFAULT_REALSENSE_SERIAL)
    parser.add_argument("--intrinsic", default=DEFAULT_INTRINSIC)
    parser.add_argument("--extrinsic", default=DEFAULT_EXTRINSIC)
    parser.add_argument("--coordinate-frame", default=DEFAULT_COORDINATE_FRAME)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-interval", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--bag-classes",
        nargs="+",
        default=["backpack", "handbag", "suitcase"],
        help="YOLO classes considered bags",
    )
    parser.add_argument(
        "--select",
        choices=("nearest", "confidence"),
        default="nearest",
        help="How to choose best_bag when multiple bags are visible",
    )
    parser.add_argument("--bag-point-x-ratio", type=float, default=0.5)
    parser.add_argument("--bag-point-y-ratio", type=float, default=0.5)
    parser.add_argument("--depth-crop-ratio", type=float, default=0.2)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--min-depth-samples", type=int, default=20)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--snapshot", default="", help="Save the last annotated frame to this image path")
    args = parser.parse_args()

    detector = RealSenseBagDetector(args)
    detector.run()


if __name__ == "__main__":
    main()
