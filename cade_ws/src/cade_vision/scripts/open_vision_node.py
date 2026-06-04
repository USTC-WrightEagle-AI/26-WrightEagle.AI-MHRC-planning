#!/usr/bin/env python3
"""
open_vision_node.py - CADE 开放视觉节点

纯特征提取器，职责：
1. 监听 /cade/task_cmd 接收大脑指令
2. 驱动 RealSense + YOLO-World 进行目标检测
3. 将 3D 坐标通过 /vision/detections_3d 发布（JSON 格式）
4. 任务完成后向 /cade/task_status 发布结果

绝对不包含：底盘控制、模型状态修改、LLM 调用。
"""

import argparse
import copy
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

try:
    import rospy
    from geometry_msgs.msg import Point, PointStamped
    from std_msgs.msg import Header, String

    ROS_AVAILABLE = True
except ImportError:
    Header = Point = PointStamped = String = None
    ROS_AVAILABLE = False

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import tf2_ros
    from tf.transformations import quaternion_matrix

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# MediaPipe + Analyzer（姿态手势管线）
try:
    from cade_vision.kits.posture_gesture import (
        MEDIAPIPE_AVAILABLE,
        MEDIAPIPE_IMPORT_ERROR,
        PostureGestureAnalyzer,
    )

    POSTURE_GESTURE_AVAILABLE = MEDIAPIPE_AVAILABLE
except ImportError:
    MEDIAPIPE_IMPORT_ERROR = None
    POSTURE_GESTURE_AVAILABLE = False

try:
    from cade_vision.analyzer import Analyzer

    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False


from cade_vision.cloth import associate_clothing
from cade_vision.tasks import CountTask, InfoTask, SearchTask
from cade_vision.tasks.base_task import BaseTask
from cade_vision.tracker import CadeTracker


class OpenVisionNode:
    """
    CADE 视觉节点 - 纯感知，只发布检测结果

    启动方式：
        rosrun cade_vision open_vision_node.py
    或
        python open_vision_node.py --model yolov8x-worldv2.pt
    """

    def __init__(self, args):
        self.args = args

        # ROS 初始化
        if ROS_AVAILABLE:
            try:
                rospy.init_node("cade_open_vision", anonymous=True)
            except rospy.exceptions.ROSException:
                pass

        # ========== Publishers ==========
        if ROS_AVAILABLE:
            # 3D 检测结果发布
            self.detections_3d_pub = rospy.Publisher(
                "/vision/detections_3d_task3", String, queue_size=10
            )
            from geometry_msgs.msg import PointStamped  # 确保头部或者这里引入了消息类型

            self.point_3d_pub = rospy.Publisher(
                "/vision/point_3d_task3", PointStamped, queue_size=10
            )
            # 任务状态发布
            self.task_status_pub = rospy.Publisher(
                "/cade/task_status_task3", String, queue_size=10
            )

        # ========== Subscriber ==========
        if ROS_AVAILABLE:
            self.task_cmd_sub = rospy.Subscriber(
                "/cade/task_cmd_task3", String, self._on_task_cmd, queue_size=10
            )

        # ========== 图像源 (realsense / file / usb_cam) ==========
        self.image_source = getattr(args, "image_source", "realsense")

        # RealSense 相关属性
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.intrinsics = None

        # file / usb_cam 相关属性
        self._static_image = None  # file 模式：单张图片
        self._file_cap = None  # file 模式：视频文件 VideoCapture
        self._file_fps = 30.0  # file 模式：视频原始帧率
        self._usb_cap = None  # usb_cam 模式的 VideoCapture
        self._loop = getattr(args, "loop", False)
        self._playback_speed = getattr(args, "playback_speed", None)

        self._init_image_source()

        # ========== YOLO 模型加载 ==========
        self.model = None
        self._model_lock = threading.Lock()
        self._clothing_classes = set()  # 动态累积的衣物类别
        self._base_classes = [  # 常开的基础类别
            "person",
            "t-shirt",
            "shirt",
            "sweater",
            "jacket",
            "coat",
            "blouse",
            "pants",
            "skirt",
            "shorts",
        ]
        if YOLO_AVAILABLE:
            startup_device = getattr(args, "device", "cuda")
            print(f"Loading YOLO model on {startup_device}: {args.model}")
            self.model = YOLO(args.model)
            self.model.to(startup_device)
            if hasattr(self.model, "set_classes"):
                self.model.set_classes(self._base_classes)
                print(f"YOLO-World base classes: {self._base_classes}")
            print(f"YOLO model loaded")

        # ========== 姿态手势管线 ==========
        self.posture_gesture = None
        if POSTURE_GESTURE_AVAILABLE:
            try:
                posture_gesture = PostureGestureAnalyzer()
                if getattr(posture_gesture, "available", False):
                    self.posture_gesture = posture_gesture
            except Exception as e:
                print(f"PostureGestureAnalyzer init failed: {e}")
        elif MEDIAPIPE_IMPORT_ERROR is not None:
            print(f"MediaPipe unavailable: {MEDIAPIPE_IMPORT_ERROR}")

        self.analyzer = None
        if ANALYZER_AVAILABLE:
            try:
                self.analyzer = Analyzer()
            except Exception as e:
                print(f"Analyzer init failed: {e}")

        # 线程池：Pose + Hands 并行推理（复用，避免每次 new Thread）
        self._pool = ThreadPoolExecutor(max_workers=2)

        # 跨帧追踪器：稳定 track_id 供 RingBuffer 时序检测
        self._tracker = CadeTracker()

        # ========== 状态管理 ==========
        self.target_class = None  # 当前目标类别
        self.task_active = False  # 是否有活跃任务
        self.task_id = None  # 当前任务 ID
        self.task_attributes = None  # 任务要求的属性过滤（如 posture/gesture）
        self.detected_objects = []  # 当前帧检测结果
        self.current_task_executor = None
        self._task_thread = None
        self._task_mapping = {
            "find_object": SearchTask,
            "find_person": SearchTask,
            "filter_by_attributes": SearchTask,
            "count_objects": CountTask,
            "count_people": CountTask,
            "get_person_info": InfoTask,
            "get_nearest_person": InfoTask,
        }
        self._lock = threading.Lock()

        # ========== 显示 ==========
        if args.display:
            cv2.namedWindow("CADE Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CADE Vision", 800, 600)

        print(f"OpenVisionNode initialized")
        print(f"  Image Source: {self.image_source}")
        print(f"  YOLO: {'Available' if self.model else 'NOT AVAILABLE'}")
        print(
            f"  MediaPipe: {'Available' if self.posture_gesture else 'NOT AVAILABLE'}"
        )
        print(f"  Analyzer: {'Available' if self.analyzer else 'NOT AVAILABLE'}")
        print(f"  ROS: {'Available' if ROS_AVAILABLE else 'NOT AVAILABLE'}")
        print(f"  Display: {args.display}")

    # ==================== 图像源初始化 ====================

    def _init_image_source(self):
        """根据 args.image_source 分支初始化图像源"""
        if self.image_source == "realsense":
            if REALSENSE_AVAILABLE:
                self._init_realsense(self.args.serial_number)
            else:
                print("Warning: pyrealsense2 not available, falling back to usb_cam")
                self.image_source = "usb_cam"
                self._init_usb_cam()
        elif self.image_source == "file":
            self._init_file_source()
        elif self.image_source == "usb_cam":
            self._init_usb_cam()
        else:
            raise ValueError(f"Unknown image source: {self.image_source}")

    def _init_file_source(self):
        """从图片或视频文件读取（开发调试用）。自动检测文件类型。"""
        file_path = getattr(self.args, "image_path", "")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

        if ext in VIDEO_EXTS:
            self._file_cap = cv2.VideoCapture(file_path)
            if not self._file_cap.isOpened():
                raise RuntimeError(f"Failed to open video: {file_path}")
            self._file_fps = self._file_cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(self._file_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(self._file_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._file_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(
                f"File source (video): {file_path}  {w}x{h}, {self._file_fps:.0f}fps, {total} frames"
            )
        else:
            self._static_image = cv2.imread(file_path)
            if self._static_image is None:
                raise ValueError(f"Failed to read image: {file_path}")
            print(
                f"File source (image): {file_path} "
                f"({self._static_image.shape[1]}x{self._static_image.shape[0]})"
            )

    def _init_usb_cam(self):
        """从 USB 摄像头读取"""
        cam_id = getattr(self.args, "usb_cam_id", 0)
        width = getattr(self.args, "width", 640)
        height = getattr(self.args, "height", 480)
        self._usb_cap = cv2.VideoCapture(cam_id)
        if not self._usb_cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera (id={cam_id})")
        self._usb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._usb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"USB camera initialized: id={cam_id}, {width}x{height}")

    # ==================== RealSense ====================

    def _init_realsense(self, serial_number="333422301212"):
        """初始化 RealSense 摄像头"""
        self._realsense_started = False
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            if serial_number:
                config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = self.pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            self.align = rs.align(rs.stream.color)

            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            self.intrinsics = color_profile.get_intrinsics()

            self._realsense_started = True

            print(f"RealSense initialized - depth scale: {self.depth_scale}")
        except Exception as e:
            print(f"RealSense init failed: {e}")
            self._realsense_started = False

    # ==================== ROS 回调 ====================

    def _on_task_cmd(self, msg: String):
        """
        任务指令回调

        消息格式：
        {
            "action": "find_object" | "find_person" | "count_objects" | ...,
            "target": "apple",
            "room": "kitchen",  // 可选
            ...
        }
        """
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as e:
            print(f"[Vision] Invalid task cmd JSON: {e}")
            return

        action = cmd.get("action", "")
        task_cls = self._task_mapping.get(action)
        if task_cls is None:
            print(f"[Vision] Unknown action: {action}")
            self._publish_status("FAILED", error=f"Unknown action: {action}")
            return

        executor = task_cls(self)
        with self._lock:
            old_executor = self.current_task_executor
            if old_executor is not None and old_executor.should_continue():
                old_executor.cancel()

            self.current_task_executor = executor
            self.task_active = True
            self.task_id = action
            self.task_attributes = self._extract_task_attributes(action, cmd)
            self.target_class = self._target_for_command(action, cmd)
            model_classes = self._build_model_classes(action, self.target_class)

        self._set_model_classes(model_classes)

        print(
            f"\n[Vision Task] {action}: target='{self.target_class}'"
            f"  attrs={self.task_attributes}"
        )

        self._task_thread = threading.Thread(
            target=self._run_task_executor, args=(executor, cmd), daemon=True
        )
        self._task_thread.start()

    def _run_task_executor(self, executor, cmd):
        try:
            executor.execute(cmd)
        except Exception as e:
            if self.finish_task(executor):
                self._publish_status("FAILED", error=str(e))

    def _extract_task_attributes(self, action: str, cmd: dict):
        """Normalize task attributes for node state and model-class planning."""
        attributes = {}
        raw_attributes = cmd.get("attributes")
        if isinstance(raw_attributes, dict):
            attributes.update(raw_attributes)

        if action == "filter_by_attributes":
            keys = [
                key
                for key in cmd.keys()
                if key not in BaseTask.RESERVED_COMMAND_KEYS and key != "attributes"
            ]
        else:
            keys = [
                key
                for key in BaseTask.PERSON_ATTRIBUTE_KEYS
                if key in cmd and key not in attributes
            ]

        for key in keys:
            attributes.setdefault(key, cmd.get(key))

        normalized = {
            key: value
            for key, value in attributes.items()
            if value is not None and value != "" and value != "unknown"
        }
        return normalized or None

    def _target_for_command(self, action: str, cmd: dict):
        """Return the active target/category used for display and search classes."""
        if action in ("find_object", "find_person"):
            default_target = "person" if action == "find_person" else ""
            target = cmd.get("target") or default_target
            return str(target).lower() if target else None
        if action in ("count_objects", "count_people"):
            default_category = "person" if action == "count_people" else ""
            category = cmd.get("category") or default_category
            return str(category).lower() if category else None
        if action == "filter_by_attributes":
            return "person"
        return None

    def _build_model_classes(self, action: str, target_class: str):
        """Build a YOLO-World class set without dropping base pipeline classes."""
        class_actions = {
            "find_object",
            "find_person",
            "filter_by_attributes",
            "count_objects",
            "count_people",
        }
        if action not in class_actions:
            return None

        classes = set(self._base_classes)
        classes.add("person")
        classes.update(self._clothing_classes)
        classes.update(self._iter_class_values(target_class))

        attrs = self.task_attributes or {}
        for cloth_type in self._iter_class_values(attrs.get("cloth_type")):
            self._clothing_classes.add(cloth_type)
            classes.add(cloth_type)

        return sorted(classes)

    @staticmethod
    def _iter_class_values(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            values = []
            for item in value:
                values.extend(OpenVisionNode._iter_class_values(item))
            return values

        class_names = []
        normalized = str(value).replace("|", ",").replace(";", ",").replace("/", ",")
        for part in normalized.split(","):
            part = part.strip().lower()
            if part:
                class_names.append(part)
        return class_names

    def _set_model_classes(self, class_names):
        if not class_names or not self.model or not hasattr(self.model, "set_classes"):
            return
        try:
            with self._model_lock:
                self.model.set_classes(class_names)
            print(f"[Vision] YOLO classes updated: {len(class_names)} classes")
        except Exception as e:
            print(f"[Vision] Failed to update YOLO classes: {e}")

    def get_latest_detections(self):
        """Return a thread-safe snapshot of the latest full detection cache."""
        with self._lock:
            return copy.deepcopy(self.detected_objects)

    def finish_task(self, executor):
        """Mark a task complete only if it is still the active executor."""
        with self._lock:
            if self.current_task_executor is not executor:
                return False
            self.task_active = False
            self.target_class = None
            self.task_id = None
            self.task_attributes = None
            self.current_task_executor = None
            return True

    def _reset_task(self):
        """重置任务状态"""
        with self._lock:
            executor = self.current_task_executor
            if executor is not None:
                executor.cancel()
            self.task_active = False
            self.target_class = None
            self.task_id = None
            self.task_attributes = None
            self.current_task_executor = None

    def _publish_detection(self, detection: dict):
        """发布检测结果"""
        if not ROS_AVAILABLE:
            print(f"[Vision] Detection: {detection}")
            return

        msg = String()
        msg.data = json.dumps(detection)
        self.detections_3d_pub.publish(msg)

        # 同时发布 PointStamped 格式（兼容旧版订阅者）
        if detection.get("position_3d"):
            x, y, z = detection["position_3d"]
            point_msg = PointStamped()
            point_msg.header = Header()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.header.frame_id = "camera_color_optical_frame"
            point_msg.point = Point(x, y, z)
            self.point_3d_pub.publish(point_msg)

    def _publish_status(self, status: str, result: dict = None, error: str = None):
        """发布任务状态"""
        status_msg = {"status": status}
        if result:
            status_msg["result"] = result
        if error:
            status_msg["error"] = error

        print(f"[Vision] Task status: {status_msg}")

        if ROS_AVAILABLE:
            msg = String()
            msg.data = json.dumps(status_msg)
            self.task_status_pub.publish(msg)

    # ==================== 3D 坐标 ====================

    def get_median_depth_in_roi(self, depth_frame, x, y, roi_size=20):
        """获取 ROI 区域中值深度"""
        width = depth_frame.get_width()
        height = depth_frame.get_height()

        x1 = max(0, int(x - roi_size // 2))
        y1 = max(0, int(y - roi_size // 2))
        x2 = min(width - 1, int(x + roi_size // 2))
        y2 = min(height - 1, int(y + roi_size // 2))

        depth_data = np.asanyarray(depth_frame.get_data())
        roi = depth_data[y1:y2, x1:x2]
        roi_meters = roi.astype(float) * self.depth_scale
        valid_depths = roi_meters[roi_meters > 0.1]
        valid_depths = valid_depths[valid_depths < 2.0]

        if len(valid_depths) == 0:
            return None
        return np.median(valid_depths)

    def get_3d_coordinates(self, depth_frame, pixel_x, pixel_y, depth_value=None):
        """2D 像素 -> 3D 世界坐标"""
        if (
            pixel_x < 0
            or pixel_y < 0
            or pixel_x >= self.intrinsics.width
            or pixel_y >= self.intrinsics.height
        ):
            return None

        try:
            if depth_value is None:
                depth = depth_frame.get_distance(int(pixel_x), int(pixel_y))
            else:
                depth = depth_value
            if depth <= 0:
                return None
            point = rs.rs2_deproject_pixel_to_point(
                self.intrinsics, [pixel_x, pixel_y], depth
            )
            return point
        except RuntimeError:
            return None

    # ==================== 主循环 ====================

    def _capture_frame(self):
        """
        统一的帧捕获入口，根据 image_source 返回 (color_image, depth_frame)。

        Returns:
            (color_image, depth_frame): depth_frame 仅 realsense 模式有效，其余为 None
        """
        if self.image_source == "realsense":
            if not getattr(self, "_realsense_started", False) or self.pipeline is None:
                # 适当加入小延时，防止没初始化成功时主循环空转把 CPU 跑满
                time.sleep(0.1)
                return None, None
            if not self.pipeline:
                return None, None

            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)

                try:
                    aligned_frames = self.align.process(frames)
                except RuntimeError as e:
                    # 打印警告日志，但绝对不向外抛出致命异常
                    print(
                        f"[Vision WARNING] RealSense align block glitched: {e}. Skipping this frame to recover..."
                    )
                    return None, None
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    return None, None
                return np.asanyarray(color_frame.get_data()), depth_frame
            except Exception as e:
                print(f"[Vision] wait_for_frames 偶尔超时或失败: {e}")
                return None, None

        elif self.image_source == "file":
            if self._file_cap is not None:
                ret, frame = self._file_cap.read()
                if not ret or frame is None:
                    if self._loop:
                        # 视频播完 → seek 回开头 → 重置追踪器/缓冲区
                        self._file_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._tracker = CadeTracker()
                        if self.posture_gesture is not None:
                            self.posture_gesture.ring_buffer.clear()
                        print(
                            "[Vision] Loop: video reset to frame 0, tracker/buffer cleared"
                        )
                        ret, frame = self._file_cap.read()
                        if not ret or frame is None:
                            return None, None
                        return frame, None
                    return None, None
                return frame, None
            if self._static_image is not None:
                return self._static_image.copy(), None
            return None, None

        elif self.image_source == "usb_cam":
            if self._usb_cap is None:
                return None, None
            ret, frame = self._usb_cap.read()
            if not ret or frame is None:
                return None, None
            return frame, None

        return None, None

    def process_frame(self):
        """处理单帧图像"""
        try:
            color_image, depth_frame = self._capture_frame()
            if color_image is None:
                return False

            # YOLO 推理
            if self.model:
                with self._model_lock:
                    results = self.model.predict(
                        source=color_image,
                        conf=self.args.conf,
                        iou=self.args.iou,
                        device=self.args.device,
                        verbose=False,
                    )
                    model_names = (
                        self.model.names.copy()
                        if hasattr(self.model.names, "copy")
                        else self.model.names
                    )
            else:
                return False

            display_image = color_image.copy()
            new_detections = []

            is_realsense = self.image_source == "realsense"
            MIN_DEPTH = 0.2
            MAX_DEPTH = 1.0

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    class_name = model_names[class_id]
                    conf = box.conf[0]

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # 3D 坐标（仅 realsense 模式计算，其他模式设为 None）
                    point_3d = None
                    if is_realsense and depth_frame is not None:
                        median_depth = self.get_median_depth_in_roi(
                            depth_frame, center_x, center_y
                        )
                        if median_depth is not None:
                            point_3d = self.get_3d_coordinates(
                                depth_frame,
                                center_x,
                                center_y,
                                depth_value=median_depth,
                            )
                        else:
                            point_3d = self.get_3d_coordinates(
                                depth_frame, center_x, center_y
                            )

                        if point_3d is not None:
                            point_3d = list(point_3d)
                            x, y, z = point_3d
                            distance = math.sqrt(x**2 + y**2 + z**2)
                            if distance < MIN_DEPTH or distance > MAX_DEPTH:
                                point_3d = None

                    obj_info = {
                        "index": len(new_detections),
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": (x1, y1, x2, y2),
                        "center": (center_x, center_y),
                        "position_3d": point_3d,
                    }
                    new_detections.append(obj_info)

            # 跨帧追踪：为每个检测分配稳定的 track_id
            track_ids = self._tracker.assign(new_detections)
            for obj, tid in zip(new_detections, track_ids):
                obj["track_id"] = tid

            # 衣物关联：将衣服框 IOU 匹配到人，提取颜色 + 类型
            associate_clothing(new_detections, color_image)

            # [MediaPipe] Pose + Hands 并行后处理（线程池，一次融合，无回退）
            if self.posture_gesture is not None:
                for obj in new_detections:
                    if obj["class_name"] != "person":
                        continue

                    x1, y1, x2, y2 = obj["bbox"]
                    person_crop = color_image[y1:y2, x1:x2]
                    if person_crop.size <= 0:
                        continue

                    obj["_pg_fut_pose"] = self._pool.submit(
                        self.posture_gesture.process_pose, person_crop
                    )
                    obj["_pg_fut_hands"] = self._pool.submit(
                        self.posture_gesture.process_hands, person_crop
                    )

                for obj in new_detections:
                    fut_pose = obj.pop("_pg_fut_pose", None)
                    fut_hands = obj.pop("_pg_fut_hands", None)
                    if fut_pose is None or fut_hands is None:
                        continue

                    pose_data = fut_pose.result()
                    hands_data = fut_hands.result()
                    pid = obj["track_id"]
                    x1, y1, x2, y2 = obj["bbox"]

                    # 从深度图提取关键点 3D 坐标（肩髋膝踝，供 posture 规则用）
                    keypoints_3d = {}
                    if (
                        depth_frame is not None
                        and pose_data is not None
                        and is_realsense
                    ):
                        lm = pose_data["landmarks"]
                        for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
                            if lm[idx][2] < 0.5:
                                continue
                            px = int(lm[idx][0] * (x2 - x1) + x1)
                            py = int(lm[idx][1] * (y2 - y1) + y1)
                            pt = self.get_3d_coordinates(depth_frame, px, py)
                            if pt is not None:
                                keypoints_3d[idx] = pt

                    pg_result = self.posture_gesture.analyze_from_landmarks(
                        pose_data,
                        person_id=pid,
                        hands_data=hands_data,
                        with_temporal=True,
                        keypoints_3d=keypoints_3d if keypoints_3d else None,
                    )

                    obj["posture"] = pg_result.get("posture", "unknown")
                    obj["gesture"] = pg_result.get("gesture", "unknown")
                    obj["landmarks"] = pg_result.get("landmarks", [])

            # 更新检测结果（始终存完整数据，过滤下放到各任务函数）
            with self._lock:
                self.detected_objects = new_detections

            # 显示
            if self.args.display:
                if not new_detections:
                    cv2.putText(
                        display_image,
                        "No objects detected",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                else:
                    with self._lock:
                        target_cls = (
                            self.target_class.lower() if self.target_class else None
                        )

                    for obj in new_detections:
                        x1, y1, x2, y2 = obj["bbox"]
                        c_name = obj["class_name"]

                        # 判断是否为当前目标
                        is_target = (
                            target_cls is not None and target_cls in c_name.lower()
                        )
                        box_color = (0, 0, 255) if is_target else (0, 255, 0)
                        thickness = 3 if is_target else 1

                        # 1. 画框
                        cv2.rectangle(
                            display_image, (x1, y1), (x2, y2), box_color, thickness
                        )

                        # 2. 整合所有注脚文本（支持多行展示）
                        lines = [
                            f"#{obj['index']} {c_name} {obj['confidence']:.2f}",
                        ]

                        # 如果有追踪 ID，加上追踪信息
                        if "track_id" in obj:
                            lines[0] += f" [ID: {obj['track_id']}]"

                        # 如果是人，加上衣服、姿态、手势信息
                        if c_name == "person":
                            cloth_str = f"Cloth: {obj.get('cloth_color', 'unk')}/{obj.get('cloth_type', 'unk')}"
                            pg_str = f"PG: {obj.get('posture', 'unk')}/{obj.get('gesture', 'unk')}"
                            lines.append(cloth_str)
                            lines.append(pg_str)

                        # 如果有 3D 坐标，加上坐标信息
                        if obj.get("position_3d") is not None:
                            cx, cy, cz = obj["position_3d"]
                            lines.append(f"XYZ: ({cx:.2f}, {cy:.2f}, {cz:.2f})m")

                        # 3. 将多行文本安全地渲染到框的上方
                        # 增加行高到 18 像素，防止上下行重叠；字体粗细调为 1
                        line_height = 18
                        curr_y = y1 - 10 - (len(lines) - 1) * line_height

                        for line in lines:
                            # 黑色轻微描边（粗细为 2，作为背景垫底，防止文字融入亮色背景）
                            cv2.putText(
                                display_image,
                                line,
                                (x1, curr_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 0, 0),
                                2,
                            )
                            # 前景绿色/红色文本（粗细调细为 1，确保字迹清晰不粘连）
                            cv2.putText(
                                display_image,
                                line,
                                (x1, curr_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                box_color,
                                1,
                            )
                            curr_y += line_height

                # 弹出窗口
                cv2.imshow("CADE Vision", display_image)

            return True

        except Exception as e:
            # print(f"\nFrame processing error: {e}")
            import traceback

            print("\n" + "=" * 50)
            print("🚨 侦测到致命异常，真凶定位如下：")
            traceback.print_exc()  # 👈 这行代码会打印出具体报错的行号和调用链路
            print("=" * 50 + "\n")
            return False

    def run(self):
        """主循环"""
        print("CADE Vision Node running...")
        print("Listening on /cade/task_cmd")
        print("Publishing detections to /vision/detections_3d")
        print("(Press 'q' to quit)")

        try:
            while True:
                if ROS_AVAILABLE and rospy.is_shutdown():
                    break

                start_time = time.time()

                if not self.process_frame():
                    time.sleep(0.01)
                    continue

                # 播放速度控制（仅 file 视频模式生效）
                if (
                    self._playback_speed is not None
                    and self.image_source == "file"
                    and self._file_cap is not None
                ):
                    ideal_interval = 1.0 / max(self._file_fps, 1.0)
                    target_interval = ideal_interval / self._playback_speed
                    elapsed = time.time() - start_time
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)

                # 检查退出
                if self.args.display:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self._reset_task()
            if self.image_source == "realsense" and self.pipeline:
                self.pipeline.stop()
            elif self.image_source == "file" and self._file_cap:
                self._file_cap.release()
            elif self.image_source == "usb_cam" and self._usb_cap:
                self._usb_cap.release()
            if self.args.display:
                cv2.destroyAllWindows()
            print("\nVision node stopped")


def main():
    parser = argparse.ArgumentParser(description="CADE Open Vision Node")
    parser.add_argument(
        "--model", type=str, default="yolov8x-worldv2.pt", help="YOLO model path"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--image-source",
        type=str,
        default="realsense",
        choices=["realsense", "file", "usb_cam"],
        help="Image source: realsense (default), file, or usb_cam",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="",
        help="Path to image/video file (for --image-source file)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop video when it ends (file video mode only)",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=None,
        help="Playback speed multiplier (e.g. 0.2 for 0.2x slow, file video only)",
    )
    parser.add_argument(
        "--usb-cam-id",
        type=int,
        default=0,
        help="USB camera device ID (for --image-source usb_cam)",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Camera capture width (usb_cam mode)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Camera capture height (usb_cam mode)"
    )
    parser.add_argument(
        "--serial-number",
        type=str,
        default="333422301212",
        help="RealSense serial number",
    )
    parser.add_argument(
        "--display", action="store_true", default=True, help="Show detection window"
    )
    parser.add_argument(
        "--no-display",
        action="store_false",
        dest="display",
        help="Hide detection window",
    )

    args = parser.parse_args()
    node = OpenVisionNode(args)
    node.run()


if __name__ == "__main__":
    main()
