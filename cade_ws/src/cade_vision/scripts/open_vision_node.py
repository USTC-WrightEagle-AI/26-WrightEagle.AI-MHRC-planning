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

import sys
import json
import time
import math
import threading
import argparse

import numpy as np
import cv2

try:
    import rospy
    from std_msgs.msg import String
    from geometry_msgs.msg import PointStamped, Point
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
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


class OpenVisionNode:
    """
    CADE 视觉节点 - 纯感知，只发布检测结果

    启动方式：
        rosrun cade_vision open_vision_node.py
    或
        python open_vision_node.py --model yolo11x-seg.pt
    """

    def __init__(self, args):
        self.args = args

        # ROS 初始化
        if ROS_AVAILABLE:
            try:
                rospy.init_node('cade_open_vision', anonymous=True)
            except rospy.exceptions.ROSException:
                pass

        # ========== Publishers ==========
        if ROS_AVAILABLE:
            # 3D 检测结果发布
            self.detections_3d_pub = rospy.Publisher(
                '/vision/detections_3d', String, queue_size=10
            )
            # 相机坐标系位置（兼容旧版）
            self.object_3d_pub = rospy.Publisher(
                '/object_3d_position', PointStamped, queue_size=10
            )
            # 任务状态发布
            self.task_status_pub = rospy.Publisher(
                '/cade/task_status', String, queue_size=10
            )

        # ========== Subscriber ==========
        if ROS_AVAILABLE:
            self.task_cmd_sub = rospy.Subscriber(
                '/cade/task_cmd', String, self._on_task_cmd, queue_size=10
            )

        # ========== RealSense 初始化 ==========
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.intrinsics = None
        if REALSENSE_AVAILABLE:
            self._init_realsense(args.serial_number)

        # ========== YOLO 模型加载 ==========
        self.model = None
        if YOLO_AVAILABLE:
            print(f"Loading YOLO model: {args.model}")
            self.model = YOLO(args.model)
            print(f"YOLO model loaded")

        # ========== TF 初始化 ==========
        self.tf_buffer = None
        self.tf_listener = None
        if TF_AVAILABLE and ROS_AVAILABLE:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ========== 相机到夹爪的变换矩阵 ==========
        self.transformation_matrix = np.array([
            [-0.02937859, -0.1152559,  0.99290129, -0.02411462],
            [ 0.01197522,  0.99321818,  0.11564701, -0.06956071],
            [-0.99949662,  0.01528775, -0.02779914,  0.01524878],
            [ 0.,          0.,          0.,          1.        ]
        ])

        # ========== 状态管理 ==========
        self.target_class = None       # 当前目标类别
        self.task_active = False       # 是否有活跃任务
        self.task_id = None            # 当前任务 ID
        self.detected_objects = []     # 当前帧检测结果
        self._lock = threading.Lock()

        # ========== 显示 ==========
        if args.display:
            cv2.namedWindow('CADE Vision', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CADE Vision', 800, 600)

        print(f"OpenVisionNode initialized")
        print(f"  YOLO: {'Available' if self.model else 'NOT AVAILABLE'}")
        print(f"  RealSense: {'Available' if self.pipeline else 'NOT AVAILABLE'}")
        print(f"  ROS: {'Available' if ROS_AVAILABLE else 'NOT AVAILABLE'}")
        print(f"  Display: {args.display}")

    # ==================== RealSense ====================

    def _init_realsense(self, serial_number="333422301212"):
        """初始化 RealSense 摄像头"""
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

            color_profile = rs.video_stream_profile(
                profile.get_stream(rs.stream.color)
            )
            self.intrinsics = color_profile.get_intrinsics()

            print(f"RealSense initialized - depth scale: {self.depth_scale}")
        except Exception as e:
            print(f"RealSense init failed: {e}")

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
            action = cmd.get("action", "")

            with self._lock:
                if action in ("find_object", "find_person"):
                    self.target_class = cmd.get("target", "").lower() if cmd.get("target") else None
                    self.task_active = True
                    self.task_id = action
                    print(f"\n[Vision Task] {action}: target='{self.target_class}'")

                    # 等待检测到目标后发布结果
                    threading.Thread(
                        target=self._execute_search_task,
                        args=(cmd,),
                        daemon=True
                    ).start()

                elif action in ("count_objects", "count_people"):
                    self.target_class = cmd.get("category", "").lower() if cmd.get("category") else None
                    self.task_active = True
                    self.task_id = action
                    print(f"\n[Vision Task] {action}: category='{self.target_class}'")

                    threading.Thread(
                        target=self._execute_count_task,
                        args=(cmd,),
                        daemon=True
                    ).start()

                elif action == "get_person_info":
                    self.task_active = True
                    self.task_id = action
                    threading.Thread(
                        target=self._execute_info_task,
                        args=(cmd,),
                        daemon=True
                    ).start()

                else:
                    print(f"[Vision] Unknown action: {action}")
                    self._publish_status("FAILED", error=f"Unknown action: {action}")

        except json.JSONDecodeError as e:
            print(f"[Vision] Invalid task cmd JSON: {e}")

    def _execute_search_task(self, cmd: dict):
        """执行搜索任务：持续检测直到找到目标"""
        timeout = cmd.get("timeout", 30.0)
        start_time = time.time()

        while time.time() - start_time < timeout:
            matches = [
                obj for obj in self.detected_objects
                if self.target_class and self.target_class in obj.get('class_name', '').lower()
                and obj.get('position_3d') is not None
            ]

            if matches:
                # 选择最近的目标
                target = min(matches, key=lambda o: o['position_3d'][2])

                # 发布检测结果到 /vision/detections_3d
                detection_msg = {
                    "type": "object_detection",
                    "name": target['class_name'],
                    "confidence": float(target['confidence']),
                    "position_3d": list(target['position_3d']),
                    "bbox": list(target['bbox']),
                }
                self._publish_detection(detection_msg)

                # 发布成功状态
                self._publish_status("SUCCESS", result=detection_msg)
                self._reset_task()
                return

            time.sleep(0.1)

        # 超时
        self._publish_status("FAILED", error=f"Target '{self.target_class}' not found")
        self._reset_task()

    def _execute_count_task(self, cmd: dict):
        """执行计数任务"""
        placement = cmd.get("placement", cmd.get("room", ""))
        category = cmd.get("category", "")

        matching = [
            obj for obj in self.detected_objects
            if category.lower() in obj.get('class_name', '').lower()
        ]

        result = {
            "type": "count_result",
            "category": category,
            "placement": placement,
            "count": len(matching),
            "items": [obj['class_name'] for obj in matching],
        }
        self._publish_status("SUCCESS", result=result)
        self._reset_task()

    def _execute_info_task(self, cmd: dict):
        """获取已检测对象/人物的信息"""
        # 直接返回当前帧检测结果
        result = {
            "type": "detection_info",
            "objects": [
                {
                    "name": obj['class_name'],
                    "confidence": float(obj['confidence']),
                    "position_3d": list(obj['position_3d']) if obj.get('position_3d') else None,
                }
                for obj in self.detected_objects
            ],
        }
        self._publish_status("SUCCESS", result=result)
        self._reset_task()

    def _reset_task(self):
        """重置任务状态"""
        with self._lock:
            self.task_active = False
            self.target_class = None
            self.task_id = None

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
            self.object_3d_pub.publish(point_msg)

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
        if (pixel_x < 0 or pixel_y < 0 or
                pixel_x >= self.intrinsics.width or
                pixel_y >= self.intrinsics.height):
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

    def process_frame(self):
        """处理单帧图像"""
        if not self.pipeline:
            return False

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return False

            color_image = np.asanyarray(color_frame.get_data())

            # YOLO 推理
            if self.model:
                results = self.model.predict(
                    source=color_image,
                    conf=self.args.conf,
                    iou=self.args.iou,
                    device=self.args.device,
                    verbose=False
                )
            else:
                return False

            display_image = color_image.copy()
            new_detections = []

            MIN_DEPTH = 0.2
            MAX_DEPTH = 1.0

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    conf = box.conf[0]

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # 3D 坐标
                    median_depth = self.get_median_depth_in_roi(
                        depth_frame, center_x, center_y
                    )
                    if median_depth is not None:
                        point_3d = self.get_3d_coordinates(
                            depth_frame, center_x, center_y, depth_value=median_depth
                        )
                    else:
                        point_3d = self.get_3d_coordinates(
                            depth_frame, center_x, center_y
                        )

                    if point_3d is not None:
                        x, y, z = point_3d
                        distance = math.sqrt(x**2 + y**2 + z**2)
                        if distance < MIN_DEPTH or distance > MAX_DEPTH:
                            continue

                    obj_info = {
                        'index': len(new_detections),
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'position_3d': point_3d,
                    }
                    new_detections.append(obj_info)

                    # 绘制
                    with self._lock:
                        is_target = (self.target_class is not None and
                                    self.target_class in class_name.lower())
                    color = (0, 0, 255) if is_target else (0, 255, 0)
                    thickness = 4 if is_target else 2

                    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)

                    label = f"#{obj_info['index']} {class_name} {conf:.2f}"
                    cv2.putText(display_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if point_3d is not None:
                        cx, cy, cz = point_3d
                        coord_text = f"({cx:.2f}, {cy:.2f}, {cz:.2f})m"
                        cv2.putText(display_image, coord_text, (x1, y2 + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # 更新检测结果
            with self._lock:
                self.detected_objects = new_detections

            # 显示
            if self.args.display:
                if not new_detections:
                    cv2.putText(display_image, "No objects detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('CADE Vision', display_image)

            return True

        except Exception as e:
            print(f"\nFrame processing error: {e}")
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

                # 检查退出
                if self.args.display:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            if self.pipeline:
                self.pipeline.stop()
            if self.args.display:
                cv2.destroyAllWindows()
            print("\nVision node stopped")


def main():
    parser = argparse.ArgumentParser(description="CADE Open Vision Node")
    parser.add_argument("--model", type=str, default="yolo11x-seg.pt",
                       help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU threshold")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--serial-number", type=str, default="333422301212",
                       help="RealSense serial number")
    parser.add_argument("--display", action="store_true", default=True,
                       help="Show detection window")
    parser.add_argument("--no-display", action="store_false", dest="display",
                       help="Hide detection window")

    args = parser.parse_args()
    node = OpenVisionNode(args)
    node.run()


if __name__ == "__main__":
    main()
