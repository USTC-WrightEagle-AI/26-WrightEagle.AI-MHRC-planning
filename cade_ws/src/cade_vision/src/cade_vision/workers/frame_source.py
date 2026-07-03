"""Frame source worker: owns the camera and publishes latest frames."""

import os
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except Exception:
    rs = None
    REALSENSE_AVAILABLE = False

import rospy
from sensor_msgs.msg import CameraInfo, Image

from cade_vision.runtime.ros_frames import make_camera_info, numpy_to_image


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class FrameSourceWorker:
    def __init__(self, args):
        rospy.init_node("cade_vision_frame_source", anonymous=True)
        self.args = args
        self.image_source = getattr(args, "image_source", "realsense")
        self.color_pub = rospy.Publisher("/vision/frame/color_task3", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/vision/frame/depth_task3", Image, queue_size=1)
        self.info_pub = rospy.Publisher("/vision/frame/camera_info_task3", CameraInfo, queue_size=1)
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.intrinsics = None
        self.static_image = None
        self.file_cap = None
        self.file_fps = 30.0
        self.usb_cap = None
        self.sequence = 0
        self._init_source()

    def _init_source(self):
        if self.image_source == "realsense":
            if REALSENSE_AVAILABLE:
                self._init_realsense()
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

    def _init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        serial_number = getattr(self.args, "serial_number", "")
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
        print(f"RealSense initialized - depth scale: {self.depth_scale}")

    def _init_file_source(self):
        file_path = getattr(self.args, "image_path", "")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext in VIDEO_EXTS:
            self.file_cap = cv2.VideoCapture(file_path)
            if not self.file_cap.isOpened():
                raise RuntimeError(f"Failed to open video: {file_path}")
            self.file_fps = self.file_cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self.static_image = cv2.imread(file_path)
            if self.static_image is None:
                raise ValueError(f"Failed to read image: {file_path}")

    def _init_usb_cam(self):
        cam_id = getattr(self.args, "usb_cam_id", 0)
        width = getattr(self.args, "width", 640)
        height = getattr(self.args, "height", 480)
        self.usb_cap = cv2.VideoCapture(cam_id)
        if not self.usb_cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera (id={cam_id})")
        self.usb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.usb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def _capture(self):
        if self.image_source == "realsense":
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            color_image = np.asanyarray(color_frame.get_data()).copy()
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(self.depth_scale)
            return color_image, depth_m
        if self.image_source == "file":
            if self.file_cap is not None:
                ret, frame = self.file_cap.read()
                if not ret or frame is None:
                    if getattr(self.args, "loop", False):
                        self.file_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.file_cap.read()
                    if not ret or frame is None:
                        return None, None
                return frame, None
            return self.static_image.copy(), None
        if self.image_source == "usb_cam":
            ret, frame = self.usb_cap.read()
            if not ret or frame is None:
                return None, None
            return frame, None
        return None, None

    def _camera_info(self, image, stamp):
        h, w = image.shape[:2]
        if self.intrinsics is not None:
            return make_camera_info(
                self.intrinsics.width,
                self.intrinsics.height,
                self.intrinsics.fx,
                self.intrinsics.fy,
                self.intrinsics.ppx,
                self.intrinsics.ppy,
                stamp=stamp,
            )
        return make_camera_info(w, h, w, w, w / 2.0, h / 2.0, stamp=stamp)

    def run(self):
        print("CADE frame source worker running")
        rate = rospy.Rate(60)
        try:
            while not rospy.is_shutdown():
                start = time.time()
                color_image, depth_m = self._capture()
                if color_image is None:
                    rate.sleep()
                    continue
                stamp = rospy.Time.now()
                self.sequence += 1
                color_msg = numpy_to_image(color_image, stamp=stamp, encoding="bgr8")
                color_msg.header.seq = self.sequence
                self.color_pub.publish(color_msg)
                self.info_pub.publish(self._camera_info(color_image, stamp))
                if depth_m is not None:
                    depth_msg = numpy_to_image(depth_m, stamp=stamp, encoding="32FC1")
                    depth_msg.header.seq = self.sequence
                    self.depth_pub.publish(depth_msg)
                if self.image_source == "file" and self.file_cap is not None:
                    target = (1.0 / max(self.file_fps, 1.0)) / max(float(getattr(self.args, "playback_speed", 1.0) or 1.0), 1e-6)
                    elapsed = time.time() - start
                    if elapsed < target:
                        rospy.sleep(target - elapsed)
                else:
                    rate.sleep()
        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            if self.file_cap is not None:
                self.file_cap.release()
            if self.usb_cap is not None:
                self.usb_cap.release()
