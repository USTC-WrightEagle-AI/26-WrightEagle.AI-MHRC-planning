"""
Camera Capture — grab RGB + 16-bit depth from RealSense on demand.

Supports two backends:
  1. pyrealsense2  — direct SDK access (preferred; no ROS topic dependency)
  2. ROS topics    — subscribe to existing /camera/* topics
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs

    _HAS_REALSENSE = True
except ImportError:
    rs = None
    _HAS_REALSENSE = False

try:
    import rospy
    from sensor_msgs.msg import CameraInfo, Image

    _HAS_ROS = True
except ImportError:
    rospy = None
    CameraInfo = Image = None
    _HAS_ROS = False


class CameraCapture:
    """On-demand RGB + 16-bit depth capture from RealSense camera."""

    def __init__(
        self,
        use_realsense: bool = True,
        rgb_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/aligned_depth_to_color/image_raw",
        camera_info_topic: str = "/camera/color/camera_info",
    ):
        self._use_realsense = use_realsense and _HAS_REALSENSE
        self._pipeline = None
        self._align = None
        self._profile = None

        # ROS fallback
        self._rgb_topic = rgb_topic
        self._depth_topic = depth_topic
        self._camera_info_topic = camera_info_topic
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_camera_info = None
        self._ros_lock = threading.Lock()

        if self._use_realsense:
            self._init_realsense()
        elif _HAS_ROS:
            self._init_ros_subscribers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        Capture one RGB + depth frame pair.

        Returns:
            rgb:          BGR uint8  (H, W, 3)
            depth:        16-bit uint16 (H, W) — millimetre units
            camera_info:  dict {width, height, fx, fy, cx, cy} or None
        """
        if self._use_realsense:
            return self._capture_realsense()
        if _HAS_ROS:
            return self._capture_ros()
        raise RuntimeError("No camera backend available (install pyrealsense2 or ROS)")

    @property
    def has_camera(self) -> bool:
        return self._use_realsense or _HAS_ROS

    # ------------------------------------------------------------------
    # pyrealsense2 backend
    # ------------------------------------------------------------------

    def _init_realsense(self) -> None:
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._profile = self._pipeline.start(config)

        # Align depth to color
        self._align = rs.align(rs.stream.color)

        # Warm up: discard first few frames for auto-exposure
        for _ in range(30):
            self._pipeline.wait_for_frames()

    def _capture_realsense(self) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to grab RealSense frames")

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())  # uint16 mm

        # Get intrinsics from the aligned color stream
        profile = color_frame.get_profile()
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        camera_info = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
        }

        return rgb, depth, camera_info

    # ------------------------------------------------------------------
    # ROS topic fallback
    # ------------------------------------------------------------------

    def _init_ros_subscribers(self) -> None:
        self._rgb_sub = rospy.Subscriber(
            self._rgb_topic, Image, self._on_rgb, queue_size=1
        )
        self._depth_sub = rospy.Subscriber(
            self._depth_topic, Image, self._on_depth, queue_size=1
        )
        self._cinfo_sub = rospy.Subscriber(
            self._camera_info_topic, CameraInfo, self._on_camera_info, queue_size=1
        )

    def _on_rgb(self, msg: Image) -> None:
        from cade_vision.runtime.ros_frames import image_to_numpy

        with self._ros_lock:
            self._latest_rgb = image_to_numpy(msg)

    def _on_depth(self, msg: Image) -> None:
        # Keep raw 16-bit — do NOT convert to float32 metres
        import numpy as np

        with self._ros_lock:
            if msg.encoding in ("16UC1", "mono16"):
                self._latest_depth = (
                    np.frombuffer(msg.data, dtype=np.uint16)
                    .reshape(msg.height, msg.width)
                    .copy()
                )
            elif msg.encoding == "32FC1":
                # Convert back to uint16 mm
                raw = (
                    np.frombuffer(msg.data, dtype=np.float32)
                    .reshape(msg.height, msg.width)
                    .copy()
                )
                self._latest_depth = (raw * 1000.0).astype(np.uint16)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._ros_lock:
            self._latest_camera_info = {
                "width": msg.width,
                "height": msg.height,
                "fx": msg.K[0],
                "fy": msg.K[4],
                "cx": msg.K[2],
                "cy": msg.K[5],
            }

    def _capture_ros(self) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        timeout = time.time() + 5.0
        while time.time() < timeout:
            with self._ros_lock:
                if (
                    self._latest_rgb is not None
                    and self._latest_depth is not None
                ):
                    return (
                        self._latest_rgb.copy(),
                        self._latest_depth.copy(),
                        self._latest_camera_info,
                    )
            time.sleep(0.01)

        raise RuntimeError(
            "Timed out waiting for RGB+depth images on ROS topics "
            f"({self._rgb_topic}, {self._depth_topic})"
        )
