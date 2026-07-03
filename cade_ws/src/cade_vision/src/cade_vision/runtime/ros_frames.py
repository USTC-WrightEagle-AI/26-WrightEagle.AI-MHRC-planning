"""Minimal ROS Image conversion without cv_bridge."""

import numpy as np

try:
    import rospy
    from sensor_msgs.msg import CameraInfo, Image

    ROS_IMAGE_AVAILABLE = True
except Exception:
    rospy = None
    CameraInfo = Image = None
    ROS_IMAGE_AVAILABLE = False


def numpy_to_image(array, stamp=None, frame_id="camera_color_optical_frame", encoding=None):
    msg = Image()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    if encoding is None:
        encoding = "bgr8" if array.ndim == 3 else "32FC1"
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(array.strides[0])
    msg.data = array.tobytes()
    return msg


def image_to_numpy(msg):
    if msg.encoding == "bgr8":
        array = np.frombuffer(msg.data, dtype=np.uint8)
        return array.reshape(msg.height, msg.width, 3).copy()
    if msg.encoding in ("rgb8",):
        array = np.frombuffer(msg.data, dtype=np.uint8)
        return array.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
    if msg.encoding == "32FC1":
        array = np.frombuffer(msg.data, dtype=np.float32)
        return array.reshape(msg.height, msg.width).copy()
    if msg.encoding in ("16UC1", "mono16"):
        array = np.frombuffer(msg.data, dtype=np.uint16)
        return array.reshape(msg.height, msg.width).astype(np.float32) * 0.001
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def make_camera_info(width, height, fx, fy, cx, cy, stamp=None, frame_id="camera_color_optical_frame"):
    msg = CameraInfo()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)
    msg.K = [float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0]
    msg.P = [float(fx), 0.0, float(cx), 0.0, 0.0, float(fy), float(cy), 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def camera_info_dict(msg):
    if msg is None:
        return None
    return {
        "width": int(msg.width),
        "height": int(msg.height),
        "fx": float(msg.K[0]),
        "fy": float(msg.K[4]),
        "cx": float(msg.K[2]),
        "cy": float(msg.K[5]),
    }

