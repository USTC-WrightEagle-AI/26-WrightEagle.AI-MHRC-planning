#!/usr/bin/env python3
"""读取当前机器人在 map 下的坐标。

参考 /home/nvidia/setgoal/get_current_map_pose.py：
- 读取 TF: map -> base_link
- 输出 x y yaw_deg
- 最后一行格式可直接复制给 send_nav_goal.py
"""

import argparse
import math
import sys

DEFAULT_TIMEOUT_SEC = 5.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取当前机器人在 map 坐标系下的位置: x y yaw_deg。"
    )
    parser.add_argument("--map-frame", default="map", help="全局坐标系，默认 map。")
    parser.add_argument("--base-frame", default="base_link", help="机器人底盘坐标系，默认 base_link。")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help="等待 TF 的秒数，默认 5.0。",
    )
    return parser.parse_args()


def load_ros_modules():
    try:
        import rospy
        import tf
        from tf.transformations import euler_from_quaternion
    except ImportError as exc:
        print(
            "缺少 ROS Python 依赖，请先 source ROS/catkin 环境后再运行: %s" % exc,
            file=sys.stderr,
        )
        raise SystemExit(2)
    return rospy, tf, euler_from_quaternion


def normalize_angle_deg(angle):
    """将角度归一化到 [-180, 180)。"""
    while angle >= 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def read_pose(rospy, tf, listener, euler_from_quaternion, map_frame, base_frame, timeout):
    try:
        listener.waitForTransform(
            map_frame,
            base_frame,
            rospy.Time(0),
            rospy.Duration(timeout),
        )
    except tf.Exception as exc:
        raise RuntimeError("等待 TF 失败: %s" % exc)

    try:
        trans, rot = listener.lookupTransform(map_frame, base_frame, rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
        raise RuntimeError("读取 TF 失败: %s" % exc)

    _, _, yaw = euler_from_quaternion(rot)
    yaw_deg = normalize_angle_deg(math.degrees(yaw))
    return float(trans[0]), float(trans[1]), float(yaw_deg)


def print_pose(x, y, yaw_deg):
    print("")
    print("Current robot pose in map:")
    print("x:       %.3f" % x)
    print("y:       %.3f" % y)
    print("yaw_deg: %.3f" % yaw_deg)
    print("")
    print("Copy this format directly into set_nav_goal.py:")
    print("%.3f %.3f %.3f" % (x, y, yaw_deg))
    print("")


def main():
    args = parse_args()
    rospy, tf, euler_from_quaternion = load_ros_modules()

    rospy.init_node("task1_get_current_map_pose", anonymous=True)
    listener = tf.TransformListener()

    try:
        rospy.loginfo("Waiting for TF: %s -> %s", args.map_frame, args.base_frame)
        x, y, yaw_deg = read_pose(
            rospy,
            tf,
            listener,
            euler_from_quaternion,
            args.map_frame,
            args.base_frame,
            args.timeout,
        )
        print_pose(x, y, yaw_deg)
    except Exception as exc:
        print("读取当前坐标失败: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
