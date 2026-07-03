#!/usr/bin/env python3
"""Print the current robot pose in the map frame."""

import argparse
import sys

import rospy

from cade_navigation.pose_utils import read_current_map_pose


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read current robot pose as x y yaw_deg."
    )
    parser.add_argument("--map-frame", default="map", help="Global frame.")
    parser.add_argument(
        "--base-frame",
        default="base_link_fusion",
        help="Robot base frame used by the navigation stack.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for TF.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("cade_get_current_map_pose", anonymous=True)
    try:
        x, y, yaw_deg = read_current_map_pose(
            map_frame=args.map_frame,
            base_frame=args.base_frame,
            timeout=args.timeout,
        )
    except Exception as exc:
        print("读取当前坐标失败: %s" % exc, file=sys.stderr)
        return 1

    print("")
    print("Current robot pose in %s:" % args.map_frame)
    print("x:       %.3f" % x)
    print("y:       %.3f" % y)
    print("yaw_deg: %.3f" % yaw_deg)
    print("")
    print("%.3f %.3f %.3f" % (x, y, yaw_deg))
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
