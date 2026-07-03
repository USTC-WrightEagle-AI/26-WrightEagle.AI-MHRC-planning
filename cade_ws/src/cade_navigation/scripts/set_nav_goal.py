#!/usr/bin/env python3
"""Send one x y yaw_deg goal to move_base."""

import argparse

import rospy

from cade_navigation.move_base_client import MoveBaseClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send x y yaw_deg navigation goal to move_base."
    )
    parser.add_argument("x", type=float, help="Target x in map frame.")
    parser.add_argument("y", type=float, help="Target y in map frame.")
    parser.add_argument("yaw_deg", type=float, help="Target yaw in degrees.")
    parser.add_argument("--map-frame", default="map", help="Goal frame.")
    parser.add_argument("--move-base", default="/move_base", help="Action server.")
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for action server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for result. 0 waits forever.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("cade_set_nav_goal", anonymous=True)
    client = MoveBaseClient(
        action_name=args.move_base,
        server_timeout=args.server_timeout,
    )
    result = client.send_goal_and_wait(
        args.x,
        args.y,
        args.yaw_deg,
        frame_id=args.map_frame,
        timeout=args.timeout,
    )
    if result.get("status") == "SUCCESS":
        rospy.loginfo("Navigation succeeded: %s", result)
        return 0

    rospy.logwarn("Navigation failed: %s", result)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
