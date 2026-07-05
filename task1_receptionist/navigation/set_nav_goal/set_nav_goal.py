#!/usr/bin/env python3
"""发送目标坐标给 move_base。

参考 /home/nvidia/setgoal/set_nav_goal.py：
- 输入格式: x y yaw_deg
- 目标坐标系默认 map
- action server 默认 /move_base
"""

import argparse
import math
import sys


GOAL_STATUS_NAMES = {
    0: "PENDING",
    1: "ACTIVE",
    2: "PREEMPTED",
    3: "SUCCEEDED",
    4: "ABORTED",
    5: "REJECTED",
    6: "PREEMPTING",
    7: "RECALLING",
    8: "RECALLED",
    9: "LOST",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="发送目标坐标 x y yaw_deg 给 move_base。"
    )
    parser.add_argument("x", type=float, help="map 坐标系下的目标 x。")
    parser.add_argument("y", type=float, help="map 坐标系下的目标 y。")
    parser.add_argument("yaw_deg", type=float, help="目标朝向，角度制。")
    parser.add_argument("--map-frame", default="map", help="目标坐标系，默认 map。")
    parser.add_argument("--move-base", default="/move_base", help="move_base action 名称。")
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=10.0,
        help="等待 move_base action server 的秒数，默认 10.0。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="等待导航结果的秒数。0 表示一直等，和参考脚本一致。",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="只发送目标，不等待导航结果。",
    )
    return parser.parse_args()


def load_common_ros_modules():
    try:
        import rospy
        from tf.transformations import quaternion_from_euler
    except ImportError as exc:
        print(
            "缺少 ROS Python 依赖，请先 source ROS/catkin 环境后再运行: %s" % exc,
            file=sys.stderr,
        )
        raise SystemExit(2)
    return rospy, quaternion_from_euler


def yaw_deg_to_quaternion(quaternion_from_euler, yaw_deg):
    """输入角度制 yaw_deg，输出四元数。"""
    yaw_rad = math.radians(yaw_deg)
    return quaternion_from_euler(0.0, 0.0, yaw_rad)


def send_goal(args):
    rospy, quaternion_from_euler = load_common_ros_modules()
    try:
        import actionlib
        from actionlib_msgs.msg import GoalStatus
        from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    except ImportError as exc:
        print("缺少 move_base/actionlib 依赖: %s" % exc, file=sys.stderr)
        return 2

    rospy.init_node("task1_set_nav_goal", anonymous=True)

    client = actionlib.SimpleActionClient(args.move_base, MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server: %s", args.move_base)
    if not client.wait_for_server(rospy.Duration(args.server_timeout)):
        rospy.logerr("Cannot connect to move_base action server.")
        return 1

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = args.map_frame
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = args.x
    goal.target_pose.pose.position.y = args.y
    goal.target_pose.pose.position.z = 0.0

    q = yaw_deg_to_quaternion(quaternion_from_euler, args.yaw_deg)
    goal.target_pose.pose.orientation.x = q[0]
    goal.target_pose.pose.orientation.y = q[1]
    goal.target_pose.pose.orientation.z = q[2]
    goal.target_pose.pose.orientation.w = q[3]

    rospy.loginfo("Sending nav goal in %s:", args.map_frame)
    rospy.loginfo("  x       = %.3f", args.x)
    rospy.loginfo("  y       = %.3f", args.y)
    rospy.loginfo("  yaw_deg = %.3f", args.yaw_deg)

    client.send_goal(goal)
    if args.no_wait:
        rospy.loginfo("Goal sent. Not waiting for result.")
        return 0

    if args.timeout and args.timeout > 0.0:
        finished = client.wait_for_result(rospy.Duration(args.timeout))
    else:
        client.wait_for_result()
        finished = True

    if not finished:
        rospy.logwarn("Navigation timeout. Canceling goal.")
        client.cancel_goal()
        return 1

    state = client.get_state()
    state_name = GOAL_STATUS_NAMES.get(state, str(state))
    if state == GoalStatus.SUCCEEDED:
        rospy.loginfo("Navigation succeeded.")
        return 0

    rospy.logwarn("Navigation failed. GoalStatus = %d (%s)", state, state_name)
    return 1

def main():
    args = parse_args()
    return send_goal(args)


if __name__ == "__main__":
    raise SystemExit(main())
