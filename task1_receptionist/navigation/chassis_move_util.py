#!/usr/bin/env python3
"""直接发布底盘速度话题的小移动工具。

默认话题是当前机器人底盘控制链使用的:
    /motion_target/target_speed_chassis  geometry_msgs/TwistStamped

示例:
    python3 chassis_move_util.py forward 0.5
    python3 chassis_move_util.py back 0.3
    python3 chassis_move_util.py left 0.2
    python3 chassis_move_util.py right 0.2
    python3 chassis_move_util.py rotate 90
    python3 chassis_move_util.py rotate -45 --angular-speed-deg 20

说明:
    这是开环速度控制工具，距离/角度由速度和发布时间估算。
    正 x 为前进，正 y 为左移，正 angular.z 为逆时针旋转。
"""

import argparse
import math
import sys


DEFAULT_TOPIC = "/motion_target/target_speed_chassis"
DEFAULT_BRAKE_TOPIC = "/motion_target/brake_mode"
DEFAULT_FRAME_ID = "base_link"


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="发布底盘速度话题，让机器人旋转指定角度或平移指定距离。"
    )
    parser.add_argument(
        "command",
        choices=("forward", "back", "left", "right", "rotate", "stop"),
        help="动作: forward/back/left/right 的 value 单位是米；rotate 的 value 单位是角度。",
    )
    parser.add_argument(
        "value",
        nargs="?",
        type=float,
        default=0.0,
        help="距离或角度。stop 可省略。rotate 支持正负角度。",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="底盘速度话题，默认 %s。" % DEFAULT_TOPIC,
    )
    parser.add_argument(
        "--msg-type",
        choices=("twist_stamped", "twist"),
        default="twist_stamped",
        help="速度消息类型，当前机器人默认使用 twist_stamped。",
    )
    parser.add_argument(
        "--frame-id",
        default=DEFAULT_FRAME_ID,
        help="TwistStamped 的 header.frame_id，默认 base_link。",
    )
    parser.add_argument(
        "--linear-speed",
        type=positive_float,
        default=0.15,
        help="平移速度 m/s，默认 0.15。",
    )
    parser.add_argument(
        "--angular-speed-deg",
        type=positive_float,
        default=25.0,
        help="旋转角速度 deg/s，默认 25。",
    )
    parser.add_argument(
        "--rate",
        type=positive_float,
        default=20.0,
        help="发布频率 Hz，默认 20。",
    )
    parser.add_argument(
        "--stop-sec",
        type=positive_float,
        default=0.5,
        help="动作结束后持续发布零速的秒数，默认 0.5。",
    )
    parser.add_argument(
        "--wait-subscriber",
        type=float,
        default=3.0,
        help="等待底盘话题订阅者的秒数，默认 3。设为 0 则不等待。",
    )
    parser.add_argument(
        "--release-brake",
        action="store_true",
        default=True,
        help="动作期间向 brake topic 发布 False 释放刹车，默认启用。",
    )
    parser.add_argument(
        "--no-release-brake",
        dest="release_brake",
        action="store_false",
        help="不发布 brake topic。",
    )
    parser.add_argument(
        "--brake-topic",
        default=DEFAULT_BRAKE_TOPIC,
        help="刹车控制话题，默认 %s。" % DEFAULT_BRAKE_TOPIC,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要发布的速度和持续时间，不真正发布 ROS 话题。",
    )
    return parser.parse_args()


def load_ros_modules():
    try:
        import rospy
        from geometry_msgs.msg import Twist, TwistStamped
        from std_msgs.msg import Bool
    except ImportError as exc:
        print(
            "缺少 ROS Python 依赖，请先 source ROS 环境，例如: "
            "source /opt/ros/noetic/setup.bash。错误: %s" % exc,
            file=sys.stderr,
        )
        raise SystemExit(2)
    return rospy, Twist, TwistStamped, Bool


def build_motion(command, value, linear_speed, angular_speed_deg):
    vx = 0.0
    vy = 0.0
    wz = 0.0
    duration = 0.0

    if command == "stop":
        return vx, vy, wz, duration

    if command in ("forward", "back", "left", "right"):
        distance = float(value)
        if distance <= 0.0:
            raise ValueError("%s 的距离必须大于 0 米" % command)
        duration = distance / linear_speed
        if command == "forward":
            vx = linear_speed
        elif command == "back":
            vx = -linear_speed
        elif command == "left":
            vy = linear_speed
        elif command == "right":
            vy = -linear_speed
        return vx, vy, wz, duration

    if command == "rotate":
        angle_deg = float(value)
        if abs(angle_deg) <= 0.0:
            raise ValueError("rotate 的角度不能为 0")
        angular_speed = math.radians(angular_speed_deg)
        duration = abs(math.radians(angle_deg)) / angular_speed
        wz = angular_speed if angle_deg > 0.0 else -angular_speed
        return vx, vy, wz, duration

    raise ValueError("未知动作: %s" % command)


def make_twist(Twist, vx, vy, wz):
    msg = Twist()
    msg.linear.x = float(vx)
    msg.linear.y = float(vy)
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = float(wz)
    return msg


def make_cmd_msg(rospy, Twist, TwistStamped, msg_type, frame_id, vx, vy, wz):
    twist = make_twist(Twist, vx, vy, wz)
    if msg_type == "twist":
        return twist

    msg = TwistStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.twist = twist
    return msg


def wait_for_subscribers(rospy, pub, topic, timeout_sec):
    if timeout_sec <= 0.0:
        return True

    deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        if pub.get_num_connections() > 0:
            return True
        rate.sleep()

    rospy.logerr("等待 %.1f 秒后仍没有订阅者: %s", timeout_sec, topic)
    return False


def publish_for_duration(
    rospy,
    Twist,
    TwistStamped,
    pub,
    brake_pub,
    args,
    vx,
    vy,
    wz,
    duration,
):
    rate = rospy.Rate(args.rate)
    end_time = rospy.Time.now() + rospy.Duration(duration)

    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        if brake_pub is not None:
            brake_pub.publish(False)
        pub.publish(
            make_cmd_msg(
                rospy,
                Twist,
                TwistStamped,
                args.msg_type,
                args.frame_id,
                vx,
                vy,
                wz,
            )
        )
        rate.sleep()


def publish_stop(rospy, Twist, TwistStamped, pub, brake_pub, args):
    rate = rospy.Rate(args.rate)
    end_time = rospy.Time.now() + rospy.Duration(args.stop_sec)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        if brake_pub is not None:
            brake_pub.publish(False)
        pub.publish(
            make_cmd_msg(
                rospy,
                Twist,
                TwistStamped,
                args.msg_type,
                args.frame_id,
                0.0,
                0.0,
                0.0,
            )
        )
        rate.sleep()


def main():
    args = parse_args()

    try:
        vx, vy, wz, duration = build_motion(
            args.command,
            args.value,
            args.linear_speed,
            args.angular_speed_deg,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        "cmd=%s value=%.3f vx=%.3f m/s vy=%.3f m/s wz=%.3f rad/s duration=%.3f s topic=%s msg_type=%s"
        % (
            args.command,
            args.value,
            vx,
            vy,
            wz,
            duration,
            args.topic,
            args.msg_type,
        )
    )

    if args.dry_run:
        return 0

    rospy, Twist, TwistStamped, Bool = load_ros_modules()
    rospy.init_node("task1_chassis_move_util", anonymous=True)

    msg_cls = TwistStamped if args.msg_type == "twist_stamped" else Twist
    pub = rospy.Publisher(args.topic, msg_cls, queue_size=10)
    brake_pub = (
        rospy.Publisher(args.brake_topic, Bool, queue_size=10)
        if args.release_brake
        else None
    )

    if not wait_for_subscribers(rospy, pub, args.topic, args.wait_subscriber):
        publish_stop(rospy, Twist, TwistStamped, pub, brake_pub, args)
        return 1

    try:
        if duration > 0.0:
            publish_for_duration(
                rospy,
                Twist,
                TwistStamped,
                pub,
                brake_pub,
                args,
                vx,
                vy,
                wz,
                duration,
            )
    finally:
        publish_stop(rospy, Twist, TwistStamped, pub, brake_pub, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
