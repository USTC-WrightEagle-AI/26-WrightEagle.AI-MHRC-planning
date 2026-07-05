#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point the left gripper toward the latest detected empty seat.

The script is intentionally conservative: it prints every target pose and only
publishes after an explicit "y" confirmation.
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEAT_JSON = SCRIPT_DIR / "latest_empty_seat.json"
DEFAULT_EXTRINSIC = SCRIPT_DIR / "camera_middle_to_leftbase.txt"
DEFAULT_CURRENT_POSE_TOPIC = "/relaxed_ik/motion_control/pose_ee_arm_left"
DEFAULT_TARGET_TOPIC = "/motion_target/target_pose_arm_left"
DEFAULT_TF_BASE_FRAME = "left_arm_base_link"
DEFAULT_TF_GRIPPER_FRAME = "left_gripper_link"

DEFAULT_ORIENTATION = (0.707, 0.0, 0.0, -0.707)
DEFAULT_RESET_POSITION = (0.1, 0.0, 0.2)
AXIS_TO_INDEX_SIGN = {
    "x": (0, 1.0),
    "+x": (0, 1.0),
    "-x": (0, -1.0),
    "y": (1, 1.0),
    "+y": (1, 1.0),
    "-y": (1, -1.0),
    "z": (2, 1.0),
    "+z": (2, 1.0),
    "-z": (2, -1.0),
}


def load_matrix(path: Path) -> List[List[float]]:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        rows.append(
            [
                float(value)
                for value in re.split(r"[,\s]+", line.replace("[", "").replace("]", ""))
                if value
            ]
        )
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"{path} 应为 4x4 外参矩阵")
    return rows


def transform_point(matrix: List[List[float]], point_xyz: Iterable[float]) -> List[float]:
    x, y, z = [float(value) for value in point_xyz]
    point = [x, y, z, 1.0]
    return [sum(matrix[row][col] * point[col] for col in range(4)) for row in range(3)]


def vector_sub(a: Iterable[float], b: Iterable[float]) -> List[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def vector_add(a: Iterable[float], b: Iterable[float]) -> List[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def vector_scale(v: Iterable[float], scale: float) -> List[float]:
    return [float(x) * scale for x in v]


def vector_norm(v: Iterable[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def normalized(v: Iterable[float]) -> List[float]:
    length = vector_norm(v)
    if length <= 1e-9:
        raise ValueError("当前夹爪位置和椅子位置太近，无法计算指向方向")
    return [float(x) / length for x in v]


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def cross(a: Iterable[float], b: Iterable[float]) -> List[float]:
    ax, ay, az = [float(value) for value in a]
    bx, by, bz = [float(value) for value in b]
    return [
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]


def projected_reference(direction: Iterable[float], preferred: Iterable[float]) -> List[float]:
    direction = normalized(direction)
    for candidate in (preferred, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
        projection = vector_sub(candidate, vector_scale(direction, dot(candidate, direction)))
        if vector_norm(projection) > 1e-6:
            return normalized(projection)
    raise ValueError("无法构造稳定姿态参考轴")


def quaternion_from_rotation_matrix(matrix: List[List[float]]) -> List[float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    quat = [qx, qy, qz, qw]
    return normalized(quat)


def pointing_orientation(
    direction: Iterable[float],
    tool_axis: str,
    up_reference: Iterable[float],
) -> List[float]:
    axis = tool_axis.strip().lower()
    if axis not in AXIS_TO_INDEX_SIGN:
        raise ValueError(f"--tool-axis 只支持: {', '.join(sorted(AXIS_TO_INDEX_SIGN))}")

    axis_index, sign = AXIS_TO_INDEX_SIGN[axis]
    tool_axis_in_base = vector_scale(normalized(direction), sign)
    basis = [None, None, None]
    basis[axis_index] = tool_axis_in_base

    if axis_index == 0:
        basis[2] = projected_reference(tool_axis_in_base, up_reference)
        basis[1] = normalized(cross(basis[2], basis[0]))
    elif axis_index == 1:
        basis[2] = projected_reference(tool_axis_in_base, up_reference)
        basis[0] = normalized(cross(basis[1], basis[2]))
    else:
        basis[1] = projected_reference(tool_axis_in_base, up_reference)
        basis[0] = normalized(cross(basis[1], basis[2]))

    # Rotation matrix columns are local X/Y/Z axes expressed in the base frame.
    rotation = [
        [basis[0][0], basis[1][0], basis[2][0]],
        [basis[0][1], basis[1][1], basis[2][1]],
        [basis[0][2], basis[1][2], basis[2][2]],
    ]
    return quaternion_from_rotation_matrix(rotation)


def fmt_xyz(xyz: Iterable[float]) -> str:
    x, y, z = [float(value) for value in xyz]
    return f"x={x:.3f}, y={y:.3f}, z={z:.3f}"


def read_empty_seat_xyz(seat_json: Path, extrinsic: Path) -> List[float]:
    payload = json.loads(seat_json.read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"空椅识别状态不是 success: {payload.get('status')}")

    seat = payload.get("best_empty_seat") or {}
    position = seat.get("position") or {}
    frame = payload.get("coordinate_frame", "")

    for key in ("leftbase_xyz_m", "left_arm_base_xyz_m", "target_frame_xyz_m", "calibrated_xyz_m"):
        xyz = position.get(key)
        if isinstance(xyz, list) and len(xyz) == 3 and frame in {"leftbase", "left_arm_base"}:
            return [float(value) for value in xyz]

    camera_xyz = position.get("camera_xyz_m")
    if isinstance(camera_xyz, list) and len(camera_xyz) == 3:
        if not extrinsic.exists():
            raise FileNotFoundError(f"需要外参文件才能把相机坐标转到 leftbase: {extrinsic}")
        return transform_point(load_matrix(extrinsic), camera_xyz)

    raise RuntimeError("latest_empty_seat.json 中没有可用的椅子坐标")


def wait_for_current_gripper_xyz(topic: str, timeout: float) -> List[float]:
    import rospy
    from geometry_msgs.msg import PoseStamped

    msg = rospy.wait_for_message(topic, PoseStamped, timeout=timeout)
    return [
        float(msg.pose.position.x),
        float(msg.pose.position.y),
        float(msg.pose.position.z),
    ]


def wait_for_current_gripper_xyz_tf(base_frame: str, gripper_frame: str, timeout: float) -> List[float]:
    import rospy
    import tf

    listener = tf.TransformListener()
    deadline = time.time() + timeout
    last_error = None
    while not rospy.is_shutdown() and time.time() < deadline:
        try:
            listener.waitForTransform(
                base_frame,
                gripper_frame,
                rospy.Time(0),
                rospy.Duration(min(0.5, max(0.01, deadline - time.time()))),
            )
            translation, _rotation = listener.lookupTransform(
                base_frame, gripper_frame, rospy.Time(0)
            )
            return [float(value) for value in translation]
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            last_error = exc
            time.sleep(0.05)

    raise RuntimeError(
        f"无法从 TF 读取当前左夹爪坐标: {base_frame} -> {gripper_frame}, last_error={last_error}"
    )


def make_pose_stamped(xyz: Iterable[float], orientation: Iterable[float]):
    import rospy
    from geometry_msgs.msg import PoseStamped

    x, y, z = [float(value) for value in xyz]
    qx, qy, qz, qw = [float(value) for value in orientation]
    msg = PoseStamped()
    msg.header.seq = 0
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = ""
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def publish_pose(topic: str, xyz: Iterable[float], orientation: Iterable[float], rate_hz: float, duration: float):
    import rospy
    from geometry_msgs.msg import PoseStamped

    pub = rospy.Publisher(topic, PoseStamped, queue_size=10)
    time.sleep(0.3)
    rate = rospy.Rate(rate_hz)
    end_time = time.time() + duration
    while not rospy.is_shutdown() and time.time() < end_time:
        msg = make_pose_stamped(xyz, orientation)
        pub.publish(msg)
        rate.sleep()


def print_target_summary(title: str, xyz: Iterable[float], orientation: Iterable[float], args: argparse.Namespace):
    qx, qy, qz, qw = [float(value) for value in orientation]
    print(f"\n{title}")
    print(f"  目标坐标: {fmt_xyz(xyz)}")
    print(f"  四元数: x={qx:.3f}, y={qy:.3f}, z={qz:.3f}, w={qw:.3f}")


def ask_yes(prompt: str, assume_yes: bool = False) -> bool:
    if assume_yes:
        print(f"{prompt}y (auto)")
        return True
    return input(prompt).strip().lower() == "y"


def parse_xyz(values: Optional[List[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("xyz 需要 3 个数字")
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point left arm toward latest empty seat")
    parser.add_argument("--seat-json", type=Path, default=DEFAULT_SEAT_JSON)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    parser.add_argument(
        "--current-source",
        choices=("tf", "topic", "manual"),
        default="tf",
        help="tf: read left gripper from TF; topic: read PoseStamped topic; manual: use --current-xyz",
    )
    parser.add_argument("--tf-base-frame", default=DEFAULT_TF_BASE_FRAME)
    parser.add_argument("--tf-gripper-frame", default=DEFAULT_TF_GRIPPER_FRAME)
    parser.add_argument("--current-pose-topic", default=DEFAULT_CURRENT_POSE_TOPIC)
    parser.add_argument("--target-topic", default=DEFAULT_TARGET_TOPIC)
    parser.add_argument("--current-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--point-distance", type=float, default=0.4, help="Meters to move from current gripper toward the chair")
    parser.add_argument("--pose-timeout", type=float, default=3.0)
    parser.add_argument("--publish-rate", type=float, default=10.0)
    parser.add_argument("--publish-duration", type=float, default=3.0)
    parser.add_argument(
        "--orientation-mode",
        choices=("point", "fixed"),
        default="point",
        help="point: compute quaternion so the selected tool axis points to the chair; fixed: use --orientation",
    )
    parser.add_argument("--tool-axis", default="+x", help="End-effector local axis used as the pointing direction, e.g. +x or -x")
    parser.add_argument("--up-reference", nargs=3, type=float, default=(0.0, 0.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--orientation", nargs=4, type=float, default=DEFAULT_ORIENTATION, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--reset-orientation", nargs=4, type=float, default=DEFAULT_ORIENTATION, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--reset-xyz", nargs=3, type=float, default=DEFAULT_RESET_POSITION, metavar=("X", "Y", "Z"))
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Assume yes for publish/reset confirmations")
    parser.add_argument("--dry-run", action="store_true", help="Only print computed targets; do not publish")
    return parser


def main():
    args = build_parser().parse_args()

    import rospy

    rospy.init_node("point_empty_seat_arm_left", anonymous=True, disable_signals=True)

    current_xyz = parse_xyz(args.current_xyz)
    current_source = args.current_source
    if current_xyz is not None:
        current_source = "manual"
    elif current_source == "tf":
        current_xyz = wait_for_current_gripper_xyz_tf(
            args.tf_base_frame, args.tf_gripper_frame, args.pose_timeout
        )
    elif current_source == "topic":
        current_xyz = wait_for_current_gripper_xyz(args.current_pose_topic, args.pose_timeout)
    else:
        raise RuntimeError("--current-source manual 需要同时提供 --current-xyz X Y Z")

    chair_xyz = read_empty_seat_xyz(args.seat_json, args.extrinsic)
    direction = vector_sub(chair_xyz, current_xyz)
    chair_distance = vector_norm(direction)
    unit_direction = normalized(direction)
    travel = args.point_distance
    if travel <= 1e-6:
        raise RuntimeError("--point-distance 必须大于 0")
    target_xyz = vector_add(current_xyz, vector_scale(unit_direction, travel))
    target_to_chair = vector_sub(chair_xyz, target_xyz)
    target_orientation = (
        pointing_orientation(target_to_chair, args.tool_axis, args.up_reference)
        if args.orientation_mode == "point"
        else [float(value) for value in args.orientation]
    )

    print("\n空椅指向规划")
    print(f"  当前左夹爪: {fmt_xyz(current_xyz)}")
    if current_source == "tf":
        print(f"  当前夹爪来源: TF {args.tf_base_frame} -> {args.tf_gripper_frame}")
    elif current_source == "topic":
        print(f"  当前夹爪来源: topic {args.current_pose_topic}")
    else:
        print("  当前夹爪来源: manual --current-xyz")
    print(f"  椅子坐标(leftbase): {fmt_xyz(chair_xyz)}")
    print(f"  指向单位向量: {fmt_xyz(unit_direction)}")
    print(f"  夹爪到椅子距离: {chair_distance:.3f} m")
    print(f"  本次伸出距离: {travel:.3f} m")
    print(f"  姿态模式: {args.orientation_mode}")
    if args.orientation_mode == "point":
        print(f"  指示轴: 末端本地 {args.tool_axis} 轴对准椅子")
    print(f"  椅子来源: {args.seat_json}")
    print(f"  外参文件: {args.extrinsic}")

    print_target_summary("准备发布指向目标", target_xyz, target_orientation, args)

    if args.dry_run:
        print("\nDry run: 不发布机械臂指令。")
        return

    if not ask_yes("输入 y 发布指向目标，其它输入取消: ", args.yes):
        print("已取消指向动作。")
        return

    publish_pose(args.target_topic, target_xyz, target_orientation, args.publish_rate, args.publish_duration)
    print(f"已发布指向目标 {args.publish_duration:.1f}s 到 {args.target_topic}")

    if args.no_reset:
        return

    reset_xyz = parse_xyz(args.reset_xyz)
    print_target_summary("准备发布复位目标", reset_xyz, args.reset_orientation, args)

    if ask_yes("输入 y 发布复位目标，其它输入跳过复位: ", args.yes):
        publish_pose(args.target_topic, reset_xyz, args.reset_orientation, args.publish_rate, args.publish_duration)
        print(f"已发布复位目标 {args.publish_duration:.1f}s 到 {args.target_topic}")
    else:
        print("已跳过复位。")


if __name__ == "__main__":
    main()
