#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move the left gripper near the latest detected hand/wrist for bag handover.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from point_empty_seat_arm import (
    DEFAULT_CURRENT_POSE_TOPIC,
    DEFAULT_EXTRINSIC,
    DEFAULT_ORIENTATION,
    DEFAULT_RESET_POSITION,
    DEFAULT_TARGET_TOPIC,
    DEFAULT_TF_BASE_FRAME,
    DEFAULT_TF_GRIPPER_FRAME,
    ask_yes,
    fmt_xyz,
    load_matrix,
    parse_xyz,
    pointing_orientation,
    print_target_summary,
    publish_pose,
    transform_point,
    vector_add,
    vector_norm,
    vector_scale,
    vector_sub,
    wait_for_current_gripper_xyz,
    wait_for_current_gripper_xyz_tf,
    normalized,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HAND_JSON = SCRIPT_DIR / "latest_handover_hand.json"
DEFAULT_LEFT_GRIPPER_TOPIC = "/motion_control/position_control_gripper_left"


def publish_gripper_position(topic: str, value: float, rate_hz: float, duration: float):
    import rospy
    from std_msgs.msg import Float32

    pub = rospy.Publisher(topic, Float32, queue_size=10)
    time.sleep(0.3)
    rate = rospy.Rate(rate_hz)
    end_time = time.time() + duration
    msg = Float32(data=float(value))
    while not rospy.is_shutdown() and time.time() < end_time:
        pub.publish(msg)
        rate.sleep()


def read_hand_target(hand_json: Path, extrinsic: Path) -> Tuple[List[float], Dict[str, Any]]:
    payload = json.loads(hand_json.read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"手腕识别状态不是 success: {payload.get('status')}")

    hand = payload.get("handover_target") or payload.get("best_hand") or {}
    position = hand.get("position") or {}
    frame = payload.get("coordinate_frame", "")

    for key in ("leftbase_xyz_m", "left_arm_base_xyz_m", "target_frame_xyz_m", "calibrated_xyz_m"):
        xyz = position.get(key)
        if isinstance(xyz, list) and len(xyz) == 3 and frame in {"leftbase", "left_arm_base"}:
            return [float(value) for value in xyz], hand

    camera_xyz = position.get("camera_xyz_m")
    if isinstance(camera_xyz, list) and len(camera_xyz) == 3:
        if not extrinsic.exists():
            raise FileNotFoundError(f"需要外参文件才能把相机坐标转到 leftbase: {extrinsic}")
        return transform_point(load_matrix(extrinsic), camera_xyz), hand

    raise RuntimeError("latest_handover_hand.json 中没有可用的接包目标坐标")


def xyz_from_position(position: Dict[str, Any]) -> List[float]:
    if not isinstance(position, dict):
        return []
    for key in ("leftbase_xyz_m", "left_arm_base_xyz_m", "target_frame_xyz_m", "calibrated_xyz_m"):
        xyz = position.get(key)
        if isinstance(xyz, list) and len(xyz) == 3:
            return [float(value) for value in xyz]
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move left arm near latest detected hand/wrist")
    parser.add_argument("--hand-json", type=Path, default=DEFAULT_HAND_JSON)
    parser.add_argument("--hand-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
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
    parser.add_argument(
        "--hand-standoff",
        type=float,
        default=0.14,
        help="Keep this many meters between the gripper target and the handover target",
    )
    parser.add_argument(
        "--max-arm-reach",
        type=float,
        default=0.70,
        help="Clamp the gripper target within this many meters from the left arm base",
    )
    parser.add_argument(
        "--hand-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Leftbase-frame offset added to the wrist point before planning",
    )
    parser.add_argument("--pose-timeout", type=float, default=3.0)
    parser.add_argument("--publish-rate", type=float, default=10.0)
    parser.add_argument("--publish-duration", type=float, default=3.0)
    parser.add_argument(
        "--orientation-mode",
        choices=("point", "fixed"),
        default="point",
        help="point: compute quaternion so the selected tool axis points to the wrist; fixed: use --orientation",
    )
    parser.add_argument("--tool-axis", default="+x", help="End-effector local axis used as the pointing direction, e.g. +x or -x")
    parser.add_argument("--up-reference", nargs=3, type=float, default=(0.0, 0.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--orientation", nargs=4, type=float, default=DEFAULT_ORIENTATION, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--reset-orientation", nargs=4, type=float, default=DEFAULT_ORIENTATION, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--reset-xyz", nargs=3, type=float, default=DEFAULT_RESET_POSITION, metavar=("X", "Y", "Z"))
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--gripper-topic", default=DEFAULT_LEFT_GRIPPER_TOPIC)
    parser.add_argument("--open-gripper-value", type=float, default=100.0)
    parser.add_argument("--close-gripper-value", type=float, default=0.0)
    parser.add_argument("--gripper-command-rate", type=float, default=10.0)
    parser.add_argument("--gripper-command-duration", type=float, default=1.0)
    parser.add_argument("--gripper-close-delay", type=float, default=5.0)
    parser.add_argument("--no-gripper", action="store_true", help="Do not publish gripper open/close commands")
    parser.add_argument("--yes", action="store_true", help="Assume yes for publish/reset confirmations")
    parser.add_argument("--dry-run", action="store_true", help="Only print computed targets; do not publish")
    return parser


def main():
    args = build_parser().parse_args()

    if args.gripper_command_rate <= 0.0:
        raise RuntimeError("--gripper-command-rate 必须大于 0")
    if args.gripper_command_duration <= 0.0:
        raise RuntimeError("--gripper-command-duration 必须大于 0")
    if args.gripper_close_delay < 0.0:
        raise RuntimeError("--gripper-close-delay 不能小于 0")
    if args.max_arm_reach <= 0.0:
        raise RuntimeError("--max-arm-reach 必须大于 0")

    current_xyz = parse_xyz(args.current_xyz)
    current_source = args.current_source
    needs_ros = not args.dry_run
    if current_xyz is not None:
        current_source = "manual"
    elif current_source == "tf":
        needs_ros = True
    elif current_source == "topic":
        needs_ros = True
    else:
        raise RuntimeError("--current-source manual 需要同时提供 --current-xyz X Y Z")

    if needs_ros:
        import rospy

        rospy.init_node("approach_handover_arm_left", anonymous=True, disable_signals=True)

    if current_xyz is None:
        if current_source == "tf":
            current_xyz = wait_for_current_gripper_xyz_tf(
                args.tf_base_frame, args.tf_gripper_frame, args.pose_timeout
            )
        elif current_source == "topic":
            current_xyz = wait_for_current_gripper_xyz(args.current_pose_topic, args.pose_timeout)

    hand_record: Dict[str, Any] = {}
    if args.hand_xyz is not None:
        raw_hand_xyz = [float(value) for value in args.hand_xyz]
    else:
        raw_hand_xyz, hand_record = read_hand_target(args.hand_json, args.extrinsic)
    hand_offset = [float(value) for value in args.hand_offset]
    hand_xyz = vector_add(raw_hand_xyz, hand_offset)

    direction = vector_sub(hand_xyz, current_xyz)
    hand_distance = vector_norm(direction)
    unit_direction = normalized(direction)
    travel = max(0.0, hand_distance - args.hand_standoff)
    planned_target_xyz = vector_add(current_xyz, vector_scale(unit_direction, travel))
    planned_target_reach = vector_norm(planned_target_xyz)
    target_xyz = planned_target_xyz
    reach_limited = planned_target_reach > args.max_arm_reach
    if reach_limited:
        target_xyz = vector_scale(planned_target_xyz, args.max_arm_reach / planned_target_reach)
    target_reach = vector_norm(target_xyz)
    target_to_hand = vector_sub(hand_xyz, target_xyz)
    target_orientation = (
        pointing_orientation(target_to_hand, args.tool_axis, args.up_reference)
        if args.orientation_mode == "point"
        else [float(value) for value in args.orientation]
    )

    print("\n接包手腕靠近规划")
    print(f"  当前左夹爪: {fmt_xyz(current_xyz)}")
    if current_source == "tf":
        print(f"  当前夹爪来源: TF {args.tf_base_frame} -> {args.tf_gripper_frame}")
    elif current_source == "topic":
        print(f"  当前夹爪来源: topic {args.current_pose_topic}")
    else:
        print("  当前夹爪来源: manual --current-xyz")
    print(f"  接包目标坐标(leftbase): {fmt_xyz(raw_hand_xyz)}")
    if hand_record:
        print(
            "  目标类型: "
            f"{hand_record.get('target_type')} "
            f"priority={hand_record.get('selection_priority')} "
            f"{hand_record.get('selection_label')}"
        )
        object_name = hand_record.get("object_class_name") or hand_record.get("bag_class_name")
        object_conf = hand_record.get("object_confidence")
        if object_conf is None:
            object_conf = hand_record.get("bag_confidence")
        print(f"  接包坐标来源: {hand_record.get('handover_position_source') or 'position'}")
        wrist_xyz = xyz_from_position(hand_record.get("wrist_position") or {})
        if wrist_xyz:
            print(f"  手腕坐标(leftbase): {fmt_xyz(wrist_xyz)}")
        object_xyz = xyz_from_position(hand_record.get("matched_object_position") or {})
        if object_xyz:
            print(f"  关联物体坐标(leftbase): {fmt_xyz(object_xyz)}")
        print(
            "  关联物体: "
            f"{object_name} "
            f"conf={object_conf} "
            f"bbox={hand_record.get('object_bbox') or hand_record.get('bag_bbox')}"
        )
        print(
            "  手-物体距离(px): "
            f"edge={hand_record.get('hand_object_distance_px') or hand_record.get('hand_bag_distance_px')} "
            f"center={hand_record.get('hand_object_center_distance_px') or hand_record.get('hand_bag_center_distance_px')}"
        )
        print(
            "  手腕: "
            f"{hand_record.get('hand_side')} "
            f"conf={hand_record.get('confidence')}"
        )
    if vector_norm(hand_offset) > 1e-9:
        print(f"  接包目标偏移后: {fmt_xyz(hand_xyz)}")
    print(f"  靠近单位向量: {fmt_xyz(unit_direction)}")
    print(f"  夹爪到接包目标距离: {hand_distance:.3f} m")
    print(f"  保留接包距离: {args.hand_standoff:.3f} m")
    print(f"  本次移动距离: {travel:.3f} m")
    print(f"  机械臂伸出限制: {args.max_arm_reach:.3f} m")
    if reach_limited:
        print(
            "  伸出距离过长，已缩回: "
            f"{planned_target_reach:.3f} m -> {target_reach:.3f} m"
        )
    else:
        print(f"  目标伸出距离: {target_reach:.3f} m")
    print(f"  姿态模式: {args.orientation_mode}")
    if args.orientation_mode == "point":
        print(f"  指示轴: 末端本地 {args.tool_axis} 轴对准接包目标")
    if args.hand_xyz is not None:
        print("  手腕来源: manual --hand-xyz")
    else:
        print(f"  接包目标来源: {args.hand_json}")
    print(f"  外参文件: {args.extrinsic}")

    if args.no_gripper:
        print("  夹爪动作: 已禁用 (--no-gripper)")
    else:
        print("  夹爪动作:")
        print(f"    topic: {args.gripper_topic}")
        print(f"    打开值: {args.open_gripper_value:.3f}")
        print(f"    关闭值: {args.close_gripper_value:.3f}")
        print(f"    每次夹爪命令持续: {args.gripper_command_duration:.1f}s")
        print(f"    机械臂目标发布完成后等待: {args.gripper_close_delay:.1f}s 再关闭夹爪")

    print_target_summary("准备发布接包手腕目标", target_xyz, target_orientation, args)

    if args.dry_run:
        print("\nDry run: 不发布机械臂或夹爪指令。")
        return

    if not ask_yes(
        "输入 y 执行接包动作: 打开夹爪 -> 发布手腕目标 -> 等待后关闭夹爪；其它输入取消: ",
        args.yes,
    ):
        print("已取消接包动作。")
        return

    if not args.no_gripper:
        publish_gripper_position(
            args.gripper_topic,
            args.open_gripper_value,
            args.gripper_command_rate,
            args.gripper_command_duration,
        )
        print(
            f"已发布打开夹爪命令 value={args.open_gripper_value:.3f} "
            f"{args.gripper_command_duration:.1f}s 到 {args.gripper_topic}"
        )

    publish_pose(args.target_topic, target_xyz, target_orientation, args.publish_rate, args.publish_duration)
    print(f"已发布接包手腕目标 {args.publish_duration:.1f}s 到 {args.target_topic}")

    if not args.no_gripper:
        print(f"等待 {args.gripper_close_delay:.1f}s 后关闭夹爪...")
        rospy.sleep(args.gripper_close_delay)
        publish_gripper_position(
            args.gripper_topic,
            args.close_gripper_value,
            args.gripper_command_rate,
            args.gripper_command_duration,
        )
        print(
            f"已发布关闭夹爪命令 value={args.close_gripper_value:.3f} "
            f"{args.gripper_command_duration:.1f}s 到 {args.gripper_topic}"
        )

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
