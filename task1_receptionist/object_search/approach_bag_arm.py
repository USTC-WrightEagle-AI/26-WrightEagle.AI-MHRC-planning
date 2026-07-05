#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move the left gripper near the latest detected bag.
"""

import argparse
import json
from pathlib import Path
from typing import List

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
DEFAULT_BAG_JSON = SCRIPT_DIR / "latest_bag.json"


def read_bag_xyz(bag_json: Path, extrinsic: Path) -> List[float]:
    payload = json.loads(bag_json.read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"包识别状态不是 success: {payload.get('status')}")

    bag = payload.get("best_bag") or {}
    position = bag.get("position") or {}
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

    raise RuntimeError("latest_bag.json 中没有可用的包坐标")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move left arm near latest detected bag")
    parser.add_argument("--bag-json", type=Path, default=DEFAULT_BAG_JSON)
    parser.add_argument("--bag-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
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
        "--bag-standoff",
        type=float,
        default=0.10,
        help="Keep this many meters between the target gripper position and the bag center",
    )
    parser.add_argument(
        "--bag-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Leftbase-frame offset added to the bag point before planning",
    )
    parser.add_argument("--pose-timeout", type=float, default=3.0)
    parser.add_argument("--publish-rate", type=float, default=10.0)
    parser.add_argument("--publish-duration", type=float, default=3.0)
    parser.add_argument(
        "--orientation-mode",
        choices=("point", "fixed"),
        default="point",
        help="point: compute quaternion so the selected tool axis points to the bag; fixed: use --orientation",
    )
    parser.add_argument("--tool-axis", default="+x", help="End-effector local axis used as the pointing direction, e.g. +x or -x")
    parser.add_argument("--up-reference", nargs=3, type=float, default=(0.0, 0.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--orientation", nargs=4, type=float, default=DEFAULT_ORIENTATION, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--reset-orientation", nargs=4, type=float, default=DEFAULT_ORIENTATION, metavar=("X", "Y", "Z", "W"))
    parser.add_argument("--reset-xyz", nargs=3, type=float, default=DEFAULT_RESET_POSITION, metavar=("X", "Y", "Z"))
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print computed targets; do not publish")
    return parser


def main():
    args = build_parser().parse_args()

    import rospy

    rospy.init_node("approach_bag_arm_left", anonymous=True, disable_signals=True)

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

    raw_bag_xyz = (
        [float(value) for value in args.bag_xyz]
        if args.bag_xyz is not None
        else read_bag_xyz(args.bag_json, args.extrinsic)
    )
    bag_offset = [float(value) for value in args.bag_offset]
    bag_xyz = vector_add(raw_bag_xyz, bag_offset)

    direction = vector_sub(bag_xyz, current_xyz)
    bag_distance = vector_norm(direction)
    unit_direction = normalized(direction)
    travel = max(0.0, bag_distance - args.bag_standoff)
    target_xyz = vector_add(current_xyz, vector_scale(unit_direction, travel))
    target_to_bag = vector_sub(bag_xyz, target_xyz)
    target_orientation = (
        pointing_orientation(target_to_bag, args.tool_axis, args.up_reference)
        if args.orientation_mode == "point"
        else [float(value) for value in args.orientation]
    )

    print("\n接包靠近规划")
    print(f"  当前左夹爪: {fmt_xyz(current_xyz)}")
    if current_source == "tf":
        print(f"  当前夹爪来源: TF {args.tf_base_frame} -> {args.tf_gripper_frame}")
    elif current_source == "topic":
        print(f"  当前夹爪来源: topic {args.current_pose_topic}")
    else:
        print("  当前夹爪来源: manual --current-xyz")
    print(f"  包坐标(leftbase): {fmt_xyz(raw_bag_xyz)}")
    if vector_norm(bag_offset) > 1e-9:
        print(f"  包目标偏移后: {fmt_xyz(bag_xyz)}")
    print(f"  靠近单位向量: {fmt_xyz(unit_direction)}")
    print(f"  夹爪到包距离: {bag_distance:.3f} m")
    print(f"  保留包边距离: {args.bag_standoff:.3f} m")
    print(f"  本次移动距离: {travel:.3f} m")
    print(f"  姿态模式: {args.orientation_mode}")
    if args.orientation_mode == "point":
        print(f"  指示轴: 末端本地 {args.tool_axis} 轴对准包")
    if args.bag_xyz is not None:
        print("  包来源: manual --bag-xyz")
    else:
        print(f"  包来源: {args.bag_json}")
    print(f"  外参文件: {args.extrinsic}")

    print_target_summary("准备发布接包目标", target_xyz, target_orientation, args)

    if args.dry_run:
        print("\nDry run: 不发布机械臂指令。")
        return

    if not ask_yes("输入 y 发布接包目标，其它输入取消: "):
        print("已取消接包动作。")
        return

    publish_pose(args.target_topic, target_xyz, target_orientation, args.publish_rate, args.publish_duration)
    print(f"已发布接包目标 {args.publish_duration:.1f}s 到 {args.target_topic}")

    if args.no_reset:
        return

    reset_xyz = parse_xyz(args.reset_xyz)
    print_target_summary("准备发布复位目标", reset_xyz, args.reset_orientation, args)
    if ask_yes("输入 y 发布复位目标，其它输入跳过复位: "):
        publish_pose(args.target_topic, reset_xyz, args.reset_orientation, args.publish_rate, args.publish_duration)
        print(f"已发布复位目标 {args.publish_duration:.1f}s 到 {args.target_topic}")
    else:
        print("已跳过复位。")


if __name__ == "__main__":
    main()
