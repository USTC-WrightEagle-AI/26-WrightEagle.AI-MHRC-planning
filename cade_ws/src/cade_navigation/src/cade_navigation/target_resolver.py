"""Shared target conversion helpers for navigation and person following."""

import json
import math

from cade_navigation.pose_utils import normalize_angle_deg, parse_position


VISION_FRAME_ALIASES = {
    "vision",
    "camera",
    "vision_camera",
    "camera_vision",
    "cade_vision",
}


def is_vision_frame(frame_id):
    if frame_id is None:
        return False
    return str(frame_id).strip().lower() in VISION_FRAME_ALIASES


def point_payload(position):
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
    }


def position_list(position):
    return [float(position[0]), float(position[1]), float(position[2])]


def goal_with_follow_distance(robot_x, robot_y, target_x, target_y, follow_distance):
    """Return a goal that stays follow_distance away from the target."""
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = math.hypot(dx, dy)
    if follow_distance <= 0.0 or distance <= follow_distance or distance == 0.0:
        return robot_x, robot_y

    unit_x = dx / distance
    unit_y = dy / distance
    return target_x - unit_x * follow_distance, target_y - unit_y * follow_distance


def camera_position_to_map(node, camera_position, tf_timeout):
    """Convert a cade_vision camera XYZ observation into map coordinates."""
    base_position, parent_position = node.transform_camera_position_to_base(
        camera_position,
        tf_timeout,
    )
    target_map_tuple = node.transform_position_to_map(
        base_position,
        node.base_frame,
        tf_timeout,
    )
    return point_payload(target_map_tuple), {
        "mode": "vision",
        "camera_position": position_list(camera_position),
        "camera_parent_frame": node.camera_parent_frame,
        "parent_position": position_list(parent_position),
        "base_frame": node.base_frame,
        "base_position": position_list(base_position),
    }


def goal_for_map_target(node, target_map, follow_distance, tf_timeout):
    """Build a move_base goal that faces a map target and keeps stand-off."""
    robot_x, robot_y, _ = node.get_robot_pose_map(tf_timeout)
    goal_x, goal_y = goal_with_follow_distance(
        robot_x,
        robot_y,
        target_map["x"],
        target_map["y"],
        float(follow_distance),
    )
    yaw_deg = math.degrees(
        math.atan2(target_map["y"] - robot_y, target_map["x"] - robot_x)
    )
    return {
        "x": float(goal_x),
        "y": float(goal_y),
        "z": 0.0,
        "yaw_deg": float(normalize_angle_deg(yaw_deg)),
        "frame_id": node.global_frame,
    }


def resolve_vision_goal(node, raw_position, follow_distance, tf_timeout):
    """Resolve vision [x,y,z] into target_map and stand-off map goal."""
    camera_position = parse_position(raw_position, "vision position")
    target_map, source_info = camera_position_to_map(
        node,
        camera_position,
        tf_timeout,
    )
    goal = goal_for_map_target(
        node,
        target_map,
        follow_distance,
        tf_timeout,
    )
    return target_map, goal, source_info


def resolve_navigation_goal(
    node,
    raw_position,
    frame_id,
    yaw_deg,
    follow_distance,
    tf_timeout,
):
    """
    Resolve either a normal 2D navigation goal or a vision XYZ target.

    Normal frames interpret position as [x, y, yaw_deg]. The vision pseudo-frame
    interprets position as cade_vision [x, y, z] and computes a stand-off goal.
    """
    if is_vision_frame(frame_id):
        target_map, goal, source_info = resolve_vision_goal(
            node,
            raw_position,
            follow_distance,
            tf_timeout,
        )
        source_info["input_frame"] = str(frame_id)
        return target_map, goal, source_info

    position = parse_position(raw_position, "position")
    yaw_from_position = yaw_from_navigation_position(raw_position)

    goal_x, goal_y, _ = node.transform_position_to_map(
        (position[0], position[1], 0.0),
        frame_id,
        tf_timeout,
    )

    if yaw_deg is None:
        if yaw_from_position is not None:
            yaw_map = node.yaw_to_map(yaw_from_position, frame_id, tf_timeout)
        else:
            yaw_map = node.yaw_to_face_point(goal_x, goal_y)
    else:
        yaw_map = node.yaw_to_map(float(yaw_deg), frame_id, tf_timeout)

    target_map = {
        "x": float(goal_x),
        "y": float(goal_y),
        "z": 0.0,
    }
    goal = {
        "x": float(goal_x),
        "y": float(goal_y),
        "z": 0.0,
        "yaw_deg": float(yaw_map),
        "frame_id": node.global_frame,
    }
    source_info = {
        "mode": "navigation_2d",
        "input_frame": frame_id,
        "input_position": position_list(position),
    }
    return target_map, goal, source_info


def yaw_from_navigation_position(raw_position):
    if hasattr(raw_position, "model_dump"):
        raw_position = raw_position.model_dump()

    if isinstance(raw_position, dict):
        for key in ("yaw_deg", "yaw", "theta"):
            if raw_position.get(key) is not None:
                return float(raw_position[key])
        if raw_position.get("z") is not None:
            return float(raw_position["z"])
        return None

    if isinstance(raw_position, (list, tuple)) and len(raw_position) >= 3:
        return float(raw_position[2])

    if isinstance(raw_position, str):
        text = raw_position.strip()
        if not text:
            return None
        if text[0] in "[{":
            try:
                return yaw_from_navigation_position(json.loads(text))
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        parts = [item for item in text.replace(",", " ").split() if item]
        if len(parts) >= 3:
            return float(parts[2])

    return None
