"""Navigation task implementation."""

import math

from cade_navigation.pose_utils import parse_float, parse_position
from cade_navigation.target_resolver import resolve_navigation_goal
from cade_navigation.tasks.base_task import BaseTask


class NavigationTask(BaseTask):
    """Navigate to a single target pose through move_base."""

    def execute(self, cmd_dict):
        raw_position = cmd_dict.get("position", cmd_dict.get("target"))
        named_target = self.node.resolve_named_location(raw_position)
        if named_target is not None:
            raw_position = named_target["position"]
        elif isinstance(raw_position, str):
            try:
                parse_position(raw_position, "position")
            except ValueError as exc:
                raise ValueError("Unknown named location: %s" % raw_position) from exc
        frame_id = cmd_dict.get("frame_id") or cmd_dict.get("source_frame")
        if frame_id is None and named_target is not None:
            frame_id = named_target.get("frame_id")
        frame_id = frame_id or self.node.base_frame

        timeout = parse_float(
            cmd_dict.get("timeout"),
            self.node.default_goal_timeout,
            "timeout",
        )
        tf_timeout = parse_float(
            cmd_dict.get("tf_timeout"),
            self.node.default_tf_timeout,
            "tf_timeout",
        )
        follow_distance = parse_float(
            cmd_dict.get("follow_distance"),
            self.node.follow_distance,
            "follow_distance",
        )
        raw_yaw = cmd_dict.get("yaw_deg")
        if raw_yaw is None and named_target is not None:
            raw_yaw = named_target.get("yaw_deg")
        yaw_deg = (
            parse_float(raw_yaw, 0.0, "yaw_deg")
            if raw_yaw is not None
            else None
        )

        target_map, goal, source_info = resolve_navigation_goal(
            self.node,
            raw_position,
            frame_id,
            yaw_deg,
            follow_distance,
            tf_timeout,
        )
        if named_target is not None:
            source_info["named_location"] = named_target.get("name")

        self.node.publish_target_marker(
            (target_map["x"], target_map["y"], target_map["z"]),
            cmd_dict.get("object_name") or cmd_dict.get("label") or "navigation_goal",
        )

        result = self.node.move_base.send_goal_and_wait(
            goal["x"],
            goal["y"],
            goal["yaw_deg"],
            frame_id=self.node.global_frame,
            timeout=timeout,
            cancel_event=self.cancel_event,
        )
        result = self._accept_named_goal_if_reached(
            result,
            target_map,
            named_target,
            cmd_dict,
        )
        return self._with_goal_result(
            result,
            "navigation",
            target_map,
            goal,
            source_info,
        )

    def _accept_named_goal_if_reached(
        self,
        status_dict,
        target_map,
        named_target,
        cmd_dict,
    ):
        if status_dict.get("status") == "SUCCESS":
            return status_dict

        if named_target is None and cmd_dict.get("accept_near_goal") is None:
            return status_dict

        accept_near_goal = self._parse_bool(
            cmd_dict.get("accept_near_goal"),
            default=named_target is not None,
        )
        if not accept_near_goal:
            return status_dict

        tolerance = parse_float(
            cmd_dict.get("success_distance_tolerance"),
            self.node.named_goal_success_tolerance,
            "success_distance_tolerance",
        )
        if tolerance <= 0.0:
            return status_dict

        try:
            robot_x, robot_y, robot_yaw = self.node.get_robot_pose_map()
        except Exception as exc:
            status_dict = dict(status_dict)
            status_dict["pose_check_error"] = str(exc)
            return status_dict

        distance = math.hypot(
            float(robot_x) - float(target_map["x"]),
            float(robot_y) - float(target_map["y"]),
        )
        if distance > tolerance:
            return status_dict

        accepted = {
            "status": "SUCCESS",
            "result": {
                "move_base_state": status_dict.get("move_base_state", "UNKNOWN"),
                "accepted_by_pose": True,
                "distance_to_goal": float(distance),
                "success_distance_tolerance": float(tolerance),
                "robot_pose": {
                    "x": float(robot_x),
                    "y": float(robot_y),
                    "yaw_deg": float(robot_yaw),
                },
            },
        }
        if status_dict.get("error"):
            accepted["result"]["move_base_error"] = status_dict["error"]
        return accepted

    @staticmethod
    def _parse_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off"):
            return False
        return bool(default)

    def _with_goal_result(self, status_dict, action, target_map, goal, source_info):
        result = dict(status_dict.get("result") or {})
        result.update(
            {
                "action": action,
                "target": target_map,
                "goal": goal,
                "source": source_info,
            }
        )
        if status_dict.get("status") != "SUCCESS":
            result.update(self.node.get_local_diagnostics("navigation_failure"))
        status_dict["result"] = result
        return status_dict
