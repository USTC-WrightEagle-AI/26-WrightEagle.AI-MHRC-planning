"""Small bounded repositioning task for local recovery and view adjustment."""

from cade_navigation.pose_utils import normalize_angle_deg, parse_float
from cade_navigation.tasks.base_task import BaseTask


MOTION_ALIASES = {
    "forward": "forward",
    "ahead": "forward",
    "backward": "backward",
    "back": "backward",
    "reverse": "backward",
    "left": "left",
    "side_left": "left",
    "right": "right",
    "side_right": "right",
    "turn_left": "turn_left",
    "rotate_left": "turn_left",
    "yaw_left": "turn_left",
    "turn_right": "turn_right",
    "rotate_right": "turn_right",
    "yaw_right": "turn_right",
}

SIDE_MOTIONS = {"left", "right"}


class RepositionTask(BaseTask):
    """
    Execute a short relative movement through move_base.

    This is intended for LLM-directed recovery and active perception, not for
    long-distance navigation.
    """

    def execute(self, cmd_dict):
        motion = self._parse_motion(cmd_dict.get("motion"))
        diagnostics_before = self.node.get_local_diagnostics("before_reposition")
        if motion in SIDE_MOTIONS:
            return self._side_motion_unavailable(
                motion,
                cmd_dict.get("reason"),
                diagnostics_before,
            )

        tf_timeout = parse_float(
            cmd_dict.get("tf_timeout"),
            self.node.default_tf_timeout,
            "tf_timeout",
        )
        timeout = parse_float(
            cmd_dict.get("timeout"),
            self.node.reposition_default_timeout,
            "timeout",
        )
        requested_distance = parse_float(
            cmd_dict.get("distance"),
            self.node.reposition_default_distance,
            "distance",
        )
        requested_angle = parse_float(
            cmd_dict.get("angle_deg"),
            self.node.reposition_default_angle_deg,
            "angle_deg",
        )

        distance = self._clamp_abs(
            requested_distance,
            self.node.reposition_min_distance,
            self.node.reposition_max_distance,
        )
        angle_deg = self._clamp_abs(
            requested_angle,
            self.node.reposition_min_angle_deg,
            self.node.reposition_max_angle_deg,
        )

        robot_x, robot_y, robot_yaw = self.node.get_robot_pose_map(
            timeout=tf_timeout,
        )
        if motion == "turn_left":
            goal_x = robot_x
            goal_y = robot_y
            goal_yaw = normalize_angle_deg(robot_yaw + angle_deg)
        elif motion == "turn_right":
            goal_x = robot_x
            goal_y = robot_y
            goal_yaw = normalize_angle_deg(robot_yaw - angle_deg)
        else:
            dx, dy = self._motion_offset(motion, distance)
            goal_x, goal_y, _ = self.node.transform_position_to_map(
                (dx, dy, 0.0),
                self.node.base_frame,
                tf_timeout,
            )
            goal_yaw = robot_yaw

        self.node.publish_target_marker(
            (goal_x, goal_y, 0.0),
            "reposition_goal",
        )
        self.node.move_base.cancel_goal_if_active()
        result = self.node.move_base.send_goal_and_wait(
            goal_x,
            goal_y,
            goal_yaw,
            frame_id=self.node.global_frame,
            timeout=timeout,
            cancel_event=self.cancel_event,
        )
        diagnostics_after = self.node.get_local_diagnostics("after_reposition")
        return self._with_result(
            result,
            motion,
            requested_distance,
            distance,
            requested_angle,
            angle_deg,
            timeout,
            (goal_x, goal_y, goal_yaw),
            cmd_dict.get("reason"),
            diagnostics_before,
            diagnostics_after,
        )

    @staticmethod
    def _parse_motion(value):
        motion = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if motion not in MOTION_ALIASES:
            raise ValueError(
                "motion must be one of: forward, backward, turn_left, turn_right"
            )
        return MOTION_ALIASES[motion]

    @staticmethod
    def _clamp_abs(value, min_value, max_value):
        value = abs(float(value))
        return max(float(min_value), min(float(max_value), value))

    @staticmethod
    def _motion_offset(motion, distance):
        if motion == "forward":
            return distance, 0.0
        if motion == "backward":
            return -distance, 0.0
        raise ValueError("Unsupported reposition motion: %s" % motion)

    @staticmethod
    def _side_motion_unavailable(motion, reason, diagnostics_before):
        result = {
            "action": "reposition",
            "motion": motion,
            "failure_reason": "side_motion_unavailable",
            "message": (
                "Sideways reposition is unavailable because the local planner "
                "does not support lateral velocity. Use turn_left, turn_right, "
                "backward, or forward."
            ),
        }
        result.update(RepositionTask._compact_reposition_diagnostics(diagnostics_before))
        if reason:
            result["reason"] = str(reason)
        return {
            "status": "FAILED",
            "error": result["message"],
            "result": result,
        }

    @staticmethod
    def _with_result(
        status_dict,
        motion,
        requested_distance,
        distance,
        requested_angle,
        angle_deg,
        timeout,
        goal,
        reason,
        diagnostics_before,
        diagnostics_after,
    ):
        result = dict(status_dict.get("result") or {})
        result.update(
            {
                "action": "reposition",
                "motion": motion,
                "requested_distance": float(requested_distance),
                "distance": float(distance),
                "requested_angle_deg": float(requested_angle),
                "angle_deg": float(angle_deg),
                "timeout": float(timeout),
                "goal": {
                    "x": float(goal[0]),
                    "y": float(goal[1]),
                    "z": 0.0,
                    "yaw_deg": float(goal[2]),
                    "frame_id": "map",
                },
            }
        )
        result.update(
            RepositionTask._compact_reposition_diagnostics(
                diagnostics_before,
                diagnostics_after,
            )
        )
        if reason:
            result["reason"] = str(reason)

        status_dict = dict(status_dict)
        status_dict["result"] = result
        return status_dict

    @staticmethod
    def _compact_reposition_diagnostics(diagnostics_before, diagnostics_after=None):
        if RepositionTask._is_verbose_diagnostics(diagnostics_before):
            payload = {"diagnostics_before": diagnostics_before}
            if diagnostics_after is not None:
                payload["diagnostics_after"] = diagnostics_after
                payload.update(diagnostics_after)
            else:
                payload.update(diagnostics_before)
            return payload

        before_summary = RepositionTask._summary(diagnostics_before)
        after_summary = RepositionTask._summary(diagnostics_after)
        payload = {}
        if before_summary:
            payload["scan_before"] = before_summary.get("scan")
            if before_summary.get("blocked"):
                payload["blocked_before"] = before_summary.get("blocked")
            if before_summary.get("caution"):
                payload["caution_before"] = before_summary.get("caution")
        if after_summary:
            payload["scan_after"] = after_summary.get("scan")
            if after_summary.get("blocked"):
                payload["blocked_after"] = after_summary.get("blocked")
            if after_summary.get("caution"):
                payload["caution_after"] = after_summary.get("caution")
        elif not before_summary:
            payload.update(diagnostics_before or {})

        suggestion = None
        if isinstance(diagnostics_after, dict):
            suggestion = diagnostics_after.get("suggested_reposition")
        if suggestion is None and isinstance(diagnostics_before, dict):
            suggestion = diagnostics_before.get("suggested_reposition")
        if suggestion:
            payload["suggested_reposition"] = suggestion
        return payload

    @staticmethod
    def _is_verbose_diagnostics(diagnostics):
        summary = RepositionTask._summary(diagnostics)
        return isinstance(summary, dict) and "sectors" in summary

    @staticmethod
    def _summary(diagnostics):
        if not isinstance(diagnostics, dict):
            return None
        return diagnostics.get("obstacle_summary")
