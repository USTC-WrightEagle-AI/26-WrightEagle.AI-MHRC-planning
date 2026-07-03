"""Continuous person following driven by cade_vision people tracks."""

import math
import time

import rospy

from cade_navigation.pose_utils import parse_float, parse_position
from cade_navigation.target_resolver import resolve_vision_goal
from cade_navigation.tasks.base_task import BaseTask


class FollowPersonTask(BaseTask):
    """
    Continuously follow a person tracked by cade_vision.

    cade_vision owns camera access and person tracking. This task subscribes to
    the people track stream cached by NavigationNode, transforms the target
    camera coordinate into map, and refreshes move_base goals while the task is
    active.
    """

    def execute(self, cmd_dict):
        if (
            cmd_dict.get("frame_id") is not None
            or cmd_dict.get("source_frame") is not None
        ):
            raise ValueError("follow_person does not accept frame_id/source_frame")

        raw_position = (
            cmd_dict.get("person_pos")
            or cmd_dict.get("location")
            or cmd_dict.get("position")
        )
        raw_position = self.node.resolve_position_alias(raw_position)
        initial_position = (
            parse_position(raw_position, "person_pos")
            if raw_position is not None
            else None
        )
        track_id = self._parse_optional_int(cmd_dict.get("track_id"), "track_id")
        if track_id is None and initial_position is None:
            raise ValueError("follow_person requires track_id or person_pos")

        duration_value = (
            cmd_dict.get("duration")
            if cmd_dict.get("duration") is not None
            else cmd_dict.get("timeout")
        )
        duration = parse_float(
            duration_value,
            self.node.default_follow_timeout,
            "duration",
        )
        if duration <= 0.0:
            duration = self.node.default_follow_timeout

        params = {
            "tf_timeout": parse_float(
                cmd_dict.get("tf_timeout"),
                self.node.default_tf_timeout,
                "tf_timeout",
            ),
            "follow_distance": parse_float(
                cmd_dict.get("follow_distance"),
                self.node.follow_distance,
                "follow_distance",
            ),
            "goal_update_distance": parse_float(
                cmd_dict.get("goal_update_distance"),
                self.node.goal_update_distance,
                "goal_update_distance",
            ),
            "loop_rate": parse_float(
                cmd_dict.get("loop_rate"),
                self.node.follow_loop_rate,
                "loop_rate",
            ),
            "vision_track_timeout": parse_float(
                cmd_dict.get("vision_track_timeout"),
                self.node.vision_track_timeout,
                "vision_track_timeout",
            ),
            "lost_timeout": parse_float(
                cmd_dict.get("lost_timeout"),
                self.node.follow_lost_timeout,
                "lost_timeout",
            ),
            "initial_match_distance": parse_float(
                cmd_dict.get("initial_match_distance"),
                self.node.follow_initial_match_distance,
                "initial_match_distance",
            ),
        }

        return self._run_follow_loop(
            initial_track_id=track_id,
            initial_position=initial_position,
            duration=duration,
            params=params,
        )

    def _run_follow_loop(
        self,
        initial_track_id,
        initial_position,
        duration,
        params,
    ):
        locked_track_id = initial_track_id
        match_position = initial_position
        last_camera_position = initial_position
        last_center_x = None
        last_goal = None
        last_target_map = None
        last_goal_payload = None
        lost_since = None
        search_since = None
        goal_updates = 0
        start_time = time.time()
        deadline = start_time + float(duration)
        rate = rospy.Rate(max(float(params["loop_rate"]), 0.2))

        self.node.publish_follow_debug(
            "LOCKING",
            track_id=locked_track_id,
            has_person_pos=initial_position is not None,
        )

        try:
            if initial_position is not None:
                (
                    last_goal,
                    last_target_map,
                    last_goal_payload,
                    goal_updates,
                ) = self._send_initial_position_goal(
                    initial_position,
                    params,
                    locked_track_id,
                    goal_updates,
                )

            while not rospy.is_shutdown():
                now = time.time()
                if self.cancel_event.is_set():
                    return self._finish(
                        "FAILED",
                        "Navigation canceled",
                        locked_track_id,
                        goal_updates,
                        last_target_map,
                        last_goal_payload,
                    )
                if now >= deadline:
                    return self._finish(
                        "SUCCESS",
                        None,
                        locked_track_id,
                        goal_updates,
                        last_target_map,
                        last_goal_payload,
                    )

                person, error, _ = self.node.vision_tracks.find_target(
                    locked_track_id,
                    match_position,
                    params["vision_track_timeout"],
                    params["initial_match_distance"],
                )

                can_rematch = (
                    person is None
                    and locked_track_id is not None
                    and last_camera_position is not None
                )
                if can_rematch:
                    person, rematch_error, _ = self.node.vision_tracks.find_target(
                        None,
                        last_camera_position,
                        params["vision_track_timeout"],
                        params["initial_match_distance"],
                    )
                    if person is not None:
                        old_id = locked_track_id
                        locked_track_id = self._parse_optional_int(
                            person.get("track_id"),
                            "track_id",
                        )
                        self.node.publish_follow_debug(
                            "RELOCKED",
                            old_track_id=old_id,
                            track_id=locked_track_id,
                        )
                    else:
                        error = rematch_error or error

                if person is None:
                    try:
                        lost_since, search_since = self._handle_lost_target(
                            error,
                            locked_track_id,
                            last_center_x,
                            lost_since,
                            search_since,
                            params["lost_timeout"],
                            has_last_goal=last_goal_payload is not None,
                        )
                    except RuntimeError as exc:
                        return self._finish(
                            "FAILED",
                            str(exc),
                            locked_track_id,
                            goal_updates,
                            last_target_map,
                            last_goal_payload,
                        )
                    rate.sleep()
                    continue

                camera_position = parse_position(
                    person.get("position_3d"),
                    "vision position_3d",
                )
                new_track_id = self._parse_optional_int(
                    person.get("track_id"),
                    "track_id",
                )
                if locked_track_id is None:
                    locked_track_id = new_track_id
                    self.node.publish_follow_debug(
                        "LOCKED",
                        track_id=locked_track_id,
                    )

                last_camera_position = camera_position
                match_position = camera_position
                last_center_x = self._center_x(person.get("center"), last_center_x)
                lost_since = None
                search_since = None
                self.node.stop_follow_search_rotation()

                target_map, goal_payload, source_info = resolve_vision_goal(
                    self.node,
                    camera_position,
                    params["follow_distance"],
                    params["tf_timeout"],
                )
                self.node.publish_target_marker(
                    (target_map["x"], target_map["y"], target_map["z"]),
                    "follow_person_target",
                )

                if self._should_update_goal(
                    last_goal,
                    goal_payload,
                    params["goal_update_distance"],
                ):
                    sent = self.node.move_base.send_goal(
                        goal_payload["x"],
                        goal_payload["y"],
                        goal_payload["yaw_deg"],
                        frame_id=self.node.global_frame,
                    )
                    if sent.get("status") == "FAILED":
                        return self._finish(
                            "FAILED",
                            sent.get("error"),
                            locked_track_id,
                            goal_updates,
                            target_map,
                            goal_payload,
                        )

                    goal_updates += 1
                    last_goal = (goal_payload["x"], goal_payload["y"])
                    last_goal_payload = goal_payload
                    self.node.publish_follow_debug(
                        "GOAL_UPDATED",
                        track_id=locked_track_id,
                        target=target_map,
                        goal=goal_payload,
                        source=source_info,
                    )

                last_target_map = target_map
                if self.node.move_base.is_failure_state():
                    state_name = self.node.move_base.get_state_name()
                    return self._finish(
                        "FAILED",
                        (
                            "move_base failed while following: %s; "
                            "goal is unreachable or blocked"
                        )
                        % state_name,
                        locked_track_id,
                        goal_updates,
                        last_target_map,
                        last_goal_payload,
                        move_base_state=state_name,
                        failure_reason="goal_unreachable_or_blocked",
                    )

                self.node.publish_follow_debug(
                    "TRACKING",
                    track_id=locked_track_id,
                    target=last_target_map,
                    goal=last_goal_payload,
                )
                rate.sleep()
        finally:
            self.node.stop_follow_search_rotation()

    def _handle_lost_target(
        self,
        error,
        locked_track_id,
        last_center_x,
        lost_since,
        search_since,
        lost_timeout,
        has_last_goal=False,
    ):
        now = time.time()
        if lost_since is None:
            lost_since = now
            search_since = now
            if not has_last_goal:
                self.node.move_base.cancel_goal_if_active()
            self.node.publish_follow_debug(
                "LOST",
                track_id=locked_track_id,
                error=error,
                keep_last_goal=bool(has_last_goal),
            )

        lost_age = now - lost_since
        if lost_age >= float(lost_timeout):
            raise RuntimeError("Target lost for %.1fs: %s" % (lost_age, error))

        if (
            self.node.follow_search_enabled
            and search_since is not None
            and now - search_since <= self.node.follow_search_timeout
        ):
            self.node.publish_follow_search_rotation(last_center_x)
            self.node.publish_follow_debug(
                "SEARCHING",
                track_id=locked_track_id,
                lost_age=lost_age,
                error=error,
            )
        else:
            self.node.stop_follow_search_rotation()

        return lost_since, search_since

    def _send_initial_position_goal(
        self,
        initial_position,
        params,
        locked_track_id,
        goal_updates,
    ):
        target_map, goal_payload, source_info = resolve_vision_goal(
            self.node,
            initial_position,
            params["follow_distance"],
            params["tf_timeout"],
        )
        self.node.publish_target_marker(
            (target_map["x"], target_map["y"], target_map["z"]),
            "follow_person_initial_target",
        )
        sent = self.node.move_base.send_goal(
            goal_payload["x"],
            goal_payload["y"],
            goal_payload["yaw_deg"],
            frame_id=self.node.global_frame,
        )
        if sent.get("status") == "FAILED":
            raise RuntimeError(sent.get("error") or "Failed to send initial follow goal")

        goal_updates += 1
        self.node.publish_follow_debug(
            "INITIAL_GOAL",
            track_id=locked_track_id,
            target=target_map,
            goal=goal_payload,
            source=source_info,
        )
        return (
            (goal_payload["x"], goal_payload["y"]),
            target_map,
            goal_payload,
            goal_updates,
        )

    @staticmethod
    def _should_update_goal(last_goal, goal_payload, threshold):
        if last_goal is None:
            return True
        distance = math.sqrt(
            (goal_payload["x"] - last_goal[0]) ** 2
            + (goal_payload["y"] - last_goal[1]) ** 2
        )
        return distance >= float(threshold)

    def _finish(
        self,
        status,
        error,
        track_id,
        goal_updates,
        last_target_map,
        last_goal_payload,
        move_base_state=None,
        failure_reason=None,
    ):
        self.node.move_base.cancel_goal_if_active()
        self.node.stop_follow_search_rotation()
        result = {
            "action": "follow_person",
            "track_id": track_id,
            "goal_updates": int(goal_updates),
            "target": last_target_map,
            "goal": last_goal_payload,
        }
        if status != "SUCCESS":
            result.update(self.node.get_local_diagnostics("follow_person_failure"))
        self.node.publish_follow_debug(
            "STOPPED" if status == "SUCCESS" else "FAILED",
            status=status,
            error=error,
            result=result,
        )
        payload = {"status": status, "result": result}
        if error:
            payload["error"] = error
        if move_base_state:
            payload["move_base_state"] = move_base_state
        if failure_reason:
            payload["failure_reason"] = failure_reason
        return payload

    @staticmethod
    def _center_x(center, fallback):
        if isinstance(center, (list, tuple)) and center:
            try:
                return float(center[0])
            except (TypeError, ValueError):
                return fallback
        return fallback

    @staticmethod
    def _parse_optional_int(value, field_name):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be an integer" % field_name) from exc
