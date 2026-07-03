"""ROS task node for CADE navigation."""

import json
import math
import os
import threading

import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from cade_navigation.local_diagnostics import LocalNavigationDiagnostics
from cade_navigation.move_base_client import MoveBaseClient
from cade_navigation.pose_utils import (
    apply_matrix4,
    lookup_pose,
    normalize_angle_deg,
    parse_matrix4,
    parse_position,
    transform_point,
)
from cade_navigation.tasks import (
    FollowPersonTask,
    NavigationTask,
    RepositionTask,
    WaitTask,
)
from cade_navigation.vision_tracks import VisionTrackCache


DEFAULT_CAMERA_TO_PARENT_MATRIX = [
    [0.0, 0.0, 1.0, 0.10],
    [-1.0, 0.0, 0.0, -0.35],
    [0.0, -1.0, 0.0, 0.07],
    [0.0, 0.0, 0.0, 1.0],
]


class NavigationNode:
    """
    Bridge CADE JSON task commands to move_base.

    The node intentionally publishes only terminal task statuses on
    /cade/task_status because cade_brain's BaseSkill returns the first status
    it receives.
    """

    def __init__(self):
        try:
            rospy.init_node("cade_navigation", anonymous=False)
        except rospy.exceptions.ROSException:
            pass

        self.cmd_topic = rospy.get_param("~cmd_topic", "/cade/task_cmd")
        self.status_topic = rospy.get_param("~status_topic", "/cade/task_status")
        self.marker_topic = rospy.get_param("~marker_topic", "/target_marker")
        self.follow_debug_topic = rospy.get_param(
            "~follow_debug_topic",
            "/cade/navigation/follow_debug",
        )
        self.vision_tracks_topic = rospy.get_param(
            "~vision_tracks_topic",
            "/vision/people_tracks_task3",
        )
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")
        self.global_frame = rospy.get_param("~global_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link_fusion")
        self.camera_parent_frame = rospy.get_param(
            "~camera_parent_frame",
            "left_arm_base_link",
        )
        camera_matrix_param = rospy.get_param(
            "~camera_to_parent_matrix",
            DEFAULT_CAMERA_TO_PARENT_MATRIX,
        )
        self.camera_to_parent_matrix = parse_matrix4(
            camera_matrix_param,
            "~camera_to_parent_matrix",
        )
        self.default_goal_timeout = float(
            rospy.get_param("~default_goal_timeout", 120.0)
        )
        self.named_goal_success_tolerance = float(
            rospy.get_param("~named_goal_success_tolerance", 0.55)
        )
        self.default_follow_timeout = float(
            rospy.get_param("~default_follow_timeout", 120.0)
        )
        self.default_tf_timeout = float(rospy.get_param("~tf_timeout", 1.0))
        self.follow_distance = float(rospy.get_param("~follow_distance", 0.8))
        self.goal_update_distance = float(
            rospy.get_param("~goal_update_distance", 0.6)
        )
        self.follow_loop_rate = float(rospy.get_param("~follow_loop_rate", 5.0))
        self.vision_track_timeout = float(
            rospy.get_param("~vision_track_timeout", 1.0)
        )
        self.follow_lost_timeout = float(
            rospy.get_param("~follow_lost_timeout", 5.0)
        )
        self.follow_initial_match_distance = float(
            rospy.get_param("~follow_initial_match_distance", 1.0)
        )
        self.follow_search_enabled = bool(
            rospy.get_param("~follow_search_enabled", True)
        )
        self.follow_search_timeout = float(
            rospy.get_param("~follow_search_timeout", 10.0)
        )
        self.follow_search_angular_speed = float(
            rospy.get_param("~follow_search_angular_speed", 0.3)
        )
        self.follow_search_deadband_px = float(
            rospy.get_param("~follow_search_deadband_px", 50.0)
        )
        self.reposition_default_distance = float(
            rospy.get_param("~reposition_default_distance", 0.35)
        )
        self.reposition_min_distance = float(
            rospy.get_param("~reposition_min_distance", 0.1)
        )
        self.reposition_max_distance = float(
            rospy.get_param("~reposition_max_distance", 0.8)
        )
        self.reposition_default_angle_deg = float(
            rospy.get_param("~reposition_default_angle_deg", 30.0)
        )
        self.reposition_min_angle_deg = float(
            rospy.get_param("~reposition_min_angle_deg", 10.0)
        )
        self.reposition_max_angle_deg = float(
            rospy.get_param("~reposition_max_angle_deg", 90.0)
        )
        self.reposition_default_timeout = float(
            rospy.get_param("~reposition_default_timeout", 8.0)
        )
        self.diagnostics_scan_topic = rospy.get_param(
            "~diagnostics_scan_topic",
            "/scan",
        )
        self.diagnostics_stale_timeout = float(
            rospy.get_param("~diagnostics_stale_timeout", 1.0)
        )
        self.diagnostics_min_valid_range = float(
            rospy.get_param("~diagnostics_min_valid_range", 0.05)
        )
        self.diagnostics_blocked_clearance = float(
            rospy.get_param("~diagnostics_blocked_clearance", 0.6)
        )
        self.diagnostics_caution_clearance = float(
            rospy.get_param("~diagnostics_caution_clearance", 0.9)
        )
        self.diagnostics_sector_half_width_deg = float(
            rospy.get_param("~diagnostics_sector_half_width_deg", 20.0)
        )
        self.diagnostics_rear_clearance_required = float(
            rospy.get_param("~diagnostics_rear_clearance_required", 0.75)
        )
        self.diagnostics_suggest_distance = float(
            rospy.get_param("~diagnostics_suggest_distance", 0.3)
        )
        self.diagnostics_suggest_angle_deg = float(
            rospy.get_param("~diagnostics_suggest_angle_deg", 30.0)
        )
        self.diagnostics_verbose_status = bool(
            rospy.get_param("~diagnostics_verbose_status", False)
        )
        self.follow_frame_center_x = float(
            rospy.get_param("~follow_frame_center_x", 320.0)
        )
        self.follow_cmd_vel_topic = rospy.get_param("~follow_cmd_vel_topic", "/cmd_vel")
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.named_locations_file = rospy.get_param("~named_locations_file", "")
        self._named_locations_file_mtime = None
        self.named_locations = rospy.get_param("~named_locations", {})
        if not isinstance(self.named_locations, dict):
            rospy.logwarn("~named_locations must be a dictionary; ignoring it")
            self.named_locations = {}
        self._refresh_named_locations_from_file(force=True)

        server_timeout = float(rospy.get_param("~server_timeout", 10.0))
        self.move_base = MoveBaseClient(
            action_name=self.move_base_action,
            server_timeout=server_timeout,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10)
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)
        self.follow_debug_pub = rospy.Publisher(
            self.follow_debug_topic,
            String,
            queue_size=10,
        )
        self.follow_search_vel_pub = rospy.Publisher(
            self.follow_cmd_vel_topic,
            Twist,
            queue_size=1,
        )
        self.local_diagnostics = LocalNavigationDiagnostics(
            scan_topic=self.diagnostics_scan_topic,
            stale_timeout=self.diagnostics_stale_timeout,
            min_valid_range=self.diagnostics_min_valid_range,
            blocked_clearance=self.diagnostics_blocked_clearance,
            caution_clearance=self.diagnostics_caution_clearance,
            sector_half_width_deg=self.diagnostics_sector_half_width_deg,
            rear_clearance_required=self.diagnostics_rear_clearance_required,
            suggested_distance=self.diagnostics_suggest_distance,
            suggested_angle_deg=self.diagnostics_suggest_angle_deg,
        )
        self.vision_tracks = VisionTrackCache(self.vision_tracks_topic)
        self.cmd_sub = rospy.Subscriber(
            self.cmd_topic,
            String,
            self._on_task_cmd,
            queue_size=10,
        )

        self._task_lock = threading.Lock()
        self._task_seq = 0
        self._current_task_id = None
        self._current_cancel_event = None

        self._task_mapping = {
            "navigation": NavigationTask,
            "follow_person": FollowPersonTask,
            "reposition": RepositionTask,
            "wait": WaitTask,
        }

        rospy.loginfo("CADE navigation node initialized")
        rospy.loginfo("  Subscribing: %s", self.cmd_topic)
        rospy.loginfo("  Publishing: %s", self.status_topic)
        rospy.loginfo("  Follow debug: %s", self.follow_debug_topic)
        rospy.loginfo("  Vision tracks: %s", self.vision_tracks_topic)
        rospy.loginfo("  move_base: %s", self.move_base_action)
        rospy.loginfo("  frames: %s <- %s", self.global_frame, self.base_frame)
        rospy.loginfo("  camera parent frame: %s", self.camera_parent_frame)
        rospy.loginfo("  named locations: %d", len(self.named_locations))
        rospy.loginfo(
            "  named goal success tolerance: %.2fm",
            self.named_goal_success_tolerance,
        )
        rospy.loginfo(
            "  reposition: distance %.2f-%.2fm default %.2fm; "
            "angle %.1f-%.1fdeg default %.1fdeg",
            self.reposition_min_distance,
            self.reposition_max_distance,
            self.reposition_default_distance,
            self.reposition_min_angle_deg,
            self.reposition_max_angle_deg,
            self.reposition_default_angle_deg,
        )
        rospy.loginfo(
            "  local diagnostics: scan=%s blocked<%.2fm caution<%.2fm compact=%s",
            self.diagnostics_scan_topic,
            self.diagnostics_blocked_clearance,
            self.diagnostics_caution_clearance,
            not self.diagnostics_verbose_status,
        )

    def _on_task_cmd(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._publish_status("FAILED", error="Invalid JSON: %s" % exc)
            return

        action = cmd.get("action", "")
        request_id = cmd.get("request_id")
        if action == "stop_navigation":
            self._cancel_active_task()
            self._publish_status(
                "SUCCESS",
                result={"action": "stop_navigation", "message": "Navigation stopped"},
                request_id=request_id,
            )
            return

        task_cls = self._task_mapping.get(action)
        if task_cls is None:
            self._publish_status(
                "FAILED",
                error="Unknown action: %s" % action,
                request_id=request_id,
            )
            return

        cancel_event = threading.Event()
        with self._task_lock:
            old_event = self._current_cancel_event
            self._task_seq += 1
            task_id = self._task_seq
            self._current_task_id = task_id
            self._current_cancel_event = cancel_event

        if old_event is not None:
            old_event.set()
            self.move_base.cancel_goal_if_active()

        rospy.loginfo("[Navigation Task] %s: %s", action, cmd)
        executor = task_cls(self, cancel_event)
        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, executor, cmd),
            daemon=True,
        )
        thread.start()

    def _run_task(self, task_id, executor, cmd):
        try:
            status_dict = executor.execute(cmd)
        except Exception as exc:
            result = {"action": cmd.get("action", "unknown")}
            if cmd.get("action") in ("navigation", "follow_person", "reposition"):
                result.update(
                    self.get_local_diagnostics("%s_exception" % cmd.get("action"))
                )
            status_dict = {
                "status": "FAILED",
                "error": str(exc),
                "result": result,
            }

        if not self._is_current_task(task_id):
            return

        self._publish_status(
            status_dict.get("status", "FAILED"),
            result=status_dict.get("result"),
            error=status_dict.get("error"),
            request_id=cmd.get("request_id"),
            extra={
                key: value
                for key, value in status_dict.items()
                if key not in ("status", "result", "error")
            },
        )
        with self._task_lock:
            if self._current_task_id == task_id:
                self._current_task_id = None
                self._current_cancel_event = None

    def _is_current_task(self, task_id) -> bool:
        with self._task_lock:
            return self._current_task_id == task_id

    def _cancel_active_task(self) -> None:
        with self._task_lock:
            cancel_event = self._current_cancel_event
            self._task_seq += 1
            self._current_task_id = None
            self._current_cancel_event = None

        if cancel_event is not None:
            cancel_event.set()
        self.move_base.cancel_goal_if_active()
        self.stop_follow_search_rotation()

    def _publish_status(self, status, result=None, error=None, extra=None, request_id=None):
        payload = {"status": status}
        if request_id:
            payload["request_id"] = request_id
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        if extra:
            payload.update(extra)

        rospy.loginfo("[Navigation Status] %s", payload)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def publish_follow_debug(self, event, **payload):
        msg_payload = {"event": event}
        msg_payload.update(payload)
        rospy.loginfo("[Follow Debug] %s", msg_payload)
        msg = String()
        msg.data = json.dumps(msg_payload, ensure_ascii=False)
        self.follow_debug_pub.publish(msg)

    def publish_follow_search_rotation(self, last_center_x=None):
        if not self.follow_search_enabled:
            return

        twist = Twist()
        if last_center_x is not None:
            search_left = (
                float(last_center_x)
                < self.follow_frame_center_x - self.follow_search_deadband_px
            )
            if search_left:
                twist.angular.z = abs(self.follow_search_angular_speed)
            else:
                twist.angular.z = -abs(self.follow_search_angular_speed)
        else:
            twist.angular.z = abs(self.follow_search_angular_speed)
        self.follow_search_vel_pub.publish(twist)

    def stop_follow_search_rotation(self):
        if not self.follow_search_enabled:
            return
        self.follow_search_vel_pub.publish(Twist())

    def get_local_diagnostics(self, context=None):
        return self.local_diagnostics.snapshot(
            context=context,
            compact=not self.diagnostics_verbose_status,
        )

    def transform_position_to_map(self, position, frame_id, timeout):
        return transform_point(
            self.tf_buffer,
            position,
            frame_id,
            self.global_frame,
            timeout=timeout,
        )

    def transform_camera_position_to_base(self, position, timeout):
        parent_position = apply_matrix4(self.camera_to_parent_matrix, position)
        base_position = transform_point(
            self.tf_buffer,
            parent_position,
            self.camera_parent_frame,
            self.base_frame,
            timeout=timeout,
        )
        return base_position, parent_position

    def resolve_position_alias(self, raw_position):
        named_target = self.resolve_named_location(raw_position)
        if named_target is not None:
            return named_target["position"]
        return raw_position

    def resolve_named_location(self, raw_position):
        if not isinstance(raw_position, str):
            return None

        self._refresh_named_locations_from_file()
        name = raw_position.strip()
        if not name:
            return None

        normalized_name = self._location_key(name)
        for location_name, value in self.named_locations.items():
            if self._location_key(location_name) == normalized_name:
                return self._named_location_payload(location_name, value)
        return None

    def _refresh_named_locations_from_file(self, force=False):
        if not self.named_locations_file:
            return
        try:
            stat = os.stat(self.named_locations_file)
        except OSError as exc:
            rospy.logwarn_throttle(
                30.0,
                "Cannot stat named locations file %s: %s",
                self.named_locations_file,
                exc,
            )
            return

        mtime = stat.st_mtime
        if not force and self._named_locations_file_mtime == mtime:
            return

        try:
            with open(self.named_locations_file, "r", encoding="utf-8") as handle:
                loaded_locations = json.load(handle)
        except Exception as exc:
            rospy.logwarn(
                "Cannot load named locations file %s: %s",
                self.named_locations_file,
                exc,
            )
            return

        if not isinstance(loaded_locations, dict):
            rospy.logwarn(
                "Named locations file %s must contain a JSON object",
                self.named_locations_file,
            )
            return

        self.named_locations = loaded_locations
        self._named_locations_file_mtime = mtime
        rospy.loginfo(
            "Loaded %d named locations from %s",
            len(self.named_locations),
            self.named_locations_file,
        )

    def _named_location_payload(self, name, value):
        frame_id = self.global_frame
        yaw_deg = None
        position_value = value
        if isinstance(value, dict):
            position_value = value.get("position", value.get("pose", value))
            frame_id = value.get("frame_id", value.get("source_frame", frame_id))
            yaw_deg = value.get("yaw_deg", value.get("yaw"))

        return {
            "name": str(name),
            "position": list(parse_position(position_value, str(name))),
            "frame_id": str(frame_id),
            "yaw_deg": yaw_deg,
        }

    @staticmethod
    def _location_key(value):
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    def get_robot_pose_map(self, timeout=None):
        return lookup_pose(
            self.tf_buffer,
            self.global_frame,
            self.base_frame,
            timeout=self.default_tf_timeout if timeout is None else timeout,
        )

    def yaw_to_map(self, yaw_deg, frame_id, timeout=None):
        """Convert yaw expressed in frame_id into global-frame yaw degrees."""
        yaw_deg = float(yaw_deg)
        if frame_id == self.global_frame:
            return normalize_angle_deg(yaw_deg)

        _, _, frame_yaw = lookup_pose(
            self.tf_buffer,
            self.global_frame,
            frame_id,
            timeout=self.default_tf_timeout if timeout is None else timeout,
        )
        return normalize_angle_deg(frame_yaw + yaw_deg)

    def yaw_to_face_point(self, x, y) -> float:
        try:
            robot_x, robot_y, _ = self.get_robot_pose_map()
            return math.degrees(math.atan2(y - robot_y, x - robot_x))
        except Exception as exc:
            rospy.logwarn("Cannot read robot pose for yaw; using 0 deg: %s", exc)
            return 0.0

    def publish_target_marker(self, position, label=None):
        if not self.publish_markers:
            return

        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = label or "cade_navigation"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 0.0
        marker.color.g = 0.2
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(5.0)
        self.marker_pub.publish(marker)

    def run(self):
        rospy.spin()


def main():
    node = NavigationNode()
    node.run()
    return 0
