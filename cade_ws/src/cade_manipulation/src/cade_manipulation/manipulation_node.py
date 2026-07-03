"""
ROS task node for CADE manipulation (grasp + dump).

Bridges /cade/task_cmd JSON commands to the GraspNet 5090 server and the
robot arm.  Follows the same pattern as cade_navigation's navigation_node.

Architecture:
  /cade/task_cmd  ──▶  ManipulationNode  ──▶  /cade/task_status
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        GraspNet      CameraCapture   ArmController
        (TCP 5090)    (RealSense)     (gripper)
"""

import json
import threading

try:
    import rospy
    from std_msgs.msg import String

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    String = None

from cade_manipulation.arm_controller import ArmController
from cade_manipulation.camera_capture import CameraCapture
from cade_manipulation.graspnet_client import GraspNetClient
from cade_manipulation.tasks import DumpTask, GraspTask


class ManipulationNode:
    """
    Bridge CADE JSON task commands to GraspNet + arm control.

    The node publishes terminal task statuses on /cade/task_status so that
    cade_brain's BaseSkill returns the first status it receives.
    """

    def __init__(self):
        # ── ROS init ────────────────────────────────────────────
        if ROS_AVAILABLE:
            try:
                rospy.init_node("cade_manipulation", anonymous=False)
            except rospy.exceptions.ROSException:
                pass

        self.cmd_topic = rospy.get_param("~cmd_topic", "/cade/task_cmd")
        self.status_topic = rospy.get_param("~status_topic", "/cade/task_status")

        # ── GraspNet 5090 ──────────────────────────────────────
        graspnet_host = rospy.get_param("~graspnet_host", "192.168.1.100")
        graspnet_port = int(rospy.get_param("~graspnet_port", 9090))
        graspnet_timeout = float(rospy.get_param("~graspnet_timeout", 10.0))
        self.graspnet = GraspNetClient(
            host=graspnet_host,
            port=graspnet_port,
            timeout=graspnet_timeout,
        )

        # ── Camera ─────────────────────────────────────────────
        use_realsense = bool(rospy.get_param("~use_realsense", True))
        rgb_topic = rospy.get_param(
            "~rgb_topic", "/camera/color/image_raw"
        )
        depth_topic = rospy.get_param(
            "~depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )
        self.camera = CameraCapture(
            use_realsense=use_realsense,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
        )

        # ── Arm ────────────────────────────────────────────────
        arm_enabled = bool(rospy.get_param("~arm_enabled", False))
        gripper_topic = rospy.get_param("~gripper_topic", "/gripper/cmd")
        default_grasp_force = float(
            rospy.get_param("~default_grasp_force", 0.6)
        )
        self.arm = ArmController(
            enabled=arm_enabled,
            gripper_topic=gripper_topic,
            default_grasp_force=default_grasp_force,
        )

        # ── Task dispatch table ────────────────────────────────
        self._task_mapping = {
            "bringMeObj": GraspTask,
            "object_dump": DumpTask,
        }

        # ── ROS pub/sub ────────────────────────────────────────
        if ROS_AVAILABLE:
            self.status_pub = rospy.Publisher(
                self.status_topic, String, queue_size=10
            )
            self.cmd_sub = rospy.Subscriber(
                self.cmd_topic,
                String,
                self._on_task_cmd,
                queue_size=10,
            )

        # ── Concurrency ────────────────────────────────────────
        self._task_lock = threading.Lock()
        self._task_seq = 0
        self._current_task_id = None
        self._current_cancel_event = None

        # ── Log ────────────────────────────────────────────────
        rospy.loginfo("CADE manipulation node initialized")
        rospy.loginfo("  Subscribing: %s", self.cmd_topic)
        rospy.loginfo("  Publishing:  %s", self.status_topic)
        rospy.loginfo("  GraspNet:    %s:%d (timeout %.1fs)",
                      graspnet_host, graspnet_port, graspnet_timeout)
        rospy.loginfo("  Camera:      %s",
                      "pyrealsense2" if self.camera.has_camera else "ROS topics")
        rospy.loginfo("  Arm:         %s",
                      "ENABLED" if self.arm.enabled else "stub (disabled)")
        rospy.loginfo("  Actions:     %s", list(self._task_mapping.keys()))

    # ── ROS callback ──────────────────────────────────────────────

    def _on_task_cmd(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._publish_status("FAILED", error="Invalid JSON: %s" % exc)
            return

        action = cmd.get("action", "")
        request_id = cmd.get("request_id")

        task_cls = self._task_mapping.get(action)
        if task_cls is None:
            # Not our action — ignore silently so other nodes can handle it
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

        rospy.loginfo("[Manipulation Task] %s: %s", action, cmd)
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
            status_dict = {
                "status": "FAILED",
                "error": str(exc),
                "result": {
                    "action": cmd.get("action", "unknown"),
                    "exception": str(exc),
                },
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

    def _publish_status(self, status, result=None, error=None,
                        extra=None, request_id=None):
        payload = {"status": status}
        if request_id:
            payload["request_id"] = request_id
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error
        if extra:
            payload.update(extra)

        rospy.loginfo("[Manipulation Status] %s", payload)
        if ROS_AVAILABLE:
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.status_pub.publish(msg)

    def run(self):
        if ROS_AVAILABLE:
            rospy.spin()


def main():
    node = ManipulationNode()
    node.run()
    return 0
