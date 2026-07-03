"""
Arm Controller — execute grasp/dump actions on the physical robot arm.

This is a STUB implementation. Replace with your actual arm driver
(MoveIt, custom serial controller, Dynamixel SDK, etc.).

The arm is expected to accept:
  - grasp_poses:  {"position": [x, y, z], "orientation": [x, y, z, w]}
  - grasp_force:  float (0.0 – 1.0)
  - release:      bool

Control flow for a typical pick-and-place:
  1. move_to_pregrasp(pose)  — approach from above
  2. move_to_grasp(pose)     — descend to object
  3. close_gripper(force)    — grip
  4. move_to_pregrasp(pose)  — lift
"""

import json
import time
from typing import Any, Dict, Optional

try:
    import rospy
    from std_msgs.msg import String

    _HAS_ROS = True
except ImportError:
    rospy = None
    String = None
    _HAS_ROS = False


class ArmController:
    """Robot arm interface (stub — replace with real hardware driver)."""

    def __init__(
        self,
        enabled: bool = False,
        gripper_topic: str = "/gripper/cmd",
        default_grasp_force: float = 0.6,
    ):
        self.enabled = enabled
        self.default_grasp_force = default_grasp_force
        self._gripper_pub = None

        if enabled and _HAS_ROS:
            self._gripper_pub = rospy.Publisher(
                gripper_topic, String, queue_size=10
            )
            rospy.loginfo("[ArmController] Arm ENABLED — gripper on %s", gripper_topic)
        elif not enabled:
            rospy.loginfo("[ArmController] Arm DISABLED — stub mode (no physical arm)")
        else:
            rospy.logwarn("[ArmController] ROS not available, arm stub only")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_grasp(
        self,
        grasp_poses: dict,
        grasp_force: Optional[float] = None,
        object_name: str = "object",
    ) -> Dict[str, Any]:
        """
        Execute a full grasp sequence.

        Args:
            grasp_poses:  {"position": [x,y,z], "orientation": [x,y,z,w]}
            grasp_force:  grip force 0-1 (None → default)
            object_name:  label for logging

        Returns:
            {"status": "SUCCESS"|"FAILED", ...}
        """
        if not self.enabled:
            return self._stub_result("grasp", object_name, grasp_poses)

        force = (
            grasp_force
            if grasp_force is not None
            else self.default_grasp_force
        )
        force = max(0.0, min(1.0, float(force)))

        try:
            # --- 1. Pregrasp approach ---
            self._move_to_pregrasp(grasp_poses)

            # --- 2. Descend to grasp ---
            self._move_to_grasp(grasp_poses)

            # --- 3. Close gripper ---
            self._close_gripper(force)

            # --- 4. Lift ---
            self._move_to_pregrasp(grasp_poses)

            return {
                "status": "SUCCESS",
                "result": {
                    "action": "grasp",
                    "object_name": object_name,
                    "grasp_poses": grasp_poses,
                    "grasp_force": force,
                },
            }
        except Exception as exc:
            return {"status": "FAILED", "error": str(exc)}

    def execute_dump(
        self,
        target_position: Optional[Any] = None,
        release_safe: bool = True,
    ) -> Dict[str, Any]:
        """
        Release the currently held object.

        Args:
            target_position:  where to place (semantic label or [x,y,z])
            release_safe:     if True, wait for stable support before releasing

        Returns:
            {"status": "SUCCESS"|"FAILED", ...}
        """
        if not self.enabled:
            return self._stub_result("dump", str(target_position or "default"))

        try:
            if release_safe:
                time.sleep(0.5)  # let arm settle

            self._open_gripper()

            return {
                "status": "SUCCESS",
                "result": {
                    "action": "dump",
                    "target_position": target_position,
                    "release_safe": release_safe,
                },
            }
        except Exception as exc:
            return {"status": "FAILED", "error": str(exc)}

    # ------------------------------------------------------------------
    # Arm motion primitives (replace with real implementation)
    # ------------------------------------------------------------------

    def _move_to_pregrasp(self, grasp_poses: dict) -> None:
        """Approach position ~10 cm above the grasp point."""
        if self._gripper_pub is not None:
            pos = grasp_poses.get("position", [0, 0, 0])
            cmd = json.dumps({
                "cmd": "move_to",
                "position": [pos[0], pos[1], pos[2] + 0.10],
                "orientation": grasp_poses.get("orientation", [0, 0, 0, 1]),
            })
            self._gripper_pub.publish(String(data=cmd))
        # Stub: simulate motion time
        time.sleep(0.5)

    def _move_to_grasp(self, grasp_poses: dict) -> None:
        """Descend to the grasp point."""
        if self._gripper_pub is not None:
            pos = grasp_poses.get("position", [0, 0, 0])
            cmd = json.dumps({
                "cmd": "move_to",
                "position": pos,
                "orientation": grasp_poses.get("orientation", [0, 0, 0, 1]),
            })
            self._gripper_pub.publish(String(data=cmd))
        time.sleep(0.5)

    def _close_gripper(self, force: float) -> None:
        """Close gripper with specified force."""
        if self._gripper_pub is not None:
            cmd = json.dumps({"cmd": "close", "force": force})
            self._gripper_pub.publish(String(data=cmd))
        time.sleep(0.3)

    def _open_gripper(self) -> None:
        """Open gripper to release object."""
        if self._gripper_pub is not None:
            cmd = json.dumps({"cmd": "open"})
            self._gripper_pub.publish(String(data=cmd))
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_result(
        action: str,
        target: str,
        detail: Any = None,
    ) -> Dict[str, Any]:
        return {
            "status": "SUCCESS",
            "result": {
                "action": action,
                "target": target,
                "detail": detail,
                "stub": True,
                "message": f"Stub {action} — arm.enabled=false",
            },
        }
