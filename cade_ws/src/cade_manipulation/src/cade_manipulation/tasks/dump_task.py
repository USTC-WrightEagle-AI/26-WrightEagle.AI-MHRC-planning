"""Dump task — object_dump implementation.

Releases the currently held object at the target position.
"""

from cade_manipulation.tasks.base_task import BaseTask


class DumpTask(BaseTask):
    """Execute an object_dump command: move to target → release gripper."""

    def execute(self, cmd_dict):
        target_position = cmd_dict.get("target_position", cmd_dict.get("position"))
        release_safe = bool(
            cmd_dict.get("release_safe", cmd_dict.get("safe", True))
        )

        # ── 1. (Optional) Navigate to placement position ──────
        # If target_position is a 3D coordinate, the arm could move there
        # before releasing. For now this is handled by the arm controller.

        # ── 2. Execute arm release ────────────────────────────
        arm = self.node.arm
        arm_result = arm.execute_dump(
            target_position=target_position,
            release_safe=release_safe,
        )

        # ── 3. Enrich result ──────────────────────────────────
        arm_result.setdefault("result", {})
        arm_result["result"].update({
            "action": "dump",
            "target_position": target_position,
            "release_safe": release_safe,
        })
        return arm_result
