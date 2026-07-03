"""Grasp task — bringMeObj implementation.

Full pipeline:
  1. Capture RGB + 16-bit depth from RealSense
  2. Send to 5090 GraspNet via TCP
  3. Parse grasp_poses response
  4. Execute grasp on robot arm
"""

from cade_manipulation.tasks.base_task import BaseTask


class GraspTask(BaseTask):
    """Execute a bringMeObj command: capture → GraspNet → arm grasp."""

    def execute(self, cmd_dict):
        object_name = cmd_dict.get("object_name", cmd_dict.get("object", "object"))
        grasp_force = cmd_dict.get("grasp_force", None)
        placement = cmd_dict.get("placement", cmd_dict.get("object_position", None))

        # ── 1. Capture camera frames ────────────────────────────
        camera = self.node.camera
        if not camera.has_camera:
            return {
                "status": "FAILED",
                "error": "No camera available for grasp capture",
            }

        try:
            rgb, depth, camera_info = camera.capture()
        except Exception as exc:
            return {
                "status": "FAILED",
                "error": f"Camera capture failed: {exc}",
                "result": {
                    "action": "grasp",
                    "object_name": object_name,
                    "stage": "camera_capture",
                },
            }

        # ── 2. Query GraspNet 5090 ─────────────────────────────
        graspnet = self.node.graspnet
        result = graspnet.query_grasp(
            text_prompt=object_name,
            rgb_image=rgb,
            depth_image=depth,
        )

        if not result.get("success"):
            return {
                "status": "FAILED",
                "error": result.get("message", "GraspNet failed"),
                "result": {
                    "action": "grasp",
                    "object_name": object_name,
                    "stage": "graspnet_query",
                    "graspnet_response": result,
                    "camera_info": camera_info,
                },
            }

        grasp_poses = result.get("grasp_poses")
        if grasp_poses is None:
            return {
                "status": "FAILED",
                "error": "GraspNet returned success=true but no grasp_poses",
                "result": {
                    "action": "grasp",
                    "object_name": object_name,
                    "stage": "graspnet_query",
                    "graspnet_response": result,
                },
            }

        # ── 3. Execute arm grasp ──────────────────────────────
        arm = self.node.arm
        arm_result = arm.execute_grasp(
            grasp_poses=grasp_poses,
            grasp_force=grasp_force,
            object_name=object_name,
        )

        if arm_result.get("status") != "SUCCESS" and arm.enabled:
            return arm_result

        # ── 4. Enrich result ──────────────────────────────────
        arm_result.setdefault("result", {})
        arm_result["result"].update({
            "action": "grasp",
            "object_name": object_name,
            "grasp_poses": grasp_poses,
            "graspnet_message": result.get("message", ""),
            "camera_info": camera_info,
            "placement": placement,
        })
        return arm_result
