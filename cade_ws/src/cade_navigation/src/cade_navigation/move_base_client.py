"""move_base action client wrapper used by cade_navigation tasks."""

import time
from typing import Dict, Optional

from cade_navigation.pose_utils import yaw_deg_to_quaternion


GOAL_STATUS_NAMES = {
    0: "PENDING",
    1: "ACTIVE",
    2: "PREEMPTED",
    3: "SUCCEEDED",
    4: "ABORTED",
    5: "REJECTED",
    6: "PREEMPTING",
    7: "RECALLING",
    8: "RECALLED",
    9: "LOST",
}

MOVE_BASE_FAILURE_STATES = {4, 5, 9}
MOVE_BASE_ACTIVE_STATES = {0, 1, 6, 7}


class MoveBaseClient:
    """Small wrapper around actionlib.SimpleActionClient('/move_base')."""

    def __init__(self, action_name: str = "/move_base",
                 server_timeout: float = 10.0):
        import actionlib
        from move_base_msgs.msg import MoveBaseAction

        self.action_name = action_name
        self.server_timeout = float(server_timeout)
        self._client = actionlib.SimpleActionClient(action_name, MoveBaseAction)
        self._server_ready = False

    def wait_for_server(self) -> bool:
        import rospy

        if self._server_ready:
            return True

        rospy.loginfo("Waiting for move_base action server: %s", self.action_name)
        self._server_ready = self._client.wait_for_server(
            rospy.Duration(self.server_timeout)
        )
        return self._server_ready

    def cancel_goal(self) -> None:
        self._client.cancel_goal()

    def cancel_goal_if_active(self) -> bool:
        """Cancel the current goal only if actionlib still tracks it as active."""
        if self.get_state() not in MOVE_BASE_ACTIVE_STATES:
            return False
        self._client.cancel_goal()
        return True

    def send_goal(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        frame_id: str = "map",
    ) -> Dict:
        """Send a MoveBaseGoal without waiting for the terminal state."""
        if not self.wait_for_server():
            return {
                "status": "FAILED",
                "error": "Cannot connect to move_base action server: %s"
                % self.action_name,
            }

        goal = self._build_goal(x, y, yaw_deg, frame_id)
        import rospy

        rospy.loginfo(
            "Sending navigation goal: frame=%s x=%.3f y=%.3f yaw_deg=%.3f",
            frame_id,
            x,
            y,
            yaw_deg,
        )
        self._client.send_goal(goal)
        return {"status": "SENT"}

    def get_state(self) -> int:
        return int(self._client.get_state())

    def get_state_name(self) -> str:
        return GOAL_STATUS_NAMES.get(self.get_state(), str(self.get_state()))

    def is_failure_state(self) -> bool:
        return self.get_state() in MOVE_BASE_FAILURE_STATES

    def send_goal_and_wait(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        frame_id: str = "map",
        timeout: Optional[float] = None,
        cancel_event=None,
    ) -> Dict:
        """Send a MoveBaseGoal and wait for a terminal state."""
        import rospy
        from actionlib_msgs.msg import GoalStatus

        sent = self.send_goal(x, y, yaw_deg, frame_id=frame_id)
        if sent.get("status") == "FAILED":
            return sent

        timeout = float(timeout or 0.0)
        deadline = time.time() + timeout if timeout > 0.0 else None
        while not rospy.is_shutdown():
            if cancel_event is not None and cancel_event.is_set():
                self._client.cancel_goal()
                return {
                    "status": "FAILED",
                    "error": "Navigation canceled",
                    "move_base_state": "PREEMPTED",
                }

            if self._client.wait_for_result(rospy.Duration(0.2)):
                break

            if deadline is not None and time.time() >= deadline:
                self._client.cancel_goal()
                return {
                    "status": "TIMEOUT",
                    "error": "Navigation timeout after %.1fs" % timeout,
                    "move_base_state": "TIMEOUT",
                }

        if rospy.is_shutdown():
            self._client.cancel_goal()
            return {
                "status": "FAILED",
                "error": "ROS shutdown while waiting for navigation result",
            }

        state = self._client.get_state()
        state_name = GOAL_STATUS_NAMES.get(state, str(state))
        if state == GoalStatus.SUCCEEDED:
            return {
                "status": "SUCCESS",
                "result": {"move_base_state": state_name},
            }

        return {
            "status": "FAILED",
            "error": "Navigation failed with GoalStatus %d (%s)"
            % (state, state_name),
            "move_base_state": state_name,
        }

    @staticmethod
    def _build_goal(x: float, y: float, yaw_deg: float, frame_id: str):
        import rospy
        from move_base_msgs.msg import MoveBaseGoal

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame_id
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(x)
        goal.target_pose.pose.position.y = float(y)
        goal.target_pose.pose.position.z = 0.0

        q = yaw_deg_to_quaternion(float(yaw_deg))
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        return goal
