"""Wait task implementation."""

import time

from cade_navigation.pose_utils import parse_float
from cade_navigation.tasks.base_task import BaseTask


class WaitTask(BaseTask):
    """Cooperative wait action for brain task sequencing."""

    def execute(self, cmd_dict):
        duration = parse_float(cmd_dict.get("duration"), 5.0, "duration")
        deadline = time.time() + duration
        while time.time() < deadline:
            if not self.should_continue():
                return {"status": "FAILED", "error": "Wait canceled"}
            time.sleep(0.05)

        return {
            "status": "SUCCESS",
            "result": {
                "action": "wait",
                "duration": duration,
            },
        }
