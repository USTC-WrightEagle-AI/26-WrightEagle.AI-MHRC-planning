"""Information lookup task executors."""

import math

from .base_task import BaseTask


class InfoTask(BaseTask):
    """Executor for get_person_info and get_nearest_person."""

    def execute(self, cmd_dict):
        action = cmd_dict.get("action", "")
        if action == "get_nearest_person":
            self._execute_nearest_person()
            return
        self._execute_person_info(cmd_dict)

    def _execute_person_info(self, cmd_dict):
        if not self.should_continue():
            return

        candidates = self.node.get_latest_detections()
        attributes = self.extract_attributes(cmd_dict)
        candidates = self.filter_candidates(candidates, attributes)

        result = {
            "type": "detection_info",
            "objects": [
                {
                    "name": obj["class_name"],
                    "confidence": float(obj["confidence"]),
                    "position_3d": list(obj["position_3d"])
                    if obj.get("position_3d")
                    else None,
                }
                for obj in candidates
            ],
        }

        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status("SUCCESS", result=result)

    def _execute_nearest_person(self):
        if not self.should_continue():
            return

        candidates = self.node.get_latest_detections()
        persons = [
            obj for obj in candidates if obj.get("class_name", "").lower() == "person"
        ]
        if not persons:
            if self.should_continue() and self.node.finish_task(self):
                self.node._publish_status("FAILED", error="No person detected")
            return

        nearest = None
        min_dist = float("inf")
        for person in persons:
            position = person.get("position_3d")
            if position is not None:
                distance = math.sqrt(
                    position[0] ** 2 + position[1] ** 2 + position[2] ** 2
                )
                if distance < min_dist:
                    min_dist = distance
                    nearest = person
        if nearest is None:
            nearest = persons[0]

        result = {
            "cloth_color": nearest.get("cloth_color", "unknown"),
            "cloth_type": nearest.get("cloth_type", "unknown"),
            "hair_color": "unknown",
            "has_glasses": False,
            "height": "unknown",
            "position_3d": nearest.get("position_3d"),
        }

        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status("SUCCESS", result=result)
