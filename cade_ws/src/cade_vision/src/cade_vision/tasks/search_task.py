"""Search-style task executors."""

import time

from .base_task import BaseTask


class SearchTask(BaseTask):
    """Executor for object/person search and attribute filtering."""

    POLL_INTERVAL = 0.1

    def execute(self, cmd_dict):
        action = cmd_dict.get("action", "")
        if action == "filter_by_attributes":
            self._execute_filter_by_attributes(cmd_dict)
            return
        self._execute_find(cmd_dict)

    def _execute_find(self, cmd_dict):
        action = cmd_dict.get("action", "")
        timeout = float(cmd_dict.get("timeout", 30.0))
        default_target = "person" if action == "find_person" else ""
        target_class = str(cmd_dict.get("target") or default_target).lower()
        attributes = self.extract_attributes(cmd_dict)
        start_time = time.time()

        while self.should_continue() and time.time() - start_time < timeout:
            candidates = self.node.get_latest_detections()
            candidates = self.filter_candidates(candidates, attributes)
            matches = [
                obj
                for obj in candidates
                if target_class
                and target_class in obj.get("class_name", "").lower()
            ]

            if self.node.image_source == "realsense":
                matches = [m for m in matches if m.get("position_3d") is not None]

            if matches:
                if self.node.image_source == "realsense":
                    target = min(matches, key=lambda o: o["position_3d"][2])
                    position = list(target["position_3d"])
                else:
                    target = max(matches, key=lambda o: o["confidence"])
                    position = None

                detection_msg = {
                    "type": "object_detection",
                    "name": target["class_name"],
                    "confidence": float(target["confidence"]),
                    "position_3d": position,
                    "bbox": list(target["bbox"]),
                }
                if self.node.finish_task(self):
                    self.node._publish_detection(detection_msg)
                    self.node._publish_status("SUCCESS", result=detection_msg)
                return

            time.sleep(self.POLL_INTERVAL)

        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status(
                "FAILED", error=f"Target '{target_class}' not found"
            )

    def _execute_filter_by_attributes(self, cmd_dict):
        attributes = self.extract_attributes(cmd_dict, include_top_level=True)
        candidates = self.node.get_latest_detections()
        persons = [
            obj for obj in candidates if obj.get("class_name", "").lower() == "person"
        ]
        matched = self.filter_candidates(persons, attributes)

        result = {
            "persons": [
                {
                    "track_id": person.get("track_id"),
                    "bbox": list(person.get("bbox", [])),
                    "position_3d": person.get("position_3d"),
                    "cloth_color": person.get("cloth_color", "unknown"),
                    "cloth_type": person.get("cloth_type", "unknown"),
                    "hair_color": "unknown",
                    "has_glasses": False,
                    "height": "unknown",
                }
                for person in matched
            ],
            "count": len(matched),
        }

        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status("SUCCESS", result=result)
