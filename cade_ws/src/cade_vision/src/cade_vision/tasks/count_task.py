"""Counting task executor."""

from .base_task import BaseTask


class CountTask(BaseTask):
    """Executor for count_objects and count_people."""

    def execute(self, cmd_dict):
        if not self.should_continue():
            return

        action = cmd_dict.get("action", "")
        placement = cmd_dict.get("placement", cmd_dict.get("room", ""))
        default_category = "person" if action == "count_people" else ""
        category = str(cmd_dict.get("category") or default_category)
        attributes = self.extract_attributes(cmd_dict)

        candidates = self.node.get_latest_detections()
        candidates = self.filter_candidates(candidates, attributes)
        matching = [
            obj
            for obj in candidates
            if category.lower() in obj.get("class_name", "").lower()
        ]

        result = {
            "type": "count_result",
            "category": category,
            "placement": placement,
            "count": len(matching),
            "items": [obj["class_name"] for obj in matching],
        }

        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status("SUCCESS", result=result)
