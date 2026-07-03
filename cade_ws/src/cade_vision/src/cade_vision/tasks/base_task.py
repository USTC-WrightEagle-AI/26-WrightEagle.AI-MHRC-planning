"""Base task abstraction for cade_vision task executors."""

from abc import ABC, abstractmethod
import threading


class BaseTask(ABC):
    """Common task executor interface."""

    RESERVED_COMMAND_KEYS = {
        "action",
        "target",
        "category",
        "timeout",
        "room",
        "placement",
        "vision_profile",
        "perception_profile",
    }

    PERSON_ATTRIBUTE_KEYS = {
        "posture",
        "gesture",
        "cloth_color",
        "cloth_type",
        "clothing",
        "identity",
        "name",
        "hair_color",
        "has_glasses",
        "height",
    }

    def __init__(self, node_context):
        self.node = node_context
        self.is_running = True
        self._state_lock = threading.Lock()

    @abstractmethod
    def execute(self, cmd_dict):
        """Run the task until success, failure, or cancellation."""

    def cancel(self):
        """Request cooperative task termination."""
        with self._state_lock:
            self.is_running = False

    def should_continue(self):
        with self._state_lock:
            return self.is_running

    def extract_attributes(self, cmd_dict, include_top_level=False):
        """Return a normalized attributes dict without mutating the command."""
        attributes = {}
        raw_attributes = cmd_dict.get("attributes")
        if isinstance(raw_attributes, dict):
            attributes.update(raw_attributes)

        if include_top_level:
            keys = [
                key
                for key in cmd_dict.keys()
                if key not in self.RESERVED_COMMAND_KEYS and key != "attributes"
            ]
        else:
            keys = [
                key
                for key in self.PERSON_ATTRIBUTE_KEYS
                if key in cmd_dict and key not in attributes
            ]

        for key in keys:
            attributes.setdefault(key, cmd_dict.get(key))

        normalized = {
            key: value
            for key, value in attributes.items()
            if value is not None and value != "" and value != "unknown"
        }
        return normalized or None

    def filter_candidates(self, candidates, attributes):
        """Apply Analyzer-based filtering with a local exact-match fallback."""
        if not attributes:
            return candidates

        analyzer = getattr(self.node, "analyzer", None)
        if analyzer is not None:
            return analyzer.filter_by_attributes(candidates, attributes)

        matched = list(candidates)
        for key, expected in attributes.items():
            expected_lower = str(expected).lower()
            matched = [
                item
                for item in matched
                if str(item.get(key, "")).lower() == expected_lower
            ]
        return matched
