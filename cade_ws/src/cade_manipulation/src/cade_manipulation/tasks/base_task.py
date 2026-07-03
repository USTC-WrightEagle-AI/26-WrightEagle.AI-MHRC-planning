"""Base task abstraction for manipulation task executors."""

from abc import ABC, abstractmethod


class BaseTask(ABC):
    """Common cooperative task executor interface."""

    def __init__(self, node_context, cancel_event):
        self.node = node_context
        self.cancel_event = cancel_event

    @abstractmethod
    def execute(self, cmd_dict):
        """Run the task and return a task status dictionary."""

    def should_continue(self) -> bool:
        return not self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()
