#!/usr/bin/env python3
"""Run one generated GPSR command through cade_brain with a deterministic LLM stub."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cade_ws/src/cade_brain/src"))

from cade_brain.controller import RobotController
from cade_brain.llm_core.schemas import RobotDecision, parse_action


COMMAND = "go to the cabinet then locate a tomato soup and take it and throw it in the trash"


class ScriptedLLM:
    def __init__(self):
        self.step = 0
        self.actions = [
            {
                "thought": "Step 1 is to go to the cabinet.",
                "reply": "I am going to the cabinet.",
                "action": {"type": "navigation", "position": "cabinet"},
            },
            {
                "thought": "The robot reached the cabinet. Now search for the tomato soup.",
                "reply": "I am looking for the tomato soup.",
                "action": {"type": "object_search", "object_name": "tomato soup"},
            },
            {
                "thought": "The tomato soup was found. Now pick it up.",
                "reply": "I found it. I will pick it up now.",
                "action": {
                    "type": "object_grasp",
                    "object_name": "tomato soup",
                    "object_position": [0.5, 0.2, 1.0],
                },
            },
            {
                "thought": "The object is held. Now throw it in the trash.",
                "reply": "I will throw it in the trash.",
                "action": {"type": "object_dump", "target_position": "trash"},
            },
            {
                "thought": "All requested steps are complete.",
                "reply": "Done. I put the tomato soup in the trash.",
                "action": None,
            },
        ]

    def get_decision(self, user_input, system_prompt, conversation_history=None):
        index = min(self.step, len(self.actions) - 1)
        self.step += 1
        item = dict(self.actions[index])
        if item["action"] is not None:
            item["action"] = parse_action(item["action"])
        return RobotDecision(**item)


class MockNavSkill:
    def go_to_location(self, target, then_find_person=False, timeout=60.0):
        return {"status": "SUCCESS", "result": {"target": target}}

    def pick_and_bring(self, object_name, placement, timeout=120.0):
        return {
            "status": "SUCCESS",
            "result": {"object_name": object_name, "placement": placement},
        }

    def execute(self, action_type, timeout=30.0, **params):
        return {"status": "SUCCESS", "result": {"action": action_type, **params}}


class MockVisionSkill:
    def find_object(self, object_name, room=None, timeout=30.0):
        return {
            "status": "SUCCESS",
            "result": {
                "name": object_name,
                "room": room,
                "position_3d": [0.5, 0.2, 1.0],
            },
        }

    def find_person(self, **params):
        return {"status": "SUCCESS", "result": params}

    def count_people(self, **params):
        return {"status": "SUCCESS", "result": {"count": 1, **params}}


def main():
    controller = RobotController(llm_client=ScriptedLLM(), show_thought=True)
    controller.nav_skill = MockNavSkill()
    controller.vision_skill = MockVisionSkill()
    decision = controller.process_input(COMMAND)
    print("\n=== TEST SUMMARY ===")
    print(f"command: {COMMAND}")
    print(f"final_reply: {decision.reply}")
    print(f"current_position: {controller.current_position}")
    print(f"successful_actions: {controller.successful_actions}")
    print(f"failed_actions: {controller.failed_actions}")
    print(f"history_messages: {len(controller.conversation_history)}")


if __name__ == "__main__":
    main()
