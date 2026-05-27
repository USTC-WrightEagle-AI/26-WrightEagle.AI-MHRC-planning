"""
Brain Controller - 中央调度器 (Refactored for ROS Pub/Sub)

整合大脑(LLM)、情景记忆(EpisodicMemory)和技能分发器(Skills)，
实现完整的感知-决策-执行循环。

WorldModel 已降级为 simple state。LLM 通过 action 主动查询，
不依赖预加载的 context injection。
"""

from typing import List, Dict, Optional
import json
import time

from cade_brain.llm_core.config import Config
from cade_brain.llm_core.llm_client import LLMClient
from cade_brain.llm_core.prompts import get_system_prompt
from cade_brain.llm_core.schemas import RobotDecision, RobotAction
from cade_brain.memory import EpisodicMemory
from cade_brain.skills.vision_skill import VisionSkill
from cade_brain.skills.nav_skill import NavSkill

from enum import Enum


class RobotState(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


class RobotController:
    """机器人主控制器 — 纯静态 prompt，action 主动查询记忆"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_mode: str = "default",
        show_thought: bool = True,
        environment_context: Optional[str] = None
    ):
        # 情景记忆（生命周期 = Controller 生命周期，开机为空）
        self.episodic = EpisodicMemory()

        # 简单状态（替代 WorldModel）
        self.current_position: str = "home"
        self.holding_object: Optional[str] = None

        # 技能分发器
        self.vision_skill = VisionSkill()
        self.nav_skill = NavSkill()

        # 状态管理
        self.state = RobotState.IDLE

        # LLM 客户端
        self.llm_client = llm_client or LLMClient()

        # 系统提示词（纯静态，不加 world_state 注入）
        self.system_prompt = get_system_prompt(prompt_mode)
        if environment_context:
            self.system_prompt += f"\n\n## Current Environment\n{environment_context}"

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

        # 显示选项
        self.show_thought = show_thought

        # 统计信息
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0

        # 最后一个 action 的执行结果（注入 observe 用）
        self._last_action_result = {}

        print(f"\n{'='*60}")
        print(f"Robot Controller initialized (ROS Pub/Sub + Episodic Memory)")
        print(f"  Episodic Memory: empty")
        print(f"  Skills: VisionSkill + NavSkill")
        print(f"{'='*60}")

    def set_state(self, state: RobotState):
        self.state = state

    def is_busy(self) -> bool:
        return self.state in (RobotState.THINKING, RobotState.SPEAKING, RobotState.EXECUTING)

    def process_input(self, user_input: str) -> RobotDecision:
        """
        处理用户输入（ReAct 循环 + 纯静态 prompt）

        1. 调用 LLM 获取决策
        2. 如有 action，执行后把结果注入对话历史，再次调用 LLM
        3. 循环直到 LLM 不再输出 action 或达到最大步数
        """
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print(f"{'='*60}")

        self.total_interactions += 1
        self.set_state(RobotState.THINKING)
        print(f"\n[Brain thinking...]")

        try:
            decision = self.llm_client.get_decision(
                user_input=user_input,
                system_prompt=self.system_prompt,
                conversation_history=self.conversation_history
            )

            if self.show_thought and decision.thought:
                print(f"\n[Thought #1]: {decision.thought}")
            if decision.reply:
                print(f"[Reply #1]: {decision.reply}")
            if decision.action:
                print(f"[Action #1]: {decision.action.type}")

            # ============ ReAct 循环 ============
            MAX_REACT_LOOPS = 5
            loop_count = 0
            final_reply = None

            while decision.action is not None and loop_count < MAX_REACT_LOOPS:
                loop_count += 1

                if decision.reply and final_reply is None:
                    final_reply = decision.reply

                action_type = decision.action.type
                action_params = decision.action.model_dump(exclude={"type"})
                action_success = self._execute_action(decision.action)

                if action_success:
                    self.successful_actions += 1
                else:
                    self.failed_actions += 1

                action_obs = {
                    "event": "action_executed",
                    "action_type": action_type,
                    "params": action_params,
                    "status": "SUCCESS" if action_success else "FAILED",
                    "result": self._last_action_result
                }
                self.observe(json.dumps(action_obs))

                decision = self.llm_client.get_decision(
                    user_input="[Continue based on the action result above]",
                    system_prompt=self.system_prompt,
                    conversation_history=self.conversation_history
                )

                label = f"#{loop_count + 1}"
                if self.show_thought and decision.thought:
                    print(f"\n[Thought {label}]: {decision.thought}")
                if decision.reply:
                    print(f"[Reply {label}]: {decision.reply}")
                if decision.action:
                    print(f"[Action {label}]: {decision.action.type}")

            if decision.reply:
                final_reply = decision.reply
            if final_reply:
                decision.reply = final_reply

            print(f"\n[ReAct] Loop ended after {loop_count} action(s)")

            self._update_conversation_history(user_input, decision)

            self.set_state(RobotState.IDLE)
            return decision

        except Exception as e:
            print(f"\nError: {e}")
            self.set_state(RobotState.ERROR)
            raise

    def _execute_action(self, action: RobotAction) -> bool:
        """执行动作（通过 Skills 分发）"""
        action_type = action.type
        self.set_state(RobotState.EXECUTING)

        try:
            result = None

            # ==================== 导航类动作 ====================
            if action_type == "goToLoc":
                result = self.nav_skill.go_to_location(
                    target=action.target,
                    then_find_person=getattr(action, 'then_find_person', False)
                )
                if result.get("status") == "SUCCESS":
                    self.current_position = action.target

            # ==================== 人物操作类动作 (委托给 VisionSkill) ====================
            elif action_type == "findPrsInRoom":
                result = self.vision_skill.find_person(
                    room=action.room,
                    gesture=getattr(action, 'gesture', None)
                )

            elif action_type == "countPrsInRoom":
                result = self.vision_skill.count_people(
                    room=action.room,
                    gesture=getattr(action, 'gesture', None)
                )

            elif action_type == "findObjInRoom":
                result = self.vision_skill.find_object(
                    object_name=action.object_name,
                    room=action.room
                )

            elif action_type == "countObjOnPlcmt":
                result = self.vision_skill.count_objects(
                    category=action.object_category,
                    placement=action.placement
                )

            # ==================== 人物操作类动作 (委托给 NavSkill) ====================
            elif action_type == "followNameFromBeacToRoom":
                result = self.nav_skill.follow_person(
                    person_name=action.person_name
                )
                if result.get("status") == "SUCCESS":
                    self.current_position = action.room

            elif action_type == "guideNameFromBeacToBeac":
                result = self.nav_skill.guide_person(
                    person_name=action.person_name,
                    from_beacon=action.from_beacon,
                    to_beacon=action.to_beacon
                )

            # ==================== 物品操作类动作 (委托给 NavSkill) ====================
            elif action_type == "bringMeObjFromPlcmt":
                result = self.nav_skill.pick_and_bring(
                    object_name=action.object_name,
                    placement=action.placement
                )

            elif action_type == "takeObjFromPlcmt":
                result = self.nav_skill.take_object(
                    object_name=action.object_name,
                    placement=action.placement
                )

            # ==================== 情景记忆类动作 ====================
            elif action_type == "bind_person":
                attrs = self.vision_skill.get_nearest_person_attributes()
                if attrs:
                    self.episodic.bind_person(action.name, attrs)
                    result = {"status": "SUCCESS",
                              "result": f"Bound {action.name} to appearance: {attrs}"}
                else:
                    result = {"status": "FAILED", "error": "No person detected nearby"}

            elif action_type == "recall":
                what = getattr(action, 'what', 'appearance')
                if what == "appearance":
                    data = self.episodic.recall_appearance(action.name)
                else:
                    data = self.episodic.recall_info(action.name)
                if data:
                    result = {"status": "SUCCESS", "result": data}
                else:
                    result = {"status": "FAILED",
                              "error": f"No memory of {action.name}"}

            elif action_type == "remember":
                self.episodic.remember(
                    action.name,
                    getattr(action, 'key', ''),
                    getattr(action, 'value', ''))
                result = {"status": "SUCCESS",
                          "result": f"Remembered {action.name}.{getattr(action,'key','')}"}

            elif action_type == "find_person_by_attributes":
                attrs = getattr(action, 'attributes', {})
                result = self.vision_skill.filter_by_attributes(attrs)

            elif action_type == "speak":
                result = {"status": "SUCCESS", "result": getattr(action, 'text', '')}

            # ==================== 保留综合动作 ====================
            elif action_type == "meetPrsAtBeac":
                result = self.nav_skill.go_to_location(target=action.beacon)
                if result.get("status") == "SUCCESS":
                    self.current_position = action.beacon

            elif action_type == "greetClothDscInRm":
                result = self.vision_skill.find_person(
                    room=action.room,
                    category=f"person wearing {action.cloth_color} clothes"
                )

            elif action_type == "countClothPrsInRoom":
                result = self.vision_skill.count_people(
                    room=action.room,
                    gesture=getattr(action, 'gesture', None),
                    category=f"person wearing {action.cloth_color} clothes"
                )

            # ==================== 兜底 ====================
            else:
                result = self.nav_skill.execute(action_type, timeout=60.0)

            success = result is not None and result.get("status") == "SUCCESS"

            if success:
                print(f"Action {action_type} succeeded: {result}")
            else:
                print(f"Action {action_type} failed: {result}")

            self._last_action_result = result
            return success

        except Exception as e:
            print(f"Action execution failed: {e}")
            return False

    def _update_conversation_history(self, user_input: str, decision: RobotDecision):
        """更新对话历史"""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        assistant_response_parts = []
        if decision.thought:
            assistant_response_parts.append(f"Thought: {decision.thought}")
        if decision.reply:
            assistant_response_parts.append(f"Reply: {decision.reply}")
        if decision.action:
            assistant_response_parts.append(f"Action: {decision.action.model_dump_json()}")
        self.conversation_history.append({
            "role": "assistant",
            "content": "\n".join(assistant_response_parts)
        })

    def observe(self, observation: str, update_world: bool = True) -> str:
        """注入外部观测到对话上下文"""
        self.conversation_history.append({
            "role": "system",
            "content": f"[OBSERVATION] {observation}"
        })
        print(f"\n[OBSERVE] {observation}")
        return observation

    def reset(self):
        """重置控制器状态"""
        self.conversation_history.clear()
        self.episodic = EpisodicMemory()
        self.current_position = "home"
        self.holding_object = None
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        print("Controller reset")

    def print_statistics(self):
        """打印统计信息"""
        print(f"\nStatistics:")
        print(f"   Total interactions: {self.total_interactions}")
        print(f"   Successful actions: {self.successful_actions}")
        print(f"   Failed actions: {self.failed_actions}")
        if self.total_interactions > 0:
            success_rate = (self.successful_actions / self.total_interactions) * 100
            print(f"   Success rate: {success_rate:.1f}%")
