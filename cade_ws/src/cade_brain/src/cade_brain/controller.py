"""
Brain Controller - 中央调度器 (Refactored for ROS Pub/Sub)

整合大脑(LLM)、动态世界模型(Memory)和技能分发器(Skills)，
实现完整的感知-决策-执行循环。

与旧版关键区别：
- 不再直接调用 self.robot 的模拟方法
- 改为通过 VisionSkill / NavSkill 向 ROS Topic 发布指令
- 世界模型开机为空，由视觉观测动态填充
"""

from typing import List, Dict, Optional
import time

from cade_brain.llm_core.config import Config
from cade_brain.llm_core.llm_client import LLMClient
from cade_brain.llm_core.prompts import get_system_prompt
from cade_brain.llm_core.schemas import RobotDecision, RobotAction
from cade_brain.memory import WorldModel
from cade_brain.skills.vision_skill import VisionSkill
from cade_brain.skills.nav_skill import NavSkill

# Re-export RobotState for compatibility
from enum import Enum


class RobotState(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


class RobotController:
    """
    机器人主控制器 (Refactored)

    职责：
    1. 接收用户输入
    2. 调用 LLM 进行决策
    3. 通过 Skills 执行动作（ROS Pub/Sub）
    4. 管理对话历史和世界模型
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_mode: str = "default",
        show_thought: bool = True,
        environment_context: Optional[str] = None
    ):
        # 动态世界模型（开机为空）
        self.world = WorldModel()

        # 技能分发器
        self.vision_skill = VisionSkill()
        self.nav_skill = NavSkill()

        # 状态管理
        self.state = RobotState.IDLE

        # LLM 客户端
        self.llm_client = llm_client or LLMClient()

        # 系统提示词
        self.system_prompt = get_system_prompt(prompt_mode)

        # 注入环境上下文
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

        print(f"\n{'='*60}")
        print(f"Robot Controller initialized (ROS Pub/Sub Mode)")
        print(f"  World Model: empty (waiting for vision input)")
        print(f"  Skills: VisionSkill + NavSkill")
        print(f"{'='*60}")

    def set_state(self, state: RobotState):
        self.state = state

    def is_busy(self) -> bool:
        return self.state in (RobotState.THINKING, RobotState.SPEAKING, RobotState.EXECUTING)

    def process_input(self, user_input: str) -> RobotDecision:
        """
        处理用户输入（完整流程）

        Args:
            user_input: 用户输入文本

        Returns:
            RobotDecision: 决策对象
        """
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print(f"{'='*60}")

        self.total_interactions += 1
        self.set_state(RobotState.THINKING)
        print(f"\n[Brain thinking...]")

        try:
            # 注入世界状态到系统提示词
            enriched_prompt = self._enrich_prompt_with_world_state()

            decision = self.llm_client.get_decision(
                user_input=user_input,
                system_prompt=enriched_prompt,
                conversation_history=self.conversation_history
            )

            # 打印决策
            if self.show_thought and decision.thought:
                print(f"\nThought: {decision.thought}")
            if decision.reply:
                print(f"Reply: {decision.reply}")
            if decision.action:
                print(f"Action: {decision.action.type}")

            # 执行动作（通过 Skills + ROS）
            if decision.action:
                success = self._execute_action(decision.action)
                if success:
                    self.successful_actions += 1
                else:
                    self.failed_actions += 1

            # 更新对话历史
            self._update_conversation_history(user_input, decision)

            self.set_state(RobotState.IDLE)
            return decision

        except Exception as e:
            print(f"\nError: {e}")
            self.set_state(RobotState.ERROR)
            raise

    def _enrich_prompt_with_world_state(self) -> str:
        """将当前世界状态注入系统提示词"""
        world_state = self.world.get_world_state()
        base_prompt = self.system_prompt.split("--- Current World State ---")[0].rstrip()
        return base_prompt + "\n\n" + world_state

    def _execute_action(self, action: RobotAction) -> bool:
        """
        执行具体动作（通过 Skills 分发）

        Args:
            action: 动作对象

        Returns:
            bool: 是否成功
        """
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
                    self.world.current_position = action.target

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
                    self.world.current_position = action.room

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

            # ==================== 世界模型查询类动作（直接从 memory 查询）====================
            elif action_type == "tellPrsInfoInLoc":
                person_name = getattr(action, 'person_name', None)
                location = action.location
                if person_name:
                    person = self.world.get_person(person_name)
                    result = {"status": "SUCCESS", "result": person}
                else:
                    people = self.world.get_people_in_location(location)
                    result = {"status": "SUCCESS", "result": {"location": location, "people": people}}

            elif action_type == "tellObjPropOnPlcmt":
                obj = self.world.get_object(action.object_name)
                if obj and getattr(action, 'property', None) in obj:
                    result = {"status": "SUCCESS", "result": {
                        "name": action.object_name,
                        "property": action.property,
                        "value": obj[action.property]
                    }}
                else:
                    result = {"status": "FAILED", "error": f"Property {action.property} not found"}

            elif action_type == "tellCatPropOnPlcmt":
                objects = self.world.get_objects_in_location(action.placement)
                matching = [o for o in objects if o.get("category") == action.category]
                if matching:
                    values = [o.get(action.property) for o in matching if action.property in o]
                    result = {"status": "SUCCESS", "result": {
                        "category": action.category,
                        "count": len(matching),
                        "items": [o["name"] for o in matching],
                        "max": max(values) if values else None,
                        "min": min(values) if values else None,
                    }}
                else:
                    result = {"status": "FAILED", "error": "No matching objects"}

            # ==================== 人物交互类动作（综合） ====================
            elif action_type == "meetPrsAtBeac":
                result = self.nav_skill.go_to_location(target=action.beacon)
                if result.get("status") == "SUCCESS":
                    self.world.current_position = action.beacon

            elif action_type == "greetNameInRm":
                person = self.world.get_person(action.person_name)
                if person:
                    result = {"status": "SUCCESS", "result": f"Greeted {action.person_name}"}
                else:
                    result = {"status": "FAILED", "error": f"Person {action.person_name} not found"}

            elif action_type == "greetClothDscInRm":
                people = self.world.get_people_in_location(action.room)
                matching = [p for p in people if p.get("cloth_color") == action.cloth_color]
                if matching:
                    result = {"status": "SUCCESS", "result": f"Greeted {matching[0]['name']}"}
                else:
                    result = {"status": "FAILED", "error": "No matching person"}

            elif action_type == "countClothPrsInRoom":
                people = self.world.get_people_in_location(action.room)
                count = sum(1 for p in people if p.get("cloth_color") == action.cloth_color)
                result = {"status": "SUCCESS", "result": {"count": count}}

            # ==================== 兜底：将其他动作委托给 NavSkill ====================
            else:
                result = self.nav_skill.execute(action_type, timeout=60.0)

            success = result is not None and result.get("status") == "SUCCESS"

            if success:
                print(f"Action {action_type} succeeded: {result}")
            else:
                print(f"Action {action_type} failed: {result}")

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
        """
        注入外部观测到对话上下文（供视觉模块、传感器等调用）

        Args:
            observation: 观测描述文本
            update_world: 是否同时刷新 world_state 摘要到系统提示词

        Returns:
            str: 当前世界状态摘要
        """
        self.conversation_history.append({
            "role": "system",
            "content": f"[OBSERVATION] {observation}"
        })
        print(f"\n[OBSERVE] {observation}")

        world_state = self.world.get_world_state()
        if update_world:
            self.system_prompt = self._enrich_prompt_with_world_state()

        print(f"\n[WORLD STATE]\n{world_state}")
        return world_state

    def reset(self):
        """重置控制器状态"""
        self.conversation_history.clear()
        self.world = WorldModel()
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
