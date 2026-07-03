"""
Brain Controller - 中央调度器 (Registry + ReAct)

整合大脑(LLM)和技能工具箱(Tools Registry)，
实现完整的感知-决策-执行循环。

记忆完全依赖时序上下文（Conversation History），
状态机已移除，情景记忆已移除。
"""

from typing import Callable, List, Dict, Optional
import json

from cade_brain.llm_core.llm_client import LLMClient
from cade_brain.llm_core.providers.system_prompt import get_system_prompt
from cade_brain.schemas import RobotDecision, RobotAction

# ★ 导入装饰器自动构建的工具箱字典
from cade_brain.skills import nav_tools, vision_tools


class RobotController:
    """机器人主控制器 — 工具箱字典 + ReAct 循环

    核心架构：
    - skills_registry：nav_tools ∪ vision_tools 合并后的动作分发表
    - conversation_history：唯一的记忆载体（时序上下文）
    - ReAct 循环：思考 → 行动 → 观测，最多 MAX_REACT_LOOPS 轮
    """

    MAX_REACT_LOOPS = 30

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_mode: str = "default",
        show_thought: bool = True,
        environment_context: Optional[str] = None,
        max_react_loops: Optional[int] = None,
        live_reply_callback: Optional[Callable[[str], None]] = None,
    ):
        # 简单位置/持有状态（被动记录，不驱动逻辑）
        self.current_position: str = "home"
        self.holding_object: Optional[str] = None

        # ★ 工具箱注册表：直接合并两个全局字典
        self.skills_registry: Dict[str, callable] = {}
        self.skills_registry.update(nav_tools)
        self.skills_registry.update(vision_tools)

        # LLM 客户端
        self.llm_client = llm_client or LLMClient()

        # 系统提示词（纯静态，不加 world_state 注入）
        self.system_prompt = get_system_prompt(prompt_mode)
        if environment_context:
            self.system_prompt += f"\n\n## Current Environment\n{environment_context}"

        # 对话历史（时序上下文 = 唯一记忆载体）
        self.conversation_history: List[Dict[str, str]] = []

        # 显示选项
        self.show_thought = show_thought
        self.max_react_loops = int(max_react_loops or self.MAX_REACT_LOOPS)
        self.live_reply_callback = live_reply_callback

        # 统计信息
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0

        # 最后一个 action 的执行结果
        self._last_action_result = {}
        self._no_progress_action_counts = {}

        print(f"\n{'='*60}")
        print(f"Robot Controller initialized (Tool Registry + ReAct)")
        print(f"  Tools Registry: {len(self.skills_registry)} functions loaded")
        print(f"  Max ReAct loops: {self.max_react_loops}")
        print(f"  Memory: conversation_history only")
        print(f"{'='*60}")

    # ─── ReAct 主循环 ─────────────────────────────────────────

    def process_input(self, user_input: str) -> RobotDecision:
        """
        处理用户输入（ReAct 循环 + 时序上下文记忆链）

        记忆链条（每轮立即写入 conversation_history）：
        ① user_input → user 消息
        ② LLM Decision → assistant 消息（Thought + Action + Reply）
        ③ Action Result → system [OBSERVATION] 消息
        ④ 下一轮 LLM 基于完整历史继续推理

        返回最后一轮 LLM 给出的最终 RobotDecision（reply 为最终回复）。
        """
        self.total_interactions += 1
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print(f"{'='*60}")

        decision = None
        loop_count = 0

        try:
            for loop_count in range(self.max_react_loops):
                # ② 获取当前决策
                prompt = (
                    user_input
                    if loop_count == 0
                    else "[Continue based on the action result above]"
                )
                print(f"\n[Brain thinking... (loop {loop_count + 1})]")

                decision = self.llm_client.get_decision(
                    user_input=prompt,
                    system_prompt=self.system_prompt,
                    conversation_history=self.conversation_history,
                )

                # 打印本轮思考过程
                label = f"#{loop_count + 1}"
                if self.show_thought and decision.thought:
                    print(f"\n[Thought {label}]: {decision.thought}")
                if decision.reply:
                    print(f"[Reply {label}]: {decision.reply}")
                if decision.action:
                    print(f"[Action {label}]: {decision.action.type}")

                # ③ 将本轮 user prompt 与 assistant decision 写入历史。
                # LLMClient 会在请求中临时追加当前 prompt；这里在成功解析后
                # 再持久化，避免同一用户输入在消息列表中出现两次。
                self.conversation_history.append(
                    {"role": "user", "content": prompt}
                )
                self._append_decision_to_history(decision)

                # ④ 检查是否终止（action 为 None → LLM 认为任务完成）
                if decision.action is None:
                    break

                if loop_count == 0 and decision.reply and self.live_reply_callback:
                    self.live_reply_callback(decision.reply)

                # ⑤ 执行动作
                action_type = decision.action.type
                action_params = decision.action.model_dump(exclude={"type"})
                action_signature = self._action_signature(decision.action)
                if self._no_progress_action_counts.get(action_signature, 0) >= 2:
                    action_success = False
                    self._last_action_result = {
                        "status": "FAILED",
                        "error": (
                            "Repeated no-progress action blocked. "
                            "Use a different observation, reposition, navigation target, "
                            "or ask the user for clarification."
                        ),
                        "blocked_repeated_action": True,
                        "action_type": action_type,
                        "params": action_params,
                    }
                    print(f"[Action] {action_type} blocked after repeated no-progress attempts")
                else:
                    action_success = self._execute_action(decision.action)
                    self._update_no_progress_counts(
                        action_signature,
                        action_type,
                        action_success,
                        self._last_action_result,
                    )

                if action_success:
                    self.successful_actions += 1
                else:
                    self.failed_actions += 1

                # ⑥ 将执行结果作为 [OBSERVATION] 追加到 conversation_history
                action_obs = {
                    "event": "action_executed",
                    "action_type": action_type,
                    "params": action_params,
                    "status": "SUCCESS" if action_success else "FAILED",
                    "result": self._last_action_result,
                }
                action_obs_json = json.dumps(action_obs, ensure_ascii=False)
                self.conversation_history.append({
                    "role": "system",
                    "content": f"[OBSERVATION] {action_obs_json}",
                })
                print(f"\n[OBSERVE] {action_obs_json}")

            # ⑦ 返回最后一轮的 decision（reply 已是最终回复）
            print(f"\n[ReAct] Loop ended after {loop_count + 1} round(s)")
            return decision

        except Exception as e:
            print(f"\nError in ReAct loop: {e}")
            raise

    # ─── 动作执行（工具箱字典直接分发） ───────────────────────

    def _execute_action(self, action: RobotAction) -> bool:
        """
        通过 Tools Registry 直接分发动作 — 无 if-elif，无 getattr。

        流程：
        1. action.type → self.skills_registry 获取函数对象
        2. action.model_dump(exclude={"type"}) → ** 解包传参
        3. 检查返回字典中的 "status" 是否为 "SUCCESS"
        """
        action_type = action.type
        tool_func = self.skills_registry.get(action_type)

        if tool_func is None:
            print(f"[Action] Unknown action type: {action_type}")
            self._last_action_result = {
                "status": "FAILED",
                "error": f"Unknown action type: {action_type}",
            }
            return False

        try:
            params = action.model_dump(exclude={"type"})
            result = tool_func(**params)

            success = result is not None and result.get("status") == "SUCCESS"
            self._last_action_result = result

            tag = "succeeded" if success else "failed"
            print(f"[Action] {action_type} {tag}: {result}")
            return success

        except Exception as e:
            print(f"[Action] {action_type} execution error: {e}")
            self._last_action_result = {"status": "FAILED", "error": str(e)}
            return False

    # ─── 辅助方法 ──────────────────────────────────────────────

    def _append_decision_to_history(self, decision: RobotDecision):
        """将 RobotDecision 序列化为文本并追加为 assistant 消息"""
        parts = []
        if decision.thought:
            parts.append(f"Thought: {decision.thought}")
        if decision.action:
            parts.append(f"Action: {decision.action.model_dump_json()}")
        if decision.reply:
            parts.append(f"Reply: {decision.reply}")
        self.conversation_history.append({
            "role": "assistant",
            "content": "\n".join(parts) if parts else "(no response)",
        })

    def _action_signature(self, action: RobotAction) -> str:
        params = action.model_dump(exclude={"type"}, exclude_none=True)
        return json.dumps(
            {"type": action.type, "params": params},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _update_no_progress_counts(
        self,
        action_signature: str,
        action_type: str,
        action_success: bool,
        result: Dict,
    ) -> None:
        if self._is_recovery_progress(action_type, action_success, result):
            self._no_progress_action_counts.clear()
            return

        if self._is_no_progress(action_type, action_success, result):
            self._no_progress_action_counts[action_signature] = (
                self._no_progress_action_counts.get(action_signature, 0) + 1
            )
        else:
            self._no_progress_action_counts.pop(action_signature, None)

    def _is_recovery_progress(
        self,
        action_type: str,
        action_success: bool,
        result: Dict,
    ) -> bool:
        if not action_success:
            return False
        if action_type in ("navigation", "reposition", "follow_person"):
            return True
        if action_type == "observe_people":
            payload = result.get("result") if isinstance(result, dict) else None
            return isinstance(payload, dict) and payload.get("count", 0) > 0
        if action_type == "find_people":
            payload = result.get("result") if isinstance(result, dict) else None
            return isinstance(payload, dict) and payload.get("count", 0) > 0
        return False

    def _is_no_progress(self, action_type: str, action_success: bool, result: Dict) -> bool:
        if not action_success:
            return True
        if action_type == "find_people":
            payload = result.get("result") if isinstance(result, dict) else None
            if isinstance(payload, dict) and payload.get("count") == 0:
                return True
        return False

    def reset(self):
        """重置控制器状态"""
        self.conversation_history.clear()
        self.current_position = "home"
        self.holding_object = None
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        self._last_action_result = {}
        self._no_progress_action_counts.clear()
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
