"""
Task1 总控 — Receptionist 接待任务

默认启动真实已实现模块:
    python task1_receptionist/task1_controller.py

使用预置脚本 (跳过真实 ASR, 用于反复测试):
    python task1_receptionist/task1_controller.py --script "Alice,橙汁,Bob,可乐" --object-search-dry-run --object-search-skip-vision

测试模式: 每个状态可直接回车跳过，也可输入 y 正常执行该状态；状态失败会记录并继续:
    python task1_receptionist/task1_controller.py --test --object-search-dry-run --object-search-skip-vision

    流程 (15 个主流程执行状态；状态 9 保留兼容但默认跳过):
    [1]  WAIT_FOR_DOORBELL_1 → 等待门铃 (guest1)
    [2]  GO_TO_DOOR          → 移动到门口
    [3]  ASK_GUEST1_INFO     → 询问 guest1 姓名和饮料，并缓存外貌
    [4]  GUIDE_GUEST1        → 带 guest1 去客厅
    [5]  POINT_EMPTY_SEAT    → 识别最近空座，导航到面向空座 1.5 米处并指向
    [6]  RETURN_TO_START     → 返回起点
    [7]  WAIT_FOR_DOORBELL_2 → 等待门铃 (guest2)
    [8]  PICK_UP_GUEST2      → 到门口接 guest2，询问姓名和饮料，并缓存外貌
    [9]  DESCRIBE_GUEST1     → 保留兼容；主流程跳过
    [10] SEAT_GUEST2         → 启动导航后向 guest2 描述 guest1 外貌，接近空座并指向
    [11] INTRODUCE_GUESTS    → 按衣着识别并面向对应客人，相互介绍两位客人
    [12] REQUEST_GUEST2_BAG  → 请求 guest2 递包，导航到 0.7 米接近点后接包
    [13] FIND_HOST           → 找到 host
    [14] FOLLOW_HOST         → 请求 host 指引并跟随
    [15] PLACE_BAG           → 到达后将包交给 host
    [16] TASK_COMPLETE       → 播报结束语
"""

import sys
import os

# Add project root to sys.path so imports work when running this script directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import re
import time
from typing import Any, Dict, List, Optional

from task1_receptionist.state_definitions import (
    Task1StateID,
    StateDefinition,
    get_execution_states,
)
from task1_receptionist.sub_modules.base_module import (
    BaseSubModule,
    DoorbellModule,
    NavigationModule,
    SpeechModule,
    VisionModule,
    ManipulationModule,
)
from task1_receptionist.sub_modules.speech_interaction import (
    SpeechInterface,
    MockSpeechInterface,
    ScriptedSpeechInterface,
    ROSSpeechInterface,
)
from task1_receptionist.sub_modules.llm_interface import (
    LLMInterface,
    LocalLLMInterface,
)


# ============================================================
# 状态 → 模块 调度表
# ============================================================

# 每个状态由哪些模块按顺序处理
TASK1_MAX_DURATION_SEC = 8 * 60


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

STATE_MODULES: Dict[Task1StateID, List[str]] = {
    Task1StateID.WAIT_FOR_DOORBELL_1: ["doorbell"],
    Task1StateID.GO_TO_DOOR:        ["navigation"],
    Task1StateID.ASK_GUEST1_INFO:   ["speech", "vision"],
    Task1StateID.GUIDE_GUEST1:      ["speech", "navigation"],
    Task1StateID.POINT_EMPTY_SEAT:  ["navigation", "manipulation"],
    Task1StateID.RETURN_TO_START:   ["navigation"],
    Task1StateID.WAIT_FOR_DOORBELL_2: ["doorbell"],
    Task1StateID.PICK_UP_GUEST2:    ["navigation", "speech", "vision"],
    Task1StateID.DESCRIBE_GUEST1:   ["speech"],
    Task1StateID.SEAT_GUEST2:       ["navigation", "manipulation"],
    Task1StateID.INTRODUCE_GUESTS:  ["speech"],
    Task1StateID.REQUEST_GUEST2_BAG:["speech", "navigation", "manipulation"],
    Task1StateID.FIND_HOST:         ["navigation", "speech", "vision"],
    Task1StateID.FOLLOW_HOST:       ["speech", "navigation"],
    Task1StateID.PLACE_BAG:         ["speech", "manipulation"],
    Task1StateID.TASK_COMPLETE:     ["speech"],
}

SKIPPED_STATES = set()

NON_FATAL_FAILURE_STATES = {
    Task1StateID.POINT_EMPTY_SEAT,
    Task1StateID.SEAT_GUEST2,
    Task1StateID.FIND_HOST,
    Task1StateID.FOLLOW_HOST,
}

TEST_SKIP_DEFAULTS: Dict[str, Any] = {
    "doorbell_1_rang": True,
    "doorbell_2_rang": True,
    "robot_at_door": True,
    "robot_at_start": True,
    "guest1_name": "Guest1",
    "guest1_drink": "water",
    "guest1_appearance": {
        "clothing": "unknown clothing",
        "visual_attributes": [],
        "source": "test_default",
    },
    "guest1_in_living_room": True,
    "empty_seat_approach_reached": True,
    "seat_pointed": True,
    "guest1_seat_map_xyz": [0.0, 0.0, 0.0],
    "guest2_at_door": True,
    "guest2_name": "Guest2",
    "guest2_drink": "water",
    "guest2_appearance": {
        "clothing": "unknown clothing",
        "visual_attributes": [],
        "clothing_signature": {"attributes": [], "keys": []},
        "source": "test_default",
    },
    "guest2_seated": True,
    "seat_number": 2,
    "guest2_empty_seat_approach_reached": True,
    "guest2_seat_pointed": True,
    "guest2_seat_map_xyz": [0.0, 0.0, 0.0],
    "guests_introduced": True,
    "handover_approach_reached": True,
    "bag_on_tray": True,
    "host_interaction_reached": True,
    "host_found": True,
    "host_location": "living_room",
    "host_guidance_requested": True,
    "nav_at_destination": True,
    "host_handoff_announced": True,
    "bag_placed": True,
}

FAILURE_CONTINUE_DEFAULTS: Dict[str, Any] = {
    **TEST_SKIP_DEFAULTS,
    "doorbell_1_rang": False,
    "doorbell_2_rang": False,
    "robot_at_door": False,
    "robot_at_start": False,
    "guest1_in_living_room": False,
    "empty_seat_approach_reached": False,
    "seat_pointed": False,
    "guest1_seat_map_xyz": None,
    "guest2_at_door": False,
    "guest2_seated": False,
    "guest2_empty_seat_approach_reached": False,
    "guest2_seat_pointed": False,
    "guest2_seat_map_xyz": None,
    "handover_approach_reached": False,
    "bag_on_tray": False,
    "host_interaction_reached": False,
    "host_found": False,
    "nav_at_destination": False,
    "bag_placed": False,
}


class Task1Runner:
    """Task1 执行器 — 状态机编排, 委托子模块执行"""

    def __init__(self,
                 speech: Optional[SpeechInterface] = None,
                 llm: Optional[LLMInterface] = None,
                 step_delay: float = 1.0,
                 object_search_dry_run: bool = False,
                 object_search_max_frames: int = 120,
                 object_search_skip_vision: bool = False,
                 test_mode: bool = False,
                 continue_on_failure: bool = False,
                 enforce_deadline: bool = False,
                 door_goal: Optional[List[float]] = None,
                 start_goal: Optional[List[float]] = None,
                 living_room_goal: Optional[List[float]] = None,
                 host_interaction_goal: Optional[List[float]] = None):
        """
        Args:
            speech: 语音交互接口 (Mock / ROS / Scripted)
            llm: LLM 信息提取接口 (Mock / ROS)
            step_delay: 状态间延迟秒数
            object_search_dry_run: object_search 操作只计算目标，不发布机械臂/夹爪指令
            object_search_max_frames: 每次识别最多处理帧数
            object_search_skip_vision: 跳过相机刷新，复用 latest_*.json
            test_mode: 每个状态开始时允许回车跳过
            continue_on_failure: 状态失败后记录失败并继续执行后续状态
            enforce_deadline: 超过规则时间后是否主动停止状态机
            door_goal: GO_TO_DOOR/PICK_UP_GUEST2 使用的门口 map 坐标 [x, y, yaw_deg]
            start_goal: RETURN_TO_START 使用的起点 map 坐标 [x, y, yaw_deg]
            living_room_goal: GUIDE_GUEST1/SEAT_GUEST2 使用的客厅 map 坐标 [x, y, yaw_deg]
            host_interaction_goal: FIND_HOST 使用的 host 交互点 map 坐标 [x, y, yaw_deg]
        """
        self.step_delay = step_delay
        self.test_mode = test_mode
        self.continue_on_failure = continue_on_failure or test_mode
        self.enforce_deadline = enforce_deadline
        self.context: Dict[str, Any] = {"robot_position": "home"}
        self._execution_states = get_execution_states()
        self._start_time: float = 0.0
        self.current_state = Task1StateID.IDLE

        # 初始化子模块
        speech_module = SpeechModule(speech, llm)
        self._modules: Dict[str, BaseSubModule] = {
            "doorbell":     DoorbellModule(),
            "navigation":   NavigationModule(
                speech=speech_module.speech,
                door_goal=door_goal,
                start_goal=start_goal,
                living_room_goal=living_room_goal,
                host_interaction_goal=host_interaction_goal,
                object_search_max_frames=object_search_max_frames,
                object_search_skip_vision=object_search_skip_vision,
            ),
            "speech":       speech_module,
            "vision":       VisionModule(),
            "manipulation": ManipulationModule(
                speech=speech_module.speech,
                object_search_dry_run=object_search_dry_run,
                object_search_max_frames=object_search_max_frames,
                object_search_skip_vision=object_search_skip_vision,
            ),
        }

    # ============================================================
    # 主入口
    # ============================================================

    def run(self):
        self._start_time = time.time()

        print()
        print("█" * 55)
        print("█  Task1 — Receptionist 接待任务 启动")
        if self.continue_on_failure:
            print("█  失败继续模式: 启用")
        if not self.enforce_deadline:
            print("█  超时自停: 禁用")
        print("█" * 55)

        completed = True
        try:
            for state_def in self._execution_states:
                if not self._execute_state(state_def):
                    completed = False
                    break
            failures = self.context.get("state_failures") or []
            if completed:
                self.current_state = Task1StateID.IDLE
                if failures:
                    self.context["task_status"] = "completed_with_failures"
                else:
                    self.context["task_status"] = "completed"
        finally:
            elapsed = time.time() - self._start_time
            failures = self.context.get("state_failures") or []
            print()
            print("█" * 55)
            if completed and failures:
                status = "完成(有状态失败)"
            else:
                status = "完成" if completed else "已停止"
            print(f"█  Task1 {status}!  总耗时: {elapsed:.1f}s")
            if failures:
                print(f"█  记录失败状态数: {len(failures)}")
            print(f"█  上下文: {json.dumps(self.context, ensure_ascii=False)}")
            print("█" * 55)
            self._cleanup()

    # ============================================================
    # 单个状态执行
    # ============================================================

    def _execute_state(self, state_def: StateDefinition) -> bool:
        idx = state_def.index
        name = state_def.state_id.value
        desc = state_def.description
        sid = state_def.state_id
        self.current_state = sid

        if self._deadline_exceeded():
            return self._fail(f"Task1 已超过规则规定的 {TASK1_MAX_DURATION_SEC}s 总时限")

        total = len(self._execution_states)
        print()
        print(f"{'═' * 60}")
        print(f"▶ 状态 [{idx:2d}/{total}] {name}")
        print(f"  描述: {desc}")
        if state_def.data_needed:
            print(f"  需要上下文: {state_def.data_needed}")
        if state_def.data_produced:
            print(f"  产出上下文: {state_def.data_produced}")
        if state_def.timeout_sec:
            if _env_flag("TASK1_INTERACTIVE", False):
                print("  等待策略: 按模块提示按 Enter 跳过")
            else:
                print("  等待策略: 自动超时/降级，不等待控制台输入")

        module_names = STATE_MODULES.get(sid, [])
        if sid in SKIPPED_STATES:
            print("  调用模块: 跳过")
        else:
            print(f"  调用模块: {' → '.join(module_names)}")
        print(f"{'─' * 60}")

        time.sleep(self.step_delay)

        if self.test_mode:
            try:
                choice = input("  [TEST] 直接回车跳过该状态；输入 y 执行该状态: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = ""
            if choice != "y":
                self._skip_state(state_def)
                return True

        if sid in SKIPPED_STATES:
            print(f"  ⏭ 已跳过: {desc}")
            self.current_state = state_def.next_state or Task1StateID.IDLE
            print(f"  当前上下文: {json.dumps(self.context, ensure_ascii=False)}")
            return True

        if not module_names:
            print(f"  ⚠ 未找到模块映射: {name}")
            return self._continue_or_fail(state_def, f"未找到模块映射: {name}")

        missing_inputs = [key for key in state_def.data_needed if not self.context.get(key)]
        if missing_inputs:
            if self.continue_on_failure:
                print(f"  ⚠ 缺少前置上下文，补默认值后继续执行: {missing_inputs}")
                filled = self._fill_context_defaults(missing_inputs)
                if filled:
                    print(f"  [CONTINUE] 补充前置上下文: {json.dumps(filled, ensure_ascii=False)}")
            else:
                print(f"  ⚠ 缺少前置上下文，停止执行: {missing_inputs}")
                return self._fail(f"状态 {name} 缺少前置上下文: {missing_inputs}")

        max_attempts = state_def.retry_on_failure + 1
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"  ↻ 重试状态 {name}: {attempt}/{max_attempts}")
            if self._execute_modules(sid, module_names):
                missing_outputs = [
                    key for key in state_def.data_produced if not self.context.get(key)
                ]
                if not missing_outputs:
                    self.current_state = state_def.next_state or Task1StateID.IDLE
                    print(f"  当前上下文: {json.dumps(self.context, ensure_ascii=False)}")
                    return True
                print(f"  ⚠ 状态关键产出未成功: {missing_outputs}")
                if self.continue_on_failure:
                    return self._continue_after_state_failure(
                        state_def,
                        f"关键产出未成功: {missing_outputs}",
                        missing_outputs=missing_outputs,
                    )
                if self._should_continue_after_missing_outputs(sid, missing_outputs):
                    self.current_state = state_def.next_state or Task1StateID.IDLE
                    self.context[f"{sid.value}_warning"] = (
                        f"关键产出未成功但继续流程: {missing_outputs}"
                    )
                    print(f"  ⚠ {name} 为非致命状态，记录失败并继续执行下一步")
                    print(f"  当前上下文: {json.dumps(self.context, ensure_ascii=False)}")
                    return True

            if self._deadline_exceeded():
                return self._fail(f"Task1 已超过规则规定的 {TASK1_MAX_DURATION_SEC}s 总时限")

        if not self.continue_on_failure:
            print(f"  ⚠ 状态执行失败，停止后续流程: {name}")
            print(f"  当前上下文: {json.dumps(self.context, ensure_ascii=False)}")
        return self._continue_or_fail(state_def, f"状态执行失败: {name}")

    def _should_continue_after_missing_outputs(
        self,
        sid: Task1StateID,
        missing_outputs: List[str],
    ) -> bool:
        return sid in NON_FATAL_FAILURE_STATES and bool(missing_outputs)

    def _skip_state(self, state_def: StateDefinition):
        self._fill_context_defaults(state_def.data_produced, overwrite=True)
        self.current_state = state_def.next_state or Task1StateID.IDLE
        print(f"  [TEST] 已跳过状态: {state_def.state_id.value}")
        if state_def.data_produced:
            skipped = {key: self.context.get(key) for key in state_def.data_produced}
            print(f"  [TEST] 补充上下文: {json.dumps(skipped, ensure_ascii=False)}")
        print(f"  当前上下文: {json.dumps(self.context, ensure_ascii=False)}")

    def _execute_modules(self, sid: Task1StateID, module_names: List[str]) -> bool:
        try:
            for mod_name in module_names:
                module = self._modules.get(mod_name)
                if module is None:
                    print(f"  ⚠ 模块未注册: {mod_name}")
                    return False
                if mod_name != "speech":
                    self._stop_conversation_gaze()
                print(f"  ┌── 模块 [{mod_name}] 开始执行 ──")
                data = module.execute(sid, self.context)
                if data:
                    print(f"  │  产出数据: {json.dumps(data, ensure_ascii=False)}")
                    self.context.update(data)
                else:
                    print("  │  无产出数据")
                print(f"  └── 模块 [{mod_name}] 执行完成 ──")
            return True
        except Exception as exc:
            print(f"  ⚠ 执行异常 [{mod_name}]: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def _stop_conversation_gaze(self) -> None:
        speech_module = self._modules.get("speech")
        stop = getattr(speech_module, "stop_guest_gaze_tracking", None)
        if callable(stop):
            stop()

    def _fill_context_defaults(
        self,
        keys: List[str],
        overwrite: bool = False,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        filled: Dict[str, Any] = {}
        defaults = defaults or TEST_SKIP_DEFAULTS
        for key in keys:
            if not overwrite and self.context.get(key):
                continue
            value = defaults.get(key, True)
            self.context[key] = value
            filled[key] = value
        return filled

    def _continue_or_fail(self, state_def: StateDefinition, reason: str) -> bool:
        if self.continue_on_failure:
            return self._continue_after_state_failure(state_def, reason)
        return self._fail(reason)

    def _continue_after_state_failure(
        self,
        state_def: StateDefinition,
        reason: str,
        missing_outputs: Optional[List[str]] = None,
    ) -> bool:
        sid = state_def.state_id
        output_keys = missing_outputs if missing_outputs is not None else state_def.data_produced
        filled = self._fill_context_defaults(
            output_keys,
            defaults=FAILURE_CONTINUE_DEFAULTS,
        )

        failure = {
            "state": sid.value,
            "index": state_def.index,
            "reason": reason,
        }
        if missing_outputs is not None:
            failure["missing_outputs"] = missing_outputs
        if filled:
            failure["filled_defaults"] = filled

        self.context.setdefault("state_failures", []).append(failure)
        self.context[f"{sid.value}_failed"] = True
        self.context[f"{sid.value}_failure_reason"] = reason
        self.context["task_status"] = "running_with_failures"
        self.current_state = state_def.next_state or Task1StateID.IDLE

        print(f"  ⚠ {reason}")
        if filled:
            print(f"  [CONTINUE] 补充默认上下文: {json.dumps(filled, ensure_ascii=False)}")
        print(f"  [CONTINUE] 状态失败已记录，继续执行下一状态: {self.current_state.value}")
        print(f"  当前上下文: {json.dumps(self.context, ensure_ascii=False)}")
        return True

    def _deadline_exceeded(self) -> bool:
        return self.enforce_deadline and time.time() - self._start_time >= TASK1_MAX_DURATION_SEC

    def _fail(self, reason: str) -> bool:
        self.current_state = Task1StateID.ERROR
        self.context["task_status"] = "error"
        self.context["task_error"] = reason
        print(f"  ⚠ {reason}，停止执行")
        return False

    # ============================================================
    # 清理
    # ============================================================

    def _cleanup(self):
        for mod in self._modules.values():
            if hasattr(mod, 'close'):
                mod.close()


# ============================================================
# 入口
# ============================================================

class RuleBasedLLMInterface(LLMInterface):
    """LLM 不可用时的轻量兜底提取器，主要用于测试流程不断住。"""

    _NAME_PATTERNS = [
        re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\s-]{0,30})", re.I),
        re.compile(r"\bi am\s+([A-Za-z][A-Za-z\s-]{0,30})", re.I),
        re.compile(r"\bi'm\s+([A-Za-z][A-Za-z\s-]{0,30})", re.I),
    ]
    _DRINKS = [
        "orange juice",
        "apple juice",
        "coffee",
        "tea",
        "cola",
        "coke",
        "water",
        "milk",
    ]

    def extract_guest_info(self, text: str, role: str = "guest") -> Dict[str, Any]:
        return {
            "name": self.extract_name(text, role),
            "drink": self.extract_drink(text, role),
        }

    def extract_name(self, text: str, role: str = "guest") -> Optional[str]:
        for pattern in self._NAME_PATTERNS:
            match = pattern.search(text or "")
            if match:
                name = match.group(1).strip().split()[0]
                print(f"  🧠 [LLM-Rule] 提取 {role} 姓名: {name}")
                return name
        return None

    def extract_drink(self, text: str, role: str = "guest") -> Optional[str]:
        lowered = (text or "").lower()
        for drink in self._DRINKS:
            if drink in lowered:
                print(f"  🧠 [LLM-Rule] 提取 {role} 饮品: {drink}")
                return drink
        return None


def create_default_speech_interface(script: Optional[List[str]] = None,
                                    interactive: bool = False) -> SpeechInterface:
    if script:
        return ScriptedSpeechInterface(script)

    try:
        speech = ROSSpeechInterface()
        print("🎤 语音接口: ROS TTS/ASR")
        return speech
    except Exception as exc:
        print(f"🎤 语音接口: ROS 不可用，降级为控制台 Mock ({exc})")
        return MockSpeechInterface(interactive=interactive)


def create_default_llm_interface() -> LLMInterface:
    try:
        return LocalLLMInterface()
    except Exception as exc:
        print(f"🧠 LLM 接口: 本地 LLM 不可用，降级为规则提取 ({exc})")
        return RuleBasedLLMInterface()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Task1 Receptionist 接待任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 流程调试：不打开相机、不发布机械臂或夹爪指令，复用 latest_*.json
  python task1_controller.py --object-search-dry-run --object-search-skip-vision

  # 使用预置语音回复和自动门铃做流程测试
  python task1_controller.py --script "Alice,橙汁,Bob,可乐" --auto-doorbell --delay 0.3 --object-search-dry-run --object-search-skip-vision

  # 临时覆盖 GO_TO_DOOR/PICK_UP_GUEST2 的门口 map 坐标
  python task1_controller.py --door-goal 1.0 2.0 90

  # 临时覆盖 RETURN_TO_START 的起点 map 坐标
  python task1_controller.py --start-goal -1.432 -0.078 169.610

  # 临时覆盖 GUIDE_GUEST1/SEAT_GUEST2 的客厅 map 坐标
  python task1_controller.py --living-room-goal -0.274 -1.151 -2.771

  # 临时覆盖 FIND_HOST 的 host 交互点 map 坐标
  python task1_controller.py --host-interaction-goal -1.432 -0.078 169.610

  # 比赛入口：默认无人值守、失败继续、不中途等待控制台输入
  python task1_controller.py

  # object_search 只计算目标，不发布机械臂或夹爪指令
  python task1_controller.py --object-search-dry-run

  # 每个状态开始时允许回车跳过；输入 y 执行时若失败也会继续后续状态
  python task1_controller.py --test --object-search-dry-run --object-search-skip-vision
        """,
    )
    parser.add_argument("--delay", type=float, default=1.0,
                        help="状态间延迟秒数 (默认 1.0, 全自动测试建议 0.3)")
    parser.add_argument("--test", action="store_true",
                        help="测试模式: 每个状态开始时直接回车跳过，输入 y 执行该状态；状态失败也继续")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="状态失败后记录失败并继续执行后续状态；默认入口已启用，保留该参数用于显式声明")
    parser.add_argument("--stop-on-failure", "--strict", dest="stop_on_failure",
                        action="store_true",
                        help="严格模式: 状态失败后停止后续流程")
    parser.add_argument("--interactive", action="store_true",
                        help="允许控制台 Enter/y 人工跳过或确认；默认比赛入口禁用人工输入")
    parser.add_argument("--enforce-deadline", action="store_true",
                        help="超过 TASK1_MAX_DURATION_SEC 后主动停止；默认不中途自停")
    parser.add_argument("--script", type=str, default=None,
                        help="预置 ASR 回复脚本, 逗号分隔 (如: Alice,橙汁,Bob,可乐)")
    parser.add_argument("--auto-doorbell", action="store_true",
                        help="自动模拟门铃 (无需按 Enter, 配合 --script 全自动测试)")
    parser.add_argument("--door-goal", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                        help="GO_TO_DOOR/PICK_UP_GUEST2 的门口 map 坐标，例如 --door-goal 1.0 2.0 90")
    parser.add_argument("--start-goal", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                        help="RETURN_TO_START 的起点 map 坐标，例如 --start-goal -1.432 -0.078 169.610")
    parser.add_argument("--living-room-goal", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                        help="GUIDE_GUEST1/SEAT_GUEST2 的客厅 map 坐标，例如 --living-room-goal -0.274 -1.151 -2.771")
    parser.add_argument("--host-interaction-goal", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                        help="FIND_HOST 的 host 交互点 map 坐标，例如 --host-interaction-goal -1.432 -0.078 169.610")
    parser.add_argument("--object-search-dry-run", action="store_true",
                        help="object_search 操作只计算/打印目标，不发布机械臂或夹爪指令")
    parser.add_argument("--object-search-max-frames", type=int, default=120,
                        help="接包手腕等通用 object_search 识别最多处理帧数 (默认 120)")
    parser.add_argument("--empty-seat-max-frames", type=int, default=None,
                        help="空座识别最多处理帧数 (默认 30，识别成功会提前退出)")
    parser.add_argument("--object-search-skip-vision", action="store_true",
                        help="跳过 RealSense 识别刷新，直接复用 latest_empty_seat/latest_handover_hand JSON")

    args = parser.parse_args()
    interactive = args.interactive or args.test
    if interactive:
        os.environ["TASK1_INTERACTIVE"] = "1"
    else:
        os.environ["TASK1_INTERACTIVE"] = "0"
    if interactive:
        os.environ.setdefault("TASK1_ASSUME_YES", "0")
    else:
        os.environ["TASK1_ASSUME_YES"] = "1"
    if args.empty_seat_max_frames is not None:
        os.environ["TASK1_EMPTY_SEAT_MAX_FRAMES"] = str(args.empty_seat_max_frames)

    # 创建语音接口
    script = None
    if args.script:
        script = [s.strip() for s in args.script.split(",") if s.strip()]

    speech = create_default_speech_interface(script=script, interactive=interactive)

    llm = create_default_llm_interface()
    continue_on_failure = (
        args.continue_on_failure
        or args.test
        or not args.stop_on_failure
    )

    runner = Task1Runner(
        speech=speech,
        llm=llm,
        step_delay=args.delay,
        object_search_dry_run=args.object_search_dry_run,
        object_search_max_frames=args.object_search_max_frames,
        object_search_skip_vision=args.object_search_skip_vision,
        test_mode=args.test,
        continue_on_failure=continue_on_failure,
        enforce_deadline=args.enforce_deadline,
        door_goal=args.door_goal,
        start_goal=args.start_goal,
        living_room_goal=args.living_room_goal,
        host_interaction_goal=args.host_interaction_goal,
    )

    if args.auto_doorbell:
        from task1_receptionist.sub_modules.base_module import DoorbellModule
        DoorbellModule.auto_simulate = True
        print("🔔 门铃自动模拟已启用 (配合 --script 全自动测试)")

    runner.run()


if __name__ == "__main__":
    main()
