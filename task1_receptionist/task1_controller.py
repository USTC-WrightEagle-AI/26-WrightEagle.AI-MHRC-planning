"""
Task1 总控 — Receptionist 接待任务

直接运行 (离线测试, 无需 roscore):
    python task1_receptionist/task1_controller.py

使用预置脚本 (跳过真实 ASR, 用于反复测试):
    python task1_receptionist/task1_controller.py --script "Alice,橙汁,Bob,可乐"

对接真实语音节点 (需要 roscore + tts_node + asr_node):
    python task1_receptionist/task1_controller.py --ros

流程 (16 个执行状态):
    [1]  WAIT_FOR_DOORBELL_1 → 等待门铃 (guest1)
    [2]  GO_TO_DOOR          → 移动到门口
    [3]  ASK_GUEST1_INFO     → 询问 guest1 姓名和饮料
    [4]  GUIDE_GUEST1        → 带 guest1 去客厅
    [5]  POINT_EMPTY_SEAT    → 指向空座请 guest1 入座
    [6]  RETURN_TO_START     → 返回起点
    [7]  WAIT_FOR_DOORBELL_2 → 等待门铃 (guest2)
    [8]  PICK_UP_GUEST2      → 到门口接 guest2
    [9]  DESCRIBE_GUEST1     → 向 guest2 描述 guest1 外貌
    [10] SEAT_GUEST2         → 带 guest2 入座
    [11] INTRODUCE_GUESTS    → 相互介绍两位客人
    [12] REQUEST_GUEST2_BAG  → 请求 guest2 放包到托盘
    [13] FIND_HOST           → 找到 host
    [14] FOLLOW_HOST         → 跟随 host
    [15] PLACE_BAG           → 把包放到指定位置
    [16] TASK_COMPLETE       → 播报结束语
"""

import sys
import os

# Add project root to sys.path so imports work when running this script directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
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
    SpeakerDOAModule,
)
from task1_receptionist.sub_modules.speech_interaction import (
    SpeechInterface,
    create_speech_interface,
)
from task1_receptionist.sub_modules.llm_interface import (
    LLMInterface,
    create_llm_interface,
)


# ============================================================
# 状态 → 模块 调度表
# ============================================================

# 每个状态由哪些模块按顺序处理
STATE_MODULES: Dict[Task1StateID, List[str]] = {
    Task1StateID.WAIT_FOR_DOORBELL_1: ["doorbell"],
    Task1StateID.GO_TO_DOOR:        ["navigation"],
    Task1StateID.ASK_GUEST1_INFO:   ["speaker_doa", "speech"],
    Task1StateID.GUIDE_GUEST1:      ["navigation"],
    Task1StateID.POINT_EMPTY_SEAT:  ["speech", "manipulation"],
    Task1StateID.RETURN_TO_START:   ["navigation"],
    Task1StateID.WAIT_FOR_DOORBELL_2: ["doorbell"],
    Task1StateID.PICK_UP_GUEST2:    ["navigation", "speaker_doa", "speech"],
    Task1StateID.DESCRIBE_GUEST1:   ["vision", "speaker_doa", "speech"],
    Task1StateID.SEAT_GUEST2:       ["navigation", "speaker_doa", "speech"],
    Task1StateID.INTRODUCE_GUESTS:  ["speaker_doa", "speech"],
    Task1StateID.REQUEST_GUEST2_BAG:["speaker_doa", "speech", "manipulation"],
    Task1StateID.FIND_HOST:         ["vision", "speaker_doa"],
    Task1StateID.FOLLOW_HOST:       ["speaker_doa", "navigation"],
    Task1StateID.PLACE_BAG:         ["manipulation"],
    Task1StateID.TASK_COMPLETE:     ["speech"],
}


class Task1Runner:
    """Task1 执行器 — 状态机编排, 委托子模块执行"""

    def __init__(self,
                 speech: Optional[SpeechInterface] = None,
                 llm: Optional[LLMInterface] = None,
                 step_delay: float = 1.0):
        """
        Args:
            speech: 语音交互接口 (Mock / ROS / Scripted)
            llm: LLM 信息提取接口 (Mock / ROS)
            step_delay: 状态间延迟秒数
        """
        self.step_delay = step_delay
        self.context: Dict[str, Any] = {"robot_position": "home"}
        self._execution_states = get_execution_states()
        self._start_time: float = 0.0

        # 初始化子模块
        self._modules: Dict[str, BaseSubModule] = {
            "doorbell":     DoorbellModule(),
            "navigation":   NavigationModule(),
            "speech":       SpeechModule(speech, llm),
            "vision":       VisionModule(),
            "manipulation": ManipulationModule(),
            "speaker_doa":  SpeakerDOAModule(),
        }

    # ============================================================
    # 主入口
    # ============================================================

    def run(self):
        self._start_time = time.time()

        print()
        print("█" * 55)
        print("█  Task1 — Receptionist 接待任务 启动")
        print("█" * 55)

        for state_def in self._execution_states:
            self._execute_state(state_def)

        elapsed = time.time() - self._start_time
        print()
        print("█" * 55)
        print(f"█  Task1 完成!  总耗时: {elapsed:.1f}s")
        print(f"█  上下文: {json.dumps(self.context, ensure_ascii=False)}")
        print("█" * 55)

        self._cleanup()

    # ============================================================
    # 单个状态执行
    # ============================================================

    def _execute_state(self, state_def: StateDefinition):
        idx = state_def.index
        name = state_def.state_id.value
        desc = state_def.description
        sid = state_def.state_id

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
            print(f"  超时: {state_def.timeout_sec}s")

        module_names = STATE_MODULES.get(sid, [])
        print(f"  调用模块: {' → '.join(module_names)}")
        print(f"{'─' * 60}")

        time.sleep(self.step_delay)

        if not module_names:
            print(f"  ⚠ 未找到模块映射: {name}")
            return

        try:
            for mod_name in module_names:
                module = self._modules.get(mod_name)
                if module is None:
                    print(f"  ⚠ 模块未注册: {mod_name}")
                    continue
                print(f"  ┌── 模块 [{mod_name}] 开始执行 ──")
                data = module.execute(sid, self.context)
                if data:
                    print(f"  │  产出数据: {json.dumps(data, ensure_ascii=False)}")
                    self.context.update(data)
                else:
                    print(f"  │  无产出数据")
                print(f"  └── 模块 [{mod_name}] 执行完成 ──")
        except Exception as e:
            print(f"  ⚠ 执行异常 [{mod_name}]: {e}")
            import traceback
            traceback.print_exc()

        print(f"  当前上下文: {json.dumps(self.context, ensure_ascii=False)}")

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

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Task1 Receptionist 接待任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 离线交互测试 (需要按 Enter 模拟门铃和输入 ASR)
  python task1_controller.py

  # 全自动离线测试 (无需任何交互)
  python task1_controller.py --script "Alice,橙汁,Bob,可乐" --auto-doorbell --delay 0.3

  # 对接真实 ROS 节点
  python task1_controller.py --ros --llm-ros
        """,
    )
    parser.add_argument("--delay", type=float, default=1.0,
                        help="状态间延迟秒数 (默认 1.0, 全自动测试建议 0.3)")
    parser.add_argument("--ros", action="store_true",
                        help="对接真实 ROS TTS + ASR 节点")
    parser.add_argument("--llm-ros", action="store_true",
                        help="对接真实 ROS LLM 节点 (需启动 llm_node.py)")
    parser.add_argument("--script", type=str, default=None,
                        help="预置 ASR 回复脚本, 逗号分隔 (如: Alice,橙汁,Bob,可乐)")
    parser.add_argument("--auto-doorbell", action="store_true",
                        help="自动模拟门铃 (无需按 Enter, 配合 --script 全自动测试)")

    args = parser.parse_args()

    # 创建语音接口
    script = None
    if args.script:
        script = [s.strip() for s in args.script.split(",") if s.strip()]

    speech = create_speech_interface(use_ros=args.ros, script=script)

    llm = create_llm_interface(use_ros=args.llm_ros)

    runner = Task1Runner(speech=speech, llm=llm, step_delay=args.delay)

    if args.auto_doorbell:
        from task1_receptionist.sub_modules.base_module import DoorbellModule
        DoorbellModule.auto_simulate = True
        print("🔔 门铃自动模拟已启用 (配合 --script 全自动测试)")

    runner.run()


if __name__ == "__main__":
    main()
