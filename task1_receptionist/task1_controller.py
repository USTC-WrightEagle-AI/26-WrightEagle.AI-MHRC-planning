"""
Task1 总控 — Receptionist 接待任务

直接运行:
    python task1_receptionist/task1_controller.py

使用预置脚本 (跳过真实 ASR, 用于反复测试):
    python task1_receptionist/task1_controller.py --script "Alice,橙汁,Bob,可乐"

对接真实语音节点 (需要 roscore + tts_node + asr_node):
    python task1_receptionist/task1_controller.py --ros

流程 (14 个执行状态):
    [1]  GO_TO_DOOR        → 移动到门口
    [2]  ASK_GUEST1_INFO   → 询问 guest1 姓名和饮料
    [3]  GUIDE_GUEST1      → 带 guest1 去客厅
    [4]  POINT_EMPTY_SEAT  → 指向空座请 guest1 入座
    [5]  RETURN_TO_START   → 返回起点
    [6]  PICK_UP_GUEST2    → 到门口接 guest2
    [7]  DESCRIBE_GUEST1   → 向 guest2 描述 guest1 外貌
    [8]  SEAT_GUEST2       → 带 guest2 入座
    [9]  INTRODUCE_GUESTS  → 相互介绍两位客人
    [10] REQUEST_GUEST2_BAG→ 请求 guest2 放包到托盘
    [11] FIND_HOST         → 找到 host
    [12] FOLLOW_HOST       → 跟随 host
    [13] PLACE_BAG         → 把包放到指定位置
    [14] TASK_COMPLETE     → 播报结束语
"""

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
    NavigationModule,
    SpeechModule,
    VisionModule,
    ManipulationModule,
)
from task1_receptionist.sub_modules.speech_interaction import (
    SpeechInterface,
    create_speech_interface,
)


# ============================================================
# 状态 → 模块 调度表
# ============================================================

# 每个状态由哪些模块按顺序处理
STATE_MODULES: Dict[Task1StateID, List[str]] = {
    Task1StateID.GO_TO_DOOR:        ["navigation"],
    Task1StateID.ASK_GUEST1_INFO:   ["speech"],
    Task1StateID.GUIDE_GUEST1:      ["navigation"],
    Task1StateID.POINT_EMPTY_SEAT:  ["manipulation"],
    Task1StateID.RETURN_TO_START:   ["navigation"],
    Task1StateID.PICK_UP_GUEST2:    ["navigation", "speech"],
    Task1StateID.DESCRIBE_GUEST1:   ["vision", "speech"],
    Task1StateID.SEAT_GUEST2:       ["navigation", "speech"],
    Task1StateID.INTRODUCE_GUESTS:  ["speech"],
    Task1StateID.REQUEST_GUEST2_BAG:["speech", "manipulation"],
    Task1StateID.FIND_HOST:         ["vision"],
    Task1StateID.FOLLOW_HOST:       ["vision", "navigation"],
    Task1StateID.PLACE_BAG:         ["manipulation"],
    Task1StateID.TASK_COMPLETE:     ["speech"],
}


class Task1Runner:
    """Task1 执行器 — 状态机编排, 委托子模块执行"""

    def __init__(self,
                 speech: Optional[SpeechInterface] = None,
                 step_delay: float = 1.0):
        """
        Args:
            speech: 语音交互接口 (Mock / ROS / Scripted)
            step_delay: 状态间延迟秒数
        """
        self.step_delay = step_delay
        self.context: Dict[str, Any] = {"robot_position": "home"}
        self._execution_states = get_execution_states()
        self._start_time: float = 0.0

        # 初始化子模块
        self._modules: Dict[str, BaseSubModule] = {
            "navigation":   NavigationModule(),
            "speech":       SpeechModule(speech),
            "vision":       VisionModule(),
            "manipulation": ManipulationModule(),
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

        print()
        print(f"{'─' * 50}")
        print(f"▶ [{idx:2d}/14] {name}")
        print(f"   {desc}")
        if state_def.data_needed:
            print(f"   需要: {state_def.data_needed}")
        if state_def.data_produced:
            print(f"   产出: {state_def.data_produced}")
        print(f"{'─' * 50}")

        # 延迟 (模拟执行时间)
        time.sleep(self.step_delay)

        # 委托子模块
        module_names = STATE_MODULES.get(sid, [])
        if not module_names:
            print(f"  ⚠ 未找到模块映射: {name}")
            return

        try:
            for mod_name in module_names:
                module = self._modules.get(mod_name)
                if module is None:
                    print(f"  ⚠ 模块未注册: {mod_name}")
                    continue
                data = module.execute(sid, self.context)
                if data:
                    self.context.update(data)
        except Exception as e:
            print(f"  ⚠ 执行异常 [{mod_name}]: {e}")
            import traceback
            traceback.print_exc()

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
  python task1_controller.py
  python task1_controller.py --script "Alice,橙汁,Bob,可乐"
  python task1_controller.py --script "张三,茶,李四,咖啡" --delay 0.5
  python task1_controller.py --ros
        """,
    )
    parser.add_argument("--delay", type=float, default=1.0,
                        help="状态间延迟秒数 (默认 1.0)")
    parser.add_argument("--ros", action="store_true",
                        help="对接真实 ROS TTS + ASR 节点")
    parser.add_argument("--script", type=str, default=None,
                        help="预置 ASR 回复脚本, 逗号分隔 (如: Alice,橙汁,Bob,可乐)")

    args = parser.parse_args()

    # 创建语音接口
    script = None
    if args.script:
        script = [s.strip() for s in args.script.split(",") if s.strip()]

    speech = create_speech_interface(use_ros=args.ros, script=script)

    runner = Task1Runner(speech=speech, step_delay=args.delay)
    runner.run()


if __name__ == "__main__":
    main()
