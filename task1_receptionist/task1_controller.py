"""
Task1 总控 — Receptionist 接待任务

直接运行即可完成完整 Task1 流程：
    python task1_receptionist/task1_controller.py

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
from typing import Any, Dict, Optional

from task1_receptionist.state_definitions import (
    StateDefinition,
    get_execution_states,
)


class Task1Runner:
    """Task1 纯 Python 执行器 — 按顺序执行所有状态"""

    def __init__(self, step_delay: float = 1.0):
        self.step_delay = step_delay
        self.context: Dict[str, Any] = {"robot_position": "home"}
        self._execution_states = get_execution_states()
        self._start_time: float = 0.0

    # ============================================================
    # 主入口
    # ============================================================

    def run(self):
        self._start_time = time.time()
        total = len(self._execution_states)

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

    # ============================================================
    # 单个状态执行
    # ============================================================

    def _execute_state(self, state_def: StateDefinition):
        idx = state_def.index
        name = state_def.state_id.value
        desc = state_def.description

        print()
        print(f"{'─' * 50}")
        print(f"▶ [{idx:2d}/14] {name}")
        print(f"   {desc}")
        if state_def.data_needed:
            print(f"   需要: {state_def.data_needed}")
        if state_def.data_produced:
            print(f"   产出: {state_def.data_produced}")
        print(f"{'─' * 50}")

        # 延迟模拟执行时间
        time.sleep(self.step_delay)

        # 调用对应处理器
        handler = getattr(self, f"_handle_{name}", None)
        if handler:
            try:
                data = handler(state_def)
                if data:
                    self.context.update(data)
            except Exception as e:
                print(f"  ⚠ 执行异常: {e}")
        else:
            print(f"  → (待实现: {name})")

    # ============================================================
    # 各状态处理器 (placeholder — 后续接入真实子模块)
    # ============================================================

    def _handle_go_to_door(self, _s: StateDefinition) -> Optional[Dict]:
        print("  🚪 导航: 移动到门口接客位置")
        return {"robot_at_door": True}

    def _handle_ask_guest1_info(self, s: StateDefinition) -> Optional[Dict]:
        print("  🎤 TTS: \"欢迎! 请问您的名字是什么?\"")
        print("  👂 ASR: 等待回复... → \"Alice\"")
        print("  🎤 TTS: \"请问您想喝什么饮料?\"")
        print("  👂 ASR: 等待回复... → \"橙汁\"")
        return {"guest1_name": "Alice", "guest1_drink": "橙汁"}

    def _handle_guide_guest1(self, s: StateDefinition) -> Optional[Dict]:
        guest = self.context.get("guest1_name", "guest1")
        print(f"  🚶 导航: 带领 {guest} 前往客厅")
        return {"guest1_in_living_room": True}

    def _handle_point_empty_seat(self, s: StateDefinition) -> Optional[Dict]:
        guest = self.context.get("guest1_name", "guest1")
        print(f"  👉 手臂: 指向空座位, 请 {guest} 入座")
        return {"guest1_seated": True, "seat_number": 1}

    def _handle_return_to_start(self, s: StateDefinition) -> Optional[Dict]:
        print("  🚶 导航: 返回起点位置, 准备接第二位客人")
        return {"robot_at_start": True}

    def _handle_pick_up_guest2(self, s: StateDefinition) -> Optional[Dict]:
        print("  🚪 导航: 移动到门口")
        print("  🎤 TTS: \"欢迎! 请问您的名字是什么?\"")
        print("  👂 ASR: 等待回复... → \"Bob\"")
        return {"guest2_at_door": True, "guest2_name": "Bob"}

    def _handle_describe_guest1(self, s: StateDefinition) -> Optional[Dict]:
        guest1 = self.context.get("guest1_name", "guest1")
        print(f"  📷 视觉: 识别 {guest1} 的外貌特征")
        print(f"  🎤 TTS: \"{guest1} 穿着红色外套, 坐在 1 号座位\"")
        return {"guest1_described": True}

    def _handle_seat_guest2(self, s: StateDefinition) -> Optional[Dict]:
        guest = self.context.get("guest2_name", "guest2")
        print(f"  🚶 导航: 带领 {guest} 前往客厅")
        print(f"  🎤 TTS: \"{guest}, 请问您想喝什么饮料?\"")
        print(f"  👂 ASR: 等待回复... → \"可乐\"")
        print(f"  👉 手臂: 指向空座位, 请 {guest} 入座")
        return {"guest2_seated": True, "guest2_drink": "可乐", "seat_number": 2}

    def _handle_introduce_guests(self, s: StateDefinition) -> Optional[Dict]:
        g1 = self.context.get("guest1_name", "guest1")
        g1d = self.context.get("guest1_drink", "?")
        g2 = self.context.get("guest2_name", "guest2")
        g2d = self.context.get("guest2_drink", "?")
        print(f"  🎤 TTS: \"{g1}, 这位是 {g2}, 他喜欢喝{g2d}。{g2}, 这位是 {g1}, 她喜欢喝{g1d}。\"")
        return {"guests_introduced": True}

    def _handle_request_guest2_bag(self, s: StateDefinition) -> Optional[Dict]:
        guest = self.context.get("guest2_name", "guest2")
        print(f"  🎤 TTS: \"{guest}, 请把您的包放在我的托盘上\"")
        print(f"  🦾 机械臂: 伸出托盘, 等待放包")
        print(f"  📷 视觉: 检测包是否放置完成")
        return {"bag_on_tray": True}

    def _handle_find_host(self, s: StateDefinition) -> Optional[Dict]:
        print("  📷 视觉: 在环境中搜索 host...")
        print("  🚶 导航: 靠近 host")
        return {"host_found": True, "host_location": "living_room"}

    def _handle_follow_host(self, s: StateDefinition) -> Optional[Dict]:
        print("  📷 视觉: 跟踪 host 位置")
        print("  🚶 导航: 跟随 host 走到指定位置")
        return {"at_destination": True, "destination": "storage_room"}

    def _handle_place_bag(self, s: StateDefinition) -> Optional[Dict]:
        print("  🦾 机械臂: 将托盘上的包放到指定位置")
        return {"bag_placed": True}

    def _handle_task_complete(self, s: StateDefinition) -> Optional[Dict]:
        print("  🎤 TTS: \"任务完成, 感谢各位的配合!\"")
        return {}


# ============================================================
# 入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Task1 Receptionist 接待任务")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="状态间延迟秒数 (默认 1.0)")
    args = parser.parse_args()

    runner = Task1Runner(step_delay=args.delay)
    runner.run()


if __name__ == "__main__":
    main()
