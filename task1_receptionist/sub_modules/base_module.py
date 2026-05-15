"""
子模块基类模板

后续各执行模块（导航、视觉、语音、抓取等）继承此基类实现。

使用方式 (以导航模块为例):

    class NavigationModule(BaseSubModule):
        def __init__(self):
            super().__init__("go_to_door")

        def execute(self, context: dict) -> dict:
            target = context.get("target", "entrance")
            success = self._robot.navigate_to(target)
            return {"robot_at_door": success}

状态处理器接口:
    每个子模块实现 execute(context) → dict, 返回产出数据,
    由 Task1Runner 在对应状态时调用并合并到上下文。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseSubModule(ABC):
    """子模块基类"""

    def __init__(self, state_name: str):
        self.state_name = state_name

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        执行本模块负责的状态

        Args:
            context: 当前任务共享上下文字典

        Returns:
            产出数据字典 (合并到总控上下文), 或 None
        """
        ...

    def cancel(self):
        """取消当前操作 (可选覆盖)"""
        pass


# ============================================================
# 示例骨架 (后续接入真实硬件/算法时填充)
# ============================================================

class NavigationModule(BaseSubModule):
    """导航子模块 — 负责 go_to_door, guide_guest1, return_to_start 等"""

    def __init__(self):
        super().__init__("go_to_door")

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state = context.get("_current_state", "")
        print(f"[NavigationModule] 导航到目标: {state}")
        return {"robot_at_door": True}


class ASRModule(BaseSubModule):
    """语音识别子模块 — 负责 ask_guest1_info 等语音交互"""

    def __init__(self):
        super().__init__("ask_guest1_info")

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print("[ASRModule] TTS播报 + ASR识别")
        return {"guest1_name": "Alice", "guest1_drink": "橙汁"}


class VisionModule(BaseSubModule):
    """视觉子模块 — 负责 find_host, describe_guest1 等"""

    def __init__(self):
        super().__init__("find_host")

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print("[VisionModule] 视觉感知")
        return {"host_found": True, "host_location": "living_room"}


class ManipulationModule(BaseSubModule):
    """操作子模块 — 负责 request_guest2_bag, place_bag 等"""

    def __init__(self):
        super().__init__("request_guest2_bag")

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print("[ManipulationModule] 机械臂操作")
        return {"bag_on_tray": True}
