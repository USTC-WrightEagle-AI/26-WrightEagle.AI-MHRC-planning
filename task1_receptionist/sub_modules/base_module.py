"""
Task1 子模块 — 各执行模块基类和具体实现

模块划分 (按能力维度):
    NavigationModule   — 导航: go_to_door, guide_guest1, return_to_start, seat_guest2...
    SpeechModule       — 语音: ask_guest1_info, pick_up_guest2, introduce_guests...
    VisionModule       — 视觉: describe_guest1, find_host, follow_host
    ManipulationModule — 操作: request_guest2_bag, place_bag

每个模块通过 execute(state_id, context) 被总控调用,
返回产出数据 dict (合并到任务上下文), 或 None。

SpeechModule 依赖 SpeechInterface (TTS + ASR),
由总控在构造时注入, 支持 Mock/ROS 双后端。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from task1_receptionist.state_definitions import Task1StateID


# ============================================================
# 基类
# ============================================================

class BaseSubModule(ABC):
    """所有子模块的抽象基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        执行指定状态

        Args:
            state_id: 当前状态 ID
            context: 任务共享上下文 (包含之前所有产出数据)

        Returns:
            本状态的产出数据 (合并到总控上下文), 或 None
        """
        ...

    def cancel(self):
        """取消当前操作 (子类可选覆盖)"""
        pass


# ============================================================
# 导航模块
# ============================================================

class NavigationModule(BaseSubModule):
    """
    导航子模块

    负责所有需要机器人移动的状态:
      GO_TO_DOOR, GUIDE_GUEST1, RETURN_TO_START,
      PICK_UP_GUEST2 (部分), SEAT_GUEST2 (部分),
      FOLLOW_HOST (部分)

    实际部署对接 move_base / nav2 actionlib。
    """

    def __init__(self):
        super().__init__("navigation")

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if state_id == Task1StateID.GO_TO_DOOR:
            print("  🚪 [Nav] 移动到门口接客位置")
            return {"robot_at_door": True}

        elif state_id == Task1StateID.GUIDE_GUEST1:
            guest = context.get("guest1_name", "guest1")
            print(f"  🚶 [Nav] 带领 {guest} 前往客厅")
            return {"guest1_in_living_room": True}

        elif state_id == Task1StateID.RETURN_TO_START:
            print("  🚶 [Nav] 返回起点位置")
            return {"robot_at_start": True}

        elif state_id == Task1StateID.PICK_UP_GUEST2:
            print("  🚪 [Nav] 移动到门口接第二位客人")
            return None  # 导航部分, 语音由 SpeechModule 处理

        elif state_id == Task1StateID.SEAT_GUEST2:
            guest = context.get("guest2_name", "guest2")
            print(f"  🚶 [Nav] 带领 {guest} 前往客厅")
            return None  # 导航部分

        elif state_id == Task1StateID.FOLLOW_HOST:
            if context.get("host_found"):
                print("  🚶 [Nav] 跟随 host 移动")
                return {"at_destination": True, "destination": "storage_room"}
            return None

        else:
            print(f"  [Nav] 未处理的状态: {state_id.value}")
            return None

    # TODO: 真实导航实现
    # def _navigate_to(self, target_pose):
    #     goal = MoveBaseGoal()
    #     goal.target_pose = target_pose
    #     self._move_base_client.send_goal(goal)
    #     self._move_base_client.wait_for_result()


# ============================================================
# 语音模块
# ============================================================

class SpeechModule(BaseSubModule):
    """
    语音交互子模块

    负责所有需要 TTS + ASR 对话的状态:
      ASK_GUEST1_INFO, PICK_UP_GUEST2 (语音部分),
      DESCRIBE_GUEST1 (TTS), SEAT_GUEST2 (语音部分),
      INTRODUCE_GUESTS, REQUEST_GUEST2_BAG (TTS),
      TASK_COMPLETE (TTS)

    依赖 SpeechInterface 注入 (Mock / ROS / Scripted)。
    """

    def __init__(self, speech=None):
        """
        Args:
            speech: SpeechInterface 实例, 为 None 时使用 MockSpeechInterface
        """
        super().__init__("speech")
        if speech is None:
            from task1_receptionist.sub_modules.speech_interaction import MockSpeechInterface
            speech = MockSpeechInterface()
        self._sp = speech

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if state_id == Task1StateID.ASK_GUEST1_INFO:
            return self._ask_guest1_info()

        elif state_id == Task1StateID.PICK_UP_GUEST2:
            return self._pick_up_guest2()

        elif state_id == Task1StateID.DESCRIBE_GUEST1:
            return self._describe_guest1(context)

        elif state_id == Task1StateID.SEAT_GUEST2:
            return self._seat_guest2(context)

        elif state_id == Task1StateID.INTRODUCE_GUESTS:
            return self._introduce_guests(context)

        elif state_id == Task1StateID.REQUEST_GUEST2_BAG:
            return self._request_guest2_bag(context)

        elif state_id == Task1StateID.TASK_COMPLETE:
            return self._task_complete()

        else:
            print(f"  [Speech] 未处理的状态: {state_id.value}")
            return None

    # ---- 各状态具体实现 ----

    def _ask_guest1_info(self) -> Dict[str, Any]:
        name = self._sp.ask("欢迎! 请问您的名字是什么?", timeout_sec=15.0)
        if not name:
            name = "Guest"

        drink = self._sp.ask(f"{name}, 请问您想喝什么饮料?", timeout_sec=15.0)
        if not drink:
            drink = "水"

        return {"guest1_name": name, "guest1_drink": drink}

    def _pick_up_guest2(self) -> Dict[str, Any]:
        self._sp.say("欢迎! 请问您的名字是什么?")
        name = self._sp.listen(timeout_sec=15.0)
        if not name:
            name = "Guest2"
        return {"guest2_at_door": True, "guest2_name": name}

    def _describe_guest1(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guest1 = context.get("guest1_name", "第一位客人")
        seat = context.get("seat_number", 1)
        # TODO: 实际接入视觉模块获取衣服颜色等特征
        self._sp.say(f"{guest1} 穿着红色外套, 坐在 {seat} 号座位。")
        return {"guest1_described": True}

    def _seat_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest = context.get("guest2_name", "guest2")
        drink = self._sp.ask(f"{guest}, 请问您想喝什么饮料?", timeout_sec=15.0)
        if not drink:
            drink = "水"
        return {"guest2_seated": True, "guest2_drink": drink, "seat_number": 2}

    def _introduce_guests(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        g1 = context.get("guest1_name", "guest1")
        g1d = context.get("guest1_drink", "?")
        g2 = context.get("guest2_name", "guest2")
        g2d = context.get("guest2_drink", "?")
        self._sp.say(f"{g1}, 这位是 {g2}, 他喜欢喝{g2d}。{g2}, 这位是 {g1}, 她喜欢喝{g1d}。")
        return {"guests_introduced": True}

    def _request_guest2_bag(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guest = context.get("guest2_name", "guest2")
        self._sp.say(f"{guest}, 请把您的包放在我的托盘上。")
        # TODO: 等待包放置完成 (视觉/力传感器)
        return {"bag_on_tray": True}

    def _task_complete(self) -> Optional[Dict[str, Any]]:
        self._sp.say("任务完成, 感谢各位的配合!")
        return {}

    def close(self):
        self._sp.close()


# ============================================================
# 视觉模块
# ============================================================

class VisionModule(BaseSubModule):
    """
    视觉子模块

    负责需要视觉感知的状态:
      DESCRIBE_GUEST1 (外貌识别),
      FIND_HOST (人物检测与识别),
      FOLLOW_HOST (人物跟踪)
    """

    def __init__(self):
        super().__init__("vision")

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if state_id == Task1StateID.DESCRIBE_GUEST1:
            print("  📷 [Vision] 识别 guest1 外貌特征")
            # TODO: 调用目标检测 / 属性识别
            # return {"guest1_appearance": {"clothing": "红色外套", "position": "1号座位"}}
            return None  # 描述由 SpeechModule 完成, 视觉只产出特征

        elif state_id == Task1StateID.FIND_HOST:
            print("  📷 [Vision] 在环境中搜索 host...")
            print("  📷 [Vision] 检测到人物, 确认为 host")
            # TODO: 调用人脸识别 / 人物重识别
            return {"host_found": True, "host_location": "living_room"}

        elif state_id == Task1StateID.FOLLOW_HOST:
            print("  📷 [Vision] 跟踪 host 位置...")
            # TODO: 人物跟踪 (KCF / DeepSORT)
            return None

        else:
            print(f"  [Vision] 未处理的状态: {state_id.value}")
            return None


# ============================================================
# 操作模块
# ============================================================

class ManipulationModule(BaseSubModule):
    """
    操作子模块

    负责需要机械臂操作的状态:
      POINT_EMPTY_SEAT (指向),
      REQUEST_GUEST2_BAG (伸出托盘 + 检测放包),
      PLACE_BAG (取包 + 放置)
    """

    def __init__(self):
        super().__init__("manipulation")

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if state_id == Task1StateID.POINT_EMPTY_SEAT:
            guest = context.get("guest1_name", "guest1")
            print(f"  👉 [Arm] 指向空座位, 请 {guest} 入座")
            # TODO: 机械臂指向预设座位
            return {"guest1_seated": True, "seat_number": 1}

        elif state_id == Task1StateID.REQUEST_GUEST2_BAG:
            print("  🦾 [Arm] 伸出托盘, 等待放包")
            # TODO: 托盘伸出 + 力传感器检测重量变化
            return None  # bag_on_tray 由 SpeechModule 的后续逻辑设置

        elif state_id == Task1StateID.PLACE_BAG:
            print("  🦾 [Arm] 将托盘上的包放到指定位置")
            # TODO: 机械臂抓取 + 放置
            return {"bag_placed": True}

        else:
            print(f"  [Arm] 未处理的状态: {state_id.value}")
            return None


# ============================================================
# 向后兼容别名
# ============================================================

ASRModule = SpeechModule  # 旧名称, 保持兼容
