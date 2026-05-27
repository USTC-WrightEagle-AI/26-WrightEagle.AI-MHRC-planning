"""
Navigation Skill - 导航技能

通过 ROS pub/sub 向导航/底盘节点发送指令并等待结果。
"""

import json
from typing import Optional, Dict, Any

try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from cade_brain.skills.base_skill import BaseSkill


class NavSkill(BaseSkill):
    """
    导航技能 - 大脑与底盘控制的桥梁

    职责：
    1. 将 LLM 导航动作翻译为 /cade/task_cmd 的 JSON 指令
    2. 等待底盘节点通过 /cade/task_status 返回结果

    绝对不包含：
    - 任何底盘控制的具体实现
    - 任何视觉处理逻辑
    """

    def __init__(self):
        super().__init__(node_name_prefix="nav_skill")

    def go_to_location(self, target: str, then_find_person: bool = False,
                       timeout: float = 60.0) -> Dict[str, Any]:
        """
        导航到指定位置

        Args:
            target: 目标位置名称
            then_find_person: 到达后是否找人
            timeout: 超时时间

        Returns:
            dict: {"status": "SUCCESS"|"FAILED", "position": ...}
        """
        return self.execute("goToLoc", timeout=timeout,
                           target=target,
                           then_find_person=then_find_person)

    def follow_person(self, person_name: Optional[str] = None,
                      gesture: Optional[str] = None,
                      location: Optional[str] = None,
                      timeout: float = 120.0) -> Dict[str, Any]:
        """
        跟随人物

        Args:
            person_name: 人物名称（可选，用于按名字跟随）
            gesture: 姿态特征（可选）
            location: 位置（可选）
            timeout: 超时时间

        Returns:
            dict: {"status": "SUCCESS"|"FAILED"}
        """
        params = {}
        if person_name:
            params["person_name"] = person_name
        if gesture:
            params["gesture"] = gesture
        if location:
            params["location"] = location
        return self.execute("follow_person", timeout=timeout, **params)

    def guide_person(self, person_name: Optional[str] = None,
                     from_beacon: Optional[str] = None,
                     to_beacon: Optional[str] = None,
                     timeout: float = 120.0) -> Dict[str, Any]:
        """
        引导人物从一个信标到另一个信标

        Args:
            person_name: 人物名称
            from_beacon: 起始信标
            to_beacon: 目标信标
            timeout: 超时时间

        Returns:
            dict: {"status": "SUCCESS"|"FAILED"}
        """
        params = {}
        if person_name:
            params["person_name"] = person_name
        if from_beacon:
            params["from_beacon"] = from_beacon
        if to_beacon:
            params["to_beacon"] = to_beacon
        return self.execute("guide_person", timeout=timeout, **params)

    def pick_and_bring(self, object_name: str, placement: str,
                       timeout: float = 120.0) -> Dict[str, Any]:
        """
        从指定位置取物品并带回

        Args:
            object_name: 物品名称
            placement: 物品所在位置
            timeout: 超时时间

        Returns:
            dict: {"status": "SUCCESS"|"FAILED"}
        """
        return self.execute("bringMeObj", timeout=timeout,
                           object_name=object_name, placement=placement)

    def take_object(self, object_name: str, placement: str,
                    timeout: float = 60.0) -> Dict[str, Any]:
        """
        从指定位置拿取物品

        Args:
            object_name: 物品名称
            placement: 物品所在位置
            timeout: 超时时间

        Returns:
            dict: {"status": "SUCCESS"|"FAILED"}
        """
        return self.execute("takeObjFromPlcmt", timeout=timeout,
                           object_name=object_name, placement=placement)
