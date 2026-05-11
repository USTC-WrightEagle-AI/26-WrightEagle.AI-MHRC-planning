"""
Vision Skill - 视觉技能

通过 ROS pub/sub 向视觉节点发送指令并等待结果。
纯指令发布，不包含任何视觉处理逻辑。
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


class VisionSkill(BaseSkill):
    """
    视觉技能 - 大脑与视觉节点的桥梁

    职责：
    1. 将 LLM 动作翻译为 /cade/task_cmd 的 JSON 指令
    2. 等待视觉节点通过 /cade/task_status 返回结果
    3. 将结果回传给 Controller 以更新 WorldModel

    绝对不包含：
    - 任何图像处理逻辑
    - 任何底盘控制逻辑
    - 任何模型状态修改
    """

    def __init__(self):
        super().__init__(node_name_prefix="vision_skill")

    def find_object(self, object_name: str, room: Optional[str] = None,
                    timeout: float = 30.0) -> Dict[str, Any]:
        """
        寻找指定物体

        Args:
            object_name: 物体名称（如 "apple", "cup"）
            room: 搜索的房间（可选）
            timeout: 超时时间

        Returns:
            dict: 包含 3D 位置等信息的检测结果
        """
        params = {"target": object_name}
        if room:
            params["room"] = room
        return self.execute("find_object", timeout=timeout, **params)

    def find_person(self, room: Optional[str] = None,
                    gesture: Optional[str] = None,
                    cloth_color: Optional[str] = None,
                    timeout: float = 30.0) -> Dict[str, Any]:
        """
        寻找人物

        Args:
            room: 搜索的房间（可选）
            gesture: 目标姿态（可选，如 "waving", "standing"）
            cloth_color: 衣服颜色（可选）
            timeout: 超时时间

        Returns:
            dict: 检测到的人物信息
        """
        params = {}
        if room:
            params["room"] = room
        if gesture:
            params["gesture"] = gesture
        if cloth_color:
            params["cloth_color"] = cloth_color
        return self.execute("find_person", timeout=timeout, **params)

    def count_objects(self, category: str, placement: str,
                      timeout: float = 30.0) -> Dict[str, Any]:
        """
        统计指定位置某类物品的数量

        Args:
            category: 物品类别
            placement: 位置
            timeout: 超时时间

        Returns:
            dict: {"count": N, ...}
        """
        return self.execute("count_objects", timeout=timeout,
                           category=category, placement=placement)

    def count_people(self, room: str, gesture: Optional[str] = None,
                     timeout: float = 30.0) -> Dict[str, Any]:
        """
        统计房间内的人数

        Args:
            room: 房间名称
            gesture: 姿态筛选（可选）
            timeout: 超时时间

        Returns:
            dict: {"count": N, ...}
        """
        params = {"room": room}
        if gesture:
            params["gesture"] = gesture
        return self.execute("count_people", timeout=timeout, **params)

    def get_person_info(self, person_name: Optional[str] = None,
                        location: Optional[str] = None,
                        timeout: float = 30.0) -> Dict[str, Any]:
        """
        获取人物详细信息

        Args:
            person_name: 人物名称（可选）
            location: 位置（可选）
            timeout: 超时时间

        Returns:
            dict: 人物信息
        """
        params = {}
        if person_name:
            params["person_name"] = person_name
        if location:
            params["location"] = location
        return self.execute("get_person_info", timeout=timeout, **params)
