"""
Robot Interface - 机器人接口定义

定义统一的机器人抽象接口，Mock和Real类都需要实现这个接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum


class RobotState(str, Enum):
    """机器人状态"""
    IDLE = "IDLE"              # 空闲
    THINKING = "THINKING"      # 思考中（LLM推理）
    EXECUTING = "EXECUTING"    # 执行中
    SPEAKING = "SPEAKING"      # 说话中（TTS播放）
    ERROR = "ERROR"            # 错误状态


class RobotInterface(ABC):
    """
    机器人抽象接口

    所有机器人类（Mock/Real）都必须实现这些方法
    """

    def __init__(self):
        self.state = RobotState.IDLE
        self.current_position: Optional[str] = None
        self.holding_object: Optional[str] = None

    @abstractmethod
    def navigate(self, target) -> bool:
        """
        导航到目标位置

        Args:
            target: 目标位置（语义标签或坐标）

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def search(self, object_name: str) -> Optional[dict]:
        """
        搜索物体

        Args:
            object_name: 物体名称

        Returns:
            dict: 找到的物体信息，如 {"name": "apple", "position": [x,y,z]}
            None: 未找到
        """
        pass

    @abstractmethod
    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        """
        抓取物体

        Args:
            object_name: 物体名称
            object_id: 物体ID（如果有多个同名物体）

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def place(self, location) -> bool:
        """
        放置物体

        Args:
            location: 放置位置

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def speak(self, content: str) -> bool:
        """
        语音输出

        Args:
            content: 要说的内容

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    def wait(self, reason: Optional[str] = None) -> bool:
        """
        等待/无操作

        Args:
            reason: 等待原因

        Returns:
            bool: 是否成功
        """
        pass

    def get_state(self) -> RobotState:
        """获取当前状态"""
        return self.state

    def set_state(self, state: RobotState):
        """设置状态"""
        self.state = state
