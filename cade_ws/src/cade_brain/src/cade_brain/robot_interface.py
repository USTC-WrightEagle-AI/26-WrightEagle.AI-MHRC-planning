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

    # ==================== 导航类动作 ====================

    @abstractmethod
    def goToLoc(self, target: str, then_find_person: bool = False) -> bool:
        """去某地（可选到达后找人）"""
        pass

    # ==================== 人物操作类动作 ====================

    @abstractmethod
    def findPrsInRoom(self, room: str, gesture: Optional[str] = None) -> Optional[dict]:
        """在房间找特定姿态/手势的人"""
        pass

    @abstractmethod
    def meetPrsAtBeac(self, person_name: str, beacon: str) -> bool:
        """在信标处见某人（按名字）"""
        pass

    @abstractmethod
    def countPrsInRoom(self, room: str, gesture: Optional[str] = None) -> int:
        """数房间里有某种姿态/手势的人数"""
        pass

    @abstractmethod
    def tellPrsInfoInLoc(self, person_name: Optional[str], location: str) -> Optional[dict]:
        """告诉我某地某人的信息"""
        pass

    @abstractmethod
    def talkInfoToGestPrsInRoom(self, room: str, gesture: str, info: str) -> bool:
        """在房间跟做手势的人交谈/传递信息"""
        pass

    @abstractmethod
    def followNameFromBeacToRoom(self, person_name: str, beacon: str, room: str) -> bool:
        """从信标跟随某人到房间"""
        pass

    @abstractmethod
    def guideNameFromBeacToBeac(self, person_name: str, from_beacon: str, to_beacon: str) -> bool:
        """从信标引导某人到另一地点"""
        pass

    @abstractmethod
    def guidePrsFromBeacToBeac(self, gesture: str, from_beacon: str, to_beacon: str) -> bool:
        """从信标引导有姿态/手势的人到另一地点"""
        pass

    @abstractmethod
    def guideClothPrsFromBeacToBeac(self, cloth_color: str, from_beacon: str, to_beacon: str) -> bool:
        """引导穿特定颜色衣服的人从信标到另一地点"""
        pass

    @abstractmethod
    def greetClothDscInRm(self, cloth_color: str, room: str) -> bool:
        """问候穿特定颜色衣服的人"""
        pass

    @abstractmethod
    def greetNameInRm(self, person_name: str, room: str) -> bool:
        """问候特定名字的人"""
        pass

    @abstractmethod
    def meetNameAtLocThenFindInRm(self, person_name: str, meet_location: str, room: str) -> bool:
        """在某地见某人然后在房间找到他们"""
        pass

    @abstractmethod
    def countClothPrsInRoom(self, cloth_color: str, room: str) -> int:
        """数房间里穿特定颜色衣服的人数"""
        pass

    @abstractmethod
    def tellPrsInfoAtLocToPrsAtLoc(self, from_person: Optional[str], from_location: str,
                                     to_person: Optional[str], to_location: str,
                                     info: Optional[str] = None) -> bool:
        """把一个地点某人的信息告诉另一地点的人"""
        pass

    @abstractmethod
    def followPrsAtLoc(self, gesture: str, location: str) -> bool:
        """跟随某地有姿态/手势的人"""
        pass

    # ==================== 物品操作类动作 ====================

    @abstractmethod
    def takeObjFromPlcmt(self, object_name: str, placement: str) -> bool:
        """从放置处拿物品"""
        pass

    @abstractmethod
    def findObjInRoom(self, object_name: str, room: str) -> Optional[dict]:
        """在房间找物品"""
        pass

    @abstractmethod
    def countObjOnPlcmt(self, object_category: str, placement: str) -> int:
        """数放置处某类物品的数量"""
        pass

    @abstractmethod
    def tellObjPropOnPlcmt(self, object_name: str, placement: str, property: str) -> Optional[dict]:
        """问放置处物品的属性（最大/最小等）"""
        pass

    @abstractmethod
    def bringMeObjFromPlcmt(self, object_name: str, placement: str) -> bool:
        """从放置处拿物品给我"""
        pass

    @abstractmethod
    def tellCatPropOnPlcmt(self, category: str, placement: str, property: str) -> Optional[dict]:
        """问放置处某类物品的属性"""
        pass

    # ==================== 动态世界模型更新 ====================

    @abstractmethod
    def add_person(self, name: str, cloth_color: Optional[str] = None,
                   gesture: Optional[str] = None, location: Optional[str] = None,
                   beacon: Optional[str] = None, info: Optional[str] = None) -> bool:
        """向世界模型添加新人物"""
        pass

    @abstractmethod
    def update_person(self, name: str, **kwargs) -> bool:
        """更新世界模型中人物的属性（位置、手势、衣服等）"""
        pass

    @abstractmethod
    def add_object(self, name: str, category: Optional[str] = None,
                   location: Optional[str] = None, placement: Optional[str] = None,
                   position: Optional[list] = None, **extra_props) -> bool:
        """向世界模型添加新物体"""
        pass

    @abstractmethod
    def update_object(self, name: str, **kwargs) -> bool:
        """更新世界模型中物体的属性（位置等）"""
        pass

    @abstractmethod
    def get_world_state(self) -> str:
        """获取当前世界模型的文本摘要（用于注入 LLM 上下文）"""
        pass

    def get_state(self) -> RobotState:
        """获取当前状态"""
        return self.state

    def set_state(self, state: RobotState):
        """设置状态"""
        self.state = state
