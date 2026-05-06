"""
数据模型定义 - Data Schemas

使用 Pydantic 严格定义机器人的动作空间和决策输出格式
"""

from typing import Literal, Union, Optional
from pydantic import BaseModel, Field


# ==================== 动作类型定义 ====================

# ==================== 导航类动作 (1种) ====================

class GoToLocAction(BaseModel):
    """导航动作 - 去某地（如"去厨房然后找人"）"""
    type: Literal["goToLoc"] = "goToLoc"
    target: str = Field(..., description="目标位置（语义标签或坐标）")
    then_find_person: Optional[bool] = Field(False, description="到达后是否寻找人")


# ==================== 人物操作类动作 (15种) ====================

class FindPrsInRoomAction(BaseModel):
    """在房间找特定姿态/手势的人"""
    type: Literal["findPrsInRoom"] = "findPrsInRoom"
    room: str = Field(..., description="房间名称")
    gesture: Optional[str] = Field(None, description="姿态/手势描述（如 waving, sitting, standing）")


class MeetPrsAtBeacAction(BaseModel):
    """在信标处见某人（按名字）"""
    type: Literal["meetPrsAtBeac"] = "meetPrsAtBeac"
    person_name: str = Field(..., description="人物名称")
    beacon: str = Field(..., description="信标/地点名称")


class CountPrsInRoomAction(BaseModel):
    """数房间里有某种姿态/手势的人数"""
    type: Literal["countPrsInRoom"] = "countPrsInRoom"
    room: str = Field(..., description="房间名称")
    gesture: Optional[str] = Field(None, description="姿态/手势描述（可选，为空则数所有人）")


class TellPrsInfoInLocAction(BaseModel):
    """告诉我某地某人的信息"""
    type: Literal["tellPrsInfoInLoc"] = "tellPrsInfoInLoc"
    person_name: Optional[str] = Field(None, description="人物名称（可选）")
    location: str = Field(..., description="地点名称")


class TalkInfoToGestPrsInRoomAction(BaseModel):
    """在房间跟做手势的人交谈/传递信息"""
    type: Literal["talkInfoToGestPrsInRoom"] = "talkInfoToGestPrsInRoom"
    room: str = Field(..., description="房间名称")
    gesture: str = Field(..., description="目标人物的手势/姿态")
    info: str = Field(..., description="要传递的信息内容")


class FollowNameFromBeacToRoomAction(BaseModel):
    """从信标跟随某人到房间"""
    type: Literal["followNameFromBeacToRoom"] = "followNameFromBeacToRoom"
    person_name: str = Field(..., description="要跟随的人物名称")
    beacon: str = Field(..., description="起始信标/地点")
    room: str = Field(..., description="目标房间")


class GuideNameFromBeacToBeacAction(BaseModel):
    """从信标引导某人到另一地点"""
    type: Literal["guideNameFromBeacToBeac"] = "guideNameFromBeacToBeac"
    person_name: str = Field(..., description="要引导的人物名称")
    from_beacon: str = Field(..., description="起始信标")
    to_beacon: str = Field(..., description="目标信标")


class GuidePrsFromBeacToBeacAction(BaseModel):
    """从信标引导有姿态/手势的人到另一地点"""
    type: Literal["guidePrsFromBeacToBeac"] = "guidePrsFromBeacToBeac"
    gesture: str = Field(..., description="姿态/手势描述")
    from_beacon: str = Field(..., description="起始信标")
    to_beacon: str = Field(..., description="目标信标")


class GuideClothPrsFromBeacToBeacAction(BaseModel):
    """引导穿特定颜色衣服的人从信标到另一地点"""
    type: Literal["guideClothPrsFromBeacToBeac"] = "guideClothPrsFromBeacToBeac"
    cloth_color: str = Field(..., description="衣服颜色")
    from_beacon: str = Field(..., description="起始信标")
    to_beacon: str = Field(..., description="目标信标")


class GreetClothDscInRmAction(BaseModel):
    """问候穿特定颜色衣服的人"""
    type: Literal["greetClothDscInRm"] = "greetClothDscInRm"
    cloth_color: str = Field(..., description="衣服颜色")
    room: str = Field(..., description="房间名称")


class GreetNameInRmAction(BaseModel):
    """问候特定名字的人"""
    type: Literal["greetNameInRm"] = "greetNameInRm"
    person_name: str = Field(..., description="人物名称")
    room: str = Field(..., description="房间名称")


class MeetNameAtLocThenFindInRmAction(BaseModel):
    """在某地见某人然后在房间找到他们"""
    type: Literal["meetNameAtLocThenFindInRm"] = "meetNameAtLocThenFindInRm"
    person_name: str = Field(..., description="人物名称")
    meet_location: str = Field(..., description="见面地点")
    room: str = Field(..., description="之后要找到该人物的房间")


class CountClothPrsInRoomAction(BaseModel):
    """数房间里穿特定颜色衣服的人数"""
    type: Literal["countClothPrsInRoom"] = "countClothPrsInRoom"
    cloth_color: str = Field(..., description="衣服颜色")
    room: str = Field(..., description="房间名称")


class TellPrsInfoAtLocToPrsAtLocAction(BaseModel):
    """把一个地点某人的信息告诉另一地点的人"""
    type: Literal["tellPrsInfoAtLocToPrsAtLoc"] = "tellPrsInfoAtLocToPrsAtLoc"
    from_person: Optional[str] = Field(None, description="信息源人物名称")
    from_location: str = Field(..., description="信息源地点")
    to_person: Optional[str] = Field(None, description="目标人物名称")
    to_location: str = Field(..., description="目标地点")
    info: Optional[str] = Field(None, description="要传递的信息")


class FollowPrsAtLocAction(BaseModel):
    """跟随某地有姿态/手势的人"""
    type: Literal["followPrsAtLoc"] = "followPrsAtLoc"
    gesture: str = Field(..., description="姿态/手势描述")
    location: str = Field(..., description="地点名称")


# ==================== 物品操作类动作 (6种) ====================

class TakeObjFromPlcmtAction(BaseModel):
    """从放置处拿物品"""
    type: Literal["takeObjFromPlcmt"] = "takeObjFromPlcmt"
    object_name: str = Field(..., description="物品名称")
    placement: str = Field(..., description="放置处/位置")


class FindObjInRoomAction(BaseModel):
    """在房间找物品"""
    type: Literal["findObjInRoom"] = "findObjInRoom"
    object_name: str = Field(..., description="物品名称")
    room: str = Field(..., description="房间名称")


class CountObjOnPlcmtAction(BaseModel):
    """数放置处某类物品的数量"""
    type: Literal["countObjOnPlcmt"] = "countObjOnPlcmt"
    object_category: str = Field(..., description="物品类别")
    placement: str = Field(..., description="放置处/位置")


class TellObjPropOnPlcmtAction(BaseModel):
    """问放置处物品的属性（最大/最小等）"""
    type: Literal["tellObjPropOnPlcmt"] = "tellObjPropOnPlcmt"
    object_name: str = Field(..., description="物品名称")
    placement: str = Field(..., description="放置处/位置")
    property: str = Field(..., description="要查询的属性（如 max, min, size, weight）")


class BringMeObjFromPlcmtAction(BaseModel):
    """从放置处拿物品给我"""
    type: Literal["bringMeObjFromPlcmt"] = "bringMeObjFromPlcmt"
    object_name: str = Field(..., description="物品名称")
    placement: str = Field(..., description="放置处/位置")


class TellCatPropOnPlcmtAction(BaseModel):
    """问放置处某类物品的属性"""
    type: Literal["tellCatPropOnPlcmt"] = "tellCatPropOnPlcmt"
    category: str = Field(..., description="物品类别名称")
    placement: str = Field(..., description="放置处/位置")
    property: str = Field(..., description="要查询的属性（如 max, min, count, size）")


# ==================== 联合动作类型 ====================

# 所有可能的动作类型
RobotAction = Union[
    GoToLocAction,
    FindPrsInRoomAction,
    MeetPrsAtBeacAction,
    CountPrsInRoomAction,
    TellPrsInfoInLocAction,
    TalkInfoToGestPrsInRoomAction,
    FollowNameFromBeacToRoomAction,
    GuideNameFromBeacToBeacAction,
    GuidePrsFromBeacToBeacAction,
    GuideClothPrsFromBeacToBeacAction,
    GreetClothDscInRmAction,
    GreetNameInRmAction,
    MeetNameAtLocThenFindInRmAction,
    CountClothPrsInRoomAction,
    TellPrsInfoAtLocToPrsAtLocAction,
    FollowPrsAtLocAction,
    TakeObjFromPlcmtAction,
    FindObjInRoomAction,
    CountObjOnPlcmtAction,
    TellObjPropOnPlcmtAction,
    BringMeObjFromPlcmtAction,
    TellCatPropOnPlcmtAction,
]


# ==================== 决策输出格式 ====================

class RobotDecision(BaseModel):
    """
    机器人决策输出格式（LLM的输出结构）

    这是LLM必须遵循的输出格式，包含思考过程、回复和动作
    """
    thought: Optional[str] = Field(  # ← 改为可选
        None,
        description="内部思考过程（CoT - Chain of Thought）"
    )
    reply: Optional[str] = Field(
        None,
        description="给用户的自然语言回复（可选）"
    )
    action: Optional[RobotAction] = Field(
        None,
        description="要执行的动作（如果是纯对话则为None）"
    )

    class Config:
        # 允许任意类型（用于处理联合类型）
        arbitrary_types_allowed = True


# ==================== 辅助函数 ====================

def parse_action(action_dict: dict) -> RobotAction:
    """
    根据 type 字段解析动作

    Args:
        action_dict: 包含 type 字段的字典

    Returns:
        对应的 Action 对象

    Raises:
        ValueError: 如果 type 不合法
    """
    action_type = action_dict.get("type")

    action_map = {
        "goToLoc": GoToLocAction,
        "findPrsInRoom": FindPrsInRoomAction,
        "meetPrsAtBeac": MeetPrsAtBeacAction,
        "countPrsInRoom": CountPrsInRoomAction,
        "tellPrsInfoInLoc": TellPrsInfoInLocAction,
        "talkInfoToGestPrsInRoom": TalkInfoToGestPrsInRoomAction,
        "followNameFromBeacToRoom": FollowNameFromBeacToRoomAction,
        "guideNameFromBeacToBeac": GuideNameFromBeacToBeacAction,
        "guidePrsFromBeacToBeac": GuidePrsFromBeacToBeacAction,
        "guideClothPrsFromBeacToBeac": GuideClothPrsFromBeacToBeacAction,
        "greetClothDscInRm": GreetClothDscInRmAction,
        "greetNameInRm": GreetNameInRmAction,
        "meetNameAtLocThenFindInRm": MeetNameAtLocThenFindInRmAction,
        "countClothPrsInRoom": CountClothPrsInRoomAction,
        "tellPrsInfoAtLocToPrsAtLoc": TellPrsInfoAtLocToPrsAtLocAction,
        "followPrsAtLoc": FollowPrsAtLocAction,
        "takeObjFromPlcmt": TakeObjFromPlcmtAction,
        "findObjInRoom": FindObjInRoomAction,
        "countObjOnPlcmt": CountObjOnPlcmtAction,
        "tellObjPropOnPlcmt": TellObjPropOnPlcmtAction,
        "bringMeObjFromPlcmt": BringMeObjFromPlcmtAction,
        "tellCatPropOnPlcmt": TellCatPropOnPlcmtAction,
    }

    if action_type not in action_map:
        raise ValueError(
            f"未知的动作类型: {action_type}. "
            f"可用动作: {list(action_map.keys())}"
        )

    return action_map[action_type](**action_dict)


# ==================== 示例数据 ====================

if __name__ == "__main__":
    # 测试用例
    print("=== 测试动作定义 ===\n")

    # 1. 导航动作
    nav_action = NavigateAction(target="kitchen")
    print(f"1. 导航（语义）: {nav_action.model_dump_json(indent=2)}\n")

    nav_action_coords = NavigateAction(target=[1.5, 2.3, 0.0])
    print(f"2. 导航（坐标）: {nav_action_coords.model_dump_json(indent=2)}\n")

    # 2. 完整决策
    decision = RobotDecision(
        thought="用户想要苹果，我需要先找到它",
        reply="好的，我这就去找苹果",
        action=SearchAction(object_name="apple")
    )
    print(f"3. 完整决策: {decision.model_dump_json(indent=2)}\n")

    # 3. 纯对话（无动作）
    chat_decision = RobotDecision(
        thought="用户在问候我",
        reply="您好！我是服务机器人LARA，很高兴为您服务",
        action=None
    )
    print(f"4. 纯对话: {chat_decision.model_dump_json(indent=2)}\n")

    print("✓ 所有测试通过！")
