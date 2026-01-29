"""
数据模型定义 - Data Schemas

使用 Pydantic 严格定义机器人的动作空间和决策输出格式
"""

from typing import Literal, Union, Optional, List
from pydantic import BaseModel, Field, field_validator


# ==================== 动作类型定义 ====================

class NavigateAction(BaseModel):
    """导航动作 - 移动到指定位置"""
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]] = Field(
        ...,
        description="目标位置，可以是语义标签(如'kitchen')或坐标[x,y,z]"
    )

    @field_validator('target')
    @classmethod
    def validate_target(cls, v):
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError("坐标必须是 [x, y, z] 格式")
            if not all(isinstance(i, (int, float)) for i in v):
                raise ValueError("坐标必须是数字")
        return v


class PickAction(BaseModel):
    """抓取动作 - 拾取物体"""
    type: Literal["pick"] = "pick"
    object_name: str = Field(..., description="物体名称")
    object_id: Optional[int] = Field(None, description="物体ID（如果有多个同名物体）")


class PlaceAction(BaseModel):
    """放置动作 - 将物体放到指定位置"""
    type: Literal["place"] = "place"
    location: Union[str, List[float]] = Field(
        ...,
        description="放置位置，可以是语义标签(如'table')或坐标"
    )


class SearchAction(BaseModel):
    """搜索动作 - 寻找物体"""
    type: Literal["search"] = "search"
    object_name: str = Field(..., description="要搜索的物体名称")


class SpeakAction(BaseModel):
    """说话动作 - 语音输出"""
    type: Literal["speak"] = "speak"
    content: str = Field(..., description="要说的内容")


class WaitAction(BaseModel):
    """等待动作 - 保持当前状态"""
    type: Literal["wait"] = "wait"
    reason: Optional[str] = Field(None, description="等待原因")


# ==================== 联合动作类型 ====================

# 所有可能的动作类型
RobotAction = Union[
    NavigateAction,
    PickAction,
    PlaceAction,
    SearchAction,
    SpeakAction,
    WaitAction
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
        "navigate": NavigateAction,
        "pick": PickAction,
        "place": PlaceAction,
        "search": SearchAction,
        "speak": SpeakAction,
        "wait": WaitAction,
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
