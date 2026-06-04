"""
Vision Actions — 视觉感知/计数类动作模型

对照 vision_skills.py 里的每个纯函数，定义同名、同参数的 Pydantic 模型。
字段名称严格对齐，确保 model_dump(exclude={"type"}) 可以直接 ** 解包传参。
"""

from typing import Literal, Optional

from pydantic import Field

from .base_action import BaseAction


class FindObjectAction(BaseAction):
    """
    Search Object State - Find a specific object in the environment.
    """

    type: Literal["find_object"] = "find_object"
    target: str = Field(
        ...,
        description="The exact name of the object to look for (e.g., bottle, chips, cup).",
    )


class FindPersonAction(BaseAction):
    """
    Search Person State - Find a specific person based on body posture, clothing color, or both.
    """

    type: Literal["find_person"] = "find_person"
    target: str = Field(
        default="person", description="The target class, usually 'person'."
    )
    gesture: Optional[str] = Field(
        None,
        description="Filter by body gesture/posture (e.g., waving, sitting, standing).",
    )
    cloth_color: Optional[str] = Field(
        None, description="Filter by clothing color (e.g., red, blue, black, white)."
    )


class CountObjectsAction(BaseAction):
    """
    Count Objects State - Count the number of a specific object category in the room.
    """

    type: Literal["count_objects"] = "count_objects"
    category: str = Field(
        ..., description="The object category name to count (e.g., bottle, cup, chair)."
    )


class CountPeopleAction(BaseAction):
    """
    Count People State - Count the number of people matching specific attributes (gesture/cloth color) in the room.
    """

    type: Literal["count_people"] = "count_people"
    category: str = Field(
        default="person", description="The target category, always 'person'."
    )
    gesture: Optional[str] = Field(
        None,
        description="Filter by body posture/gesture (e.g., waving, sitting, standing).",
    )
    cloth_color: Optional[str] = Field(
        None, description="Filter by clothing color (e.g., red, blue, white)."
    )


class NameRecognitionAction(BaseAction):
    """
    人名识别状态 - 通过姓名识别人物
    """

    type: Literal["name_recognition"] = "name_recognition"
    name: str = Field(..., description="要识别的人物名称")
    bind_to_appearance: bool = Field(
        default=True, description="是否将名字与视觉外观绑定"
    )


