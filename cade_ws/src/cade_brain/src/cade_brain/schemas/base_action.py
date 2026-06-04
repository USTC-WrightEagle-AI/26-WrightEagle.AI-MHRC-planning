"""
Base Action - 动作基类

所有原子动作模型的基类，提供统一的 type 字段。
"""

from typing import Optional, Union, List
from pydantic import BaseModel, Field


class Position3D(BaseModel):
    """3D position in the robot/world coordinate frame."""
    x: float = Field(..., description="X 坐标")
    y: float = Field(..., description="Y 坐标")
    z: float = Field(..., description="Z 坐标")


class BaseAction(BaseModel):
    """
    动作基类 - 所有原子动作模型的基类

    子类必须用 Literal 定义 type 字段的默认值，
    确保与对应纯函数的 __name__ 一致。
    """
    type: str = Field(..., description="动作类型标识符")

    class Config:
        # 允许任意类型（如 Union[Position3D, List[float]]）
        arbitrary_types_allowed = True
