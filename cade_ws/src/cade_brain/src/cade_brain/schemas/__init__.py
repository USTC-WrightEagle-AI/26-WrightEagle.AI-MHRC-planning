"""
Schemas Package — 动作模型与决策输出

提供统一的 RobotAction 联合类型和 RobotDecision 决策模型。
所有动作模型与纯函数工具箱严格对齐，通过 inspect 动态扫描，彻底杜绝硬编码与双重维护。
"""

import inspect
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, Field

from .base_action import BaseAction, Position3D
from .nav_actions import *
from .vision_actions import *

# ==========================================
# ⚡ 核心核心：通过反射机制（Reflection）动态构建动作映射表
# ==========================================
_ACTION_MODEL_MAP: Dict[str, Type[BaseAction]] = {}

# 动态扫描当前包内所有的模型类
# 只要是继承自 BaseAction 且定义了具体 type 的类，全部自动注册
import cade_brain.schemas.nav_actions as nav_mods
import cade_brain.schemas.vision_actions as vis_mods

for module in [nav_mods, vis_mods]:
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseAction) and obj is not BaseAction:
            # 自动提取你在 Pydantic 类里锁定的 type 字面量默认值（例如 "goToLoc" 或 "clothes_recognition"）
            try:
                action_type = obj.model_fields["type"].default
            except AttributeError:
                action_type = obj.__fields__["type"].default

            if action_type and action_type != "PydanticUndefined":
                _ACTION_MODEL_MAP[action_type] = obj


# ==========================================
# ── RobotAction 联合类型（利用动态生成的类列表动态组装）
# ==========================================
# 动态将扫描到的所有具体 Pydantic 类组合成一个 Union 联合类型
RobotAction = Union[tuple(_ACTION_MODEL_MAP.values())]


# ── RobotDecision 决策模型 ────────────────────────────────────
class RobotDecision(BaseModel):
    """
    机器人决策输出格式（LLM 的输出结构）
    """

    thought: Optional[str] = Field(
        None, description="内部思考过程（CoT）- 解释如何分解任务"
    )
    reply: Optional[str] = Field(
        None, description="给用户的自然语言回复（必须使用英语）"
    )
    action: Optional[RobotAction] = Field(
        None, description="要执行的原子动作（单个动作）"
    )

    class Config:
        arbitrary_types_allowed = True


# ==========================================
# ── 统一容错解析函数（纯动态，拒绝补丁屎山）
# ==========================================
def parse_action(action_dict: Dict[str, Any]) -> RobotAction:
    """
    根据 type 字段动态将动作字典反序列化为具体的 Pydantic 模型实例。
    """
    action_dict = dict(action_dict)
    action_type = action_dict.get("type")

    # 1. 查找动态注册表
    model_cls = _ACTION_MODEL_MAP.get(action_type)
    if model_cls is None:
        raise ValueError(
            f"未知的动作类型: {action_type}. "
            f"当前最新注册的可用动作列表: {list(_ACTION_MODEL_MAP.keys())}"
        )

    # 2. 物理实例化并进行 Pydantic 强类型安检
    return model_cls(**action_dict)


# ── 导出列表 ──────────────────────────────────────────────────
__all__ = [
    "BaseAction",
    "Position3D",
    "RobotAction",
    "RobotDecision",
    "parse_action",
]
