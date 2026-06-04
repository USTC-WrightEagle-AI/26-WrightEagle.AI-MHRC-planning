"""
Navigation Actions — 导航/控制类动作模型

对照 nav_skills.py 里的每个纯函数，定义同名、同参数的 Pydantic 模型。
字段名称严格对齐，确保 model_dump(exclude={"type"}) 可以直接 ** 解包传参。
"""

from typing import Any, List, Literal, Optional, Union

from pydantic import Field

from .base_action import BaseAction, Position3D



class NavigationAction(BaseAction):
    """
    导航状态 - 移动到指定3d位置,若成功则返回success
    """

    type: Literal["navigation"] = "navigation"
    position: str = Field(
        ...,
        description="目标位置，具体的3D坐标（x,y,z）",
    )


class PersonTrackingAction(BaseAction):
    """
    人物追踪状态 - 持续追踪特定人物,同时包括了视觉和导航部分，结束后输出success
    """

    type: Literal["follow_person"] = "follow_person"
    person_pos: Optional[str] = Field(None, description="人物位置（可选）")
    duration: Optional[float] = Field(
        None, description="追踪持续时间（秒，None表示持续直到中断）"
    )



class ObjectGraspAction(BaseAction):
    """
    物体抓取状态 - 抓取指定物体,,若成功则返回success
    """

    type: Literal["bringMeObj"] = "bringMeObj"
    object_name: str = Field(..., description="要抓取的物体名称")
    object_position: Optional[Union[Position3D, List[float]]] = Field(
        None, description="物体位置（如果不提供，则先搜索）"
    )
    grasp_force: Optional[float] = Field(None, description="抓取力度（0-1，可选）")


class ObjectDumpAction(BaseAction):
    """
    物体放置状态 - 释放当前抓取的物体,,若成功则返回success
    """

    type: Literal["object_dump"] = "object_dump"
    target_position: Union[str, Position3D, List[float]] = Field(
        ..., description="放置位置（语义标签如'table'，或3D坐标[x,y,z]）"
    )
    release_safe: bool = Field(
        default=True, description="是否安全释放（先检测下方是否有支撑面）"
    )


class TalkingAction(BaseAction):
    """
    对话状态 - 响应用户对话
    """

    type: Literal["talking"] = "talking"
    text: str = Field(..., description="要说的文本内容")
    wait_for_response: bool = Field(default=False, description="是否等待用户响应")


class WaitAction(BaseAction):
    """
    等待状态 - 等待指定时间或事件
    """

    type: Literal["wait"] = "wait"
    duration: float = Field(..., description="等待时间（秒）")
    interruptible: bool = Field(default=True, description="是否可被中断")


class ListeningAction(BaseAction):
    """
    监听状态 - 持续监听音频流
    """

    type: Literal["listening"] = "listening"
    continuous: bool = Field(default=True, description="是否持续监听")
    vad_enabled: bool = Field(default=True, description="是否启用语音活动检测")
    wake_word: Optional[str] = Field(None, description="唤醒词（可选）")


class IdleAction(BaseAction):
    """
    空闲状态 - 机器人待机，等待指令
    """

    type: Literal["idle"] = "idle"
    low_power_mode: bool = Field(default=False, description="是否进入低功耗模式")
    summary: str = Field(..., description="任务总结")
