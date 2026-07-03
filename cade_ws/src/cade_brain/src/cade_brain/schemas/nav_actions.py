"""
Navigation Actions — 导航/控制类动作模型

对照 nav_skills.py 里的每个纯函数，定义同名、同参数的 Pydantic 模型。
字段名称严格对齐，确保 model_dump(exclude={"type"}) 可以直接 ** 解包传参。
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field

from .base_action import BaseAction, Position3D


CoordinateInput = Union[str, Position3D, List[float], Dict[str, Any]]


class NavigationAction(BaseAction):
    """
    导航状态 - 一次性移动到指定目标，成功后结束。

    普通导航使用 map/base_link 等 2D 导航坐标：[x, y, yaw_deg]。
    从 observe_people/find_people 得到的 position_3d 可以直接传入；
    brain 会自动按 vision/camera 3D 坐标 [x, y, z] 转发。
    """

    type: Literal["navigation"] = "navigation"
    position: CoordinateInput = Field(
        ...,
        description=(
            "导航目标。frame_id 为 map/base_link 时是 2D 导航坐标 "
            "[x,y,yaw_deg]；observe_people/find_people 返回的 position_3d "
            "[x,y,z] 可直接传入，brain 会自动按 vision 坐标处理。"
            "可用命名地点包括 sofa, side tables, desk, desk lamp, office, "
            "bathroom, bedroom, kitchen, tv stand, trash, table。"
        ),
    )
    frame_id: Optional[str] = Field(
        default=None,
        description=(
            "位置所在坐标系。可用 map/base_link 等普通 TF frame；"
            "也可用特殊值 'vision' 表示 position 来自 vision position_3d。"
        ),
    )
    yaw_deg: Optional[float] = Field(
        default=None,
        description="目标朝向角度；不提供则由导航节点自动面向目标",
    )
    timeout: Optional[float] = Field(
        default=None,
        description="导航等待超时时间（秒）",
    )
    follow_distance: Optional[float] = Field(
        default=None,
        description=(
            "当 frame_id='vision' 时使用，表示移动到视觉目标附近并保持的距离（米）。"
        ),
    )


class PersonTrackingAction(BaseAction):
    """
    人物持续跟随 - 根据 vision 返回的 track_id/person_pos 锁定人物并持续跟随。

    person_pos 是 observe_people 返回的 vision/camera 3D 坐标 [x,y,z]，
    只能原样传给 follow_person 或 frame_id="vision" 的 navigation。
    duration 是最大跟随时间；到期成功结束。move_base_state=ABORTED
    表示路径不可达或被障碍物阻挡，不等于视觉丢失。
    """

    type: Literal["follow_person"] = "follow_person"
    track_id: Optional[int] = Field(
        None,
        description=(
            "vision 返回的人物 track_id。若 observe_people 结果中有 track_id，"
            "调用 follow_person 时应原样传入。"
        ),
    )
    person_pos: Optional[CoordinateInput] = Field(
        None,
        description=(
            "observe_people 返回的人物 vision/camera 3D 位置 [x,y,z]，"
            "用于初始锁定或 track_id 丢失时重匹配；不可当成普通 navigation 的 [x,y,yaw_deg]。"
        ),
    )
    duration: Optional[float] = Field(
        None, description="持续跟随时间（秒）"
    )
    follow_distance: Optional[float] = Field(
        default=None,
        description="跟随时与目标保持的距离（米）",
    )
    timeout: Optional[float] = Field(
        default=None,
        description="跟随导航等待超时时间（秒）",
    )


class StopNavigationAction(BaseAction):
    """
    停止导航状态 - 取消当前导航或跟随任务。
    """

    type: Literal["stop_navigation"] = "stop_navigation"


class RepositionAction(BaseAction):
    """
    微移动/重定位状态 - 用于导航失败恢复、避开近处障碍或调整观察视角。

    这是短距离动作，不应用作长距离导航。导航失败返回 diagnostics 或
    suggested_reposition 时，优先按建议选择 motion；动作后重新 observe
    或 retry 原目标。底盘不支持横移，因此没有 left/right。
    """

    type: Literal["reposition"] = "reposition"
    motion: Literal[
        "forward",
        "backward",
        "turn_left",
        "turn_right",
    ] = Field(
        ...,
        description=(
            "微移动方向。forward/backward 是相对机器人当前朝向的小幅前后移动；"
            "turn_left/turn_right 是原地旋转。不要输出 left/right。"
        ),
    )
    distance: Optional[float] = Field(
        default=None,
        description="平移距离（米）。默认 0.35，导航节点会限制在安全范围内。",
    )
    angle_deg: Optional[float] = Field(
        default=None,
        description="旋转角度（度）。默认 30，导航节点会限制在安全范围内。",
    )
    timeout: Optional[float] = Field(
        default=None,
        description="等待动作完成的超时时间（秒）。默认 8。",
    )
    reason: Optional[str] = Field(
        default=None,
        description="动作原因，仅用于日志，例如 improve_view 或 recover_from_blocked_navigation。",
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


class WaitAction(BaseAction):
    """
    等待状态 - 等待指定时间或事件
    """

    type: Literal["wait"] = "wait"
    duration: float = Field(..., description="等待时间（秒）")
    interruptible: bool = Field(default=True, description="是否可被中断")
