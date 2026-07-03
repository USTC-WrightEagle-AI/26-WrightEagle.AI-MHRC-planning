"""
Navigation Skills — 导航/控制类原子动作（纯函数）

每个函数对应一个 action.type，通过 @register_nav_tool 自动注册到 nav_tools。
函数名即动作类型名，零双重维护。
"""

from typing import Any, Dict, Optional

from cade_brain.skills import HardwareContext, register_nav_tool

# ── 底层 ROS 指令封装（内部复用） ───────────────────────────────


def _execute(
    action_type: str,
    wait_timeout: float = 30.0,
    **params: Any,
) -> Dict[str, Any]:
    """通过 HardwareContext 发布 ROS 指令并等待结果"""
    return HardwareContext().execute(
        action_type,
        wait_timeout=wait_timeout,
        **params,
    )


def _normalize_navigation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Treat bare vision 3D targets as camera coordinates unless caller says otherwise."""
    if (
        params.get("frame_id") is not None
        or params.get("source_frame") is not None
        or params.get("yaw_deg") is not None
    ):
        return params

    position = params.get("position")
    vision_position = _extract_vision_position(position)
    if vision_position is None:
        return params

    params["position"] = vision_position
    params["frame_id"] = "vision"
    return params


def _extract_vision_position(value: Any) -> Optional[Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    if isinstance(value, dict):
        point = value.get("position_3d")
        if _is_numeric_triplet(point):
            return [float(v) for v in point]
        if _is_xyz_dict(value):
            return [float(value["x"]), float(value["y"]), float(value["z"])]
        return None

    if _is_numeric_triplet(value):
        return [float(v) for v in value]
    return None


def _is_numeric_triplet(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return True


def _is_xyz_dict(value: Dict[str, Any]) -> bool:
    if not all(key in value for key in ("x", "y", "z")):
        return False
    try:
        float(value["x"])
        float(value["y"])
        float(value["z"])
    except (TypeError, ValueError):
        return False
    return True


# ── 导航类动作 ──────────────────────────────────────────────────


@register_nav_tool
def navigation(
    position: Any,
    timeout: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """导航到目标位置"""
    params: Dict[str, Any] = {"position": position}
    if timeout is not None:
        params["timeout"] = timeout
    params.update({key: value for key, value in kwargs.items() if value is not None})
    params = _normalize_navigation_params(params)

    wait_timeout = float(timeout) + 10.0 if timeout is not None else 1000.0
    return _execute("navigation", wait_timeout=wait_timeout, **params)


@register_nav_tool
def follow_person(
    person_pos: Optional[Any] = None,
    duration: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """追踪/跟随人物"""
    params: Dict[str, Any] = {}
    if person_pos is not None:
        params["person_pos"] = person_pos
    if duration is not None:
        params["duration"] = duration
    params.update({key: value for key, value in kwargs.items() if value is not None})

    timeout = (duration or 120.0) + 10
    return _execute("follow_person", wait_timeout=timeout, **params)


@register_nav_tool
def stop_navigation(**kwargs) -> Dict[str, Any]:
    """停止当前导航任务"""
    return _execute("stop_navigation", wait_timeout=5.0)


@register_nav_tool
def reposition(
    motion: str,
    distance: Optional[float] = None,
    angle_deg: Optional[float] = None,
    timeout: Optional[float] = None,
    reason: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """短距离重定位，用于导航恢复、避障或调整观察视角；不支持横移。"""
    params: Dict[str, Any] = {"motion": motion}
    if distance is not None:
        params["distance"] = distance
    if angle_deg is not None:
        params["angle_deg"] = angle_deg
    if timeout is not None:
        params["timeout"] = timeout
    if reason:
        params["reason"] = reason
    params.update({key: value for key, value in kwargs.items() if value is not None})

    wait_timeout = float(timeout) + 2.0 if timeout is not None else 10.0
    return _execute("reposition", wait_timeout=wait_timeout, **params)


@register_nav_tool
def bringMeObj(
    object_name: str,
    object_position: Optional[Any] = None,
    grasp_force: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """抓取物体"""
    params: Dict[str, Any] = {
        "object_name": object_name,
        "placement": object_position,
    }
    if grasp_force is not None:
        params["grasp_force"] = grasp_force
    return _execute("bringMeObj", wait_timeout=120.0, **params)


@register_nav_tool
def object_dump(
    target_position: Optional[Any] = None, release_safe: bool = True, **kwargs
) -> Dict[str, Any]:
    """放置/释放物体"""
    return _execute(
        "object_dump",
        wait_timeout=60.0,
        target_position=target_position,
        release_safe=release_safe,
    )


@register_nav_tool
def wait(duration: float = 5.0, interruptible: bool = True, **kwargs) -> Dict[str, Any]:
    """等待指定时间"""
    return _execute(
        "wait",
        wait_timeout=duration + 10,
        duration=duration,
        interruptible=interruptible,
    )
