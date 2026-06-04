"""
Navigation Skills — 导航/控制类原子动作（纯函数）

每个函数对应一个 action.type，通过 @register_nav_tool 自动注册到 nav_tools。
函数名即动作类型名，零双重维护。
"""

from typing import Any, Dict, Optional

from cade_brain.skills import HardwareContext, register_nav_tool

# ── 底层 ROS 指令封装（内部复用） ───────────────────────────────


def _execute(action_type: str, timeout: float = 30.0, **params: Any) -> Dict[str, Any]:
    """通过 HardwareContext 发布 ROS 指令并等待结果"""
    return HardwareContext().execute(action_type, timeout=timeout, **params)


# ── 导航类动作 ──────────────────────────────────────────────────


@register_nav_tool
def navigation(position: str, **kwargs) -> Dict[str, Any]:
    """导航到目标位置"""

    return _execute("navigation", timeout=1000.0, target=position)


@register_nav_tool
def follow_person(
    person_pos: Optional[str] = None,
    duration: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """追踪/跟随人物"""
    params: Dict[str, Any] = {}
    if person_pos:
        params["location"] = person_pos
    timeout = (duration or 120.0) + 10
    return _execute("follow_person", timeout=timeout, **params)


@register_nav_tool
def bringMeObj(
    object_name: str,
    object_position: Optional[Any] = None,
    grasp_force: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """抓取物体"""
    return _execute(
        "bringMeObj", timeout=120.0, object_name=object_name, placement=object_position
    )


@register_nav_tool
def object_dump(
    target_position: Optional[Any] = None, release_safe: bool = True, **kwargs
) -> Dict[str, Any]:
    """放置/释放物体"""
    return _execute(
        "object_dump",
        timeout=60.0,
        target_position=target_position,
        release_safe=release_safe,
    )


@register_nav_tool
def talking(
    text: str = "", wait_for_response: bool = False, **kwargs
) -> Dict[str, Any]:
    """对话/语音输出（本地处理，无 ROS 调用）"""
    return {"status": "SUCCESS", "result": text}


@register_nav_tool
def wait(duration: float = 5.0, interruptible: bool = True, **kwargs) -> Dict[str, Any]:
    """等待指定时间"""
    return _execute(
        "wait", timeout=duration + 10, duration=duration, interruptible=interruptible
    )


@register_nav_tool
def listening(
    continuous: bool = True,
    vad_enabled: bool = True,
    wake_word: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """进入监听模式"""
    params: Dict[str, Any] = {
        "continuous": continuous,
        "vad_enabled": vad_enabled,
    }
    if wake_word:
        params["wake_word"] = wake_word
    return _execute("listening", timeout=30.0, **params)


@register_nav_tool
def idle(low_power_mode: bool = False, summary: str = "", **kwargs) -> Dict[str, Any]:
    """进入空闲状态（本地处理，无 ROS 调用）"""
    return {"status": "SUCCESS", "result": summary or "idle"}
