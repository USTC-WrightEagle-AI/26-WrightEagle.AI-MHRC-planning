"""
Vision Skills — 视觉感知/计数类原子动作（纯函数）

每个函数对应一个 action.type，通过 @register_vision_tool 自动注册到 vision_tools。
函数名即动作类型名，零双重维护。
"""

from typing import Any, Dict, Optional

from cade_brain.skills import HardwareContext, register_vision_tool

# ── 底层 ROS 指令封装（内部复用） ───────────────────────────────


def _execute(action_type: str, timeout: float = 30.0, **params: Any) -> Dict[str, Any]:
    """通过 HardwareContext 发布 ROS 指令并等待结果"""
    return HardwareContext().execute(action_type, timeout=timeout, **params)


# ── 视觉感知类动作 ─────────────────────────────────────────────


@register_vision_tool
def find_object(target: str, **kwargs) -> Dict[str, Any]:
    """
    寻找物体的核心物理技能
    """
    # 完美对齐底层的契约
    payload = {
        "action": "find_object",
        "target": target.lower(),
    }
    # 发送到 /cade/task_cmd 并等待返回状态
    return _execute(payload["action"], timeout=45.0, **payload)


@register_vision_tool
def find_person(
    target: str = "person",
    gesture: Optional[str] = None,
    cloth_color: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    寻找人类的核心物理技能（支持手势与衣服颜色动态注入）
    """
    # 组装底层代码需要的 attributes 字典
    attributes = {}
    if gesture:
        attributes["posture"] = gesture  # 对齐底层对于手势/姿态的解析
    if cloth_color:
        attributes["cloth_type"] = cloth_color  # 关键：对齐底层的 cloth_type 注入逻辑

    payload = {
        "action": "find_person",
        "target": target.lower(),
        "attributes": attributes if attributes else None,
    }

    # 发送到话题，点火！
    return _execute(payload["action"], timeout=45.0, **payload)


@register_vision_tool
def count_objects(category: str, **kwargs) -> Dict[str, Any]:
    """
    统计特定物体的数量
    """
    payload = {
        "action": "count_objects",
        "category": category.lower(),
        "attributes": None,
    }
    return _execute(payload["action"], timeout=45.0, **payload)


@register_vision_tool
def count_people(
    gesture: Optional[str] = None, cloth_color: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    统计特定条件的人数（完美契合底层的 category='person' 和 attributes 注入）
    """
    # 组装底层代码需要的 attributes 字典
    attributes = {}
    if gesture:
        attributes["posture"] = gesture
    if cloth_color:
        attributes["cloth_type"] = cloth_color

    payload = {
        "action": "count_people",
        "category": "person",  # 显式、死死固定为底层需要的 'person'
        "attributes": attributes if attributes else None,
    }

    return _execute(payload["action"], timeout=45.0, **payload)


@register_vision_tool
def name_recognition(
    name: str, bind_to_appearance: bool = True, **kwargs
) -> Dict[str, Any]:
    """通过名字识别人物（获取最近人物属性并附带 name）"""
    result = _execute("get_nearest_person", timeout=5.0)
    if result and result.get("status") == "SUCCESS":
        result["name"] = name
        return result
    return result or {"status": "FAILED", "error": "No person detected nearby"}



