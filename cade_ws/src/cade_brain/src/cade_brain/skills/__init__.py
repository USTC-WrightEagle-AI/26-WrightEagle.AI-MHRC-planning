"""
Skills Toolbox — 函数即工具 (Function as a Tool)

全局工具箱字典 + 注册装饰器 + 硬件上下文单例。

使用方式：
    from cade_brain.skills import register_nav_tool, register_vision_tool

    @register_nav_tool
    def navigation(position: str, **kwargs) -> dict:
        ...

装饰器在函数定义时自动将其写入对应的工具箱字典，
函数名 (__name__) 即为 action.type，零双重维护。
"""

from typing import Dict, Callable, Any

# ── 全局工具箱字典 ──────────────────────────────────────────────

nav_tools: Dict[str, Callable] = {}
vision_tools: Dict[str, Callable] = {}


# ── 注册装饰器 ─────────────────────────────────────────────────

def register_nav_tool(func: Callable) -> Callable:
    """将导航/控制类函数自动注册到 nav_tools 字典"""
    nav_tools[func.__name__] = func
    return func


def register_vision_tool(func: Callable) -> Callable:
    """将视觉感知/计数类函数自动注册到 vision_tools 字典"""
    vision_tools[func.__name__] = func
    return func


# ── 硬件上下文单例 ──────────────────────────────────────────────

class HardwareContext:
    """
    全局 ROS 硬件桥接器（单例）。

    封装 BaseSkill 的 ROS pub/sub 基础设施，
    供所有纯函数共享同一个物理连接。

    延迟初始化：首次调用 execute() 时才创建 BaseSkill 实例，
    确保 rospy.init_node() 已先被调用。
    """

    _instance: "HardwareContext | None" = None

    def __new__(cls) -> "HardwareContext":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._base = None
        return cls._instance

    def _ensure_initialized(self) -> None:
        if self._base is None:
            from cade_brain.skills.base_skill import BaseSkill
            self._base = BaseSkill(node_name_prefix="hardware_ctx")
            print("[HardwareContext] ROS bridge initialized")

    def execute(self, action_type: str, wait_timeout: float = 30.0,
                **params: Any) -> Dict[str, Any]:
        """发布指令并等待结果（委托给 BaseSkill.execute）"""
        self._ensure_initialized()
        return self._base.execute(
            action_type,
            wait_timeout=wait_timeout,
            **params,
        )


class VisionHardwareContext:
    """
    Vision 专用 ROS 桥接器。

    cade_vision task3 使用独立 topic，不能复用导航/通用硬件通道。
    """

    _instance: "VisionHardwareContext | None" = None

    def __new__(cls) -> "VisionHardwareContext":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._base = None
        return cls._instance

    def _ensure_initialized(self) -> None:
        if self._base is None:
            from cade_brain.skills.base_skill import BaseSkill
            self._base = BaseSkill(
                node_name_prefix="vision_hardware_ctx",
                cmd_topic="/cade/task_cmd_task3",
                status_topic="/cade/task_status_task3",
            )
            print("[VisionHardwareContext] ROS bridge initialized")

    def warmup(self) -> None:
        """Create ROS pub/sub handles before the first vision action."""
        self._ensure_initialized()

    def execute(self, action_type: str, wait_timeout: float = 30.0,
                **params: Any) -> Dict[str, Any]:
        """发布视觉指令并等待 cade_vision task3 返回状态。"""
        self._ensure_initialized()
        return self._base.execute(
            action_type,
            wait_timeout=wait_timeout,
            **params,
        )


# ── 导入技能模块以触发装饰器注册 ──────────────────────────────
# 当本包被导入时，自动加载子模块，@register_nav_tool / @register_vision_tool
# 装饰器随即执行，将函数写入 nav_tools / vision_tools 字典。

import cade_brain.skills.nav_skills    # noqa: F401, E402
import cade_brain.skills.vision_skills  # noqa: F401, E402
