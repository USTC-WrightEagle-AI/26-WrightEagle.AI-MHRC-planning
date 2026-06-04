"""
Base Skill - 技能基类

所有技能共享 ROS 发布/等待模式：
1. 预先在 __init__ 注册 /cade/task_status 订阅者（主线程，ROS master 认可）
2. execute() 清空缓存 → 发布 /cade/task_cmd → 轮询等待回调结果
"""

import json
import time
from typing import Optional, Dict, Any

try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class BaseSkill:
    """技能基类 - 提供 ROS pub/sub 基础设施"""

    def __init__(self, node_name_prefix: str = "skill"):
        self._cmd_pub = None
        self._status_sub = None
        self._last_status = None
        self._node_initialized = False
        if ROS_AVAILABLE:
            self._init_ros(node_name_prefix)

    def _init_ros(self, prefix: str):
        """初始化 ROS 通信（运行在主线程，ROS master 认可）"""
        try:
            self._cmd_pub = rospy.Publisher('/cade/task_cmd', String, queue_size=10)
            self._status_sub = rospy.Subscriber(
                '/cade/task_status', String, self._on_status
            )
            self._node_initialized = True
        except Exception as e:
            print(f"[BaseSkill] ROS 初始化失败: {e}")

    def _on_status(self, msg: String):
        """订阅者回调，将最新状态存入 self._last_status"""
        try:
            self._last_status = json.loads(msg.data)
            print(f"[Skill] Received status: {self._last_status}")
        except json.JSONDecodeError:
            self._last_status = {"status": "ERROR", "error": "Invalid JSON"}

    def _publish_command(self, action_type: str, **params) -> None:
        """发布任务指令到 /cade/task_cmd"""
        cmd = json.dumps({"action": action_type, **params})
        if self._cmd_pub is not None:
            msg = String()
            msg.data = cmd
            self._cmd_pub.publish(msg)
            print(f"[Skill] Published to /cade/task_cmd: {cmd}")
        else:
            print(f"[Skill] (no ROS) Would publish: {cmd}")

    def execute(self, action_type: str, timeout: float = 30.0, **params) -> Dict[str, Any]:
        """
        执行一个技能：清空缓存 → 发布指令 → 轮询等待回调结果

        Subscriber 已在 __init__ 中预先注册（主线程），execute 可在任意
        线程调用而不会出现 ROS master 不认 daemon 线程 subscriber 的问题。

        Args:
            action_type: 动作类型
            timeout: 等待超时（秒）
            **params: 动作参数

        Returns:
            dict: {"status": "SUCCESS"|"FAILED"|"TIMEOUT", "result": ...}
        """
        if not ROS_AVAILABLE:
            print("[Skill] (no ROS) Simulating task status: SUCCESS")
            return {"status": "SUCCESS", "result": "simulated"}

        # 清空上次缓存，避免读到旧消息
        self._last_status = None
        self._publish_command(action_type, **params)

        # 轮询等待回调写入 self._last_status
        poll_interval = 0.05
        elapsed = 0.0
        while self._last_status is None and elapsed < timeout:
            if hasattr(rospy, 'sleep'):
                rospy.sleep(poll_interval)
            else:
                time.sleep(poll_interval)
            elapsed += poll_interval

        if self._last_status is not None:
            return self._last_status
        else:
            print(f"[Skill] Timeout waiting for status (>{timeout}s)")
            return {"status": "TIMEOUT", "error": f"No response within {timeout}s"}
