"""
Base Skill - 技能基类

所有技能共享 ROS 发布/等待模式：
1. 发布 JSON 指令到 /cade/task_cmd
2. 等待 /cade/task_status 返回结果
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
        self._node_initialized = False
        if ROS_AVAILABLE:
            self._init_ros(node_name_prefix)

    def _init_ros(self, prefix: str):
        """延迟初始化 ROS 通信"""
        try:
            self._cmd_pub = rospy.Publisher('/cade/task_cmd', String, queue_size=10)
            self._node_initialized = True
        except Exception as e:
            print(f"[BaseSkill] ROS 初始化失败: {e}")

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

    def _wait_for_status(self, timeout: float = 30.0) -> Dict[str, Any]:
        """等待 /cade/task_status 返回结果"""
        if not ROS_AVAILABLE:
            print("[Skill] (no ROS) Simulating task status: SUCCESS")
            return {"status": "SUCCESS", "result": "simulated"}

        try:
            msg = rospy.wait_for_message('/cade/task_status', String, timeout=timeout)
            result = json.loads(msg.data)
            print(f"[Skill] Received status: {result}")
            return result
        except rospy.ROSException as e:
            print(f"[Skill] Timeout waiting for status: {e}")
            return {"status": "TIMEOUT", "error": str(e)}

    def execute(self, action_type: str, timeout: float = 30.0, **params) -> Dict[str, Any]:
        """
        执行一个技能：发布指令 + 等待结果

        Args:
            action_type: 动作类型
            timeout: 等待超时（秒）
            **params: 动作参数

        Returns:
            dict: {"status": "SUCCESS"|"FAILED"|"TIMEOUT", "result": ...}
        """
        self._publish_command(action_type, **params)
        return self._wait_for_status(timeout=timeout)
