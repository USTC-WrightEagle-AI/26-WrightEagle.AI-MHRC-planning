"""
Base Skill - 技能基类

所有技能共享 ROS 发布/等待模式：
1. 预先在 __init__ 注册状态订阅者（主线程，ROS master 认可）
2. execute() 清空缓存 → 发布命令 → 轮询等待回调结果
"""

import json
import threading
import time
import uuid
from typing import Optional, Dict, Any

try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    String = Any


class BaseSkill:
    """技能基类 - 提供 ROS pub/sub 基础设施"""

    def __init__(
        self,
        node_name_prefix: str = "skill",
        cmd_topic: str = "/cade/task_cmd",
        status_topic: str = "/cade/task_status",
    ):
        self._cmd_pub = None
        self._status_sub = None
        self._last_status = None
        self._node_initialized = False
        self._cmd_topic = cmd_topic
        self._status_topic = status_topic
        self._execute_lock = threading.Lock()
        self._status_cv = threading.Condition()
        self._active_request_id = None
        if ROS_AVAILABLE:
            self._init_ros(node_name_prefix)

    def _init_ros(self, prefix: str):
        """初始化 ROS 通信（运行在主线程，ROS master 认可）"""
        try:
            self._cmd_pub = rospy.Publisher(self._cmd_topic, String, queue_size=10)
            self._status_sub = rospy.Subscriber(
                self._status_topic, String, self._on_status
            )
            self._node_initialized = True
        except Exception as e:
            print(f"[BaseSkill] ROS 初始化失败: {e}")

    def _on_status(self, msg: String):
        """订阅者回调，将最新状态存入 self._last_status"""
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            status = {"status": "ERROR", "error": "Invalid JSON"}

        with self._status_cv:
            request_id = status.get("request_id")
            active_request_id = self._active_request_id

            if active_request_id is None:
                print(f"[Skill] Ignoring stale status with no active request: {status}")
                return

            if request_id is not None and request_id != active_request_id:
                print(
                    "[Skill] Ignoring status for request_id=%s while waiting for %s: %s"
                    % (request_id, active_request_id, status)
                )
                return

            self._last_status = status
            print(f"[Skill] Received status: {self._last_status}")
            self._status_cv.notify_all()

    def _publish_command(self, action_type: str, request_id: str, **params) -> None:
        """发布任务指令到配置的 command topic。"""
        cmd = json.dumps({"action": action_type, "request_id": request_id, **params})
        if self._cmd_pub is not None:
            self._wait_for_command_subscribers()
            msg = String()
            msg.data = cmd
            self._cmd_pub.publish(msg)
            print(f"[Skill] Published to {self._cmd_topic}: {cmd}")
        else:
            print(f"[Skill] (no ROS) Would publish: {cmd}")

    def _wait_for_command_subscribers(self, timeout: float = 2.0) -> None:
        """
        Wait briefly for ROS TCP connections before publishing.

        rospy can drop the first message if a Publisher is created and used
        immediately. This matters for lazily initialized skill contexts.
        """
        if self._cmd_pub is None or not hasattr(self._cmd_pub, "get_num_connections"):
            return

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self._cmd_pub.get_num_connections() > 0:
                    return
            except Exception:
                return

            if hasattr(rospy, "sleep"):
                rospy.sleep(0.05)
            else:
                time.sleep(0.05)

        print(
            f"[Skill WARNING] No subscribers connected to {self._cmd_topic}; "
            "publishing anyway"
        )

    def execute(
        self,
        action_type: str,
        wait_timeout: float = 30.0,
        **params,
    ) -> Dict[str, Any]:
        """
        执行一个技能：清空缓存 → 发布指令 → 轮询等待回调结果

        Subscriber 已在 __init__ 中预先注册（主线程），execute 可在任意
        线程调用而不会出现 ROS master 不认 daemon 线程 subscriber 的问题。

        Args:
            action_type: 动作类型
            wait_timeout: 等待状态回复的超时（秒）
            **params: 动作参数

        Returns:
            dict: {"status": "SUCCESS"|"FAILED"|"TIMEOUT", "result": ...}
        """
        if not ROS_AVAILABLE:
            print("[Skill] (no ROS) Simulating task status: SUCCESS")
            return {"status": "SUCCESS", "result": "simulated"}

        request_id = params.pop("request_id", None) or uuid.uuid4().hex

        with self._execute_lock:
            with self._status_cv:
                self._last_status = None
                self._active_request_id = request_id

            try:
                self._publish_command(action_type, request_id=request_id, **params)

                deadline = time.time() + float(wait_timeout)
                with self._status_cv:
                    while self._last_status is None:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        self._status_cv.wait(timeout=min(0.1, remaining))

                    if self._last_status is not None:
                        return self._last_status

                print(f"[Skill] Timeout waiting for status (>{wait_timeout}s)")
                return {
                    "status": "TIMEOUT",
                    "request_id": request_id,
                    "error": f"No response within {wait_timeout}s",
                }
            finally:
                with self._status_cv:
                    self._active_request_id = None
