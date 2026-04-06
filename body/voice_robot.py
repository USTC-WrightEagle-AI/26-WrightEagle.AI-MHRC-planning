"""
Voice Robot - 语音专用躯干

继承自 RobotInterface，实现一个只说不做的机器人体
用于语音交互场景，所有物理动作虚拟化
"""

import rospy
from typing import Optional
from body.robot_interface import RobotInterface, RobotState


class VoiceRobot(RobotInterface):
    """
    语音机器人

    特点：
    - 语音输出通过 ROS 话题 /tts
    - 物理动作虚拟化（仅日志记录）
    - 状态管理支持回声抑制
    """

    def __init__(self, name: str = "CADE"):
        """
        初始化语音机器人

        Args:
            name: 机器人名称
        """
        super().__init__()
        self.name = name
        self.current_position = "lab"  # 虚拟位置
        self.holding_object = None

        rospy.loginfo(f"[{self.name}] VoiceRobot 初始化成功")
        rospy.loginfo(f"  模式: 纯语音交互（无物理动作）")

    def navigate(self, target) -> bool:
        """
        虚拟导航 - 仅记录日志

        Args:
            target: 目标位置

        Returns:
            bool: 始终返回 True
        """
        self.set_state(RobotState.EXECUTING)
        rospy.loginfo(f"[NAVIGATE] 虚拟导航 -> {target}")
        self.current_position = str(target)
        self.set_state(RobotState.IDLE)
        return True

    def search(self, object_name: str) -> Optional[dict]:
        """
        虚拟搜索 - 返回虚拟结果

        Args:
            object_name: 物体名称

        Returns:
            dict: 虚拟物体信息
        """
        self.set_state(RobotState.EXECUTING)
        rospy.loginfo(f"[SEARCH] 虚拟搜索: {object_name}")

        # 返回虚拟物体信息
        result = {
            "name": object_name,
            "location": self.current_position,
            "position": [0.5, 0.5, 1.0]  # 虚拟坐标
        }

        self.set_state(RobotState.IDLE)
        return result

    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        """
        虚拟抓取 - 仅记录日志

        Args:
            object_name: 物体名称
            object_id: 物体ID

        Returns:
            bool: 始终返回 True
        """
        self.set_state(RobotState.EXECUTING)
        rospy.loginfo(f"[PICK] 虚拟抓取: {object_name}")

        self.holding_object = object_name
        self.set_state(RobotState.IDLE)
        return True

    def place(self, location) -> bool:
        """
        虚拟放置 - 仅记录日志

        Args:
            location: 放置位置

        Returns:
            bool: 始终返回 True
        """
        self.set_state(RobotState.EXECUTING)
        rospy.loginfo(f"[PLACE] 虚拟放置 -> {location}")

        self.holding_object = None
        self.set_state(RobotState.IDLE)
        return True

    def speak(self, content: str) -> bool:
        """
        语音输出 - 由 RosVoiceBridge 处理实际的 TTS

        注意：此方法不直接发布到 /tts 话题
        实际的 TTS 发布由 RosVoiceBridge 在收到 LLM 回复后执行

        Args:
            content: 要说的内容

        Returns:
            bool: 始终返回 True
        """
        self.set_state(RobotState.SPEAKING)
        rospy.loginfo(f"[SPEAK] 语音输出: \"{content}\"")

        # 注意：不在这里发布 TTS，由 Bridge 统一处理
        self.set_state(RobotState.IDLE)
        return True

    def wait(self, reason: Optional[str] = None) -> bool:
        """
        等待 - 仅记录日志

        Args:
            reason: 等待原因

        Returns:
            bool: 始终返回 True
        """
        msg = f"[WAIT] 等待中"
        if reason:
            msg += f" 原因: {reason}"
        rospy.loginfo(msg)

        self.set_state(RobotState.IDLE)
        return True

    def get_status(self) -> dict:
        """获取机器人状态"""
        return {
            "name": self.name,
            "state": self.state.value,
            "position": self.current_position,
            "holding": self.holding_object,
        }

    def is_busy(self) -> bool:
        """
        检查机器人是否忙碌

        用于回声抑制：当机器人正在思考或说话时，
        应该忽略 ASR 输入

        Returns:
            bool: True 表示忙碌，不应处理新的语音输入
        """
        return self.state in (RobotState.THINKING, RobotState.SPEAKING, RobotState.EXECUTING)
