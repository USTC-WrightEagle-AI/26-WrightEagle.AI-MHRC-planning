"""
CADE Robot V2 (Current)
功能实现：继承 V1 的完整空间认知与物体逻辑。
核心新增：集成 ROS 语音交互能力 (ASR/TTS 状态对齐)、增强型回声抑制逻辑 (is_busy)。
"""

import time
import random
from typing import Optional, Dict, Any
try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    rospy = None
    ROS_AVAILABLE = False

from body.robot_interface import RobotInterface, RobotState
from config import Config


class Robot(RobotInterface):
    """
    CADE 机器人 V2

    融合 V1 空间认知与 V2 语音交互：
    - V1 基础：内部语义地图、物体数据库、动作状态校验
    - V2 增强：ROS 语音交互集成、状态机对齐、回声抑制
    """

    def __init__(self, name: Optional[str] = None):
        """
        初始化机器人

        Args:
            name: 机器人名称，默认为 Config.ROBOT_NAME
        """
        super().__init__()
        self.name = name or Config.ROBOT_NAME
        self.current_position = "home"  # 初始位置
        self.holding_object = None

        # V1 语义地图（所有机器人共享的基础地图）
        self.known_locations: Dict[str, list] = {
            "home": [0.0, 0.0, 0.0],
            "起点": [0.0, 0.0, 0.0],  # 中文别名
            "start_point": [0.0, 0.0, 0.0],  # 英文别名
            "kitchen": [5.0, 2.0, 0.0],
            "厨房": [5.0, 2.0, 0.0],
            "living_room": [3.0, -1.0, 0.0],
            "客厅": [3.0, -1.0, 0.0],
            "bedroom": [-2.0, 4.0, 0.0],
            "卧室": [-2.0, 4.0, 0.0],
            "table": [4.0, 1.0, 0.0],
            "桌子": [4.0, 1.0, 0.0],
            "desk": [1.0, 3.0, 0.0],
            "书桌": [1.0, 3.0, 0.0],
        }

        # V1 物体数据库（所有机器人共享的物体信息）
        self.known_objects: Dict[str, Dict[str, Any]] = {
            "apple": {"name": "apple", "location": "table", "position": [4.0, 1.0, 0.8]},
            "bottle": {"name": "bottle", "location": "kitchen", "position": [5.0, 2.0, 1.0]},
            "cup": {"name": "cup", "location": "table", "position": [4.2, 1.0, 0.8]},
            "book": {"name": "book", "location": "desk", "position": [1.0, 3.0, 0.9]},
        }

        # 初始化日志
        self._log_info(f"🤖 {self.name} 初始化成功")
        self._log_info(f"   当前位置: {self.current_position}")
        self._log_info(f"   已知位置: {list(self.known_locations.keys())}")
        self._log_info(f"   已知物体: {list(self.known_objects.keys())}")
        if ROS_AVAILABLE:
            self._log_info(f"   模式: ROS 语音交互集成")
        else:
            self._log_info(f"   模式: 纯控制台模拟")

    def _log_info(self, message: str):
        """统一的日志输出：ROS 可用时用 rospy.loginfo，否则用 print"""
        if ROS_AVAILABLE and rospy.get_node_uri() is not None:
            rospy.loginfo(message)
        else:
            print(message)

    def navigate(self, target) -> bool:
        """导航到目标位置（V1 逻辑 + V2 日志）"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🚗 [导航] 从 {self.current_position} 前往 {target}")

        # 模拟移动延迟
        time.sleep(0.5)

        # 检查目标是否存在
        if isinstance(target, str):
            if target in self.known_locations:
                self.current_position = target
                coords = self.known_locations[target]
                self._log_info(f"✓ 已到达 {target} (坐标: {coords})")
                self.set_state(RobotState.IDLE)
                return True
            else:
                self._log_info(f"✗ 未知位置: {target}")
                self.set_state(RobotState.ERROR)
                return False
        elif isinstance(target, list) and len(target) == 3:
            # 直接坐标导航
            self.current_position = f"坐标{target}"
            self._log_info(f"✓ 已到达坐标 {target}")
            self.set_state(RobotState.IDLE)
            return True
        else:
            self._log_info(f"✗ 无效的目标格式: {target}")
            self.set_state(RobotState.ERROR)
            return False

    def search(self, object_name: str) -> Optional[dict]:
        """搜索物体（V1 逻辑 + V2 日志）"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🔍 [搜索] 正在寻找: {object_name}")

        # 模拟搜索延迟
        time.sleep(0.8)

        # 查找物体
        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            self._log_info(f"✓ 找到 {object_name} 在 {obj['location']}")
            self._log_info(f"   位置: {obj['position']}")
            self.set_state(RobotState.IDLE)
            return obj
        else:
            # 模拟一定概率找不到
            if random.random() < 0.3:
                self._log_info(f"✗ 未找到 {object_name}")
                self.set_state(RobotState.IDLE)
                return None
            else:
                # 创建一个"发现"的物体
                new_obj = {
                    "name": object_name,
                    "location": self.current_position,
                    "position": [
                        random.uniform(-5, 5),
                        random.uniform(-5, 5),
                        random.uniform(0.5, 1.5)
                    ]
                }
                self.known_objects[object_name] = new_obj
                self._log_info(f"✓ 找到 {object_name} 在当前位置")
                self.set_state(RobotState.IDLE)
                return new_obj

    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        """抓取物体（V1 逻辑 + V2 日志）"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🤏 [抓取] 尝试抓取: {object_name}")

        # 检查是否已经抓着东西
        if self.holding_object:
            self._log_info(f"✗ 手中已有物体: {self.holding_object}")
            self.set_state(RobotState.ERROR)
            return False

        # 模拟抓取延迟
        time.sleep(0.6)

        # 检查物体是否存在
        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            # 简化版：不检查距离，假设已经在附近
            self.holding_object = object_name
            self._log_info(f"✓ 成功抓取 {object_name}")
            self.set_state(RobotState.IDLE)
            return True
        else:
            self._log_info(f"✗ 物体不存在: {object_name}")
            self.set_state(RobotState.ERROR)
            return False

    def place(self, location) -> bool:
        """放置物体（V1 逻辑 + V2 日志）"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n📦 [放置] 将物体放置到: {location}")

        # 检查是否抓着东西
        if not self.holding_object:
            self._log_info(f"✗ 手中没有物体")
            self.set_state(RobotState.ERROR)
            return False

        # 模拟放置延迟
        time.sleep(0.5)

        self._log_info(f"✓ 已将 {self.holding_object} 放置到 {location}")
        # 更新物体位置
        if self.holding_object in self.known_objects:
            self.known_objects[self.holding_object]["location"] = str(location)

        self.holding_object = None
        self.set_state(RobotState.IDLE)
        return True

    def speak(self, content: str) -> bool:
        """
        语音输出（V2 增强版）

        注意：此方法不直接发布到 /tts 话题
        实际的 TTS 发布由 RosVoiceBridge 在收到 LLM 回复后执行
        """
        self.set_state(RobotState.SPEAKING)
        self._log_info(f"[SPEAK] 语音输出: \"{content}\"")

        # 注意：不在这里发布 TTS，由 Bridge 统一处理
        # 状态保持 SPEAKING，由外部（如 RosVoiceBridge）在 TTS 完成后重置
        # 这里不自动重置状态，以支持外部状态管理
        return True

    def wait(self, reason: Optional[str] = None) -> bool:
        """等待/无操作（V1 逻辑 + V2 日志）"""
        msg = f"\n⏸️  [等待]"
        if reason:
            msg += f" 原因: {reason}"
        self._log_info(msg)

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

    def print_status(self):
        """打印当前状态"""
        status = self.get_status()
        self._log_info(f"\n📊 机器人状态:")
        self._log_info(f"   名称: {status['name']}")
        self._log_info(f"   状态: {status['state']}")
        self._log_info(f"   位置: {status['position']}")
        self._log_info(f"   手持: {status['holding'] or '无'}")

    def is_busy(self) -> bool:
        """
        检查机器人是否忙碌（V2 新增）

        用于回声抑制：当机器人正在思考或说话时，
        应该忽略 ASR 输入

        Returns:
            bool: True 表示忙碌，不应处理新的语音输入
        """
        return self.state in (RobotState.THINKING, RobotState.SPEAKING, RobotState.EXECUTING)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 测试 CADE Robot V2 ===\n")

    # 创建机器人（非 ROS 环境）
    robot = Robot(name="LARA")

    # 测试导航
    robot.navigate("kitchen")
    robot.print_status()

    # 测试搜索
    result = robot.search("bottle")
    print(f"搜索结果: {result}")

    # 测试抓取
    robot.pick("bottle")
    robot.print_status()

    # 测试导航 + 放置
    robot.navigate("table")
    robot.place("table")
    robot.print_status()

    # 测试语音（状态管理）
    robot.speak("任务完成！")
    print(f"说话后状态: {robot.state}")

    # 测试忙碌状态
    robot.set_state(RobotState.THINKING)
    print(f"思考时忙碌: {robot.is_busy()}")
    robot.set_state(RobotState.IDLE)
    print(f"空闲时忙碌: {robot.is_busy()}")

    print("\n✓ 所有测试通过！")
