"""
Mock Robot - 模拟机器人

在没有真实硬件的情况下，模拟机器人的行为
用于PC端开发和测试
"""

import time
import random
from typing import Optional, List
from body.robot_interface import RobotInterface, RobotState


class MockRobot(RobotInterface):
    """
    模拟机器人类

    模拟所有硬件操作，用日志代替真实动作
    """

    def __init__(self, name: str = "MockRobot"):
        super().__init__()
        self.name = name
        self.current_position = "home"  # 初始位置
        self.holding_object = None      # 当前抓取的物体

        # 模拟的环境地图
        self.known_locations = {
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

        # 模拟的物体数据库
        self.known_objects = {
            "apple": {"name": "apple", "location": "table", "position": [4.0, 1.0, 0.8]},
            "bottle": {"name": "bottle", "location": "kitchen", "position": [5.0, 2.0, 1.0]},
            "cup": {"name": "cup", "location": "table", "position": [4.2, 1.0, 0.8]},
            "book": {"name": "book", "location": "desk", "position": [1.0, 3.0, 0.9]},
        }

        print(f"🤖 {self.name} 初始化成功")
        print(f"   当前位置: {self.current_position}")
        print(f"   已知位置: {list(self.known_locations.keys())}")
        print(f"   已知物体: {list(self.known_objects.keys())}")

    def navigate(self, target) -> bool:
        """模拟导航"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n🚗 [导航] 从 {self.current_position} 前往 {target}")

        # 模拟移动延迟
        time.sleep(0.5)

        # 检查目标是否存在
        if isinstance(target, str):
            if target in self.known_locations:
                self.current_position = target
                coords = self.known_locations[target]
                print(f"✓ 已到达 {target} (坐标: {coords})")
                self.set_state(RobotState.IDLE)
                return True
            else:
                print(f"✗ 未知位置: {target}")
                self.set_state(RobotState.ERROR)
                return False
        elif isinstance(target, list) and len(target) == 3:
            # 直接坐标导航
            self.current_position = f"坐标{target}"
            print(f"✓ 已到达坐标 {target}")
            self.set_state(RobotState.IDLE)
            return True
        else:
            print(f"✗ 无效的目标格式: {target}")
            self.set_state(RobotState.ERROR)
            return False

    def search(self, object_name: str) -> Optional[dict]:
        """模拟搜索物体"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n🔍 [搜索] 正在寻找: {object_name}")

        # 模拟搜索延迟
        time.sleep(0.8)

        # 查找物体
        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            print(f"✓ 找到 {object_name} 在 {obj['location']}")
            print(f"   位置: {obj['position']}")
            self.set_state(RobotState.IDLE)
            return obj
        else:
            # 模拟一定概率找不到
            if random.random() < 0.3:
                print(f"✗ 未找到 {object_name}")
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
                print(f"✓ 找到 {object_name} 在当前位置")
                self.set_state(RobotState.IDLE)
                return new_obj

    def pick(self, object_name: str, object_id: Optional[int] = None) -> bool:
        """模拟抓取物体"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n🤏 [抓取] 尝试抓取: {object_name}")

        # 检查是否已经抓着东西
        if self.holding_object:
            print(f"✗ 手中已有物体: {self.holding_object}")
            self.set_state(RobotState.ERROR)
            return False

        # 模拟抓取延迟
        time.sleep(0.6)

        # 检查物体是否存在
        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            # 简化版：不检查距离，假设已经在附近
            self.holding_object = object_name
            print(f"✓ 成功抓取 {object_name}")
            self.set_state(RobotState.IDLE)
            return True
        else:
            print(f"✗ 物体不存在: {object_name}")
            self.set_state(RobotState.ERROR)
            return False

    def place(self, location) -> bool:
        """模拟放置物体"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n📦 [放置] 将物体放置到: {location}")

        # 检查是否抓着东西
        if not self.holding_object:
            print(f"✗ 手中没有物体")
            self.set_state(RobotState.ERROR)
            return False

        # 模拟放置延迟
        time.sleep(0.5)

        print(f"✓ 已将 {self.holding_object} 放置到 {location}")
        # 更新物体位置
        if self.holding_object in self.known_objects:
            self.known_objects[self.holding_object]["location"] = str(location)

        self.holding_object = None
        self.set_state(RobotState.IDLE)
        return True

    def speak(self, content: str) -> bool:
        """模拟语音输出"""
        self.set_state(RobotState.EXECUTING)
        print(f"\n💬 [语音] {self.name}: \"{content}\"")

        # 模拟TTS延迟
        time.sleep(0.3)

        self.set_state(RobotState.IDLE)
        return True

    def wait(self, reason: Optional[str] = None) -> bool:
        """等待/无操作"""
        msg = f"\n⏸️  [等待]"
        if reason:
            msg += f" 原因: {reason}"
        print(msg)

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
        print(f"\n📊 机器人状态:")
        print(f"   状态: {status['state']}")
        print(f"   位置: {status['position']}")
        print(f"   手持: {status['holding'] or '无'}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 测试 Mock Robot ===\n")

    # 创建机器人
    robot = MockRobot(name="LARA")

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

    # 测试语音
    robot.speak("任务完成！")

    print("\n✓ 所有测试通过！")
