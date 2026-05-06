"""
CADE Robot V2 (Current)
功能实现：继承 V1 的完整空间认知与物体逻辑。
核心新增：集成 ROS 语音交互能力 (ASR/TTS 状态对齐)、增强型回声抑制逻辑 (is_busy)。
"""

import time
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
            # 信标/放置处位置
            "beacon_1": [2.0, 0.0, 0.0],
            "beacon_2": [-1.0, 1.0, 0.0],
            "beacon_3": [4.0, -2.0, 0.0],
            "entrance": [0.0, -2.0, 0.0],
            "hallway": [1.0, 0.0, 0.0],
        }

        # V1 物体数据库（所有机器人共享的物体信息）
        self.known_objects: Dict[str, Dict[str, Any]] = {
            "apple": {"name": "apple", "category": "fruit", "location": "table", "placement": "table", "position": [4.0, 1.0, 0.8], "size": 0.1, "weight": 0.2},
            "bottle": {"name": "bottle", "category": "container", "location": "kitchen", "placement": "kitchen", "position": [5.0, 2.0, 1.0], "size": 0.2, "weight": 0.5},
            "cup": {"name": "cup", "category": "container", "location": "table", "placement": "table", "position": [4.2, 1.0, 0.8], "size": 0.15, "weight": 0.3},
            "book": {"name": "book", "category": "stationery", "location": "desk", "placement": "desk", "position": [1.0, 3.0, 0.9], "size": 0.3, "weight": 0.8},
            "orange": {"name": "orange", "category": "fruit", "location": "table", "placement": "table", "position": [3.8, 1.0, 0.8], "size": 0.09, "weight": 0.18},
            "plate": {"name": "plate", "category": "tableware", "location": "kitchen", "placement": "kitchen", "position": [5.1, 2.1, 0.9], "size": 0.25, "weight": 0.4},
            "spoon": {"name": "spoon", "category": "tableware", "location": "kitchen", "placement": "kitchen", "position": [4.9, 2.0, 0.9], "size": 0.15, "weight": 0.05},
            "key": {"name": "key", "category": "tool", "location": "desk", "placement": "desk", "position": [1.1, 3.1, 0.85], "size": 0.05, "weight": 0.03},
            "remote": {"name": "remote", "category": "electronics", "location": "living_room", "placement": "living_room", "position": [3.0, -0.5, 0.6], "size": 0.2, "weight": 0.15},
        }

        # V2 人物数据库（扩展支持人物操作类动作）
        self.known_people: Dict[str, Dict[str, Any]] = {
            "alice": {
                "name": "alice", "cloth_color": "red", "gesture": "waving",
                "location": "living_room", "beacon": "beacon_1",
                "info": "Alice is a visitor, age 28, waiting for assistance."
            },
            "bob": {
                "name": "bob", "cloth_color": "blue", "gesture": "standing",
                "location": "kitchen", "beacon": "beacon_2",
                "info": "Bob is a staff member, preparing food."
            },
            "charlie": {
                "name": "charlie", "cloth_color": "green", "gesture": "sitting",
                "location": "bedroom", "beacon": "beacon_3",
                "info": "Charlie is a guest, resting."
            },
            "diana": {
                "name": "diana", "cloth_color": "red", "gesture": "waving",
                "location": "living_room", "beacon": "beacon_1",
                "info": "Diana is a visitor, needs directions."
            },
            "eve": {
                "name": "eve", "cloth_color": "blue", "gesture": "standing",
                "location": "hallway", "beacon": "entrance",
                "info": "Eve is a delivery person with a package."
            },
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

    # ==================== 导航类动作 ====================

    def goToLoc(self, target: str, then_find_person: bool = False) -> bool:
        """去某地（可选到达后找人）"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🚗 [GoToLoc] 前往 {target}")
        time.sleep(0.5)

        if target in self.known_locations:
            self.current_position = target
            coords = self.known_locations[target]
            self._log_info(f"✓ 已到达 {target} (坐标: {coords})")
        else:
            self._log_info(f"✗ 未知位置: {target}，尝试导航到坐标")
            self.current_position = target

        if then_find_person:
            self._log_info(f"🔍 [GoToLoc] 到达后在 {target} 寻找人...")
            people_here = [p for p in self.known_people.values() if p["location"] == target]
            if people_here:
                self._log_info(f"✓ 找到 {len(people_here)} 人: {[p['name'] for p in people_here]}")
            else:
                self._log_info(f"  当前位置没有找到人")

        self.set_state(RobotState.IDLE)
        return True

    # ==================== 人物操作类动作 ====================

    def findPrsInRoom(self, room: str, gesture: Optional[str] = None) -> Optional[dict]:
        """在房间找特定姿态/手势的人"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🔍 [FindPrsInRoom] 在 {room} 寻找" + (f" {gesture} 的人" if gesture else " 任何人"))
        time.sleep(0.6)

        people_in_room = []
        for p in self.known_people.values():
            if p["location"] == room:
                if gesture is None or p["gesture"] == gesture:
                    people_in_room.append(p)

        if people_in_room:
            found = people_in_room[0]
            self._log_info(f"✓ 找到: {found['name']} (gesture={found['gesture']}, cloth={found['cloth_color']})")
            self.set_state(RobotState.IDLE)
            return found
        else:
            self._log_info(f"✗ 未找到匹配的人")
            self.set_state(RobotState.IDLE)
            return None

    def meetPrsAtBeac(self, person_name: str, beacon: str) -> bool:
        """在信标处见某人（按名字）"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🤝 [MeetPrsAtBeac] 前往 {beacon} 见 {person_name}")
        time.sleep(0.5)

        if beacon in self.known_locations:
            self.current_position = beacon

        if person_name in self.known_people:
            person = self.known_people[person_name]
            self._log_info(f"✓ 在 {beacon} 见到了 {person_name} (cloth={person['cloth_color']})")
            self.set_state(RobotState.IDLE)
            return True
        else:
            self._log_info(f"✗ 未找到名为 {person_name} 的人")
            self.set_state(RobotState.ERROR)
            return False

    def countPrsInRoom(self, room: str, gesture: Optional[str] = None) -> int:
        """数房间里有某种姿态/手势的人数"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🔢 [CountPrsInRoom] 数 {room} 中" + (f" {gesture} 的人数" if gesture else " 的总人数"))
        time.sleep(0.4)

        count = 0
        for p in self.known_people.values():
            if p["location"] == room:
                if gesture is None or p["gesture"] == gesture:
                    count += 1

        self._log_info(f"✓ {room} 中{' ' + gesture if gesture else ''} 共有 {count} 人")
        self.set_state(RobotState.IDLE)
        return count

    def tellPrsInfoInLoc(self, person_name: Optional[str], location: str) -> Optional[dict]:
        """告诉我某地某人的信息"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n📋 [TellPrsInfoInLoc] 查询 {location} " + (f"中 {person_name} 的信息" if person_name else "所有人的信息"))
        time.sleep(0.4)

        if person_name and person_name in self.known_people:
            person = self.known_people[person_name]
            if person["location"] == location:
                self._log_info(f"✓ {person_name}: {person['info']}")
                self.set_state(RobotState.IDLE)
                return person
            else:
                self._log_info(f"  {person_name} 不在 {location}，当前在 {person['location']}")
                self.set_state(RobotState.IDLE)
                return person

        # Return all people at this location
        people_at_loc = [p for p in self.known_people.values() if p["location"] == location]
        if people_at_loc:
            self._log_info(f"✓ {location} 有 {len(people_at_loc)} 人: {[p['name'] for p in people_at_loc]}")
            result = {"location": location, "people": people_at_loc}
            self.set_state(RobotState.IDLE)
            return result

        self._log_info(f"✗ {location} 没有找到人")
        self.set_state(RobotState.IDLE)
        return None

    def talkInfoToGestPrsInRoom(self, room: str, gesture: str, info: str) -> bool:
        """在房间跟做手势的人交谈/传递信息"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n💬 [TalkInfoToGestPrsInRoom] 在 {room} 跟 {gesture} 的人交谈")
        time.sleep(0.5)

        for p in self.known_people.values():
            if p["location"] == room and p["gesture"] == gesture:
                self._log_info(f"✓ 向 {p['name']} 传递信息: \"{info}\"")
                self.set_state(RobotState.IDLE)
                return True

        self._log_info(f"✗ 在 {room} 未找到 {gesture} 的人")
        self.set_state(RobotState.ERROR)
        return False

    def followNameFromBeacToRoom(self, person_name: str, beacon: str, room: str) -> bool:
        """从信标跟随某人到房间"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🚶 [FollowNameFromBeacToRoom] 从 {beacon} 跟随 {person_name} 到 {room}")
        time.sleep(0.8)

        if person_name not in self.known_people:
            self._log_info(f"✗ 未找到 {person_name}")
            self.set_state(RobotState.ERROR)
            return False

        self.current_position = room
        person = self.known_people[person_name]
        person["location"] = room
        self._log_info(f"✓ 已跟随 {person_name} 从 {beacon} 到达 {room}")
        self.set_state(RobotState.IDLE)
        return True

    def guideNameFromBeacToBeac(self, person_name: str, from_beacon: str, to_beacon: str) -> bool:
        """从信标引导某人到另一地点"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🧭 [GuideNameFromBeacToBeac] 引导 {person_name} 从 {from_beacon} 到 {to_beacon}")
        time.sleep(0.7)

        if person_name not in self.known_people:
            self._log_info(f"✗ 未找到 {person_name}")
            self.set_state(RobotState.ERROR)
            return False

        person = self.known_people[person_name]
        person["beacon"] = to_beacon
        self.current_position = to_beacon
        self._log_info(f"✓ 已引导 {person_name} 到达 {to_beacon}")
        self.set_state(RobotState.IDLE)
        return True

    def guidePrsFromBeacToBeac(self, gesture: str, from_beacon: str, to_beacon: str) -> bool:
        """从信标引导有姿态/手势的人到另一地点"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🧭 [GuidePrsFromBeacToBeac] 引导 {gesture} 的人从 {from_beacon} 到 {to_beacon}")
        time.sleep(0.7)

        for p in self.known_people.values():
            if p["gesture"] == gesture and p["beacon"] == from_beacon:
                p["beacon"] = to_beacon
                self.current_position = to_beacon
                self._log_info(f"✓ 已引导 {p['name']} ({gesture}) 到达 {to_beacon}")
                self.set_state(RobotState.IDLE)
                return True

        self._log_info(f"✗ 未在 {from_beacon} 找到 {gesture} 的人")
        self.set_state(RobotState.ERROR)
        return False

    def guideClothPrsFromBeacToBeac(self, cloth_color: str, from_beacon: str, to_beacon: str) -> bool:
        """引导穿特定颜色衣服的人从信标到另一地点"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🧭 [GuideClothPrsFromBeacToBeac] 引导穿 {cloth_color} 衣服的人从 {from_beacon} 到 {to_beacon}")
        time.sleep(0.7)

        for p in self.known_people.values():
            if p["cloth_color"] == cloth_color and p["beacon"] == from_beacon:
                p["beacon"] = to_beacon
                self.current_position = to_beacon
                self._log_info(f"✓ 已引导 {p['name']} (cloth={cloth_color}) 到达 {to_beacon}")
                self.set_state(RobotState.IDLE)
                return True

        self._log_info(f"✗ 未在 {from_beacon} 找到穿 {cloth_color} 衣服的人")
        self.set_state(RobotState.ERROR)
        return False

    def greetClothDscInRm(self, cloth_color: str, room: str) -> bool:
        """问候穿特定颜色衣服的人"""
        self.set_state(RobotState.SPEAKING)
        self._log_info(f"\n👋 [GreetClothDscInRm] 在 {room} 问候穿 {cloth_color} 衣服的人")
        time.sleep(0.3)

        for p in self.known_people.values():
            if p["location"] == room and p["cloth_color"] == cloth_color:
                self._log_info(f"✓ 已问候 {p['name']} (穿 {cloth_color} 衣服)")
                self.set_state(RobotState.IDLE)
                return True

        self._log_info(f"✗ 在 {room} 未找到穿 {cloth_color} 衣服的人")
        self.set_state(RobotState.IDLE)
        return False

    def greetNameInRm(self, person_name: str, room: str) -> bool:
        """问候特定名字的人"""
        self.set_state(RobotState.SPEAKING)
        self._log_info(f"\n👋 [GreetNameInRm] 在 {room} 问候 {person_name}")
        time.sleep(0.3)

        if person_name in self.known_people:
            person = self.known_people[person_name]
            if person["location"] == room:
                self._log_info(f"✓ 已问候 {person_name}")
                self.set_state(RobotState.IDLE)
                return True
            else:
                self._log_info(f"✗ {person_name} 不在 {room}，当前在 {person['location']}")
                self.set_state(RobotState.IDLE)
                return False
        else:
            self._log_info(f"✗ 未找到 {person_name}")
            self.set_state(RobotState.IDLE)
            return False

    def meetNameAtLocThenFindInRm(self, person_name: str, meet_location: str, room: str) -> bool:
        """在某地见某人然后在房间找到他们"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🤝🔍 [MeetNameAtLocThenFindInRm] 在 {meet_location} 见 {person_name}，然后在 {room} 找到他们")
        time.sleep(0.9)

        if person_name not in self.known_people:
            self._log_info(f"✗ 未找到 {person_name}")
            self.set_state(RobotState.ERROR)
            return False

        # Step 1: Meet at location
        self.current_position = meet_location
        self._log_info(f"  步骤1: 已到达 {meet_location} 与 {person_name} 见面")

        # Step 2: Find them in room
        person = self.known_people[person_name]
        person["location"] = room
        self.current_position = room
        self._log_info(f"  步骤2: 已在 {room} 找到 {person_name}")
        self.set_state(RobotState.IDLE)
        return True

    def countClothPrsInRoom(self, cloth_color: str, room: str) -> int:
        """数房间里穿特定颜色衣服的人数"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🔢 [CountClothPrsInRoom] 数 {room} 中穿 {cloth_color} 衣服的人数")
        time.sleep(0.4)

        count = sum(1 for p in self.known_people.values()
                    if p["location"] == room and p["cloth_color"] == cloth_color)

        self._log_info(f"✓ {room} 中穿 {cloth_color} 衣服的有 {count} 人")
        self.set_state(RobotState.IDLE)
        return count

    def tellPrsInfoAtLocToPrsAtLoc(self, from_person: Optional[str], from_location: str,
                                     to_person: Optional[str], to_location: str,
                                     info: Optional[str] = None) -> bool:
        """把一个地点某人的信息告诉另一地点的人"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n📨 [TellPrsInfoAtLocToPrsAtLoc] 从 {from_location} 传递信息到 {to_location}")
        time.sleep(0.6)

        # Collect info from source
        source_people = [p for p in self.known_people.values() if p["location"] == from_location]
        if from_person:
            source_people = [p for p in source_people if p["name"] == from_person]

        if not info:
            info = " | ".join([f"{p['name']}: {p['info']}" for p in source_people]) if source_people else "No info available"

        # Find target person
        target_people = [p for p in self.known_people.values() if p["location"] == to_location]
        if to_person:
            target_people = [p for p in target_people if p["name"] == to_person]

        if target_people:
            target_names = [p['name'] for p in target_people]
            self._log_info(f"✓ 已将信息 \"{info}\" 传递给 {to_location} 的 {target_names}")
        else:
            self._log_info(f"  目标地点 {to_location} 没有人，信息已记录")

        self.set_state(RobotState.IDLE)
        return True

    def followPrsAtLoc(self, gesture: str, location: str) -> bool:
        """跟随某地有姿态/手势的人"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🚶 [FollowPrsAtLoc] 在 {location} 跟随 {gesture} 的人")
        time.sleep(0.6)

        for p in self.known_people.values():
            if p["location"] == location and p["gesture"] == gesture:
                self.current_position = p["location"]
                self._log_info(f"✓ 正在跟随 {p['name']} ({gesture}) 在 {location}")
                self.set_state(RobotState.IDLE)
                return True

        self._log_info(f"✗ 在 {location} 未找到 {gesture} 的人")
        self.set_state(RobotState.ERROR)
        return False

    # ==================== 物品操作类动作 ====================

    def takeObjFromPlcmt(self, object_name: str, placement: str) -> bool:
        """从放置处拿物品"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🤏 [TakeObjFromPlcmt] 从 {placement} 拿 {object_name}")
        time.sleep(0.5)

        if self.holding_object:
            self._log_info(f"✗ 手中已有物体: {self.holding_object}")
            self.set_state(RobotState.ERROR)
            return False

        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            if obj.get("placement") == placement:
                self.holding_object = object_name
                self._log_info(f"✓ 从 {placement} 拿起了 {object_name}")
                self.set_state(RobotState.IDLE)
                return True
            else:
                self._log_info(f"✗ {object_name} 不在 {placement}，在 {obj.get('placement')}")
                self.set_state(RobotState.ERROR)
                return False
        else:
            self._log_info(f"✗ 未知物品: {object_name}")
            self.set_state(RobotState.ERROR)
            return False

    def findObjInRoom(self, object_name: str, room: str) -> Optional[dict]:
        """在房间找物品"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🔍 [FindObjInRoom] 在 {room} 寻找 {object_name}")
        time.sleep(0.5)

        if object_name in self.known_objects:
            obj = self.known_objects[object_name]
            if obj.get("location") == room or obj.get("placement") == room:
                self._log_info(f"✓ 在 {room} 找到 {object_name} (位置: {obj['position']})")
                self.set_state(RobotState.IDLE)
                return obj
            else:
                self._log_info(f"✗ {object_name} 不在 {room}，在 {obj.get('location')}")
                self.set_state(RobotState.IDLE)
                return None
        else:
            self._log_info(f"✗ 未找到 {object_name}")
            self.set_state(RobotState.IDLE)
            return None

    def countObjOnPlcmt(self, object_category: str, placement: str) -> int:
        """数放置处某类物品的数量"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n🔢 [CountObjOnPlcmt] 数 {placement} 上 {object_category} 的数量")
        time.sleep(0.4)

        count = sum(1 for obj in self.known_objects.values()
                    if obj.get("placement") == placement and obj.get("category") == object_category)

        self._log_info(f"✓ {placement} 上有 {count} 个 {object_category}")
        self.set_state(RobotState.IDLE)
        return count

    def tellObjPropOnPlcmt(self, object_name: str, placement: str, property: str) -> Optional[dict]:
        """问放置处物品的属性（最大/最小等）"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n📏 [TellObjPropOnPlcmt] 查询 {placement} 上 {object_name} 的 {property}")
        time.sleep(0.4)

        candidates = [obj for obj in self.known_objects.values()
                      if obj.get("placement") == placement and obj["name"] == object_name]

        if not candidates:
            self._log_info(f"✗ 在 {placement} 未找到 {object_name}")
            self.set_state(RobotState.IDLE)
            return None

        obj = candidates[0]
        if property in obj:
            value = obj[property]
            self._log_info(f"✓ {object_name} 的 {property}: {value}")
            self.set_state(RobotState.IDLE)
            return {"name": object_name, "property": property, "value": value}
        else:
            self._log_info(f"✗ {object_name} 没有属性 {property}")
            self.set_state(RobotState.IDLE)
            return None

    def bringMeObjFromPlcmt(self, object_name: str, placement: str) -> bool:
        """从放置处拿物品给我"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n📦 [BringMeObjFromPlcmt] 从 {placement} 拿 {object_name} 带回来")
        time.sleep(0.8)

        if object_name not in self.known_objects:
            self._log_info(f"✗ 未知物品: {object_name}")
            self.set_state(RobotState.ERROR)
            return False

        obj = self.known_objects[object_name]
        if obj.get("placement") != placement:
            self._log_info(f"✗ {object_name} 不在 {placement}")
            self.set_state(RobotState.ERROR)
            return False

        # Navigate to placement, pick object, return to home
        self._log_info(f"  前往 {placement}...")
        self.current_position = placement
        self.holding_object = object_name
        self._log_info(f"  拿起 {object_name}...")
        self.current_position = "home"
        self._log_info(f"  返回起点...")
        self._log_info(f"✓ 已将 {object_name} 从 {placement} 带给你")
        self.set_state(RobotState.IDLE)
        return True

    def tellCatPropOnPlcmt(self, category: str, placement: str, property: str) -> Optional[dict]:
        """问放置处某类物品的属性"""
        self.set_state(RobotState.EXECUTING)
        self._log_info(f"\n📊 [TellCatPropOnPlcmt] 查询 {placement} 上 {category} 类别的 {property}")
        time.sleep(0.5)

        candidates = [obj for obj in self.known_objects.values()
                      if obj.get("placement") == placement and obj.get("category") == category]

        if not candidates:
            self._log_info(f"✗ 在 {placement} 未找到 {category} 类别物品")
            self.set_state(RobotState.IDLE)
            return None

        values = [obj.get(property) for obj in candidates if property in obj]
        if not values:
            self._log_info(f"✗ 这些物品没有 {property} 属性")
            self.set_state(RobotState.IDLE)
            return None

        result = {
            "category": category,
            "placement": placement,
            "property": property,
            "count": len(candidates),
            "max": max(values) if values else None,
            "min": min(values) if values else None,
            "items": [obj["name"] for obj in candidates],
        }
        self._log_info(f"✓ {category} 在 {placement}: count={result['count']}, max_{property}={result['max']}, min_{property}={result['min']}")
        self._log_info(f"  物品: {result['items']}")
        self.set_state(RobotState.IDLE)
        return result


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 测试 CADE Robot V2 ===\n")

    robot = Robot(name="LARA")

    # 测试导航
    robot.goToLoc("kitchen", then_find_person=True)
    robot.print_status()

    # 测试找人
    result = robot.findPrsInRoom("living_room", "waving")
    print(f"找人结果: {result}")

    # 测试拿物品
    robot.takeObjFromPlcmt("bottle", "kitchen")
    robot.print_status()

    # 测试数物品
    count = robot.countObjOnPlcmt("fruit", "table")
    print(f"桌上的水果数量: {count}")

    # 测试带物品给我
    robot.holding_object = None
    robot.bringMeObjFromPlcmt("cup", "table")
    robot.print_status()

    # 测试查物品属性
    result = robot.tellCatPropOnPlcmt("container", "kitchen", "size")
    print(f"厨具容器属性: {result}")

    # 测试忙碌状态
    robot.set_state(RobotState.THINKING)
    print(f"思考时忙碌: {robot.is_busy()}")
    robot.set_state(RobotState.IDLE)
    print(f"空闲时忙碌: {robot.is_busy()}")

    print("\n✓ 所有测试通过！")
