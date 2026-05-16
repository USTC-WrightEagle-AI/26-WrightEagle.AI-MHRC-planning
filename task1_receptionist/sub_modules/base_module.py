"""
Task1 子模块 — 各执行模块基类和具体实现

模块划分 (按能力维度):
    DoorbellModule    — 门铃: wait_for_doorbell_1, wait_for_doorbell_2
    NavigationModule   — 导航: go_to_door, guide_guest1, return_to_start, seat_guest2...
    SpeechModule       — 语音: ask_guest1_info, pick_up_guest2, introduce_guests...
    VisionModule       — 视觉: describe_guest1, find_host, follow_host
    ManipulationModule — 操作: point_empty_seat, wait_for_bag, place_bag
    SpeakerDOAModule   — 说话人识别 + 声源定位: enroll, locate, track, face_speaker

每个模块通过 execute(state_id, context) 被总控调用,
返回产出数据 dict (合并到任务上下文), 或 None。

所有模块通过 ROSTopicBridge 与外部 ROS 节点通信:
    发送指令到 command 话题 → 等待 result 话题的响应。
ROS 不可用时自动降级为模拟行为。

话题名称和指令字符串定义在 topic_names.py, 方便统一修改。
"""

import json
import queue
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from task1_receptionist.state_definitions import Task1StateID
from task1_receptionist.sub_modules.topic_names import (
    DOORBELL_TOPIC,
    DOORBELL_TIMEOUT,
    NAV_COMMAND_TOPIC,
    NAV_RESULT_TOPIC,
    NAV_TIMEOUT,
    NAV_CMD_GO_TO_DOOR,
    NAV_CMD_GO_TO_LIVING_ROOM,
    NAV_CMD_GO_TO_START,
    NAV_CMD_FOLLOW_PERSON,
    VISION_COMMAND_TOPIC,
    VISION_RESULT_TOPIC,
    VISION_TIMEOUT,
    VISION_CMD_DESCRIBE_PERSON,
    VISION_CMD_FIND_HOST,
    VISION_CMD_TRACK_PERSON,
    MANIP_COMMAND_TOPIC,
    MANIP_RESULT_TOPIC,
    MANIP_TIMEOUT,
    MANIP_CMD_POINT_SEAT,
    MANIP_CMD_WAIT_FOR_BAG,
    MANIP_CMD_PLACE_BAG,
    SPEAKER_DOA_ENROLL_TOPIC,
    SPEAKER_DOA_COMMAND_TOPIC,
    SPEAKER_DOA_RESULT_TOPIC,
    SPEAKER_DOA_TIMEOUT,
    SPEAKER_DOA_FACE_TIMEOUT,
    SPEAKER_DOA_CMD_LOCATE,
    SPEAKER_DOA_CMD_TRACK,
    SPEAKER_DOA_CMD_FACE_SPEAKER,
    NAV_CMD_TURN_TO_ANGLE,
)


# ============================================================
# ROS 话题桥接工具
# ============================================================

class ROSTopicBridge:
    """
    ROS 话题桥接 — 发送指令到 command 话题, 等待 result 话题的响应

    封装 "发布指令 → 等待结果" 的通用模式。
    ROS 不可用时 call() 返回 None, 由调用方决定降级策略。
    """

    def __init__(self, command_topic: str, result_topic: str, timeout: float = 120.0):
        self._command_topic = command_topic
        self._result_topic = result_topic
        self._default_timeout = timeout
        self._pub = None
        self._sub = None
        self._result_queue: queue.Queue = queue.Queue()
        self._initialized = False
        self._rospy = None
        self._String = None

    @property
    def available(self) -> bool:
        return self._initialized

    def _ensure_init(self) -> bool:
        if self._initialized:
            return True
        try:
            import rospy
            from std_msgs.msg import String
            if not rospy.core.is_initialized():
                return False
            self._rospy = rospy
            self._String = String
            self._pub = rospy.Publisher(self._command_topic, String, queue_size=10)
            self._sub = rospy.Subscriber(self._result_topic, String, self._on_result, queue_size=10)
            time.sleep(0.3)
            self._initialized = True
            rospy.loginfo(
                f"[ROSBridge] 已连接: cmd={self._command_topic}, result={self._result_topic}"
            )
            return True
        except Exception:
            return False

    def _on_result(self, msg):
        try:
            self._result_queue.put_nowait(msg.data)
        except queue.Full:
            pass

    def _drain_queue(self):
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def call(self, command: str, timeout: Optional[float] = None) -> Optional[dict]:
        """
        发送指令, 等待结果

        Args:
            command: 指令字符串 (如 "导航到门口")
            timeout: 超时秒数, None 使用默认值

        Returns:
            结果 dict (解析 JSON), 或 None (超时/ROS不可用)
        """
        if not self._ensure_init():
            return None

        self._drain_queue()
        print(f"    [ROS] 发布: \"{command}\" → {self._command_topic}")
        print(f"    [ROS] 等待: {self._result_topic} (超时 {timeout or self._default_timeout}s)")
        self._pub.publish(self._String(data=command))
        self._rospy.loginfo(f"[ROSBridge] 发送指令: {command} → {self._command_topic}")

        t = timeout if timeout is not None else self._default_timeout
        try:
            raw = self._result_queue.get(timeout=t)
            try:
                result = json.loads(raw)
                print(f"    [ROS] 收到: {result}")
                self._rospy.loginfo(f"[ROSBridge] 收到结果: {result}")
                return result
            except (json.JSONDecodeError, TypeError):
                print(f"    [ROS] 收到 (raw): {raw}")
                return {"status": "success", "raw": raw}
        except queue.Empty:
            print(f"    [ROS] ⏰ 超时 ({t}s): {command}")
            self._rospy.logwarn(f"[ROSBridge] 超时 ({t}s): {command}")
            return None


# ============================================================
# 基类
# ============================================================

class BaseSubModule(ABC):
    """所有子模块的抽象基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        执行指定状态

        Args:
            state_id: 当前状态 ID
            context: 任务共享上下文 (包含之前所有产出数据)

        Returns:
            本状态的产出数据 (合并到总控上下文), 或 None
        """
        ...

    def cancel(self):
        """取消当前操作 (子类可选覆盖)"""
        pass


# ============================================================
# 门铃模块
# ============================================================

class DoorbellModule(BaseSubModule):
    """
    门铃子模块 — 等待门铃信号

    订阅 /doorbell/detected 话题 (asr_tts/doorbell_node.py 发布),
    收到 JSON 消息 {"detected": true, "label": "Doorbell", "probability": 0.85, ...}
    即表示有人按门铃。
    ROS 不可用时降级为控制台等待用户输入。
    """

    auto_simulate = False

    def __init__(self):
        super().__init__("doorbell")
        self._sub = None
        self._bell_queue: queue.Queue = queue.Queue()
        self._initialized = False
        self._rospy = None

    def _ensure_init(self) -> bool:
        if self._initialized:
            return True
        try:
            import rospy
            from std_msgs.msg import String
            if not rospy.core.is_initialized():
                return False
            self._rospy = rospy
            self._sub = rospy.Subscriber(DOORBELL_TOPIC, String, self._on_doorbell, queue_size=10)
            time.sleep(0.3)
            self._initialized = True
            rospy.loginfo(f"[Doorbell] 已订阅: {DOORBELL_TOPIC}")
            return True
        except Exception:
            return False

    def _on_doorbell(self, msg):
        try:
            data = json.loads(msg.data)
            if data.get("detected"):
                self._bell_queue.put_nowait(data)
        except (json.JSONDecodeError, TypeError):
            self._bell_queue.put_nowait({"detected": True, "raw": msg.data})

    def _drain_queue(self):
        while True:
            try:
                self._bell_queue.get_nowait()
            except queue.Empty:
                break

    def _wait_for_doorbell(self, label: str, timeout: float = DOORBELL_TIMEOUT) -> Dict[str, Any]:
        if self._ensure_init():
            self._drain_queue()
            print(f"  🔔 [Doorbell] 等待门铃 ({label})...")
            print(f"  🔔 [Doorbell] 订阅话题: {DOORBELL_TOPIC}, 超时: {timeout}s")
            self._rospy.loginfo(f"[Doorbell] 等待门铃: {label}")
            try:
                data = self._bell_queue.get(timeout=timeout)
                lbl = data.get("label", "unknown")
                prob = data.get("probability", 0)
                print(f"  🔔 [Doorbell] ✅ 收到门铃信号! label={lbl}, prob={prob}")
                return {"doorbell_rang": True}
            except queue.Empty:
                print(f"  🔔 [Doorbell] ⏰ 超时 ({timeout}s), 继续执行")
                return {"doorbell_rang": False}

        print(f"  🔔 [Doorbell] ⚠️ ROS 不可用, 模拟等待门铃 ({label})...")
        print(f"  🔔 [Doorbell] (真实环境: 订阅 {DOORBELL_TOPIC}, 等待 doorbell_node 发布)")

        if self.auto_simulate:
            print(f"  🔔 [Doorbell] ✅ 自动模拟门铃 ({label})")
            return {"doorbell_rang": True}

        try:
            result = input(f"  🔔 按 Enter 模拟门铃 ({label}): ")
            print(f"  🔔 [Doorbell] ✅ 收到门铃信号 (模拟)")
            return {"doorbell_rang": True}
        except (EOFError, KeyboardInterrupt):
            print(f"  🔔 [Doorbell] 跳过等待, 继续执行")
            return {"doorbell_rang": True}

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if state_id == Task1StateID.WAIT_FOR_DOORBELL_1:
            result = self._wait_for_doorbell("guest1")
            return {"doorbell_1_rang": result["doorbell_rang"]}
        elif state_id == Task1StateID.WAIT_FOR_DOORBELL_2:
            result = self._wait_for_doorbell("guest2")
            return {"doorbell_2_rang": result["doorbell_rang"]}
        return None


# ============================================================
# 导航模块
# ============================================================

class NavigationModule(BaseSubModule):
    """
    导航子模块 — 通过 ROS 话题与导航服务器通信

    发送指令到 /navigation/command, 等待 /navigation/result 响应。
    ROS 不可用时降级为模拟导航 (time.sleep)。
    """

    def __init__(self):
        super().__init__("navigation")
        self._bridge = ROSTopicBridge(NAV_COMMAND_TOPIC, NAV_RESULT_TOPIC, NAV_TIMEOUT)

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, callable] = {
            Task1StateID.GO_TO_DOOR: self._go_to_door,
            Task1StateID.GUIDE_GUEST1: self._guide_guest1,
            Task1StateID.RETURN_TO_START: self._return_to_start,
            Task1StateID.PICK_UP_GUEST2: self._pick_up_guest2,
            Task1StateID.SEAT_GUEST2: self._seat_guest2,
            Task1StateID.FOLLOW_HOST: self._follow_host,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    def _navigate(self, command: str, timeout: Optional[float] = None) -> dict:
        result = self._bridge.call(command, timeout=timeout)
        if result is None:
            print(f"  🚪 [Nav] ⚠️ ROS 不可用, 模拟: {command}")
            time.sleep(5)
            return {"status": "success"}
        return result

    def _go_to_door(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚪 [Nav] 导航到门口...")
        result = self._navigate(NAV_CMD_GO_TO_DOOR)
        if result.get("status") == "success":
            print("  🚪 [Nav] 已到达门口")
            return {"robot_at_door": True}
        print(f"  🚪 [Nav] 导航失败: {result.get('error', 'unknown')}")
        return {"robot_at_door": False}

    def _guide_guest1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest = context.get("guest1_name", "guest1")
        print(f"  🚶 [Nav] 带 {guest} 去客厅...")
        result = self._navigate(NAV_CMD_GO_TO_LIVING_ROOM)
        if result.get("status") == "success":
            print(f"  🚶 [Nav] 已带 {guest} 到达客厅")
            return {"guest1_in_living_room": True}
        return {"guest1_in_living_room": False}

    def _return_to_start(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚶 [Nav] 返回起点...")
        result = self._navigate(NAV_CMD_GO_TO_START)
        if result.get("status") == "success":
            print("  🚶 [Nav] 已返回起点")
            return {"robot_at_start": True}
        return {"robot_at_start": False}

    def _pick_up_guest2(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print("  🚪 [Nav] 导航到门口接 guest2...")
        result = self._navigate(NAV_CMD_GO_TO_DOOR)
        if result.get("status") == "success":
            print("  🚪 [Nav] 已到达门口")
        return None

    def _seat_guest2(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guest = context.get("guest2_name", "guest2")
        print(f"  🚶 [Nav] 带 {guest} 去客厅...")
        result = self._navigate(NAV_CMD_GO_TO_LIVING_ROOM)
        if result.get("status") == "success":
            print(f"  🚶 [Nav] 已带 {guest} 到达客厅")
        return None

    def _follow_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚶 [Nav] 跟随 host...")
        result = self._navigate(NAV_CMD_FOLLOW_PERSON, timeout=180.0)
        if result.get("status") == "success":
            dest = result.get("data", {}).get("destination", "storage_room")
            print(f"  🚶 [Nav] 已跟随 host 到达: {dest}")
            return {"at_destination": True, "destination": dest}
        return {"at_destination": False}


# ============================================================
# 语音模块
# ============================================================

class SpeechModule(BaseSubModule):
    """
    语音交互子模块

    负责所有需要 TTS + ASR 对话的状态:
      ASK_GUEST1_INFO, PICK_UP_GUEST2 (语音部分),
      DESCRIBE_GUEST1 (TTS), SEAT_GUEST2 (语音部分),
      INTRODUCE_GUESTS, REQUEST_GUEST2_BAG (TTS),
      POINT_EMPTY_SEAT (TTS), TASK_COMPLETE (TTS)

    依赖 SpeechInterface 注入 (Mock / ROS / Scripted)。
    """

    def __init__(self, speech=None, llm=None):
        super().__init__("speech")
        if speech is None:
            from task1_receptionist.sub_modules.speech_interaction import MockSpeechInterface
            speech = MockSpeechInterface()
        self._sp = speech

        if llm is None:
            from task1_receptionist.sub_modules.llm_interface import MockLLMInterface
            llm = MockLLMInterface()
        self._llm = llm

        self._pub_enroll = None
        self._rospy = None
        self._String = None

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, callable] = {
            Task1StateID.ASK_GUEST1_INFO: self._ask_guest1_info,
            Task1StateID.PICK_UP_GUEST2: self._pick_up_guest2,
            Task1StateID.DESCRIBE_GUEST1: self._describe_guest1,
            Task1StateID.POINT_EMPTY_SEAT: self._point_empty_seat,
            Task1StateID.SEAT_GUEST2: self._seat_guest2,
            Task1StateID.INTRODUCE_GUESTS: self._introduce_guests,
            Task1StateID.REQUEST_GUEST2_BAG: self._request_guest2_bag,
            Task1StateID.TASK_COMPLETE: self._task_complete,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    # ---- 各状态具体实现 ----

    def _ask_guest1_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  💬 [Speech] 示例: 客人应该说 'My name is Alice' 或 'I am Alice'")
        self._enroll_speaker("guest1", "guest1", duration=5.0)
        raw_name = self._sp.ask("Welcome! May I have your name please?", timeout_sec=60.0)

        print(f"  💬 [Speech] 示例: 客人应该说 'Orange juice please' 或 'I would like some cola'")
        guest_label = raw_name or "Guest"
        raw_drink = self._sp.ask(f"{guest_label}, what would you like to drink?", timeout_sec=60.0)

        name = "Guest"
        drink = "water"

        try:
            combined_text = f"{raw_name or ''}. {raw_drink or ''}".strip()
            if combined_text:
                info = self._llm.extract_guest_info(combined_text, role="guest1")
                if info.get("name"):
                    name = info["name"]
                if info.get("drink"):
                    drink = info["drink"]
        except Exception as e:
            print(f"  ⚠ [Speech] LLM 提取失败, 使用原始文本: {e}")
            if raw_name:
                name = raw_name
            if raw_drink:
                drink = raw_drink

        if name == "Guest" and raw_name:
            name = raw_name
        if drink == "water" and raw_drink:
            drink = raw_drink

        return {"guest1_name": name, "guest1_drink": drink}

    def _pick_up_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  💬 [Speech] 示例: 客人应该说 'My name is Bob' 或 'I am Bob'")
        self._enroll_speaker("guest2", "guest2", duration=5.0)
        self._sp.say("Welcome! May I have your name please?")
        raw_name = self._sp.listen(timeout_sec=60.0)

        name = "Guest2"
        try:
            if raw_name:
                extracted = self._llm.extract_name(raw_name, role="guest2")
                if extracted:
                    name = extracted
        except Exception as e:
            print(f"  ⚠ [Speech] LLM 提取失败, 使用原始文本: {e}")
            if raw_name:
                name = raw_name

        if name == "Guest2" and raw_name:
            name = raw_name

        return {"guest2_at_door": True, "guest2_name": name}

    def _describe_guest1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest1 = context.get("guest1_name", "the first guest")
        appearance = context.get("guest1_appearance", {})
        clothing = appearance.get("clothing", "unknown clothing")
        position = appearance.get("position", "an empty seat")
        self._sp.say(f"{guest1} is wearing {clothing}, sitting at {position}.")
        return {"guest1_described": True}

    def _point_empty_seat(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guest = context.get("guest1_name", "guest")
        self._sp.say(f"{guest}, please take a seat here.")
        return None

    def _seat_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest = context.get("guest2_name", "guest2")
        print(f"  💬 [Speech] 示例: 客人应该说 'Coffee please' 或 'I would like some tea'")
        self._sp.say(f"{guest}, please follow me to the living room.")
        raw_drink = self._sp.ask("What would you like to drink?", timeout_sec=60.0)

        drink = "water"
        try:
            if raw_drink:
                extracted = self._llm.extract_drink(raw_drink, role="guest2")
                if extracted:
                    drink = extracted
        except Exception as e:
            print(f"  ⚠ [Speech] LLM 提取失败, 使用原始文本: {e}")
            if raw_drink:
                drink = raw_drink

        if drink == "water" and raw_drink:
            drink = raw_drink

        return {"guest2_seated": True, "guest2_drink": drink, "seat_number": 2}

    def _introduce_guests(self, context: Dict[str, Any]) -> Dict[str, Any]:
        g1 = context.get("guest1_name", "guest1")
        g1d = context.get("guest1_drink", "?")
        g2 = context.get("guest2_name", "guest2")
        g2d = context.get("guest2_drink", "?")
        self._sp.say(f"{g1}, this is {g2}. He likes {g2d}. {g2}, this is {g1}. She likes {g1d}.")
        return {"guests_introduced": True}

    def _request_guest2_bag(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guest = context.get("guest2_name", "guest2")
        self._sp.say(f"{guest}, please place your bag on my tray.")
        return None

    def _task_complete(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._sp.say("Task completed. Thank you all for your cooperation!")
        return {}

    # ---- 声纹录入 ----

    def _enroll_speaker(self, speaker_name: str, role: str, duration: float = 5.0):
        if self._pub_enroll is None:
            try:
                import rospy
                from std_msgs.msg import String
                if not rospy.core.is_initialized():
                    return
                self._rospy = rospy
                self._String = String
                self._pub_enroll = rospy.Publisher(SPEAKER_DOA_ENROLL_TOPIC, String, queue_size=10)
                time.sleep(0.5)
            except Exception:
                return

        cmd = json.dumps({
            "command": "enroll",
            "speaker_name": f"{role}_{speaker_name}",
            "duration": duration,
        })
        self._rospy.loginfo(f"[SpeakerDOA] 发送录入指令: {cmd}")
        self._pub_enroll.publish(self._String(data=cmd))

        print(f"  🎙️ [SpeakerDOA] 正在录入 {role} 声纹 ({speaker_name}), 请说话 {duration} 秒...")
        time.sleep(duration + 1)
        print(f"  🎙️ [SpeakerDOA] {role} 声纹录入完成")

    def close(self):
        self._sp.close()
        if hasattr(self._llm, 'close'):
            self._llm.close()


# ============================================================
# 视觉模块
# ============================================================

class VisionModule(BaseSubModule):
    """
    视觉子模块 — 通过 ROS 话题与视觉服务器通信

    发送指令到 /vision/command, 等待 /vision/result 响应。
    ROS 不可用时降级为模拟视觉识别。
    """

    def __init__(self):
        super().__init__("vision")
        self._bridge = ROSTopicBridge(VISION_COMMAND_TOPIC, VISION_RESULT_TOPIC, VISION_TIMEOUT)

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, callable] = {
            Task1StateID.DESCRIBE_GUEST1: self._describe_guest1,
            Task1StateID.FIND_HOST: self._find_host,
            Task1StateID.FOLLOW_HOST: self._track_person,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    def _call_vision(self, command: str, timeout: Optional[float] = None) -> Optional[dict]:
        result = self._bridge.call(command, timeout=timeout)
        if result is None:
            print(f"  📷 [Vision] ⚠️ ROS 不可用, 模拟: {command}")
            time.sleep(5)
            return None
        return result

    def _describe_guest1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  📷 [Vision] 识别 guest1 外貌...")
        result = self._call_vision(VISION_CMD_DESCRIBE_PERSON)
        if result and result.get("status") == "success":
            data = result.get("data", {})
            clothing = data.get("clothing", "unknown clothing")
            position = data.get("position", "unknown position")
            print(f"  📷 [Vision] 识别完成: {clothing}, {position}")
            return {"guest1_appearance": data}
        print("  📷 [Vision] 使用模拟数据: 红色外套, 1号座位")
        return {"guest1_appearance": {"clothing": "red jacket", "position": "seat 1"}}

    def _find_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  📷 [Vision] 搜索 host...")
        result = self._call_vision(VISION_CMD_FIND_HOST, timeout=180.0)
        if result and result.get("status") == "success":
            data = result.get("data", {})
            location = data.get("location", "unknown")
            print(f"  📷 [Vision] 找到 host, 位置: {location}")
            return {"host_found": True, "host_location": location}
        print("  📷 [Vision] 使用模拟数据: host 在客厅")
        return {"host_found": True, "host_location": "living_room"}

    def _track_person(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print("  📷 [Vision] 跟踪人物...")
        result = self._call_vision(VISION_CMD_TRACK_PERSON)
        if result and result.get("status") == "success":
            print("  📷 [Vision] 跟踪完成")
        return None


# ============================================================
# 操作模块
# ============================================================

class ManipulationModule(BaseSubModule):
    """
    操作子模块 — 通过 ROS 话题与操作服务器通信

    发送指令到 /manipulation/command, 等待 /manipulation/result 响应。
    ROS 不可用时降级为模拟操作。
    """

    def __init__(self):
        super().__init__("manipulation")
        self._bridge = ROSTopicBridge(MANIP_COMMAND_TOPIC, MANIP_RESULT_TOPIC, MANIP_TIMEOUT)

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, callable] = {
            Task1StateID.POINT_EMPTY_SEAT: self._point_empty_seat,
            Task1StateID.REQUEST_GUEST2_BAG: self._wait_for_bag,
            Task1StateID.PLACE_BAG: self._place_bag,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    def _call_manip(self, command: str, timeout: Optional[float] = None) -> dict:
        result = self._bridge.call(command, timeout=timeout)
        if result is None:
            print(f"  🦾 [Manip] ⚠️ ROS 不可用, 模拟: {command}")
            time.sleep(5)
            return {"status": "success"}
        return result

    def _point_empty_seat(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🦾 [Manip] 指向空座...")
        result = self._call_manip(MANIP_CMD_POINT_SEAT)
        if result.get("status") == "success":
            print("  🦾 [Manip] 已指向空座")
            return {"seat_pointed": True}
        return {"seat_pointed": False}

    def _wait_for_bag(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🦾 [Manip] 等待放包到托盘...")
        result = self._call_manip(MANIP_CMD_WAIT_FOR_BAG, timeout=120.0)
        if result.get("status") == "success":
            print("  🦾 [Manip] 检测到包已放置")
            return {"bag_on_tray": True}
        return {"bag_on_tray": False}

    def _place_bag(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🦾 [Manip] 放包到指定位置...")
        result = self._call_manip(MANIP_CMD_PLACE_BAG)
        if result.get("status") == "success":
            print("  🦾 [Manip] 放包完成")
            return {"bag_placed": True}
        return {"bag_placed": False}


# ============================================================
# 说话人识别 + 声源定位模块
# ============================================================

class SpeakerDOAModule(BaseSubModule):
    """
    说话人识别 + 声源定位子模块 — 通过 ROS 话题与 SpeakerDOA 节点通信

    负责:
      ASK_GUEST1_INFO   — 面向 guest1 说话人
      PICK_UP_GUEST2    — 面向 guest2 说话人
      DESCRIBE_GUEST1   — 面向 guest2 说话人
      SEAT_GUEST2       — 面向 guest2 说话人
      INTRODUCE_GUESTS  — 面向说话人
      REQUEST_GUEST2_BAG— 面向 guest2 说话人
      FIND_HOST         — 通过 DOA 声源定位寻找 host
      FOLLOW_HOST       — 通过 DOA 声源跟踪跟随 host

    面向说话人: 订阅 /speaker_doa/result, 获取 angle_deg,
    通过 /navigation/command 发送转向指令。

    声纹录入通过 /speaker_doa/enroll 话题发送,
    定位/跟踪指令通过 /speaker_doa/command 话题发送。
    """

    def __init__(self):
        super().__init__("speaker_doa")
        self._bridge = ROSTopicBridge(
            SPEAKER_DOA_COMMAND_TOPIC, SPEAKER_DOA_RESULT_TOPIC, SPEAKER_DOA_TIMEOUT
        )
        self._nav_bridge = ROSTopicBridge(
            NAV_COMMAND_TOPIC, NAV_RESULT_TOPIC, NAV_TIMEOUT
        )
        self._pub_enroll = None
        self._rospy = None
        self._String = None
        self._enroll_initialized = False
        self._doa_sub = None
        self._doa_queue: queue.Queue = queue.Queue()
        self._doa_initialized = False

    def _ensure_enroll_init(self) -> bool:
        if self._enroll_initialized:
            return True
        try:
            import rospy
            from std_msgs.msg import String
            if not rospy.core.is_initialized():
                return False
            self._rospy = rospy
            self._String = String
            self._pub_enroll = rospy.Publisher(SPEAKER_DOA_ENROLL_TOPIC, String, queue_size=10)
            time.sleep(0.3)
            self._enroll_initialized = True
            return True
        except Exception:
            return False

    def _ensure_doa_sub_init(self) -> bool:
        if self._doa_initialized:
            return True
        try:
            import rospy
            from std_msgs.msg import String
            if not rospy.core.is_initialized():
                return False
            self._rospy = rospy
            self._String = String
            self._doa_sub = rospy.Subscriber(
                SPEAKER_DOA_RESULT_TOPIC, String, self._on_doa_result, queue_size=10
            )
            time.sleep(0.3)
            self._doa_initialized = True
            rospy.loginfo(f"[SpeakerDOA] 已订阅: {SPEAKER_DOA_RESULT_TOPIC}")
            return True
        except Exception:
            return False

    def _on_doa_result(self, msg):
        try:
            data = json.loads(msg.data)
            if data.get("status") == "recognized" and data.get("angle_deg") is not None:
                self._doa_queue.put_nowait(data)
        except (json.JSONDecodeError, TypeError):
            pass

    def _drain_doa_queue(self):
        while True:
            try:
                self._doa_queue.get_nowait()
            except queue.Empty:
                break

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, callable] = {
            Task1StateID.ASK_GUEST1_INFO: self._face_speaker_guest1,
            Task1StateID.PICK_UP_GUEST2: self._face_speaker_guest2,
            Task1StateID.DESCRIBE_GUEST1: self._face_speaker_guest2,
            Task1StateID.SEAT_GUEST2: self._face_speaker_guest2,
            Task1StateID.INTRODUCE_GUESTS: self._face_speaker_conversation,
            Task1StateID.REQUEST_GUEST2_BAG: self._face_speaker_guest2,
            Task1StateID.FIND_HOST: self._find_host,
            Task1StateID.FOLLOW_HOST: self._follow_host,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    # ============================================================
    # 面向说话人
    # ============================================================

    def _face_speaker(self, label: str, timeout: float = SPEAKER_DOA_FACE_TIMEOUT) -> Optional[float]:
        if self._ensure_doa_sub_init():
            self._drain_doa_queue()
            print(f"  🎙️ [SpeakerDOA] 等待声源定位 ({label})...")
            print(f"  🎙️ [SpeakerDOA] 订阅话题: {SPEAKER_DOA_RESULT_TOPIC}, 超时: {timeout}s")
            try:
                data = self._doa_queue.get(timeout=timeout)
                angle = data.get("angle_deg", 0)
                speaker_name = data.get("speaker_name", "unknown")
                print(f"  🎙️ [SpeakerDOA] ✅ 检测到说话人: {speaker_name}, 角度: {angle}°")

                turn_cmd = f"{NAV_CMD_TURN_TO_ANGLE} {angle}"
                print(f"  🎙️ [SpeakerDOA] 发送转向指令: \"{turn_cmd}\" → {NAV_COMMAND_TOPIC}")
                result = self._nav_bridge.call(turn_cmd, timeout=15.0)
                if result and result.get("status") == "success":
                    print(f"  🎙️ [SpeakerDOA] ✅ 已转向说话人 ({angle}°)")
                else:
                    print(f"  🎙️ [SpeakerDOA] ⚠️ 转向结果未知, 继续执行")
                return angle
            except queue.Empty:
                print(f"  🎙️ [SpeakerDOA] ⏰ 未检测到说话人 ({timeout}s), 跳过转向")
                return None

        print(f"  🎙️ [SpeakerDOA] ⚠️ ROS 不可用, 模拟面向说话人 ({label})")
        print(f"  🎙️ [SpeakerDOA] (真实环境: 订阅 {SPEAKER_DOA_RESULT_TOPIC}, 获取 angle_deg 后发送转向指令)")
        time.sleep(1)
        return None

    def _face_speaker_guest1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        angle = self._face_speaker("guest1")
        result = {"facing_guest1": True}
        if angle is not None:
            result["guest1_angle"] = angle
        return result

    def _face_speaker_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        angle = self._face_speaker("guest2")
        result = {"facing_guest2": True}
        if angle is not None:
            result["guest2_angle"] = angle
        return result

    def _face_speaker_conversation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        angle = self._face_speaker("conversation")
        result = {"facing_speaker": True}
        if angle is not None:
            result["speaker_angle"] = angle
        return result

    # ============================================================
    # 声纹录入
    # ============================================================

    def _enroll_speaker(self, speaker_name: str, role: str, duration: float = 5.0):
        if not self._ensure_enroll_init():
            print(f"  🎙️ [SpeakerDOA] ⚠️ ROS 不可用, 跳过声纹录入: {role}")
            return

        cmd = json.dumps({
            "command": "enroll",
            "speaker_name": f"{role}_{speaker_name}",
            "duration": duration,
        })
        self._rospy.loginfo(f"[SpeakerDOA] 发送录入指令: {cmd}")
        self._pub_enroll.publish(self._String(data=cmd))

        print(f"  🎙️ [SpeakerDOA] 正在录入 {role} 声纹 ({speaker_name}), 请说话 {duration} 秒...")
        print(f"  🎙️ [SpeakerDOA] 发布话题: {SPEAKER_DOA_ENROLL_TOPIC}")
        time.sleep(duration + 1)
        print(f"  🎙️ [SpeakerDOA] {role} 声纹录入完成")

    # ============================================================
    # 寻找 host / 跟随 host
    # ============================================================

    def _find_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("host_found"):
            print("  🎙️ [SpeakerDOA] 视觉已找到 host, 录入声纹...")
            self._enroll_speaker("host", "host", duration=5.0)
            return {"host_found": True, "host_location": context.get("host_location", "unknown")}

        print("  🎙️ [SpeakerDOA] 通过声源定位寻找 host...")
        result = self._bridge.call(SPEAKER_DOA_CMD_LOCATE, timeout=60.0)
        if result is None:
            print("  🎙️ [SpeakerDOA] ⚠️ ROS 不可用, 模拟查找 host")
            time.sleep(5)
            return {"host_found": True, "host_location": "living_room"}

        if result.get("status") == "success":
            data = result.get("data", {})
            angle = data.get("angle_deg", 0)
            speaker = data.get("speaker_name", "unknown")
            print(f"  🎙️ [SpeakerDOA] 检测到声源: {speaker}, 角度: {angle}°")
            self._enroll_speaker("host", "host", duration=5.0)
            return {"host_found": True, "host_location": f"angle_{angle}", "host_angle": angle}

        print("  🎙️ [SpeakerDOA] 声源定位失败")
        return {"host_found": False}

    def _follow_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🎙️ [SpeakerDOA] 声源跟踪跟随 host...")
        result = self._bridge.call(SPEAKER_DOA_CMD_TRACK, timeout=120.0)
        if result is None:
            print("  🎙️ [SpeakerDOA] ⚠️ ROS 不可用, 模拟跟随")
            time.sleep(5)
            return {"at_destination": True, "destination": "storage_room"}

        if result.get("status") == "success":
            data = result.get("data", {})
            dest = data.get("destination", "storage_room")
            print(f"  🎙️ [SpeakerDOA] 跟随完成, 到达: {dest}")
            return {"at_destination": True, "destination": dest}

        return {"at_destination": False}


# ============================================================
# 兼容别名
# ============================================================

ASRModule = SpeechModule
