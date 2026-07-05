"""
Task1 子模块 — 各执行模块基类和具体实现

模块划分 (按能力维度):
    DoorbellModule    — 门铃: wait_for_doorbell_1, wait_for_doorbell_2
    NavigationModule   — 导航: go_to_door, guide_guest1, return_to_start, seat_guest2...
    SpeechModule       — 语音: ask_guest1_info, pick_up_guest2, introduce_guests...
    VisionModule       — 视觉: describe_guest1, find_host, follow_host
    ManipulationModule — 操作: point_empty_seat, wait_for_bag, place_bag

每个模块通过 execute(state_id, context) 被总控调用,
返回产出数据 dict (合并到任务上下文), 或 None。

导航和视觉模块通过 ROSTopicBridge 与外部 ROS 节点通信:
    发送指令到 command 话题 → 等待 result 话题的响应。
操作模块直接调用 object_search 里的空座识别、接包和夹爪脚本。

话题名称和指令字符串定义在 topic_names.py, 方便统一修改。
"""

import json
import math
import os
import queue
import re
import select
import shlex
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from task1_receptionist.state_definitions import Task1StateID
from task1_receptionist.sub_modules.topic_names import (
    DOORBELL_TOPIC,
    DOORBELL_TIMEOUT,
    NAV_COMMAND_TOPIC,
    NAV_RESULT_TOPIC,
    NAV_TIMEOUT,
    NAV_CMD_FOLLOW_PERSON,
    VISION_COMMAND_TOPIC,
    VISION_RESULT_TOPIC,
    VISION_TIMEOUT,
    VISION_CMD_DESCRIBE_PERSON,
    VISION_CMD_FIND_HOST,
    VISION_CMD_TRACK_PERSON,
    ASR_TOPIC,
    ASR_SEGMENT_TOPIC,
)

NAVIGATION_DIR = Path(__file__).resolve().parents[1] / "navigation"
NAV_SET_GOAL_SCRIPT = NAVIGATION_DIR / "set_nav_goal" / "set_nav_goal.py"
NAV_CHASSIS_MOVE_SCRIPT = NAVIGATION_DIR / "chassis_move_util.py"

# 固定导航点位集中写在这里，格式为 [x, y, yaw_deg]。
# 需要临时改门口、起点、客厅等坐标时，优先改这个表。
DEFAULT_NAV_GOALS: Dict[str, Optional[List[float]]] = {
    "door": [1.922, -1.477, -90.825],
    "start": [0.769, 0.526, 0],
    "living_room": [0.769, 0.526, 0],
    "host_interaction": [1.922, -1.477, 90],
}

# 导航前底盘微调表。所有距离单位为米，旋转单位为度。
# forward_m: 正数前进，负数后退；left_m: 正数左移，负数右移；
# rotate_deg: 正数逆时针，负数顺时针。
# order 控制执行顺序，可选项为 "forward"、"left"、"rotate"。
DEFAULT_PRE_NAV_ADJUSTMENTS: Dict[str, Dict[str, Any]] = {
    "go_to_door": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
    "guide_guest1_living_room": {
        "enabled": False,
        "forward_m": -0.2,
        "left_m": 0.0,
        "rotate_deg": 180.0,
        "order": ["forward", "left", "rotate"],
    },
    "point_empty_seat_approach": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
    "return_to_start": {
        "enabled": False,
        "forward_m": -0.2,
        "left_m": 0.0,
        "rotate_deg": 180.0,
        "order": ["forward", "left", "rotate"],
    },
    "pick_up_guest2_door": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
    "seat_guest2_living_room": {
        "enabled": False,
        "forward_m": -0.2,
        "left_m": 0.0,
        "rotate_deg": 180.0,
        "order": ["forward", "left", "rotate"],
    },
    "seat_guest2_empty_seat_approach": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
    "handover_approach": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
    "find_host_interaction": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
    "follow_host": {
        "enabled": False,
        "forward_m": 0.0,
        "left_m": 0.0,
        "rotate_deg": 0.0,
        "order": ["forward", "left", "rotate"],
    },
}

OBJECT_SEARCH_DIR = Path(__file__).resolve().parents[1] / "object_search"
OBJECT_SEARCH_OUTPUT_JSON = OBJECT_SEARCH_DIR / "latest_empty_seat.json"
OBJECT_SEARCH_HAND_JSON = OBJECT_SEARCH_DIR / "latest_handover_hand.json"
OBJECT_SEARCH_BAG_JSON = OBJECT_SEARCH_DIR / "latest_bag.json"
OBJECT_SEARCH_PEOPLE_JSON = OBJECT_SEARCH_DIR / "latest_people.json"
OBJECT_SEARCH_CACHE_STATUS_JSON = OBJECT_SEARCH_DIR / "latest_vision_cache_status.json"
OBJECT_SEARCH_RUN_EMPTY = OBJECT_SEARCH_DIR / "run_object_search.sh"
OBJECT_SEARCH_RUN_HAND = OBJECT_SEARCH_DIR / "run_hand_search.sh"
OBJECT_SEARCH_RUN_VISION_CACHE = OBJECT_SEARCH_DIR / "run_vision_cache.sh"
OBJECT_SEARCH_POINT_ARM = OBJECT_SEARCH_DIR / "point_empty_seat_arm.py"
OBJECT_SEARCH_APPROACH_HANDOVER_ARM = OBJECT_SEARCH_DIR / "approach_handover_arm.py"
OBJECT_SEARCH_CAMERA_TO_LEFTBASE = OBJECT_SEARCH_DIR / "camera_middle_to_leftbase.txt"
POINTING_TARGET_FRAME = "left_arm_base"
LEFT_GRIPPER_POSITION_TOPIC = "/motion_control/position_control_gripper_left"
DEFAULT_NAV_BASE_FRAME = "base_link"
DEFAULT_EMPTY_SEAT_LEFTBASE_FRAME = "left_arm_base_link"
DEFAULT_EMPTY_SEAT_MAX_FRAMES = 30
DEFAULT_OBJECT_SEARCH_CACHE_MAX_AGE_SEC = 3.0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _interactive_enabled() -> bool:
    return _env_flag("TASK1_INTERACTIVE", False)


def _assume_yes_enabled() -> bool:
    return _env_flag("TASK1_ASSUME_YES", not _interactive_enabled())


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _known_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {
        "unknown",
        "unknown clothing",
        "none",
        "null",
        "n/a",
    }


def _dedupe_texts(values: List[str], limit: Optional[int] = None) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not _known_text(text):
            continue
        key = " ".join(text.lower().replace(";", " ").replace(",", " ").split())
        if key in seen:
            continue
        deduped.append(text)
        seen.add(key)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _clean_clothing_type(value: Any) -> str:
    text = str(value).strip()
    if not _known_text(text):
        return ""
    # Fashionpedia labels can be verbose, for example "top, t-shirt, sweatshirt".
    return text.split(",")[0].strip()


def _normalize_clothing_phrase(value: Any) -> str:
    text = str(value).strip(" ,")
    if not _known_text(text):
        return ""
    if "," in text:
        text = text.split(",", 1)[0].strip()
    return text


def _split_clothing_text(value: Any) -> List[str]:
    if not _known_text(value):
        return []
    phrases: List[str] = []
    for part in str(value).split(";"):
        text = _normalize_clothing_phrase(part)
        if _known_text(text):
            phrases.append(text)
    return phrases


def _format_clothing_item(item: Dict[str, Any]) -> Optional[str]:
    color = item.get("color")
    ctype = _clean_clothing_type(item.get("type") or item.get("category"))
    if _known_text(color) and _known_text(ctype):
        return f"{color} {ctype}"
    if _known_text(ctype):
        return str(ctype)
    if _known_text(color):
        return f"{color} clothing"
    return None


def _guest1_visual_attributes(appearance: Dict[str, Any], limit: int = 4) -> List[str]:
    clothing_attributes: List[str] = []
    for key in ("clothing_summary", "cloth_summary"):
        clothing_attributes.extend(_split_clothing_text(appearance.get(key)))

    for item in appearance.get("cloth_items") or []:
        if not isinstance(item, dict):
            continue
        formatted = _format_clothing_item(item)
        if formatted:
            clothing_attributes.append(formatted)

    color = appearance.get("cloth_color")
    ctype = appearance.get("cloth_type")
    if _known_text(color) and _known_text(ctype):
        for color_part, type_part in zip(str(color).split("/"), str(ctype).split("/")):
            clean_type = _clean_clothing_type(type_part)
            if _known_text(color_part) and _known_text(clean_type):
                clothing_attributes.append(f"{color_part.strip()} {clean_type}")

    if not clothing_attributes:
        clothing = appearance.get("clothing")
        if isinstance(clothing, list):
            for item in clothing:
                if isinstance(item, dict):
                    formatted = _format_clothing_item(item)
                    if formatted:
                        clothing_attributes.append(formatted)
                else:
                    clothing_attributes.extend(_split_clothing_text(item))
        else:
            clothing_attributes.extend(_split_clothing_text(clothing))

    attributes = _dedupe_texts(clothing_attributes, limit=limit)
    if not attributes:
        raw_attrs = appearance.get("visual_attributes")
        if isinstance(raw_attrs, list):
            raw_phrases: List[str] = []
            for item in raw_attrs:
                raw_phrases.extend(_split_clothing_text(item))
            attributes = _dedupe_texts(raw_phrases, limit=limit)

    if appearance.get("has_glasses") is True:
        attributes.append("glasses")

    return _dedupe_texts(attributes, limit=limit)


def _canonical_clothing_phrase(value: Any) -> str:
    text = str(value).strip().lower()
    if not _known_text(text):
        return ""
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _guest_clothing_signature(appearance: Dict[str, Any], limit: int = 8) -> Dict[str, Any]:
    if not isinstance(appearance, dict):
        appearance = {}
    attributes = _guest1_visual_attributes(appearance, limit=limit)
    keys = _dedupe_texts(
        [_canonical_clothing_phrase(attribute) for attribute in attributes],
        limit=None,
    )
    return {
        "attributes": attributes,
        "keys": keys,
    }


def _guest_clothing_match_score(
    target_appearance: Dict[str, Any],
    candidate_person: Dict[str, Any],
) -> Tuple[float, str]:
    target_keys = set((_guest_clothing_signature(target_appearance).get("keys") or []))
    candidate_keys = set((_guest_clothing_signature(candidate_person).get("keys") or []))
    if not target_keys:
        return 0.0, "target_has_no_clothing_signature"
    if not candidate_keys:
        return 0.0, "candidate_has_no_clothing_signature"

    exact = target_keys & candidate_keys
    score = float(len(exact) * 10)
    partial_matches = []
    for target_key in target_keys:
        for candidate_key in candidate_keys:
            if target_key == candidate_key:
                continue
            if target_key in candidate_key or candidate_key in target_key:
                partial_matches.append((target_key, candidate_key))
                score += 3.0

    coverage = len(exact) / max(1, len(target_keys))
    if exact:
        reason = f"exact={sorted(exact)} coverage={coverage:.2f}"
    elif partial_matches:
        reason = f"partial={partial_matches[:2]} coverage={coverage:.2f}"
    else:
        reason = "no_overlap"
    return score, reason


def _join_natural(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _speech_clothing_phrase(text: str) -> str:
    lowered = text.lower().strip()
    if lowered in {"glasses", "sunglasses"}:
        return lowered
    article_free_words = ("pants", "jeans", "shorts", "trousers", "shoes", "socks")
    if any(word in lowered.split() for word in article_free_words):
        return text
    if lowered.startswith(("a ", "an ", "the ")):
        return text
    return f"an {text}" if lowered[:1] in {"a", "e", "i", "o", "u"} else f"a {text}"


def _guest1_description_sentence(context: Dict[str, Any]) -> tuple:
    guest1 = context.get("guest1_name") or "the first guest"
    guest2 = context.get("guest2_name") or "guest"
    appearance = context.get("guest1_appearance")
    if not isinstance(appearance, dict):
        appearance = {}

    attributes = _guest1_visual_attributes(appearance)
    if attributes:
        clothing_phrases = [_speech_clothing_phrase(item) for item in attributes if item != "glasses"]
        glasses = "glasses" in {item.lower() for item in attributes}
        details = _join_natural(clothing_phrases)
        if glasses:
            details = f"{details}, and glasses" if details else "glasses"
        return (
            f"{guest2}, {guest1} is wearing {details}.",
            True,
        )

    return (
        f"{guest2}, I could not get a reliable clothing description for {guest1}.",
        False,
    )


def _fresh_json_cache(path: Path, max_age_sec: float) -> Optional[float]:
    if max_age_sec <= 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return None
    age = max(0.0, time.time() - float(timestamp))
    return age if age <= max_age_sec else None


def _fresh_vision_cache_status(max_age_sec: float) -> Optional[float]:
    if max_age_sec <= 0:
        return None
    try:
        payload = json.loads(OBJECT_SEARCH_CACHE_STATUS_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if payload.get("status") != "running":
        return None
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return None
    age = max(0.0, time.time() - float(timestamp))
    return age if age <= max_age_sec else None


def _object_search_cache_state(
    result_path: Path,
    label: str,
    max_age_sec: float,
    log_prefix: str,
) -> str:
    age = _fresh_json_cache(result_path, max_age_sec)
    if age is not None:
        print(
            f"{log_prefix} 复用后台{label}缓存: "
            f"{result_path.name} age={age:.1f}s"
        )
        return "fresh"

    status_age = _fresh_vision_cache_status(max_age_sec)
    if status_age is not None:
        print(
            f"{log_prefix} 后台视觉缓存正在运行，但 {result_path.name} "
            f"尚未在 {max_age_sec:.1f}s 内刷新；本次不重复打开相机 "
            f"(status age={status_age:.1f}s)"
        )
        return "daemon_running"

    return "miss"


class PeopleVisionCacheManager:
    _process: Optional[subprocess.Popen] = None
    _last_start_time = 0.0

    @classmethod
    def ensure_running(
        cls,
        log_prefix: str,
        max_age_sec: float,
    ) -> bool:
        if _fresh_json_cache(OBJECT_SEARCH_PEOPLE_JSON, max_age_sec) is not None:
            return True
        if _fresh_vision_cache_status(max_age_sec) is not None:
            return True
        if cls._process is not None and cls._process.poll() is None:
            return True
        if time.time() - cls._last_start_time < 8.0:
            return False

        cls._last_start_time = time.time()
        if not OBJECT_SEARCH_RUN_VISION_CACHE.exists():
            print(f"{log_prefix} 人物视觉缓存启动脚本不存在: {OBJECT_SEARCH_RUN_VISION_CACHE}")
            return False

        argv = [
            "bash",
            str(OBJECT_SEARCH_RUN_VISION_CACHE),
            "--no-display",
            "--disable-empty-seat",
            "--disable-handover-hand",
            "--disable-bag",
            "--people-every",
            os.environ.get("TASK1_PEOPLE_CACHE_EVERY", "0.3"),
            "--people-conf",
            os.environ.get("TASK1_PEOPLE_SEARCH_CONF", "0.2"),
        ]
        argv.extend(_split_env_args("TASK1_PEOPLE_CACHE_ARGS", log_prefix))
        log_path = Path(
            os.environ.get(
                "TASK1_PEOPLE_CACHE_LOG",
                str(OBJECT_SEARCH_DIR / "latest_people_cache.log"),
            )
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        printable = " ".join(shlex.quote(str(part)) for part in argv)
        print(f"{log_prefix} 人物缓存不新鲜，启动轻量人物视觉缓存")
        print(f"    $ {printable}")
        try:
            log_file = open(log_path, "ab")
            cls._process = subprocess.Popen(
                [str(part) for part in argv],
                cwd=str(OBJECT_SEARCH_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return True
        except Exception as exc:
            print(f"{log_prefix} 启动人物视觉缓存失败: {exc}")
            return False


def _split_env_args(env_name: str, log_prefix: str) -> List[str]:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return []
    try:
        return shlex.split(value)
    except ValueError as exc:
        print(f"{log_prefix} {env_name} 参数解析失败: {exc}")
        return []


def _load_matrix4(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        values = [
            float(value)
            for value in line.strip("[]").replace(",", " ").split()
            if value
        ]
        rows.append(values)
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"{path} 应为 4x4 外参矩阵")
    return rows


def _apply_matrix4(matrix: List[List[float]], xyz: List[float]) -> List[float]:
    vector = [float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0]
    result = [sum(matrix[row][col] * vector[col] for col in range(4)) for row in range(4)]
    w = result[3] if abs(result[3]) > 1e-9 else 1.0
    return [result[0] / w, result[1] / w, result[2] / w]


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

    def publish(self, command: str) -> bool:
        """只发布指令, 不等待 result。ROS 不可用时返回 False。"""
        if not self._ensure_init():
            return False

        self._drain_queue()
        print(f"    [ROS] 发布: \"{command}\" → {self._command_topic}")
        self._pub.publish(self._String(data=command))
        self._rospy.loginfo(f"[ROSBridge] 发送指令: {command} → {self._command_topic}")
        return True

    def call(self, command: str, timeout: Optional[float] = None) -> Optional[dict]:
        """
        发送指令, 等待结果

        Args:
            command: 指令字符串 (如 "导航到门口")
            timeout: 等待结果秒数；None 使用默认超时，不再等待控制台输入

        Returns:
            结果 dict (解析 JSON), 或 None (超时/ROS不可用)
        """
        if not self.publish(command):
            return None

        try:
            wait_timeout = self._default_timeout if timeout is None else timeout
            if wait_timeout is None or wait_timeout <= 0.0:
                raw = self._result_queue.get_nowait()
            else:
                raw = self._result_queue.get(timeout=max(0.0, wait_timeout))
            try:
                result = json.loads(raw)
                print(f"    [ROS] 收到: {result}")
                self._rospy.loginfo(f"[ROSBridge] 收到结果: {result}")
                return result
            except (json.JSONDecodeError, TypeError):
                print(f"    [ROS] 收到 (raw): {raw}")
                return {"status": "success", "raw": raw}
        except queue.Empty:
            wait_timeout = self._default_timeout if timeout is None else timeout
            if wait_timeout is not None and wait_timeout > 0.0:
                print(f"    [ROS] 等待 {self._result_topic} 超时 {wait_timeout:.1f}s，继续降级")
            else:
                print(f"    [ROS] 未收到 {self._result_topic}，继续降级")
            self._rospy.loginfo(f"[ROSBridge] 等待结果超时: {command}")
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
    ROS 可用时自动等待 /doorbell/detected；仅交互模式允许控制台 Enter 手动跳过。
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

    @staticmethod
    def _stdin_enter_requested() -> bool:
        if not _interactive_enabled():
            return False
        try:
            if not sys.stdin or sys.stdin.closed:
                return False
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return False
            return sys.stdin.readline() != ""
        except Exception:
            return False

    @staticmethod
    def _doorbell_timeout_sec() -> float:
        raw = os.environ.get("TASK1_DOORBELL_TIMEOUT_SEC", "").strip()
        if not raw:
            return DOORBELL_TIMEOUT
        try:
            return float(raw)
        except ValueError:
            print(f"  🔔 [Doorbell] TASK1_DOORBELL_TIMEOUT_SEC 参数无效: {raw}")
            return DOORBELL_TIMEOUT

    def _wait_for_doorbell(self, label: str) -> Dict[str, Any]:
        if self.auto_simulate:
            print(f"  🔔 [Doorbell] ✅ 自动模拟门铃 ({label})")
            return {"doorbell_rang": True}

        if self._ensure_init():
            self._drain_queue()
            print(f"  🔔 [Doorbell] 已连接 ROS 门铃话题: {DOORBELL_TOPIC}")
            timeout_sec = self._doorbell_timeout_sec()
            deadline = None if timeout_sec <= 0 else time.time() + timeout_sec
            if deadline is None:
                if _interactive_enabled():
                    print(f"  🔔 [Doorbell] 等待 {label} 门铃；按 Enter 可手动跳过")
                else:
                    print(f"  🔔 [Doorbell] 等待 {label} 门铃")
            else:
                if _interactive_enabled():
                    print(f"  🔔 [Doorbell] 等待 {label} 门铃，最多 {timeout_sec:.1f}s；按 Enter 可手动跳过")
                else:
                    print(f"  🔔 [Doorbell] 等待 {label} 门铃，最多 {timeout_sec:.1f}s")

            while True:
                try:
                    event = self._bell_queue.get(timeout=0.1)
                    label_name = event.get("label") or event.get("raw") or "Doorbell"
                    probability = event.get("probability")
                    prob_text = f", prob={probability}" if probability is not None else ""
                    print(f"  🔔 [Doorbell] ✅ 检测到门铃 ({label}): {label_name}{prob_text}")
                    return {"doorbell_rang": True, "doorbell_event": event}
                except queue.Empty:
                    pass

                if self._stdin_enter_requested():
                    print(f"  🔔 [Doorbell] ✅ 手动确认门铃 ({label})")
                    return {"doorbell_rang": True, "manual": True}

                if self._rospy and self._rospy.is_shutdown():
                    print("  🔔 [Doorbell] ROS 已关闭，停止等待门铃")
                    return {"doorbell_rang": False, "timeout": False}

                if deadline is not None and time.time() >= deadline:
                    print(f"  🔔 [Doorbell] ⚠️ 等待门铃超时 ({label})")
                    return {"doorbell_rang": False, "timeout": True}
        else:
            print(f"  🔔 [Doorbell] ROS 门铃话题不可用")

        if not _interactive_enabled():
            print(f"  🔔 [Doorbell] 非交互模式: 无门铃事件，按超时/失败处理并交给状态机降级")
            return {"doorbell_rang": False, "unavailable": True}

        try:
            input(f"  🔔 门铃事件 ({label}) 完成后按 Enter 继续: ")
            print(f"  🔔 [Doorbell] ✅ 手动确认门铃 ({label})")
            return {"doorbell_rang": True, "manual": True}
        except (EOFError, KeyboardInterrupt):
            print(f"  🔔 [Doorbell] 输入中断, 按未检测到门铃处理")
            return {"doorbell_rang": False, "unavailable": True}

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if state_id == Task1StateID.WAIT_FOR_DOORBELL_1:
            result = self._wait_for_doorbell("guest1")
            data = {"doorbell_1_rang": result["doorbell_rang"]}
            if result.get("doorbell_event"):
                data["doorbell_1_event"] = result["doorbell_event"]
            if result.get("manual"):
                data["doorbell_1_manual"] = True
            return data
        elif state_id == Task1StateID.WAIT_FOR_DOORBELL_2:
            result = self._wait_for_doorbell("guest2")
            data = {"doorbell_2_rang": result["doorbell_rang"]}
            if result.get("doorbell_event"):
                data["doorbell_2_event"] = result["doorbell_event"]
            if result.get("manual"):
                data["doorbell_2_manual"] = True
            return data
        return None


# ============================================================
# 导航模块
# ============================================================

class NavigationModule(BaseSubModule):
    """
    导航子模块

    GO_TO_DOOR、GUIDE_GUEST1、RETURN_TO_START、PICK_UP_GUEST2、SEAT_GUEST2
    和 FIND_HOST
    的固定点位部分直接调用 navigation/set_nav_goal/set_nav_goal.py。POINT_EMPTY_SEAT
    和 SEAT_GUEST2 的空座接近点由 object_search 结果动态计算后发送给 move_base。
    """

    def __init__(
        self,
        speech=None,
        door_goal: Optional[List[float]] = None,
        start_goal: Optional[List[float]] = None,
        living_room_goal: Optional[List[float]] = None,
        host_interaction_goal: Optional[List[float]] = None,
        object_search_max_frames: int = 120,
        object_search_skip_vision: bool = False,
    ):
        super().__init__("navigation")
        self._speech = speech
        self._bridge = ROSTopicBridge(NAV_COMMAND_TOPIC, NAV_RESULT_TOPIC, NAV_TIMEOUT)
        self._nav_goals = {
            "door": self._configured_goal("door", "门口坐标", override=door_goal),
            "start": self._configured_goal("start", "起点坐标", override=start_goal),
            "living_room": self._configured_goal(
                "living_room",
                "客厅坐标",
                override=living_room_goal,
            ),
            "host_interaction": self._configured_goal(
                "host_interaction",
                "host 交互点坐标",
                override=host_interaction_goal,
            ),
        }
        self._nav_map_frame = os.environ.get("TASK1_NAV_MAP_FRAME", "map")
        self._nav_base_frame = os.environ.get("TASK1_NAV_BASE_FRAME", DEFAULT_NAV_BASE_FRAME)
        self._move_base_action = os.environ.get("TASK1_MOVE_BASE_ACTION", "/move_base")
        self._nav_server_timeout = _env_float("TASK1_NAV_SERVER_TIMEOUT", 10.0)
        self._nav_goal_timeout = _env_float(
            "TASK1_NAV_GOAL_TIMEOUT",
            _env_float("TASK1_DOOR_NAV_TIMEOUT", 120.0),
        )
        self._nav_goal_no_wait = _env_flag(
            "TASK1_NAV_GOAL_NO_WAIT",
            _env_flag("TASK1_DOOR_NAV_NO_WAIT", False),
        )
        self._pre_nav_adjust_linear_speed = _env_float(
            "TASK1_PRE_NAV_ADJUST_LINEAR_SPEED", 0.15
        )
        self._pre_nav_adjust_angular_speed_deg = _env_float(
            "TASK1_PRE_NAV_ADJUST_ANGULAR_SPEED_DEG", 25.0
        )
        self._pre_nav_adjust_wait_subscriber = _env_float(
            "TASK1_PRE_NAV_ADJUST_WAIT_SUBSCRIBER", 3.0
        )
        self._pre_nav_adjust_dry_run = _env_flag("TASK1_PRE_NAV_ADJUST_DRY_RUN", False)
        self._object_search_max_frames = _env_int(
            "TASK1_OBJECT_SEARCH_MAX_FRAMES", object_search_max_frames
        )
        self._empty_seat_max_frames = _env_int(
            "TASK1_EMPTY_SEAT_MAX_FRAMES", DEFAULT_EMPTY_SEAT_MAX_FRAMES
        )
        self._object_search_skip_vision = object_search_skip_vision or _env_flag(
            "TASK1_OBJECT_SEARCH_SKIP_VISION", False
        )
        self._object_search_prefer_cache = _env_flag(
            "TASK1_OBJECT_SEARCH_PREFER_CACHE", True
        )
        self._object_search_cache_max_age = _env_float(
            "TASK1_OBJECT_SEARCH_CACHE_MAX_AGE_SEC",
            DEFAULT_OBJECT_SEARCH_CACHE_MAX_AGE_SEC,
        )
        self._empty_seat_approach_distance = _env_float(
            "TASK1_EMPTY_SEAT_APPROACH_DISTANCE", 1.5
        )
        self._handover_approach_distance = _env_float(
            "TASK1_HANDOVER_APPROACH_DISTANCE", 0.7
        )
        self._handover_recognition_delay_sec = max(
            0.0,
            _env_float("TASK1_HANDOVER_RECOGNITION_DELAY_SEC", 5.0),
        )
        self._handover_cache_wait_sec = max(
            0.0,
            _env_float("TASK1_HANDOVER_CACHE_WAIT_SEC", 6.0),
        )
        self._empty_seat_tf_timeout = _env_float("TASK1_EMPTY_SEAT_TF_TIMEOUT", 3.0)
        self._seat_guest2_description_delay_sec = max(
            0.0,
            _env_float("TASK1_SEAT_GUEST2_DESCRIPTION_DELAY_SEC", 1.0),
        )
        self._empty_seat_leftbase_frame = os.environ.get(
            "TASK1_EMPTY_SEAT_LEFTBASE_FRAME",
            DEFAULT_EMPTY_SEAT_LEFTBASE_FRAME,
        )
        self._empty_seat_camera_to_leftbase = Path(
            os.environ.get(
                "TASK1_EMPTY_SEAT_CAMERA_TO_LEFTBASE",
                str(OBJECT_SEARCH_CAMERA_TO_LEFTBASE),
            )
        )
        self._host_follower = HostFollowController()

    def _normalize_goal(
        self,
        values: Optional[List[float]],
        label: str = "目标坐标",
    ) -> Optional[List[float]]:
        if values is None:
            return None
        if len(values) != 3:
            print(f"  🚪 [Nav] {label}必须是 3 个数: x y yaw_deg")
            return None
        try:
            return [float(values[0]), float(values[1]), float(values[2])]
        except (TypeError, ValueError):
            print(f"  🚪 [Nav] {label}解析失败: {values}")
            return None

    def _parse_goal_text(self, text: str, label: str) -> Optional[List[float]]:
        parts = text.replace(",", " ").split()
        if len(parts) != 3:
            print(f"  🚪 [Nav] {label} 格式应为: x y yaw_deg")
            return None
        return self._normalize_goal(parts, label)

    def _goal_from_env(self, key: str, label: str) -> Optional[List[float]]:
        prefix = f"TASK1_{key.upper()}"
        packed = os.environ.get(f"{prefix}_GOAL", "").strip()
        if packed:
            return self._parse_goal_text(packed, f"{prefix}_GOAL")

        x = os.environ.get(f"{prefix}_X")
        y = os.environ.get(f"{prefix}_Y")
        yaw = os.environ.get(f"{prefix}_YAW_DEG") or os.environ.get(f"{prefix}_YAW")
        if x is None and y is None and yaw is None:
            return None
        if x is None or y is None or yaw is None:
            print(f"  🚪 [Nav] {prefix}_X/{prefix}_Y/{prefix}_YAW_DEG 需要同时设置")
            return None
        return self._normalize_goal([x, y, yaw], label)

    def _configured_goal(
        self,
        key: str,
        label: str,
        override: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        return (
            self._normalize_goal(override, label)
            or self._goal_from_env(key, label)
            or self._normalize_goal(DEFAULT_NAV_GOALS.get(key), label)
        )

    def _extra_args(self, env_name: str) -> List[str]:
        value = os.environ.get(env_name, "").strip()
        if not value:
            return []
        try:
            return shlex.split(value)
        except ValueError as exc:
            print(f"  🚪 [Nav] {env_name} 参数解析失败: {exc}")
            return []

    def _run_empty_seat_search(self) -> bool:
        cache_state = (
            _object_search_cache_state(
                OBJECT_SEARCH_OUTPUT_JSON,
                "空座识别",
                self._object_search_cache_max_age,
                "  🚪 [Nav:EmptySeat]",
            )
            if self._object_search_prefer_cache
            else "miss"
        )
        if cache_state == "fresh":
            pass
        elif cache_state == "daemon_running":
            return False
        elif self._object_search_skip_vision:
            print("  🚪 [Nav:EmptySeat] 跳过空座视觉刷新，复用 latest_empty_seat.json")
        else:
            argv = [
                "bash",
                str(OBJECT_SEARCH_RUN_EMPTY),
                "--no-display",
                "--max-frames",
                str(self._empty_seat_max_frames),
                "--stop-on-success",
            ]
            argv.extend(self._extra_args("TASK1_EMPTY_SEAT_SEARCH_ARGS"))
            printable = " ".join(shlex.quote(str(part)) for part in argv)
            print("  🚪 [Nav:EmptySeat] 运行空座识别")
            print(f"    $ {printable}")
            try:
                completed = subprocess.run(
                    [str(part) for part in argv],
                    cwd=str(OBJECT_SEARCH_DIR),
                )
            except FileNotFoundError as exc:
                print(f"  🚪 [Nav:EmptySeat] 空座识别启动失败: {exc}")
                return False
            if completed.returncode != 0:
                print(f"  🚪 [Nav:EmptySeat] 空座识别失败: returncode={completed.returncode}")
                return False

        target = self._load_empty_seat_target_for_nav()
        return target is not None

    def _read_handover_payload_for_nav(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(OBJECT_SEARCH_HAND_JSON.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"  🚪 [Nav:Handover] 未找到接包识别结果: {OBJECT_SEARCH_HAND_JSON}")
            return None
        except json.JSONDecodeError as exc:
            print(f"  🚪 [Nav:Handover] 接包识别结果不是合法 JSON: {exc}")
            return None
        except OSError as exc:
            print(f"  🚪 [Nav:Handover] 读取接包识别结果失败: {exc}")
            return None

    def _handover_payload_success(
        self,
        payload: Optional[Dict[str, Any]],
        min_timestamp: Optional[float] = None,
    ) -> bool:
        if not payload or payload.get("status") != "success":
            return False
        if min_timestamp is None:
            return True
        timestamp = payload.get("timestamp")
        return isinstance(timestamp, (int, float)) and float(timestamp) >= min_timestamp - 0.05

    def _wait_for_handover_hand_cache_for_nav(
        self,
        min_timestamp: Optional[float] = None,
    ) -> bool:
        if self._handover_cache_wait_sec <= 0.0:
            return False
        deadline = time.time() + self._handover_cache_wait_sec
        last_status = None
        print(
            "  🚪 [Nav:Handover] 等待后台接包目标刷新 "
            f"最多 {self._handover_cache_wait_sec:.1f}s"
        )
        while time.time() < deadline:
            payload = self._read_handover_payload_for_nav()
            if self._handover_payload_success(payload, min_timestamp=min_timestamp):
                return True
            if payload:
                last_status = payload.get("status")
            time.sleep(0.2)
        print(f"  🚪 [Nav:Handover] 等待接包目标超时: last_status={last_status}")
        return False

    def _run_handover_hand_search_for_nav(
        self,
        min_timestamp: Optional[float] = None,
    ) -> bool:
        cache_state = (
            _object_search_cache_state(
                OBJECT_SEARCH_HAND_JSON,
                "拿包手腕识别",
                self._object_search_cache_max_age,
                "  🚪 [Nav:Handover]",
            )
            if self._object_search_prefer_cache
            else "miss"
        )
        if cache_state == "fresh":
            payload = self._read_handover_payload_for_nav()
            if self._handover_payload_success(payload, min_timestamp=min_timestamp):
                return True
            return self._wait_for_handover_hand_cache_for_nav(min_timestamp=min_timestamp)
        if cache_state == "daemon_running":
            return self._wait_for_handover_hand_cache_for_nav(min_timestamp=min_timestamp)

        if self._object_search_skip_vision:
            print("  🚪 [Nav:Handover] 跳过拿包手腕视觉刷新，复用 latest_handover_hand.json")
            payload = self._read_handover_payload_for_nav()
            return self._handover_payload_success(payload, min_timestamp=min_timestamp)

        argv = [
            "bash",
            str(OBJECT_SEARCH_RUN_HAND),
            "--no-display",
            "--max-frames",
            str(self._object_search_max_frames),
        ]
        argv.extend(self._extra_args("TASK1_HAND_SEARCH_ARGS"))
        printable = " ".join(shlex.quote(str(part)) for part in argv)
        print("  🚪 [Nav:Handover] 运行拿包手腕识别")
        print(f"    $ {printable}")
        try:
            completed = subprocess.run(
                [str(part) for part in argv],
                cwd=str(OBJECT_SEARCH_DIR),
            )
        except FileNotFoundError as exc:
            print(f"  🚪 [Nav:Handover] 拿包手腕识别启动失败: {exc}")
            return False
        if completed.returncode != 0:
            print(f"  🚪 [Nav:Handover] 拿包手腕识别失败: returncode={completed.returncode}")
            return False

        payload = self._read_handover_payload_for_nav()
        return self._handover_payload_success(payload, min_timestamp=min_timestamp)

    def _load_empty_seat_target_for_nav(self) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(OBJECT_SEARCH_OUTPUT_JSON.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"  🚪 [Nav:EmptySeat] 未找到空座识别结果: {OBJECT_SEARCH_OUTPUT_JSON}")
            return None
        except json.JSONDecodeError as exc:
            print(f"  🚪 [Nav:EmptySeat] 空座识别结果不是合法 JSON: {exc}")
            return None

        if payload.get("status") != "success":
            print(f"  🚪 [Nav:EmptySeat] 空座识别状态不是 success: {payload.get('status')}")
            return None

        seat = payload.get("best_empty_seat") or {}
        position = seat.get("position") or {}
        camera_xyz = position.get("camera_xyz_m")
        leftbase_xyz = position.get("leftbase_xyz_m") or position.get("left_arm_base_xyz_m")
        target_frame_xyz = position.get("target_frame_xyz_m")
        calibrated_xyz = position.get("calibrated_xyz_m")
        xyz = (
            leftbase_xyz
            or target_frame_xyz
            or calibrated_xyz
            or camera_xyz
        )
        if not isinstance(xyz, list) or len(xyz) != 3:
            print("  🚪 [Nav:EmptySeat] 空座识别结果缺少可用目标坐标")
            return None

        return {
            "frame_id": payload.get("coordinate_frame") or POINTING_TARGET_FRAME,
            "xyz_m": [float(value) for value in xyz],
            "camera_xyz_m": (
                [float(value) for value in camera_xyz]
                if isinstance(camera_xyz, list) and len(camera_xyz) == 3
                else None
            ),
            "leftbase_xyz_m": (
                [float(value) for value in leftbase_xyz]
                if isinstance(leftbase_xyz, list) and len(leftbase_xyz) == 3
                else None
            ),
            "target_frame_xyz_m": (
                [float(value) for value in target_frame_xyz]
                if isinstance(target_frame_xyz, list) and len(target_frame_xyz) == 3
                else None
            ),
            "calibrated_xyz_m": (
                [float(value) for value in calibrated_xyz]
                if isinstance(calibrated_xyz, list) and len(calibrated_xyz) == 3
                else None
            ),
            "class_name": seat.get("class_name"),
            "confidence": seat.get("confidence"),
            "depth_m": seat.get("depth_m"),
            "bbox": seat.get("bbox"),
            "source_json": str(OBJECT_SEARCH_OUTPUT_JSON),
            "timestamp": payload.get("timestamp"),
        }

    def _load_handover_target_for_nav(self) -> Optional[Dict[str, Any]]:
        payload = self._read_handover_payload_for_nav()
        if not payload or payload.get("status") != "success":
            status = None if payload is None else payload.get("status")
            print(f"  🚪 [Nav:Handover] 接包识别状态不是 success: {status}")
            return None

        target_record = payload.get("handover_target") or payload.get("best_hand") or {}
        position = target_record.get("position") or {}
        camera_xyz = position.get("camera_xyz_m")
        leftbase_xyz = position.get("leftbase_xyz_m") or position.get("left_arm_base_xyz_m")
        target_frame_xyz = position.get("target_frame_xyz_m")
        calibrated_xyz = position.get("calibrated_xyz_m")
        xyz = leftbase_xyz or target_frame_xyz or calibrated_xyz or camera_xyz
        if not isinstance(xyz, list) or len(xyz) != 3:
            print("  🚪 [Nav:Handover] 接包识别结果缺少可用目标坐标")
            return None

        return {
            "frame_id": payload.get("coordinate_frame") or POINTING_TARGET_FRAME,
            "xyz_m": [float(value) for value in xyz],
            "camera_xyz_m": (
                [float(value) for value in camera_xyz]
                if isinstance(camera_xyz, list) and len(camera_xyz) == 3
                else None
            ),
            "leftbase_xyz_m": (
                [float(value) for value in leftbase_xyz]
                if isinstance(leftbase_xyz, list) and len(leftbase_xyz) == 3
                else None
            ),
            "target_frame_xyz_m": (
                [float(value) for value in target_frame_xyz]
                if isinstance(target_frame_xyz, list) and len(target_frame_xyz) == 3
                else None
            ),
            "calibrated_xyz_m": (
                [float(value) for value in calibrated_xyz]
                if isinstance(calibrated_xyz, list) and len(calibrated_xyz) == 3
                else None
            ),
            "target_type": target_record.get("target_type"),
            "selection_label": target_record.get("selection_label"),
            "selection_priority": target_record.get("selection_priority"),
            "handover_position_source": target_record.get("handover_position_source"),
            "handover_depth_m": target_record.get("handover_depth_m"),
            "wrist_position": target_record.get("wrist_position"),
            "matched_object_position": target_record.get("matched_object_position"),
            "hand_side": target_record.get("hand_side"),
            "confidence": target_record.get("confidence"),
            "object_class_name": target_record.get("object_class_name"),
            "object_confidence": target_record.get("object_confidence"),
            "bag_class_name": target_record.get("bag_class_name"),
            "bag_confidence": target_record.get("bag_confidence"),
            "hand_object_distance_px": target_record.get("hand_object_distance_px"),
            "hand_bag_distance_px": target_record.get("hand_bag_distance_px"),
            "source_json": str(OBJECT_SEARCH_HAND_JSON),
            "timestamp": payload.get("timestamp"),
        }

    def _empty_seat_tf_frame_candidates(self, frame_id: str) -> List[str]:
        configured = os.environ.get("TASK1_EMPTY_SEAT_TF_FRAMES", "").strip()
        candidates: List[str] = []
        if configured:
            candidates.extend(part.strip() for part in configured.split(",") if part.strip())
        if frame_id:
            candidates.append(frame_id)
        if frame_id in {"leftbase", "left_arm_base"}:
            candidates.extend(["left_arm_base_link"])

        deduped: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _leftbase_frame_candidates(self) -> List[str]:
        configured = os.environ.get("TASK1_EMPTY_SEAT_LEFTBASE_FRAME", "").strip()
        candidates = [configured] if configured else []
        candidates.extend([self._empty_seat_leftbase_frame, DEFAULT_EMPTY_SEAT_LEFTBASE_FRAME])
        return [candidate for index, candidate in enumerate(candidates) if candidate and candidate not in candidates[:index]]

    def _camera_xyz_to_leftbase(self, camera_xyz: List[float]) -> Optional[List[float]]:
        try:
            matrix = _load_matrix4(self._empty_seat_camera_to_leftbase)
            return _apply_matrix4(matrix, camera_xyz)
        except FileNotFoundError:
            print(f"  🚪 [Nav:EmptySeat] 未找到相机到 leftbase 外参: {self._empty_seat_camera_to_leftbase}")
            return None
        except (ValueError, OSError) as exc:
            print(f"  🚪 [Nav:EmptySeat] 相机到 leftbase 外参读取失败: {exc}")
            return None

    def _load_ros_tf_modules(self):
        try:
            import rospy
            import tf
            from geometry_msgs.msg import PointStamped
            from tf.transformations import euler_from_quaternion
        except ImportError as exc:
            print(f"  🚪 [Nav:EmptySeat] 缺少 ROS/TF Python 依赖: {exc}")
            return None
        return rospy, tf, PointStamped, euler_from_quaternion

    def _ensure_ros_node_for_nav(self, rospy) -> None:
        if not rospy.core.is_initialized():
            rospy.init_node("task1_empty_seat_navigation", anonymous=True, disable_signals=True)

    def _transform_point_with_tf(
        self,
        source_frame: str,
        target_frame: str,
        xyz: List[float],
        label: str,
        log_prefix: str = "  🚪 [Nav:EmptySeat]",
    ) -> Optional[List[float]]:
        modules = self._load_ros_tf_modules()
        if modules is None:
            return None
        rospy, tf, PointStamped, _ = modules
        self._ensure_ros_node_for_nav(rospy)

        listener = tf.TransformListener()
        time.sleep(0.3)

        point = PointStamped()
        point.header.frame_id = source_frame
        point.header.stamp = rospy.Time(0)
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        try:
            listener.waitForTransform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(self._empty_seat_tf_timeout),
            )
            mapped = listener.transformPoint(target_frame, point)
            return [
                float(mapped.point.x),
                float(mapped.point.y),
                float(mapped.point.z),
            ]
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            print(f"{log_prefix} {label} TF 转换失败 {source_frame}->{target_frame}: {exc}")
            return None

    def _transform_empty_seat_to_base_link(self, target: Dict[str, Any]) -> Optional[List[float]]:
        camera_xyz = target.get("camera_xyz_m")
        leftbase_xyz: Optional[List[float]] = None
        if isinstance(camera_xyz, list) and len(camera_xyz) == 3:
            leftbase_xyz = self._camera_xyz_to_leftbase(camera_xyz)
            if leftbase_xyz is not None:
                print(
                    "  🚪 [Nav:EmptySeat] tracker-style 相机点转 leftbase: "
                    f"{[round(value, 3) for value in leftbase_xyz]} m"
                )

        if leftbase_xyz is None:
            for key in ("leftbase_xyz_m", "target_frame_xyz_m", "calibrated_xyz_m"):
                xyz = target.get(key)
                if isinstance(xyz, list) and len(xyz) == 3:
                    leftbase_xyz = [float(value) for value in xyz]
                    print(
                        "  🚪 [Nav:EmptySeat] 复用 JSON leftbase 点: "
                        f"{[round(value, 3) for value in leftbase_xyz]} m"
                    )
                    break

        if leftbase_xyz is not None:
            for source_frame in self._leftbase_frame_candidates():
                base_xyz = self._transform_point_with_tf(
                    source_frame,
                    self._nav_base_frame,
                    leftbase_xyz,
                    "leftbase 到 base_link",
                )
                if base_xyz is not None:
                    print(
                        "  🚪 [Nav:EmptySeat] 空座 base_link 坐标: "
                        f"x={base_xyz[0]:.3f}, y={base_xyz[1]:.3f}, z={base_xyz[2]:.3f}"
                    )
                    return base_xyz

        source_frame = str(target.get("frame_id") or "").strip()
        xyz = target.get("xyz_m")
        if source_frame and isinstance(xyz, list) and len(xyz) == 3:
            if source_frame == self._nav_base_frame:
                return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            for candidate in self._empty_seat_tf_frame_candidates(source_frame):
                base_xyz = self._transform_point_with_tf(
                    candidate,
                    self._nav_base_frame,
                    [float(value) for value in xyz],
                    "空座源 frame 到 base_link",
                )
                if base_xyz is not None:
                    return base_xyz

        return None

    def _handover_camera_xyz_to_leftbase(self, camera_xyz: List[float]) -> Optional[List[float]]:
        try:
            matrix = _load_matrix4(self._empty_seat_camera_to_leftbase)
            return _apply_matrix4(matrix, camera_xyz)
        except FileNotFoundError:
            print(f"  🚪 [Nav:Handover] 未找到相机到 leftbase 外参: {self._empty_seat_camera_to_leftbase}")
            return None
        except (ValueError, OSError) as exc:
            print(f"  🚪 [Nav:Handover] 相机到 leftbase 外参读取失败: {exc}")
            return None

    def _transform_handover_target_to_base_link(self, target: Dict[str, Any]) -> Optional[List[float]]:
        camera_xyz = target.get("camera_xyz_m")
        leftbase_xyz: Optional[List[float]] = None
        if isinstance(camera_xyz, list) and len(camera_xyz) == 3:
            leftbase_xyz = self._handover_camera_xyz_to_leftbase(camera_xyz)
            if leftbase_xyz is not None:
                print(
                    "  🚪 [Nav:Handover] tracker-style 相机点转 leftbase: "
                    f"{[round(value, 3) for value in leftbase_xyz]} m"
                )

        if leftbase_xyz is None:
            for key in ("leftbase_xyz_m", "target_frame_xyz_m", "calibrated_xyz_m"):
                xyz = target.get(key)
                if isinstance(xyz, list) and len(xyz) == 3:
                    leftbase_xyz = [float(value) for value in xyz]
                    print(
                        "  🚪 [Nav:Handover] 复用 JSON leftbase 点: "
                        f"{[round(value, 3) for value in leftbase_xyz]} m"
                    )
                    break

        if leftbase_xyz is not None:
            for source_frame in self._leftbase_frame_candidates():
                base_xyz = self._transform_point_with_tf(
                    source_frame,
                    self._nav_base_frame,
                    leftbase_xyz,
                    "接包 leftbase 到 base_link",
                    log_prefix="  🚪 [Nav:Handover]",
                )
                if base_xyz is not None:
                    print(
                        "  🚪 [Nav:Handover] 接包目标 base_link 坐标: "
                        f"x={base_xyz[0]:.3f}, y={base_xyz[1]:.3f}, z={base_xyz[2]:.3f}"
                    )
                    return base_xyz

        source_frame = str(target.get("frame_id") or "").strip()
        xyz = target.get("xyz_m")
        if source_frame and isinstance(xyz, list) and len(xyz) == 3:
            if source_frame == self._nav_base_frame:
                return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            for candidate in self._empty_seat_tf_frame_candidates(source_frame):
                base_xyz = self._transform_point_with_tf(
                    candidate,
                    self._nav_base_frame,
                    [float(value) for value in xyz],
                    "接包源 frame 到 base_link",
                    log_prefix="  🚪 [Nav:Handover]",
                )
                if base_xyz is not None:
                    return base_xyz

        return None

    def _project_base_point_to_map(
        self,
        base_xyz: List[float],
        current_pose: List[float],
    ) -> List[float]:
        robot_x, robot_y, yaw_deg = current_pose[:3]
        yaw = math.radians(float(yaw_deg))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        base_x, base_y, base_z = [float(value) for value in base_xyz[:3]]
        return [
            float(robot_x) + base_x * cos_yaw - base_y * sin_yaw,
            float(robot_y) + base_x * sin_yaw + base_y * cos_yaw,
            base_z,
        ]

    def _transform_empty_seat_to_map(self, target: Dict[str, Any]) -> Optional[List[float]]:
        modules = self._load_ros_tf_modules()
        if modules is None:
            return None
        rospy, tf, PointStamped, _ = modules
        self._ensure_ros_node_for_nav(rospy)

        source_frame = str(target.get("frame_id") or "").strip()
        xyz = target.get("xyz_m")
        if not isinstance(xyz, list) or len(xyz) != 3:
            print("  🚪 [Nav:EmptySeat] 空座目标坐标无效")
            return None
        if source_frame == self._nav_map_frame:
            return [float(xyz[0]), float(xyz[1]), float(xyz[2])]

        listener = tf.TransformListener()
        time.sleep(0.3)

        errors: List[str] = []
        for candidate in self._empty_seat_tf_frame_candidates(source_frame):
            point = PointStamped()
            point.header.frame_id = candidate
            point.header.stamp = rospy.Time(0)
            point.point.x = float(xyz[0])
            point.point.y = float(xyz[1])
            point.point.z = float(xyz[2])
            try:
                listener.waitForTransform(
                    self._nav_map_frame,
                    candidate,
                    rospy.Time(0),
                    rospy.Duration(self._empty_seat_tf_timeout),
                )
                mapped = listener.transformPoint(self._nav_map_frame, point)
                return [
                    float(mapped.point.x),
                    float(mapped.point.y),
                    float(mapped.point.z),
                ]
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                errors.append(f"{candidate}: {exc}")

        print("  🚪 [Nav:EmptySeat] 无法把空座坐标转换到 map")
        for error in errors:
            print(f"    - {error}")
        return None

    def _transform_handover_target_to_map(self, target: Dict[str, Any]) -> Optional[List[float]]:
        modules = self._load_ros_tf_modules()
        if modules is None:
            return None
        rospy, tf, PointStamped, _ = modules
        self._ensure_ros_node_for_nav(rospy)

        source_frame = str(target.get("frame_id") or "").strip()
        xyz = target.get("xyz_m")
        if not isinstance(xyz, list) or len(xyz) != 3:
            print("  🚪 [Nav:Handover] 接包目标坐标无效")
            return None
        if source_frame == self._nav_map_frame:
            return [float(xyz[0]), float(xyz[1]), float(xyz[2])]

        listener = tf.TransformListener()
        time.sleep(0.3)

        errors: List[str] = []
        for candidate in self._empty_seat_tf_frame_candidates(source_frame):
            point = PointStamped()
            point.header.frame_id = candidate
            point.header.stamp = rospy.Time(0)
            point.point.x = float(xyz[0])
            point.point.y = float(xyz[1])
            point.point.z = float(xyz[2])
            try:
                listener.waitForTransform(
                    self._nav_map_frame,
                    candidate,
                    rospy.Time(0),
                    rospy.Duration(self._empty_seat_tf_timeout),
                )
                mapped = listener.transformPoint(self._nav_map_frame, point)
                return [
                    float(mapped.point.x),
                    float(mapped.point.y),
                    float(mapped.point.z),
                ]
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
                errors.append(f"{candidate}: {exc}")

        print("  🚪 [Nav:Handover] 无法把接包目标转换到 map")
        for error in errors:
            print(f"    - {error}")
        return None

    def _read_current_map_pose(
        self,
        log_prefix: str = "  🚪 [Nav:EmptySeat]",
    ) -> Optional[List[float]]:
        modules = self._load_ros_tf_modules()
        if modules is None:
            return None
        rospy, tf, _, euler_from_quaternion = modules
        self._ensure_ros_node_for_nav(rospy)

        listener = tf.TransformListener()
        time.sleep(0.3)
        try:
            listener.waitForTransform(
                self._nav_map_frame,
                self._nav_base_frame,
                rospy.Time(0),
                rospy.Duration(self._empty_seat_tf_timeout),
            )
            trans, rot = listener.lookupTransform(
                self._nav_map_frame,
                self._nav_base_frame,
                rospy.Time(0),
            )
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            print(f"{log_prefix} 读取当前机器人 map 位姿失败: {exc}")
            return None

        _, _, yaw = euler_from_quaternion(rot)
        return [float(trans[0]), float(trans[1]), math.degrees(float(yaw))]

    def _normalize_angle_deg(self, angle: float) -> float:
        while angle >= 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def _compute_target_approach_goal(
        self,
        target_map_xyz: List[float],
        current_pose: List[float],
        distance: float,
    ) -> List[float]:
        target_x, target_y = float(target_map_xyz[0]), float(target_map_xyz[1])
        robot_x, robot_y = float(current_pose[0]), float(current_pose[1])
        dx = robot_x - target_x
        dy = robot_y - target_y
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            yaw_rad = math.radians(float(current_pose[2]))
            dx = -math.cos(yaw_rad)
            dy = -math.sin(yaw_rad)
            norm = 1.0

        distance = max(0.1, float(distance))
        goal_x = target_x + dx / norm * distance
        goal_y = target_y + dy / norm * distance
        goal_yaw = math.degrees(math.atan2(target_y - goal_y, target_x - goal_x))
        return [goal_x, goal_y, self._normalize_angle_deg(goal_yaw)]

    def _compute_empty_seat_approach_goal(
        self,
        seat_map_xyz: List[float],
        current_pose: List[float],
    ) -> List[float]:
        return self._compute_target_approach_goal(
            seat_map_xyz,
            current_pose,
            self._empty_seat_approach_distance,
        )

    def _get_pre_nav_adjustment(self, key: Optional[str]) -> Dict[str, Any]:
        if not key:
            return {}
        config = DEFAULT_PRE_NAV_ADJUSTMENTS.get(key, {})
        return dict(config)

    def _pre_nav_motion_commands(
        self,
        config: Dict[str, Any],
    ) -> List[tuple]:
        commands: Dict[str, tuple] = {}

        forward_m = float(config.get("forward_m", 0.0) or 0.0)
        if abs(forward_m) > 1e-6:
            commands["forward"] = (
                "forward" if forward_m > 0.0 else "back",
                abs(forward_m),
            )

        left_m = float(config.get("left_m", 0.0) or 0.0)
        if abs(left_m) > 1e-6:
            commands["left"] = (
                "left" if left_m > 0.0 else "right",
                abs(left_m),
            )

        rotate_deg = float(config.get("rotate_deg", 0.0) or 0.0)
        if abs(rotate_deg) > 1e-6:
            commands["rotate"] = ("rotate", rotate_deg)

        order = config.get("order") or ["forward", "left", "rotate"]
        result: List[tuple] = []
        for name in order:
            command = commands.pop(str(name), None)
            if command:
                result.append(command)
        result.extend(commands.values())
        return result

    def _run_pre_nav_adjustment(
        self,
        key: Optional[str],
        label: str,
    ) -> Dict[str, Any]:
        config = self._get_pre_nav_adjustment(key)
        if not config or not config.get("enabled", False):
            return {"status": "skipped", "key": key}

        motions = self._pre_nav_motion_commands(config)
        if not motions:
            print(f"  🚪 [NavAdjust] {label}: 已启用但没有配置移动量，跳过")
            return {"status": "success", "key": key, "motions": []}

        print(f"  🚪 [NavAdjust] {label}: 执行导航前底盘微调 ({key})")
        executed = []
        for command, value in motions:
            argv = [
                sys.executable,
                str(NAV_CHASSIS_MOVE_SCRIPT),
                command,
                f"{value:.6f}",
                "--linear-speed",
                str(self._pre_nav_adjust_linear_speed),
                "--angular-speed-deg",
                str(self._pre_nav_adjust_angular_speed_deg),
                "--wait-subscriber",
                str(self._pre_nav_adjust_wait_subscriber),
            ]
            if self._pre_nav_adjust_dry_run:
                argv.append("--dry-run")
            argv.extend(self._extra_args("TASK1_PRE_NAV_ADJUST_ARGS"))

            printable = " ".join(shlex.quote(str(part)) for part in argv)
            print(f"    $ {printable}")
            try:
                completed = subprocess.run(argv, cwd=str(NAV_CHASSIS_MOVE_SCRIPT.parent))
            except FileNotFoundError as exc:
                return {"status": "failed", "key": key, "error": str(exc), "motions": executed}
            except Exception as exc:
                return {"status": "failed", "key": key, "error": str(exc), "motions": executed}

            executed.append({"command": command, "value": value})
            if completed.returncode != 0:
                return {
                    "status": "failed",
                    "key": key,
                    "error": f"chassis_move_util.py returncode={completed.returncode}",
                    "motions": executed,
                }

        return {"status": "success", "key": key, "motions": executed}

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
            Task1StateID.GO_TO_DOOR: self._go_to_door,
            Task1StateID.GUIDE_GUEST1: self._guide_guest1,
            Task1StateID.POINT_EMPTY_SEAT: self._approach_empty_seat_guest1,
            Task1StateID.RETURN_TO_START: self._return_to_start,
            Task1StateID.PICK_UP_GUEST2: self._pick_up_guest2,
            Task1StateID.SEAT_GUEST2: self._seat_guest2,
            Task1StateID.REQUEST_GUEST2_BAG: self._approach_handover_guest2,
            Task1StateID.FIND_HOST: self._go_to_host_interaction,
            Task1StateID.FOLLOW_HOST: self._follow_host,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    def _send_nav_goal(
        self,
        key: str,
        label: str,
        goal: List[float],
        adjust_key: Optional[str] = None,
        after_start: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ) -> dict:
        adjust_result = self._run_pre_nav_adjustment(adjust_key, label)
        if adjust_result.get("status") == "failed":
            return {
                "status": "failed",
                "error": "pre_nav_adjust_failed",
                "data": {"adjustment": adjust_result},
            }

        x, y, yaw_deg = goal
        argv = [
            sys.executable,
            str(NAV_SET_GOAL_SCRIPT),
            f"{x:.6f}",
            f"{y:.6f}",
            f"{yaw_deg:.6f}",
            "--map-frame",
            self._nav_map_frame,
            "--move-base",
            self._move_base_action,
            "--server-timeout",
            str(self._nav_server_timeout),
        ]
        if self._nav_goal_timeout and self._nav_goal_timeout > 0.0:
            argv.extend(["--timeout", str(self._nav_goal_timeout)])
        if self._nav_goal_no_wait:
            argv.append("--no-wait")

        printable = " ".join(shlex.quote(str(part)) for part in argv)
        print(f"  🚪 [Nav] 使用 set_nav_goal.py 发送{label}")
        print(
            "  🚪 [Nav] 目标: "
            f"frame={self._nav_map_frame}, x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.1f}deg"
        )
        print(f"    $ {printable}")
        callback_data: Dict[str, Any] = {}
        try:
            if after_start is None:
                completed = subprocess.run(
                    argv,
                    cwd=str(NAV_SET_GOAL_SCRIPT.parent),
                    timeout=None,
                )
                returncode = completed.returncode
            else:
                process = subprocess.Popen(
                    argv,
                    cwd=str(NAV_SET_GOAL_SCRIPT.parent),
                )
                time.sleep(self._seat_guest2_description_delay_sec)
                if process.poll() is None or process.returncode == 0:
                    try:
                        data = after_start()
                        if data:
                            callback_data.update(data)
                    except Exception as exc:
                        print(f"  💬 [Speech] 导航后播报 guest1 外貌失败: {exc}")
                        callback_data["guest1_described"] = False
                returncode = process.wait()
        except FileNotFoundError as exc:
            return {"status": "failed", "error": f"set_nav_goal.py not found: {exc}"}
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

        if returncode == 0:
            data = {
                "destination": key,
                "goal": {"x": x, "y": y, "yaw_deg": yaw_deg},
            }
            data.update(callback_data)
            return {
                "status": "success",
                "data": data,
            }
        return {
            "status": "failed",
            "error": f"set_nav_goal.py returncode={returncode}",
        }

    def _navigate_to_named_goal(
        self,
        key: str,
        label: str,
        adjust_key: Optional[str] = None,
    ) -> dict:
        goal = self._nav_goals.get(key)
        if not goal:
            env_name = f"TASK1_{key.upper()}_GOAL"
            arg_name = f"--{key.replace('_', '-')}-goal"
            print(f"  🚪 [Nav] 未配置{label}，无法执行导航")
            print(f"  🚪 [Nav] 配置方式 1: python task1_controller.py {arg_name} X Y YAW_DEG")
            print(f"  🚪 [Nav] 配置方式 2: export {env_name}='X Y YAW_DEG'")
            print("  🚪 [Nav] 配置方式 3: 修改 base_module.py 里的 DEFAULT_NAV_GOALS")
            return {"status": "failed", "error": f"missing_{key}_goal"}

        return self._send_nav_goal(key, label, goal, adjust_key=adjust_key)

    def _navigate_to_door_goal(self, adjust_key: Optional[str] = None) -> dict:
        return self._navigate_to_named_goal("door", "门口目标", adjust_key=adjust_key)

    def _navigate_to_start_goal(self, adjust_key: Optional[str] = None) -> dict:
        return self._navigate_to_named_goal("start", "起点目标", adjust_key=adjust_key)

    def _navigate_to_living_room_goal(self, adjust_key: Optional[str] = None) -> dict:
        return self._navigate_to_named_goal("living_room", "客厅目标", adjust_key=adjust_key)

    def _navigate_to_living_room_goal_with_description(
        self,
        context: Dict[str, Any],
        adjust_key: Optional[str] = None,
    ) -> dict:
        goal = self._nav_goals.get("living_room")
        if not goal:
            return self._navigate_to_living_room_goal(adjust_key=adjust_key)
        return self._send_nav_goal(
            "living_room",
            "客厅目标",
            goal,
            adjust_key=adjust_key,
            after_start=lambda: self._describe_guest1_after_nav_start(context),
        )

    def _describe_guest1_after_nav_start(self, context: Dict[str, Any]) -> Dict[str, Any]:
        sentence, described = _guest1_description_sentence(context)
        if self._speech is None:
            print(f"  💬 [Speech] 未注入语音接口，跳过播报: {sentence}")
            return {"guest1_described": False}
        print("  💬 [Speech] 客厅导航已启动，向 guest2 描述 guest1 外貌")
        self._speech.say(sentence)
        return {"guest1_described": described}

    def _navigate_to_host_interaction_goal(self, adjust_key: Optional[str] = None) -> dict:
        return self._navigate_to_named_goal(
            "host_interaction",
            "host 交互点目标",
            adjust_key=adjust_key,
        )

    def _navigate_to_empty_seat_approach(
        self,
        label: str,
        adjust_key: Optional[str] = None,
    ) -> dict:
        print(
            f"  🚪 [Nav:EmptySeat] 为 {label} 识别最近空座并计算 "
            f"{self._empty_seat_approach_distance:.1f}m 接近点"
        )
        adjust_result = self._run_pre_nav_adjustment(adjust_key, "空座识别/接近目标")
        if adjust_result.get("status") == "failed":
            return {
                "status": "failed",
                "error": "pre_nav_adjust_failed",
                "data": {"adjustment": adjust_result},
            }

        if not self._run_empty_seat_search():
            return {"status": "failed", "error": "empty_seat_search_failed"}

        target = self._load_empty_seat_target_for_nav()
        if not target:
            return {"status": "failed", "error": "missing_empty_seat_target"}

        print(
            "  🚪 [Nav:EmptySeat] 最近空座目标 "
            f"{target['frame_id']}: {target['xyz_m']} m"
        )
        current_pose = self._read_current_map_pose()
        if current_pose is None:
            return {
                "status": "failed",
                "error": "current_pose_failed",
                "data": {"empty_seat_target": target},
            }

        seat_base_xyz = self._transform_empty_seat_to_base_link(target)
        if seat_base_xyz is not None:
            seat_map_xyz = self._project_base_point_to_map(seat_base_xyz, current_pose)
        else:
            print("  🚪 [Nav:EmptySeat] tracker-style base_link 转换失败，尝试直接转换到 map")
            seat_map_xyz = self._transform_empty_seat_to_map(target)
            if seat_map_xyz is None:
                return {
                    "status": "failed",
                    "error": "empty_seat_tf_failed",
                    "data": {
                        "empty_seat_target": target,
                        "empty_seat_base_xyz": seat_base_xyz,
                    },
                }
            seat_base_xyz = None

        approach_goal = self._compute_empty_seat_approach_goal(seat_map_xyz, current_pose)
        print(
            "  🚪 [Nav:EmptySeat] 空座 map 坐标: "
            f"x={seat_map_xyz[0]:.3f}, y={seat_map_xyz[1]:.3f}, z={seat_map_xyz[2]:.3f}"
        )
        print(
            "  🚪 [Nav:EmptySeat] 接近点: "
            f"x={approach_goal[0]:.3f}, y={approach_goal[1]:.3f}, "
            f"yaw={approach_goal[2]:.1f}deg, distance={self._empty_seat_approach_distance:.2f}m"
        )

        result = self._send_nav_goal(
            "empty_seat_approach",
            "空座接近目标",
            approach_goal,
        )
        result.setdefault("data", {})
        result["data"].update(
            {
                "goal": {
                    "x": approach_goal[0],
                    "y": approach_goal[1],
                    "yaw_deg": approach_goal[2],
                },
                "empty_seat_target": target,
                "empty_seat_base_xyz": seat_base_xyz,
                "empty_seat_map_xyz": seat_map_xyz,
                "approach_distance_m": self._empty_seat_approach_distance,
            }
        )
        return result

    def _navigate_to_handover_approach(
        self,
        label: str,
        adjust_key: Optional[str] = None,
    ) -> dict:
        print(
            f"  🚪 [Nav:Handover] 为 {label} 识别接包目标并计算 "
            f"{self._handover_approach_distance:.1f}m 接近点"
        )
        adjust_result = self._run_pre_nav_adjustment(adjust_key, "接包识别/接近目标")
        if adjust_result.get("status") == "failed":
            return {
                "status": "failed",
                "error": "pre_nav_adjust_failed",
                "data": {"adjustment": adjust_result},
            }

        if self._handover_recognition_delay_sec > 0.0:
            print(
                "  🚪 [Nav:Handover] 已提示客人递包，等待 "
                f"{self._handover_recognition_delay_sec:.1f}s 后开始识别"
            )
            time.sleep(self._handover_recognition_delay_sec)

        search_start_time = time.time()
        if not self._run_handover_hand_search_for_nav(min_timestamp=search_start_time):
            return {"status": "failed", "error": "handover_target_search_failed"}

        target = self._load_handover_target_for_nav()
        if not target:
            return {"status": "failed", "error": "missing_handover_target"}

        object_name = target.get("object_class_name") or target.get("bag_class_name") or "none"
        object_conf = target.get("object_confidence")
        if object_conf is None:
            object_conf = target.get("bag_confidence")
        print(
            "  🚪 [Nav:Handover] 接包目标 "
            f"{target['frame_id']}: {target['xyz_m']} m, "
            f"type={target.get('target_type')} "
            f"source={target.get('handover_position_source')} "
            f"object={object_name} "
            f"conf={object_conf}"
        )

        current_pose = self._read_current_map_pose(log_prefix="  🚪 [Nav:Handover]")
        if current_pose is None:
            return {"status": "failed", "error": "current_pose_failed", "target": target}

        target_base_xyz = self._transform_handover_target_to_base_link(target)
        if target_base_xyz is not None:
            target_map_xyz = self._project_base_point_to_map(target_base_xyz, current_pose)
        else:
            print("  🚪 [Nav:Handover] base_link 转换失败，尝试直接转换到 map")
            target_map_xyz = self._transform_handover_target_to_map(target)
            if target_map_xyz is None:
                return {"status": "failed", "error": "handover_target_tf_failed", "target": target}
            target_base_xyz = None

        approach_goal = self._compute_target_approach_goal(
            target_map_xyz,
            current_pose,
            self._handover_approach_distance,
        )
        print(
            "  🚪 [Nav:Handover] 接包目标 map 坐标: "
            f"x={target_map_xyz[0]:.3f}, y={target_map_xyz[1]:.3f}, z={target_map_xyz[2]:.3f}"
        )
        print(
            "  🚪 [Nav:Handover] 接近点: "
            f"x={approach_goal[0]:.3f}, y={approach_goal[1]:.3f}, "
            f"yaw={approach_goal[2]:.1f}deg, distance={self._handover_approach_distance:.2f}m"
        )

        result = self._send_nav_goal(
            "handover_approach",
            "接包接近目标",
            approach_goal,
        )
        if result.get("status") == "success":
            result.setdefault("data", {})
            result["data"].update(
                {
                    "handover_hand_target": target,
                    "handover_target_base_xyz": target_base_xyz,
                    "handover_target_map_xyz": target_map_xyz,
                    "approach_distance_m": self._handover_approach_distance,
                    "completed_time": time.time(),
                }
            )
        return result

    def _navigate(self, command: str, timeout: Optional[float] = None) -> dict:
        if self._bridge.publish(command):
            print(f"  🚪 [Nav] 已发送导航指令: {command}")
            if _interactive_enabled():
                try:
                    input("  🚪 [Nav] 导航完成后按 Enter 继续: ")
                except (EOFError, KeyboardInterrupt):
                    print("  🚪 [Nav] 输入中断, 继续执行")
            else:
                wait_sec = timeout if timeout is not None else self._nav_goal_timeout
                if wait_sec and wait_sec > 0.0:
                    print(f"  🚪 [Nav] 非交互模式: 已发出导航指令，等待由下游/超时控制 ({wait_sec:.1f}s)")
            return {"status": "success"}

        print(f"  🚪 [Nav] ⚠️ ROS 不可用, 模拟: {command}")
        time.sleep(5)
        return {"status": "success"}

    def _go_to_door(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚪 [Nav] 导航到门口...")
        result = self._navigate_to_door_goal(adjust_key="go_to_door")
        if result.get("status") == "success":
            print("  🚪 [Nav] 已到达门口")
            return {"robot_at_door": True, "robot_position": "door"}
        print(f"  🚪 [Nav] 导航失败: {result.get('error', 'unknown')}")
        return {"robot_at_door": False, "robot_position": context.get("robot_position", "unknown")}

    def _guide_guest1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest = context.get("guest1_name", "guest1")
        print(f"  🚶 [Nav] 带 {guest} 去客厅...")
        result = self._navigate_to_living_room_goal(adjust_key="guide_guest1_living_room")
        if result.get("status") == "success":
            print(f"  🚶 [Nav] 已带 {guest} 到达客厅")
            return {"guest1_in_living_room": True, "robot_position": "living_room"}
        print(f"  🚶 [Nav] 导航失败: {result.get('error', 'unknown')}")
        return {"guest1_in_living_room": False, "robot_position": context.get("robot_position", "unknown")}

    def _approach_empty_seat_guest1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest = context.get("guest1_name", "guest1")
        result = self._navigate_to_empty_seat_approach(
            guest,
            adjust_key="point_empty_seat_approach",
        )
        data = result.get("data") or {}
        if result.get("status") == "success":
            return {
                "empty_seat_approach_reached": True,
                "robot_position": "empty_seat_approach",
                "empty_seat_approach_goal": data.get("goal"),
                "empty_seat_target": data.get("empty_seat_target"),
                "empty_seat_base_xyz": data.get("empty_seat_base_xyz"),
                "empty_seat_map_xyz": data.get("empty_seat_map_xyz"),
                "guest1_seat_target": data.get("empty_seat_target"),
                "guest1_seat_base_xyz": data.get("empty_seat_base_xyz"),
                "guest1_seat_map_xyz": data.get("empty_seat_map_xyz"),
            }
        print(f"  🚪 [Nav:EmptySeat] 接近空座失败: {result.get('error', 'unknown')}")
        return {
            "empty_seat_approach_reached": False,
            "robot_position": context.get("robot_position", "unknown"),
            "empty_seat_approach_goal": data.get("goal"),
            "empty_seat_target": data.get("empty_seat_target"),
            "empty_seat_base_xyz": data.get("empty_seat_base_xyz"),
            "empty_seat_map_xyz": data.get("empty_seat_map_xyz"),
            "guest1_seat_target": data.get("empty_seat_target"),
            "guest1_seat_base_xyz": data.get("empty_seat_base_xyz"),
            "guest1_seat_map_xyz": data.get("empty_seat_map_xyz"),
            "empty_seat_approach_error": result.get("error", "unknown"),
        }

    def _return_to_start(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚶 [Nav] 返回起点...")
        result = self._navigate_to_start_goal(adjust_key="return_to_start")
        if result.get("status") == "success":
            print("  🚶 [Nav] 已返回起点")
            return {"robot_at_start": True, "robot_position": "home"}
        print(f"  🚶 [Nav] 导航失败: {result.get('error', 'unknown')}")
        return {"robot_at_start": False, "robot_position": context.get("robot_position", "unknown")}

    def _pick_up_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚪 [Nav] 导航到门口接 guest2...")
        result = self._navigate_to_door_goal(adjust_key="pick_up_guest2_door")
        if result.get("status") == "success":
            print("  🚪 [Nav] 已到达门口")
            return {"guest2_at_door": True, "robot_position": "door"}
        print(f"  🚪 [Nav] 导航失败: {result.get('error', 'unknown')}")
        return {"guest2_at_door": False, "robot_position": context.get("robot_position", "unknown")}

    def _seat_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest = context.get("guest2_name", "guest2")
        print(f"  🚶 [Nav] 带 {guest} 去客厅...")
        living_result = self._navigate_to_living_room_goal_with_description(
            context,
            adjust_key="seat_guest2_living_room",
        )
        if living_result.get("status") != "success":
            print(f"  🚶 [Nav] 导航失败: {living_result.get('error', 'unknown')}")
            return {"guest2_seated": False, "robot_position": context.get("robot_position", "unknown")}

        print(f"  🚶 [Nav] 已带 {guest} 到达客厅，继续接近空座")
        approach_result = self._navigate_to_empty_seat_approach(
            guest,
            adjust_key="seat_guest2_empty_seat_approach",
        )
        data = approach_result.get("data") or {}
        if approach_result.get("status") == "success":
            living_data = living_result.get("data") or {}
            return {
                "guest2_seated": True,
                "seat_number": 2,
                "guest1_described": bool(living_data.get("guest1_described")),
                "robot_position": "empty_seat_approach",
                "guest2_empty_seat_approach_reached": True,
                "guest2_empty_seat_approach_goal": data.get("goal"),
                "guest2_empty_seat_target": data.get("empty_seat_target"),
                "guest2_empty_seat_base_xyz": data.get("empty_seat_base_xyz"),
                "guest2_empty_seat_map_xyz": data.get("empty_seat_map_xyz"),
                "guest2_seat_target": data.get("empty_seat_target"),
                "guest2_seat_base_xyz": data.get("empty_seat_base_xyz"),
                "guest2_seat_map_xyz": data.get("empty_seat_map_xyz"),
            }

        print(f"  🚪 [Nav:EmptySeat] 接近空座失败: {approach_result.get('error', 'unknown')}")
        return {
            "guest2_seated": False,
            "guest2_empty_seat_approach_reached": False,
            "robot_position": "living_room",
            "guest1_described": bool((living_result.get("data") or {}).get("guest1_described")),
            "guest2_empty_seat_approach_goal": data.get("goal"),
            "guest2_empty_seat_target": data.get("empty_seat_target"),
            "guest2_empty_seat_base_xyz": data.get("empty_seat_base_xyz"),
            "guest2_empty_seat_map_xyz": data.get("empty_seat_map_xyz"),
            "guest2_seat_target": data.get("empty_seat_target"),
            "guest2_seat_base_xyz": data.get("empty_seat_base_xyz"),
            "guest2_seat_map_xyz": data.get("empty_seat_map_xyz"),
            "guest2_empty_seat_approach_error": approach_result.get("error", "unknown"),
        }

    def _approach_handover_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        guest = context.get("guest2_name", "guest2")
        result = self._navigate_to_handover_approach(
            guest,
            adjust_key="handover_approach",
        )
        data = result.get("data") or {}
        if result.get("status") == "success":
            return {
                "handover_approach_reached": True,
                "robot_position": "handover_approach",
                "handover_approach_goal": data.get("goal"),
                "handover_hand_target": data.get("handover_hand_target"),
                "handover_target_base_xyz": data.get("handover_target_base_xyz"),
                "handover_target_map_xyz": data.get("handover_target_map_xyz"),
                "handover_approach_distance_m": data.get("approach_distance_m"),
                "handover_approach_completed_time": data.get("completed_time"),
            }
        print(f"  🚪 [Nav:Handover] 接近接包目标失败: {result.get('error', 'unknown')}")
        return {
            "handover_approach_reached": False,
            "robot_position": context.get("robot_position", "unknown"),
            "handover_approach_error": result.get("error", "unknown"),
        }

    def _go_to_host_interaction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚶 [Nav] 前往 host 交互点...")
        result = self._navigate_to_host_interaction_goal(adjust_key="find_host_interaction")
        if result.get("status") == "success":
            print("  🚶 [Nav] 已到达 host 交互点")
            return {
                "host_interaction_reached": True,
                "robot_position": "host_interaction",
            }
        print(f"  🚶 [Nav] host 交互点导航失败: {result.get('error', 'unknown')}")
        return {
            "host_interaction_reached": False,
            "robot_position": context.get("robot_position", "unknown"),
        }

    def _follow_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  🚶 [Nav] 跟随 host...")
        adjust_result = self._run_pre_nav_adjustment("follow_host", "跟随 host")
        if adjust_result.get("status") == "failed":
            print(f"  🚶 [Nav] 跟随前底盘微调失败: {adjust_result.get('error', 'unknown')}")
            return {"nav_at_destination": False}
        result = self._host_follower.run()
        if result.get("status") == "success":
            dest = result.get("destination", "host_destination")
            print(f"  🚶 [Nav] 已跟随 host 到达: {dest}")
            return {
                "nav_at_destination": True,
                "destination": dest,
                "robot_position": dest,
                "host_follow_result": result,
            }
        print(f"  🚶 [Nav] 跟随 host 失败: {result.get('error', 'unknown')}")
        return {"nav_at_destination": False, "host_follow_result": result}


class DoorGuestGazeTracker:
    """
    Door conversation visual servo.

    It reads latest_people.json refreshed by vision_cache_daemon.py and publishes a
    small yaw velocity so the nearest guest stays near the image center.
    """

    def __init__(self):
        self.enabled = _env_flag("TASK1_DOOR_GAZE_TRACK_ENABLED", True)
        self.people_json = Path(
            os.environ.get("TASK1_DOOR_GAZE_PEOPLE_JSON", str(OBJECT_SEARCH_PEOPLE_JSON))
        )
        self.cache_max_age = _env_float("TASK1_DOOR_GAZE_CACHE_MAX_AGE_SEC", 2.5)
        self.fallback_image_width = _env_float("TASK1_DOOR_GAZE_IMAGE_WIDTH", 640.0)
        self.deadband_ratio = max(
            0.0,
            min(0.9, _env_float("TASK1_DOOR_GAZE_CENTER_DEADBAND_RATIO", 0.12)),
        )
        self.max_angular_speed = math.radians(
            _env_float("TASK1_DOOR_GAZE_MAX_ANGULAR_SPEED_DEG", 18.0)
        )
        self.min_angular_speed = math.radians(
            _env_float("TASK1_DOOR_GAZE_MIN_ANGULAR_SPEED_DEG", 4.0)
        )
        self.match_min_score = max(0.0, _env_float("TASK1_GUEST_GAZE_MATCH_MIN_SCORE", 3.0))
        self.rate_hz = max(1.0, _env_float("TASK1_DOOR_GAZE_RATE_HZ", 8.0))
        self.log_interval = max(0.0, _env_float("TASK1_DOOR_GAZE_LOG_INTERVAL", 1.5))
        self.topic = os.environ.get("TASK1_DOOR_GAZE_TOPIC", "/motion_target/target_speed_chassis")
        self.msg_type = os.environ.get("TASK1_DOOR_GAZE_MSG_TYPE", "twist_stamped")
        if self.msg_type not in {"twist_stamped", "twist"}:
            self.msg_type = "twist_stamped"
        self.frame_id = os.environ.get("TASK1_DOOR_GAZE_FRAME_ID", "base_link")
        self.brake_topic = os.environ.get("TASK1_DOOR_GAZE_BRAKE_TOPIC", "/motion_target/brake_mode")
        self.release_brake = _env_flag("TASK1_DOOR_GAZE_RELEASE_BRAKE", True)
        self.dry_run = _env_flag("TASK1_DOOR_GAZE_DRY_RUN", False)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_log_time = 0.0
        self._last_tf_error_log_time = 0.0
        self._ros_ready = False
        self._ros_failed = False
        self._rospy = None
        self._Twist = None
        self._TwistStamped = None
        self._Bool = None
        self._pub = None
        self._brake_pub = None
        self._target_appearance: Optional[Dict[str, Any]] = None
        self._fallback_map_xyz: Optional[List[float]] = None
        self._tf = None
        self._tf_listener = None
        self._euler_from_quaternion = None
        self.map_frame = os.environ.get(
            "TASK1_INTRO_GAZE_MAP_FRAME",
            os.environ.get("TASK1_NAV_MAP_FRAME", "map"),
        )
        self.base_frame = os.environ.get(
            "TASK1_INTRO_GAZE_BASE_FRAME",
            os.environ.get("TASK1_NAV_BASE_FRAME", DEFAULT_NAV_BASE_FRAME),
        )
        self.seat_fallback_enabled = _env_flag("TASK1_INTRO_GAZE_SEAT_FALLBACK_ENABLED", True)
        self.seat_fallback_deadband = math.radians(
            max(0.0, _env_float("TASK1_INTRO_GAZE_SEAT_DEADBAND_DEG", 8.0))
        )
        self.seat_tf_timeout = max(0.05, _env_float("TASK1_INTRO_GAZE_TF_TIMEOUT", 0.35))

    def start(
        self,
        label: str,
        target_appearance: Optional[Dict[str, Any]] = None,
        fallback_map_xyz: Optional[List[float]] = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._target_appearance = dict(target_appearance or {})
            self._fallback_map_xyz = self._coerce_xyz(fallback_map_xyz)
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(label,),
                daemon=True,
            )
            self._thread.start()
        print(f"  👀 [DoorGaze] 开始门口对话目光跟随: {label}")

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout=1.0)
        self._publish_stop(repeat=3)
        with self._lock:
            self._thread = None
            self._target_appearance = None
            self._fallback_map_xyz = None
        print("  👀 [DoorGaze] 已停止目光跟随")

    def _run(self, label: str) -> None:
        period = 1.0 / self.rate_hz
        while not self._stop_event.is_set():
            try:
                wz, reason = self._compute_angular_velocity()
                self._publish_wz(wz)
                self._maybe_log(wz, reason)
            except Exception as exc:
                print(f"  👀 [DoorGaze] 跟随线程异常 ({label}): {exc}")
                self._publish_stop(repeat=2)
                time.sleep(period)
            self._stop_event.wait(period)

    def _load_people_payload(self) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(self.people_json.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            print(f"  👀 [DoorGaze] 人物缓存 JSON 解析失败: {exc}")
            return None
        timestamp = payload.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return None
        if time.time() - float(timestamp) > self.cache_max_age:
            return None
        if payload.get("status") != "success":
            return None
        return payload

    @staticmethod
    def _coerce_xyz(values: Any) -> Optional[List[float]]:
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            return None
        try:
            z = float(values[2]) if len(values) >= 3 else 0.0
            return [float(values[0]), float(values[1]), z]
        except (TypeError, ValueError):
            return None

    def _select_target_person(self, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        target_appearance = self._target_appearance or {}
        best_person = payload.get("best_person") or {}
        if not target_appearance:
            return (best_person or None), "best_person"

        raw_people = payload.get("people")
        people = raw_people if isinstance(raw_people, list) else []
        if not people and best_person:
            people = [best_person]

        best_match: Optional[Tuple[float, int, Dict[str, Any], str]] = None
        for person in people:
            if not isinstance(person, dict):
                continue
            score, reason = _guest_clothing_match_score(target_appearance, person)
            if score < self.match_min_score:
                continue
            area = int(person.get("bbox_area_px") or 0)
            candidate = (score, area, person, reason)
            if best_match is None or candidate[:2] > best_match[:2]:
                best_match = candidate

        if best_match is not None:
            score, _area, person, reason = best_match
            return person, f"clothing_match score={score:.1f} {reason}"
        if self._fallback_map_xyz is not None:
            return None, "no_clothing_match"
        return best_person, "no_clothing_match_fallback_best_person"

    def _compute_angular_velocity(self) -> tuple:
        payload = self._load_people_payload()
        if not payload:
            return self._compute_seat_fallback_velocity("no_fresh_person")

        person, select_reason = self._select_target_person(payload)
        if not person:
            return self._compute_seat_fallback_velocity(select_reason)
        bbox = person.get("bbox")
        center = person.get("bbox_center_px")
        image_width = float(payload.get("image_width") or self.fallback_image_width)
        if isinstance(center, list) and len(center) >= 1:
            center_x = float(center[0])
        elif isinstance(bbox, list) and len(bbox) == 4:
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        else:
            return self._compute_seat_fallback_velocity(f"{select_reason}; missing_bbox")

        half_width = max(image_width / 2.0, 1.0)
        offset_ratio = (center_x - half_width) / half_width
        if abs(offset_ratio) <= self.deadband_ratio:
            return 0.0, f"{select_reason}; centered offset={offset_ratio:.2f}"

        direction = -1.0 if offset_ratio > 0.0 else 1.0
        speed = self.max_angular_speed * min(1.0, abs(offset_ratio))
        speed = max(self.min_angular_speed, min(self.max_angular_speed, speed))
        return direction * speed, f"{select_reason}; tracking offset={offset_ratio:.2f}"

    def _compute_seat_fallback_velocity(self, reason: str) -> tuple:
        if not self.seat_fallback_enabled:
            return 0.0, f"{reason}; seat_fallback_disabled"
        seat_xyz = self._fallback_map_xyz
        if seat_xyz is None:
            return 0.0, f"{reason}; no_seat_fallback"

        pose = self._read_current_map_pose()
        if pose is None:
            return 0.0, f"{reason}; seat_fallback_no_tf"

        dx = float(seat_xyz[0]) - float(pose[0])
        dy = float(seat_xyz[1]) - float(pose[1])
        distance = math.hypot(dx, dy)
        if distance < 0.05:
            return 0.0, f"{reason}; seat_fallback_too_close"

        target_yaw = math.atan2(dy, dx)
        yaw_error = self._normalize_angle_rad(target_yaw - math.radians(float(pose[2])))
        if abs(yaw_error) <= self.seat_fallback_deadband:
            return 0.0, f"{reason}; seat_fallback_centered yaw_error={math.degrees(yaw_error):.1f}deg"

        speed = self.max_angular_speed * min(1.0, abs(yaw_error) / math.radians(90.0))
        speed = max(self.min_angular_speed, min(self.max_angular_speed, speed))
        direction = 1.0 if yaw_error > 0.0 else -1.0
        return (
            direction * speed,
            f"{reason}; seat_fallback yaw_error={math.degrees(yaw_error):.1f}deg",
        )

    @staticmethod
    def _normalize_angle_rad(angle: float) -> float:
        while angle >= math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _ensure_tf_listener(self) -> bool:
        if self.dry_run:
            return False
        if self._tf_listener is not None:
            return True
        if not self._ensure_ros():
            return False
        try:
            import tf
            from tf.transformations import euler_from_quaternion

            self._tf = tf
            self._euler_from_quaternion = euler_from_quaternion
            self._tf_listener = tf.TransformListener()
            time.sleep(0.1)
            return True
        except Exception as exc:
            self._log_tf_error(f"TF 不可用，无法按座位坐标兜底转向: {exc}")
            return False

    def _read_current_map_pose(self) -> Optional[List[float]]:
        if not self._ensure_tf_listener():
            return None
        try:
            self._tf_listener.waitForTransform(
                self.map_frame,
                self.base_frame,
                self._rospy.Time(0),
                self._rospy.Duration(self.seat_tf_timeout),
            )
            trans, rot = self._tf_listener.lookupTransform(
                self.map_frame,
                self.base_frame,
                self._rospy.Time(0),
            )
            _, _, yaw = self._euler_from_quaternion(rot)
            return [float(trans[0]), float(trans[1]), math.degrees(float(yaw))]
        except (
            self._tf.Exception,
            self._tf.LookupException,
            self._tf.ConnectivityException,
            self._tf.ExtrapolationException,
        ) as exc:
            self._log_tf_error(f"读取当前 map 位姿失败，跳过座位兜底转向: {exc}")
            return None
        except Exception as exc:
            self._log_tf_error(f"座位兜底转向失败: {exc}")
            return None

    def _log_tf_error(self, message: str) -> None:
        now = time.time()
        interval = self.log_interval if self.log_interval > 0.0 else 1.5
        if now - self._last_tf_error_log_time >= interval:
            self._last_tf_error_log_time = now
            print(f"  👀 [DoorGaze] {message}")

    def _ensure_ros(self) -> bool:
        if self.dry_run:
            return True
        if self._ros_ready:
            return True
        if self._ros_failed:
            return False
        try:
            import rospy
            from geometry_msgs.msg import Twist, TwistStamped
            from std_msgs.msg import Bool

            if not rospy.core.is_initialized():
                rospy.init_node("task1_door_gaze_tracker", anonymous=True, disable_signals=True)

            msg_cls = TwistStamped if self.msg_type == "twist_stamped" else Twist
            self._pub = rospy.Publisher(self.topic, msg_cls, queue_size=10)
            self._brake_pub = (
                rospy.Publisher(self.brake_topic, Bool, queue_size=10)
                if self.release_brake
                else None
            )
            self._rospy = rospy
            self._Twist = Twist
            self._TwistStamped = TwistStamped
            self._Bool = Bool
            self._ros_ready = True
            time.sleep(0.2)
            return True
        except Exception as exc:
            print(f"  👀 [DoorGaze] ROS 底盘速度发布不可用，跳过目光跟随: {exc}")
            self._ros_failed = True
            return False

    def _make_twist(self, wz: float):
        msg = self._Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        return msg

    def _make_cmd_msg(self, wz: float):
        twist = self._make_twist(wz)
        if self.msg_type == "twist":
            return twist
        msg = self._TwistStamped()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.twist = twist
        return msg

    def _publish_wz(self, wz: float) -> None:
        if self.dry_run:
            return
        if not self._ensure_ros():
            return
        if self._brake_pub is not None:
            self._brake_pub.publish(False)
        self._pub.publish(self._make_cmd_msg(wz))

    def _publish_stop(self, repeat: int = 1) -> None:
        if self.dry_run:
            return
        if not self._ensure_ros():
            return
        for _ in range(max(1, repeat)):
            if self._brake_pub is not None:
                self._brake_pub.publish(False)
            self._pub.publish(self._make_cmd_msg(0.0))
            time.sleep(0.05)

    def _maybe_log(self, wz: float, reason: str) -> None:
        if self.log_interval <= 0.0:
            return
        now = time.time()
        if now - self._last_log_time < self.log_interval:
            return
        self._last_log_time = now
        print(
            "  👀 [DoorGaze] "
            f"{reason}, wz={math.degrees(wz):.1f}deg/s, source={self.people_json.name}"
        )


class HostFollowController:
    """
    Local host-follow controller for the final FOLLOW_HOST state.

    It follows the nearest person from latest_people.json and listens to ASR
    commands from the host. The robot keeps the public prompt natural while
    accepting a broad set of start/stop phrases internally.
    """

    FOLLOW_PATTERNS = [
        re.compile(pattern, re.I)
        for pattern in (
            r"\bfollow\b",
            r"\bfollow me\b",
            r"\bplease follow\b",
            r"\bcome with me\b",
            r"\bgo with me\b",
            r"\bcome on\b",
            r"\bthis way\b",
            r"\bover here\b",
            r"\blet'?s go\b",
            r"\bstart\b",
            r"\bgo ahead\b",
        )
    ]
    STOP_PATTERNS = [
        re.compile(pattern, re.I)
        for pattern in (
            r"\bstop\b",
            r"\bstop following\b",
            r"\bstop following me\b",
            r"\bhalt\b",
            r"\bwait\b",
            r"\bwe are here\b",
            r"\bwe're here\b",
            r"\bwe have arrived\b",
            r"\bwe have reached\b",
            r"\bwe arrived\b",
            r"\bwe are arrived\b",
            r"\bhere we are\b",
            r"\barrived\b",
            r"\bdestination\b",
            r"\bthis is our destination\b",
            r"\bthis is the place\b",
            r"\bthis is where\b",
            r"\bthis is it\b",
            r"\byou can stop\b",
            r"\byou may stop\b",
            r"\bstop here\b",
            r"\bstop now\b",
            r"\bplease stop\b",
            r"\benough\b",
            r"\bthat'?s enough\b",
            r"\bfinish(?:ed)?\b",
            r"\bleave it here\b",
            r"\bplace it here\b",
            r"\bput it here\b",
            r"\bdrop it here\b",
            r"^here[.!? ]*$",
        )
    ]

    def __init__(self):
        self.enabled = _env_flag("TASK1_HOST_FOLLOW_ENABLED", True)
        self.people_json = Path(
            os.environ.get("TASK1_HOST_FOLLOW_PEOPLE_JSON", str(OBJECT_SEARCH_PEOPLE_JSON))
        )
        self.topic = os.environ.get("TASK1_HOST_FOLLOW_TOPIC", "/motion_target/target_speed_chassis")
        self.msg_type = os.environ.get("TASK1_HOST_FOLLOW_MSG_TYPE", "twist_stamped")
        if self.msg_type not in {"twist_stamped", "twist"}:
            self.msg_type = "twist_stamped"
        self.frame_id = os.environ.get("TASK1_HOST_FOLLOW_FRAME_ID", "base_link")
        self.brake_topic = os.environ.get("TASK1_HOST_FOLLOW_BRAKE_TOPIC", "/motion_target/brake_mode")
        self.release_brake = _env_flag("TASK1_HOST_FOLLOW_RELEASE_BRAKE", True)
        self.extra_stop_topic = os.environ.get("TASK1_HOST_FOLLOW_EXTRA_STOP_TOPIC", "/cmd_vel").strip()
        self.stop_hold_sec = max(0.0, _env_float("TASK1_HOST_FOLLOW_STOP_HOLD_SEC", 0.8))
        self.dry_run = _env_flag("TASK1_HOST_FOLLOW_DRY_RUN", False)
        self.follow_mode = os.environ.get("TASK1_HOST_FOLLOW_MODE", "nav_goal").strip().lower()
        if self.follow_mode not in {"nav_goal", "velocity"}:
            print(f"  🚶 [HostFollow] 未知跟随模式 {self.follow_mode!r}，改用 nav_goal")
            self.follow_mode = "nav_goal"

        self.cache_max_age = _env_float("TASK1_HOST_FOLLOW_CACHE_MAX_AGE_SEC", 2.0)
        self.rate_hz = max(1.0, _env_float("TASK1_HOST_FOLLOW_RATE_HZ", 10.0))
        self.max_duration = _env_float("TASK1_HOST_FOLLOW_TIMEOUT_SEC", 180.0)
        self.lost_timeout = max(
            0.0,
            _env_float("TASK1_HOST_FOLLOW_LOST_TIMEOUT_SEC", 30.0),
        )
        self.wait_start_sec = _env_float("TASK1_HOST_FOLLOW_WAIT_START_SIGNAL_SEC", 8.0)
        self.auto_start = _env_flag("TASK1_HOST_FOLLOW_AUTO_START", False)
        self.start_on_person = _env_flag("TASK1_HOST_FOLLOW_START_ON_PERSON", True)
        self.start_center_ratio = max(
            0.0,
            min(1.0, _env_float("TASK1_HOST_FOLLOW_START_CENTER_RATIO", 0.55)),
        )
        self.timeout_is_success = _env_flag("TASK1_HOST_FOLLOW_TIMEOUT_IS_SUCCESS", False)
        self.log_interval = max(0.0, _env_float("TASK1_HOST_FOLLOW_LOG_INTERVAL", 1.0))
        self.log_asr = _env_flag("TASK1_HOST_FOLLOW_LOG_ASR", True)

        self.target_distance = _env_float("TASK1_HOST_FOLLOW_TARGET_DISTANCE_M", 1.2)
        self.distance_deadband = _env_float("TASK1_HOST_FOLLOW_DISTANCE_DEADBAND_M", 0.20)
        self.max_linear_speed = _env_float("TASK1_HOST_FOLLOW_MAX_LINEAR_SPEED", 0.25)
        self.min_linear_speed = _env_float("TASK1_HOST_FOLLOW_MIN_LINEAR_SPEED", 0.05)
        self.linear_kp = _env_float("TASK1_HOST_FOLLOW_LINEAR_KP", 0.45)
        self.reverse_enabled = _env_flag("TASK1_HOST_FOLLOW_REVERSE_ENABLED", False)
        self.reverse_speed = _env_float("TASK1_HOST_FOLLOW_REVERSE_SPEED", 0.08)

        self.nav_map_frame = os.environ.get("TASK1_HOST_FOLLOW_MAP_FRAME", os.environ.get("TASK1_NAV_MAP_FRAME", "map"))
        self.nav_base_frame = os.environ.get("TASK1_HOST_FOLLOW_BASE_FRAME", os.environ.get("TASK1_NAV_BASE_FRAME", DEFAULT_NAV_BASE_FRAME))
        self.nav_move_base_action = os.environ.get("TASK1_HOST_FOLLOW_MOVE_BASE_ACTION", os.environ.get("TASK1_MOVE_BASE_ACTION", "/move_base"))
        self.nav_server_timeout = _env_float("TASK1_HOST_FOLLOW_NAV_SERVER_TIMEOUT", 10.0)
        self.nav_tf_timeout = _env_float("TASK1_HOST_FOLLOW_NAV_TF_TIMEOUT_SEC", 1.0)
        self.nav_goal_update_distance = _env_float("TASK1_HOST_FOLLOW_NAV_UPDATE_DISTANCE_M", 0.6)
        self.nav_goal_update_interval = max(0.0, _env_float("TASK1_HOST_FOLLOW_NAV_UPDATE_INTERVAL_SEC", 1.0))
        self.nav_initial_match_distance = _env_float("TASK1_HOST_FOLLOW_INITIAL_MATCH_DISTANCE_M", 1.0)
        self.nav_relock_enabled = _env_flag("TASK1_HOST_FOLLOW_RELOCK_ENABLED", True)
        self.nav_search_enabled = _env_flag("TASK1_HOST_FOLLOW_SEARCH_ENABLED", True)
        self.nav_search_timeout = _env_float("TASK1_HOST_FOLLOW_SEARCH_TIMEOUT_SEC", 10.0)
        self.nav_search_angular_speed = _env_float("TASK1_HOST_FOLLOW_SEARCH_ANGULAR_SPEED", 0.3)
        self.nav_goal_failure_limit = max(1, _env_int("TASK1_HOST_FOLLOW_NAV_FAILURE_LIMIT", 3))
        self.nav_goal_retry_interval = max(
            0.2,
            _env_float("TASK1_HOST_FOLLOW_NAV_RETRY_INTERVAL_SEC", 2.0),
        )
        self.nav_abort_is_terminal = _env_flag("TASK1_HOST_FOLLOW_NAV_ABORT_IS_TERMINAL", False)
        self.leftbase_frame = os.environ.get(
            "TASK1_HOST_FOLLOW_LEFTBASE_FRAME",
            os.environ.get("TASK1_EMPTY_SEAT_LEFTBASE_FRAME", DEFAULT_EMPTY_SEAT_LEFTBASE_FRAME),
        )
        self.camera_to_leftbase = Path(
            os.environ.get(
                "TASK1_HOST_FOLLOW_CAMERA_TO_LEFTBASE",
                os.environ.get("TASK1_EMPTY_SEAT_CAMERA_TO_LEFTBASE", str(OBJECT_SEARCH_CAMERA_TO_LEFTBASE)),
            )
        )

        self.center_deadband_ratio = max(
            0.0,
            min(0.9, _env_float("TASK1_HOST_FOLLOW_CENTER_DEADBAND_RATIO", 0.10)),
        )
        self.max_angular_speed = math.radians(
            _env_float("TASK1_HOST_FOLLOW_MAX_ANGULAR_SPEED_DEG", 22.0)
        )
        self.min_angular_speed = math.radians(
            _env_float("TASK1_HOST_FOLLOW_MIN_ANGULAR_SPEED_DEG", 4.0)
        )

        self._ros_ready = False
        self._ros_failed = False
        self._rospy = None
        self._Twist = None
        self._TwistStamped = None
        self._Bool = None
        self._String = None
        self._PointStamped = None
        self._MoveBaseGoal = None
        self._GoalStatus = None
        self._quaternion_from_euler = None
        self._euler_from_quaternion = None
        self._tf = None
        self._tf_listener = None
        self._move_base_client = None
        self._move_base_ready = False
        self._pub = None
        self._extra_stop_pub = None
        self._brake_pub = None
        self._asr_sub = None
        self._asr_segment_sub = None
        self._command_queue: queue.Queue = queue.Queue()
        self._last_log_time = 0.0

    def run(self) -> Dict[str, Any]:
        if not self.enabled:
            print("  🚶 [HostFollow] 已禁用，本状态按成功处理")
            return {
                "status": "success",
                "destination": "host_destination",
                "reason": "disabled",
            }

        if not self._ensure_ros():
            return {
                "status": "failed",
                "error": "ros_unavailable",
                "destination": None,
            }
        PeopleVisionCacheManager.ensure_running(
            "  🚶 [HostFollow]",
            self.cache_max_age,
        )

        self._drain_commands()
        if _interactive_enabled():
            print("  🚶 [HostFollow] 等待 host 开始带路信号；控制台 Enter 可直接开始跟随")
        else:
            print("  🚶 [HostFollow] 等待 host 开始带路信号")
        start_result = self._wait_for_start_signal()
        if start_result.get("status") == "stopped":
            self._publish_stop(repeat=5)
            return {
                "status": "success",
                "destination": "host_destination",
                "reason": "stop_before_start",
                "stop_command": start_result.get("text"),
            }
        if start_result.get("status") == "failed":
            self._publish_stop(repeat=5)
            return start_result

        if _interactive_enabled():
            print("  🚶 [HostFollow] 开始跟随 host；控制台 Enter 表示已到达终点")
        else:
            print("  🚶 [HostFollow] 开始跟随 host")
        return self._follow_loop(start_result)

    def _wait_for_start_signal(self) -> Dict[str, Any]:
        started_at = time.time()
        signal_deadline = started_at + max(0.0, self.wait_start_sec)
        hard_deadline = started_at + max(1.0, self.max_duration)
        manual_wait_announced = False

        while time.time() < hard_deadline:
            command = self._get_command_nowait()
            if command:
                if command["kind"] == "stop":
                    return {"status": "stopped", "text": command["text"]}
                if command["kind"] == "follow":
                    print(f"  🚶 [HostFollow] 收到开始信号: {command['text']}")
                    return {"status": "started", "start_command": command["text"]}
            if self.start_on_person:
                payload, _ = self._load_people_payload_with_reason()
                if payload:
                    person = payload.get("best_person") or {}
                    offset_ratio = self._extract_offset_ratio(payload, person)
                    if abs(offset_ratio) <= self.start_center_ratio:
                        print(
                            "  🚶 [HostFollow] 看到前方人物，直接开始跟随: "
                            f"offset={offset_ratio:.2f}, depth={person.get('depth_m')}"
                        )
                        return {
                            "status": "started",
                            "start_command": "visual_person",
                            "start_person": person,
                        }
            if self._stdin_requested():
                print("  🚶 [HostFollow] 控制台 Enter: 开始跟随")
                return {"status": "started", "start_command": "manual_enter"}
            if time.time() >= signal_deadline:
                if self.auto_start:
                    print("  🚶 [HostFollow] 未听到开始信号，自动开始跟随")
                    return {"status": "started", "start_command": "auto_timeout"}
                if not self._stdin_available():
                    return {"status": "failed", "error": "start_signal_timeout"}
                if not manual_wait_announced:
                    print("  🚶 [HostFollow] 未听到开始信号，继续等待；按 Enter 开始跟随")
                    manual_wait_announced = True
            time.sleep(0.1)

        return {"status": "failed", "error": "start_signal_timeout"}

    def _follow_loop(self, start_result: Dict[str, Any]) -> Dict[str, Any]:
        if self.follow_mode == "nav_goal":
            return self._follow_loop_nav_goal(start_result)
        return self._follow_loop_velocity(start_result)

    def _follow_loop_velocity(self, start_result: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.time()
        deadline = started_at + max(1.0, self.max_duration)
        lost_since: Optional[float] = None
        last_person: Optional[Dict[str, Any]] = None
        period = 1.0 / self.rate_hz

        while time.time() < deadline:
            command = self._get_command_nowait()
            if command and command["kind"] == "stop":
                self._publish_stop(repeat=8)
                print(f"  🚶 [HostFollow] 收到停止/到达信号: {command['text']}")
                return {
                    "status": "success",
                    "destination": "host_destination",
                    "reason": "host_stop_signal",
                    "start_command": start_result.get("start_command"),
                    "stop_command": command["text"],
                    "duration_sec": round(time.time() - started_at, 1),
                    "last_person": last_person,
                }
            if self._stdin_requested():
                self._publish_stop(repeat=8)
                print("  🚶 [HostFollow] 控制台 Enter: 到达终点，进入下一状态")
                return {
                    "status": "success",
                    "destination": "host_destination",
                    "reason": "manual_enter",
                    "start_command": start_result.get("start_command"),
                    "duration_sec": round(time.time() - started_at, 1),
                    "last_person": last_person,
                }

            payload, reject_reason = self._load_people_payload_with_reason()
            if not payload:
                self._publish_stop()
                if lost_since is None:
                    lost_since = time.time()
                lost_elapsed = time.time() - lost_since
                self._maybe_log(
                    0.0,
                    0.0,
                    f"waiting_for_person {reject_reason} lost={lost_elapsed:.1f}s",
                )
                if self.lost_timeout > 0.0 and lost_elapsed >= self.lost_timeout:
                    self._publish_stop(repeat=8)
                    return {
                        "status": "failed",
                        "error": "person_lost",
                        "duration_sec": round(time.time() - started_at, 1),
                        "last_person": last_person,
                    }
                time.sleep(period)
                continue

            lost_since = None
            person = payload.get("best_person") or {}
            last_person = person
            vx, wz, reason = self._compute_velocity(payload, person)
            self._publish_velocity(vx, wz)
            self._maybe_log(vx, wz, reason)
            time.sleep(period)

        self._publish_stop(repeat=8)
        if self.timeout_is_success:
            return {
                "status": "success",
                "destination": "host_destination",
                "reason": "timeout_treated_as_success",
                "duration_sec": round(time.time() - started_at, 1),
                "last_person": last_person,
            }
        return {
            "status": "failed",
            "error": "follow_timeout",
            "duration_sec": round(time.time() - started_at, 1),
            "last_person": last_person,
        }

    def _follow_loop_nav_goal(self, start_result: Dict[str, Any]) -> Dict[str, Any]:
        if not self._ensure_move_base_client():
            return {
                "status": "failed",
                "error": "move_base_unavailable",
                "destination": None,
            }

        started_at = time.time()
        deadline = started_at + max(1.0, self.max_duration)
        period = 1.0 / self.rate_hz
        start_person = start_result.get("start_person") or {}
        locked_track_id = self._person_track_id(start_person)
        match_camera_xyz = self._person_camera_xyz(start_person)
        last_camera_xyz = match_camera_xyz
        last_center_offset = None
        last_person: Optional[Dict[str, Any]] = start_person if start_person else None
        last_target_map: Optional[List[float]] = None
        last_goal: Optional[List[float]] = None
        last_goal_sent_at = 0.0
        lost_since: Optional[float] = None
        search_since: Optional[float] = None
        goal_updates = 0
        nav_send_failures = 0
        nav_motion_failures = 0
        nav_retry_after = 0.0
        last_failure_state: Optional[str] = None

        print(
            "  🚶 [HostFollow] 使用导航目标跟随 host: "
            f"mode=nav_goal, track_id={locked_track_id}, target_distance={self.target_distance:.2f}m"
        )

        while time.time() < deadline:
            command = self._get_command_nowait()
            if command and command["kind"] == "stop":
                self._cancel_move_base_goal()
                self._publish_stop(repeat=8)
                print(f"  🚶 [HostFollow] 收到停止/到达信号: {command['text']}")
                return {
                    "status": "success",
                    "destination": "host_destination",
                    "reason": "host_stop_signal",
                    "start_command": start_result.get("start_command"),
                    "stop_command": command["text"],
                    "duration_sec": round(time.time() - started_at, 1),
                    "track_id": locked_track_id,
                    "goal_updates": goal_updates,
                    "last_person": last_person,
                    "last_target_map_xyz": last_target_map,
                    "last_goal": last_goal,
                }
            if self._stdin_requested():
                self._cancel_move_base_goal()
                self._publish_stop(repeat=8)
                print("  🚶 [HostFollow] 控制台 Enter: 到达终点，进入下一状态")
                return {
                    "status": "success",
                    "destination": "host_destination",
                    "reason": "manual_enter",
                    "start_command": start_result.get("start_command"),
                    "duration_sec": round(time.time() - started_at, 1),
                    "track_id": locked_track_id,
                    "goal_updates": goal_updates,
                    "last_person": last_person,
                    "last_target_map_xyz": last_target_map,
                    "last_goal": last_goal,
                }

            payload, reject_reason = self._load_people_payload_with_reason()
            person, select_reason, selected_track_id = self._select_nav_follow_person(
                payload,
                locked_track_id,
                match_camera_xyz,
                last_camera_xyz,
            )
            if person is None:
                now = time.time()
                if lost_since is None:
                    lost_since = now
                    search_since = now
                    if last_goal is None:
                        self._cancel_move_base_goal()
                    print(
                        "  🚶 [HostFollow] 暂时丢失 host: "
                        f"{select_reason or reject_reason}"
                    )
                lost_elapsed = now - lost_since
                self._maybe_log(
                    0.0,
                    0.0,
                    f"nav_waiting_for_person {select_reason or reject_reason} lost={lost_elapsed:.1f}s",
                )
                if self.lost_timeout > 0.0 and lost_elapsed >= self.lost_timeout:
                    self._cancel_move_base_goal()
                    self._publish_stop(repeat=8)
                    return {
                        "status": "failed",
                        "error": "person_lost",
                        "duration_sec": round(time.time() - started_at, 1),
                        "track_id": locked_track_id,
                        "goal_updates": goal_updates,
                        "last_person": last_person,
                        "last_target_map_xyz": last_target_map,
                        "last_goal": last_goal,
                    }
                if last_goal is None and self.nav_search_enabled and search_since is not None:
                    if now - search_since <= self.nav_search_timeout:
                        self._publish_search_rotation(last_center_offset)
                    else:
                        self._publish_stop()
                time.sleep(period)
                continue

            if selected_track_id is not None and selected_track_id != locked_track_id:
                if locked_track_id is None:
                    print(f"  🚶 [HostFollow] 已锁定 host track_id={selected_track_id}")
                else:
                    print(f"  🚶 [HostFollow] host 重新锁定: {locked_track_id} -> {selected_track_id}")
                locked_track_id = selected_track_id

            last_person = person
            last_center_offset = self._extract_offset_ratio(payload or {}, person)
            camera_xyz = self._person_camera_xyz(person)
            if camera_xyz is not None:
                last_camera_xyz = camera_xyz
                match_camera_xyz = camera_xyz
            lost_since = None
            search_since = None

            target_map = self._person_to_map_xyz(person)
            current_pose = self._read_current_map_pose()
            if target_map is None or current_pose is None:
                self._maybe_log(
                    0.0,
                    0.0,
                    "nav_waiting_for_tf_or_person_coordinate",
                )
                time.sleep(period)
                continue

            goal = self._compute_follow_nav_goal(target_map, current_pose)
            now = time.time()
            if now < nav_retry_after:
                self._maybe_log(
                    0.0,
                    0.0,
                    f"nav_retry_waiting {last_failure_state or 'move_base_failure'} "
                    f"retry_in={nav_retry_after - now:.1f}s",
                )
                time.sleep(period)
                continue

            goal_shift = (
                float("inf")
                if last_goal is None
                else self._xy_distance(goal, last_goal)
            )
            should_update = (
                last_goal is None
                or (
                    goal_shift >= self.nav_goal_update_distance
                    and now - last_goal_sent_at >= self.nav_goal_update_interval
                )
            )
            sent_goal_this_cycle = False
            if should_update:
                sent = self._send_move_base_goal(goal)
                if not sent:
                    nav_send_failures += 1
                    if nav_send_failures >= self.nav_goal_failure_limit:
                        self._cancel_move_base_goal()
                        self._publish_stop(repeat=8)
                        return {
                            "status": "failed",
                            "error": "nav_goal_send_failed",
                            "duration_sec": round(time.time() - started_at, 1),
                            "track_id": locked_track_id,
                            "goal_updates": goal_updates,
                            "last_person": last_person,
                            "last_target_map_xyz": last_target_map,
                            "last_goal": last_goal,
                        }
                else:
                    nav_send_failures = 0
                    sent_goal_this_cycle = True
                    goal_updates += 1
                    last_goal = goal
                    last_goal_sent_at = now
                    print(
                        "  🚶 [HostFollow] 导航目标刷新: "
                        f"track_id={locked_track_id}, target=({target_map[0]:.2f},{target_map[1]:.2f}), "
                        f"goal=({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.1f}deg), reason={select_reason}"
                    )

            last_target_map = target_map
            failure_state = None if sent_goal_this_cycle else self._move_base_failure_state()
            if failure_state:
                nav_motion_failures += 1
                last_failure_state = failure_state
                print(
                    "  🚶 [HostFollow] move_base 当前失败状态: "
                    f"{failure_state}；取消本次目标，{self.nav_goal_retry_interval:.1f}s 后按最新人物坐标重试 "
                    f"(motion_failures={nav_motion_failures})"
                )
                self._cancel_move_base_goal()
                last_goal = None
                last_goal_sent_at = 0.0
                nav_retry_after = time.time() + self.nav_goal_retry_interval
                if self.nav_abort_is_terminal and nav_motion_failures >= self.nav_goal_failure_limit:
                    self._cancel_move_base_goal()
                    self._publish_stop(repeat=8)
                    return {
                        "status": "failed",
                        "error": f"move_base_failed: {failure_state}",
                        "duration_sec": round(time.time() - started_at, 1),
                        "track_id": locked_track_id,
                        "goal_updates": goal_updates,
                        "last_person": last_person,
                        "last_target_map_xyz": last_target_map,
                        "last_goal": last_goal,
                    }
                time.sleep(period)
                continue
            if last_failure_state is not None:
                last_failure_state = None
                nav_motion_failures = 0

            self._maybe_log(
                0.0,
                0.0,
                f"nav_tracking track_id={locked_track_id} target_map=({target_map[0]:.2f},{target_map[1]:.2f})",
            )
            time.sleep(period)

        self._cancel_move_base_goal()
        self._publish_stop(repeat=8)
        if self.timeout_is_success:
            return {
                "status": "success",
                "destination": "host_destination",
                "reason": "timeout_treated_as_success",
                "duration_sec": round(time.time() - started_at, 1),
                "track_id": locked_track_id,
                "goal_updates": goal_updates,
                "last_person": last_person,
                "last_target_map_xyz": last_target_map,
                "last_goal": last_goal,
            }
        return {
            "status": "failed",
            "error": "follow_timeout",
            "duration_sec": round(time.time() - started_at, 1),
            "track_id": locked_track_id,
            "goal_updates": goal_updates,
            "last_person": last_person,
            "last_target_map_xyz": last_target_map,
            "last_goal": last_goal,
        }

    def _load_people_payload(self) -> Optional[Dict[str, Any]]:
        payload, _ = self._load_people_payload_with_reason()
        return payload

    def _load_people_payload_with_reason(self) -> tuple:
        try:
            payload = json.loads(self.people_json.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None, f"no_people_cache: {self.people_json}"
        except json.JSONDecodeError as exc:
            print(f"  🚶 [HostFollow] 人物缓存 JSON 解析失败: {exc}")
            return None, f"bad_people_cache_json: {exc}"

        timestamp = payload.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return None, "people_cache_missing_timestamp"
        age = time.time() - float(timestamp)
        if age > self.cache_max_age:
            return None, f"stale_people_cache age={age:.1f}s"
        status = payload.get("status")
        if status != "success":
            count = payload.get("person_count")
            return None, f"people_cache_status={status} count={count}"
        if not payload.get("best_person"):
            return None, "people_cache_missing_best_person"
        return payload, "success"

    def _select_nav_follow_person(
        self,
        payload: Optional[Dict[str, Any]],
        locked_track_id: Optional[int],
        match_camera_xyz: Optional[List[float]],
        last_camera_xyz: Optional[List[float]],
    ) -> tuple:
        if not payload:
            return None, "no_people_payload", locked_track_id

        people = payload.get("people")
        if not isinstance(people, list):
            people = []
        best_person = payload.get("best_person")
        if isinstance(best_person, dict) and best_person:
            best_id = self._person_track_id(best_person)
            if best_id is None or all(self._person_track_id(person) != best_id for person in people):
                people = [best_person, *people]

        candidates = [person for person in people if isinstance(person, dict)]
        if not candidates:
            return None, "people_cache_empty", locked_track_id

        if locked_track_id is not None:
            for person in candidates:
                if self._person_track_id(person) == locked_track_id:
                    if self._person_has_nav_point(person):
                        return person, "track_id", locked_track_id
                    return None, f"track_id={locked_track_id}_missing_3d_coordinate", locked_track_id

            if self.nav_relock_enabled:
                reference = last_camera_xyz or match_camera_xyz
                person, distance = self._nearest_person_by_camera_xyz(candidates, reference)
                if person is not None and distance <= self.nav_initial_match_distance:
                    return person, f"relock_nearest_camera_distance={distance:.2f}m", self._person_track_id(person)
            return None, f"track_id={locked_track_id}_not_visible", locked_track_id

        if match_camera_xyz is not None:
            person, distance = self._nearest_person_by_camera_xyz(candidates, match_camera_xyz)
            if person is not None and distance <= self.nav_initial_match_distance:
                return person, f"initial_nearest_camera_distance={distance:.2f}m", self._person_track_id(person)

        if isinstance(best_person, dict) and self._person_has_nav_point(best_person):
            return best_person, "best_person", self._person_track_id(best_person)

        for person in candidates:
            if self._person_has_nav_point(person):
                return person, "first_person_with_3d_coordinate", self._person_track_id(person)
        return None, "no_person_with_3d_coordinate", locked_track_id

    @staticmethod
    def _coerce_xyz(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None

    def _person_track_id(self, person: Any) -> Optional[int]:
        if not isinstance(person, dict):
            return None
        try:
            value = person.get("track_id")
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _person_camera_xyz(self, person: Any) -> Optional[List[float]]:
        if not isinstance(person, dict):
            return None
        position = person.get("position") if isinstance(person.get("position"), dict) else {}
        for key in ("position_3d", "camera_xyz_m"):
            xyz = self._coerce_xyz(person.get(key))
            if xyz is not None:
                return xyz
        return self._coerce_xyz(position.get("camera_xyz_m"))

    def _person_leftbase_xyz(self, person: Any) -> Optional[List[float]]:
        if not isinstance(person, dict):
            return None
        position = person.get("position") if isinstance(person.get("position"), dict) else {}
        for key in ("leftbase_xyz_m", "left_arm_base_xyz_m"):
            xyz = self._coerce_xyz(person.get(key))
            if xyz is not None:
                return xyz
            xyz = self._coerce_xyz(position.get(key))
            if xyz is not None:
                return xyz
        return None

    def _person_has_nav_point(self, person: Any) -> bool:
        if not isinstance(person, dict):
            return False
        if self._person_leftbase_xyz(person) is not None:
            return True
        if self._person_camera_xyz(person) is not None:
            return True
        if self._coerce_xyz(person.get("xyz_m")) is not None and person.get("frame_id"):
            return True
        return False

    def _nearest_person_by_camera_xyz(
        self,
        people: List[Dict[str, Any]],
        reference_xyz: Optional[List[float]],
    ) -> tuple:
        if reference_xyz is None:
            return None, float("inf")
        best_person = None
        best_distance = float("inf")
        for person in people:
            xyz = self._person_camera_xyz(person)
            if xyz is None:
                continue
            distance = self._xyz_distance(xyz, reference_xyz)
            if distance < best_distance:
                best_person = person
                best_distance = distance
        return best_person, best_distance

    @staticmethod
    def _xyz_distance(a: List[float], b: List[float]) -> float:
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )

    @staticmethod
    def _xy_distance(a: List[float], b: List[float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _leftbase_frame_candidates(self) -> List[str]:
        configured = os.environ.get("TASK1_HOST_FOLLOW_TF_FRAMES", "").strip()
        candidates: List[str] = []
        if configured:
            candidates.extend(part.strip() for part in configured.split(",") if part.strip())
        candidates.extend([self.leftbase_frame, DEFAULT_EMPTY_SEAT_LEFTBASE_FRAME])
        return [candidate for index, candidate in enumerate(candidates) if candidate and candidate not in candidates[:index]]

    def _camera_xyz_to_leftbase(self, camera_xyz: List[float]) -> Optional[List[float]]:
        try:
            matrix = _load_matrix4(self.camera_to_leftbase)
            return _apply_matrix4(matrix, camera_xyz)
        except FileNotFoundError:
            print(f"  🚶 [HostFollow] 缺少相机到 leftbase 外参: {self.camera_to_leftbase}")
            return None
        except (ValueError, OSError) as exc:
            print(f"  🚶 [HostFollow] 相机到 leftbase 外参读取失败: {exc}")
            return None

    def _person_to_map_xyz(self, person: Dict[str, Any]) -> Optional[List[float]]:
        leftbase_xyz = self._person_leftbase_xyz(person)
        if leftbase_xyz is None:
            camera_xyz = self._person_camera_xyz(person)
            if camera_xyz is not None:
                leftbase_xyz = self._camera_xyz_to_leftbase(camera_xyz)

        if leftbase_xyz is not None:
            for source_frame in self._leftbase_frame_candidates():
                mapped = self._transform_point_with_tf(
                    source_frame,
                    self.nav_map_frame,
                    leftbase_xyz,
                    "host leftbase 到 map",
                )
                if mapped is not None:
                    return mapped

        xyz = self._coerce_xyz(person.get("xyz_m"))
        frame_id = str(person.get("frame_id") or "").strip()
        if xyz is not None and frame_id:
            if frame_id == self.nav_map_frame:
                return xyz
            return self._transform_point_with_tf(
                frame_id,
                self.nav_map_frame,
                xyz,
                "host frame 到 map",
            )
        return None

    def _transform_point_with_tf(
        self,
        source_frame: str,
        target_frame: str,
        xyz: List[float],
        label: str,
    ) -> Optional[List[float]]:
        if not self._ensure_ros():
            return None
        if self._tf_listener is None:
            self._tf_listener = self._tf.TransformListener()
            time.sleep(0.2)
        point = self._PointStamped()
        point.header.frame_id = source_frame
        point.header.stamp = self._rospy.Time(0)
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        try:
            self._tf_listener.waitForTransform(
                target_frame,
                source_frame,
                self._rospy.Time(0),
                self._rospy.Duration(self.nav_tf_timeout),
            )
            mapped = self._tf_listener.transformPoint(target_frame, point)
            return [float(mapped.point.x), float(mapped.point.y), float(mapped.point.z)]
        except (
            self._tf.Exception,
            self._tf.LookupException,
            self._tf.ConnectivityException,
            self._tf.ExtrapolationException,
        ) as exc:
            print(f"  🚶 [HostFollow] {label} TF 转换失败 {source_frame}->{target_frame}: {exc}")
            return None

    def _read_current_map_pose(self) -> Optional[List[float]]:
        if not self._ensure_ros():
            return None
        if self._tf_listener is None:
            self._tf_listener = self._tf.TransformListener()
            time.sleep(0.2)
        try:
            self._tf_listener.waitForTransform(
                self.nav_map_frame,
                self.nav_base_frame,
                self._rospy.Time(0),
                self._rospy.Duration(self.nav_tf_timeout),
            )
            trans, rot = self._tf_listener.lookupTransform(
                self.nav_map_frame,
                self.nav_base_frame,
                self._rospy.Time(0),
            )
        except (
            self._tf.Exception,
            self._tf.LookupException,
            self._tf.ConnectivityException,
            self._tf.ExtrapolationException,
        ) as exc:
            print(f"  🚶 [HostFollow] 读取机器人 map 位姿失败: {exc}")
            return None
        _, _, yaw = self._euler_from_quaternion(rot)
        return [float(trans[0]), float(trans[1]), math.degrees(float(yaw))]

    def _compute_follow_nav_goal(
        self,
        target_map_xyz: List[float],
        current_pose: List[float],
    ) -> List[float]:
        target_x, target_y = float(target_map_xyz[0]), float(target_map_xyz[1])
        robot_x, robot_y = float(current_pose[0]), float(current_pose[1])
        dx = robot_x - target_x
        dy = robot_y - target_y
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            yaw_rad = math.radians(float(current_pose[2]))
            dx = -math.cos(yaw_rad)
            dy = -math.sin(yaw_rad)
            norm = 1.0
        distance = max(0.1, float(self.target_distance))
        goal_x = target_x + dx / norm * distance
        goal_y = target_y + dy / norm * distance
        goal_yaw = math.degrees(math.atan2(target_y - goal_y, target_x - goal_x))
        return [goal_x, goal_y, self._normalize_angle_deg(goal_yaw)]

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        while angle >= 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def _publish_search_rotation(self, last_center_offset: Optional[float]) -> None:
        direction = -1.0 if last_center_offset and last_center_offset > 0.0 else 1.0
        self._publish_velocity(0.0, direction * abs(self.nav_search_angular_speed))

    def _ensure_move_base_client(self) -> bool:
        if self.dry_run:
            return True
        if self._move_base_ready:
            return True
        if not self._ensure_ros():
            return False
        try:
            import actionlib
            from actionlib_msgs.msg import GoalStatus
            from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

            self._MoveBaseGoal = MoveBaseGoal
            self._GoalStatus = GoalStatus
            self._move_base_client = actionlib.SimpleActionClient(
                self.nav_move_base_action,
                MoveBaseAction,
            )
            if not self._move_base_client.wait_for_server(
                self._rospy.Duration(self.nav_server_timeout)
            ):
                print(f"  🚶 [HostFollow] 无法连接 move_base action: {self.nav_move_base_action}")
                return False
            self._move_base_ready = True
            return True
        except Exception as exc:
            print(f"  🚶 [HostFollow] move_base action 初始化失败: {exc}")
            return False

    def _send_move_base_goal(self, goal: List[float]) -> bool:
        if self.dry_run:
            print(
                "  🚶 [HostFollow] DRY-RUN 导航目标: "
                f"x={goal[0]:.3f}, y={goal[1]:.3f}, yaw={goal[2]:.1f}deg"
            )
            return True
        if not self._ensure_move_base_client():
            return False
        goal_msg = self._MoveBaseGoal()
        goal_msg.target_pose.header.frame_id = self.nav_map_frame
        goal_msg.target_pose.header.stamp = self._rospy.Time.now()
        goal_msg.target_pose.pose.position.x = float(goal[0])
        goal_msg.target_pose.pose.position.y = float(goal[1])
        goal_msg.target_pose.pose.position.z = 0.0
        quat = self._quaternion_from_euler(0.0, 0.0, math.radians(float(goal[2])))
        goal_msg.target_pose.pose.orientation.x = quat[0]
        goal_msg.target_pose.pose.orientation.y = quat[1]
        goal_msg.target_pose.pose.orientation.z = quat[2]
        goal_msg.target_pose.pose.orientation.w = quat[3]
        self._move_base_client.send_goal(goal_msg)
        return True

    def _cancel_move_base_goal(self) -> None:
        if self.dry_run:
            return
        try:
            if self._move_base_client is not None:
                if self._GoalStatus is not None:
                    try:
                        state = self._move_base_client.get_state()
                        terminal_states = {
                            self._GoalStatus.SUCCEEDED,
                            self._GoalStatus.PREEMPTED,
                            self._GoalStatus.ABORTED,
                            self._GoalStatus.REJECTED,
                            self._GoalStatus.RECALLED,
                            self._GoalStatus.LOST,
                        }
                        if state in terminal_states:
                            return
                    except Exception:
                        pass
                self._move_base_client.cancel_goal()
        except Exception:
            pass

    def _move_base_failure_state(self) -> Optional[str]:
        if self.dry_run or self._move_base_client is None or self._GoalStatus is None:
            return None
        try:
            state = self._move_base_client.get_state()
        except Exception:
            return None
        failed_states = {
            self._GoalStatus.ABORTED: "ABORTED",
            self._GoalStatus.REJECTED: "REJECTED",
            self._GoalStatus.LOST: "LOST",
        }
        return failed_states.get(state)

    def _compute_velocity(self, payload: Dict[str, Any], person: Dict[str, Any]) -> tuple:
        offset_ratio = self._extract_offset_ratio(payload, person)
        if abs(offset_ratio) <= self.center_deadband_ratio:
            wz = 0.0
        else:
            direction = -1.0 if offset_ratio > 0.0 else 1.0
            angular_speed = self.max_angular_speed * min(1.0, abs(offset_ratio))
            angular_speed = max(
                self.min_angular_speed,
                min(self.max_angular_speed, angular_speed),
            )
            wz = direction * angular_speed

        depth_m = person.get("depth_m")
        vx = 0.0
        if isinstance(depth_m, (int, float)):
            distance_error = float(depth_m) - self.target_distance
            if distance_error > self.distance_deadband:
                vx = min(self.max_linear_speed, self.linear_kp * distance_error)
                vx = max(self.min_linear_speed, vx)
            elif self.reverse_enabled and distance_error < -self.distance_deadband:
                vx = -min(self.reverse_speed, self.linear_kp * abs(distance_error))

        reason = (
            f"depth={person.get('depth_m')}m offset={offset_ratio:.2f} "
            f"target={self.target_distance:.2f}m"
        )
        return vx, wz, reason

    @staticmethod
    def _extract_offset_ratio(payload: Dict[str, Any], person: Dict[str, Any]) -> float:
        image_width = float(payload.get("image_width") or 640.0)
        raw_offset_ratio = person.get("horizontal_offset_ratio")
        if isinstance(raw_offset_ratio, (int, float)):
            return float(raw_offset_ratio)
        else:
            bbox = person.get("bbox") or person.get("bbox_xyxy")
            center = person.get("bbox_center_px") or person.get("center_px")
            if isinstance(center, list) and center:
                center_x = float(center[0])
            elif isinstance(person.get("center_x"), (int, float)):
                center_x = float(person["center_x"])
            elif isinstance(bbox, list) and len(bbox) == 4:
                center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            else:
                center_x = image_width / 2.0
            return (center_x - image_width / 2.0) / max(image_width / 2.0, 1.0)

    def _ensure_ros(self) -> bool:
        if self.dry_run:
            return True
        if self._ros_ready:
            return True
        if self._ros_failed:
            return False
        try:
            import rospy
            import tf
            from geometry_msgs.msg import PointStamped, Twist, TwistStamped
            from std_msgs.msg import Bool, String
            from tf.transformations import euler_from_quaternion, quaternion_from_euler

            if not rospy.core.is_initialized():
                rospy.init_node("task1_host_follow_controller", anonymous=True, disable_signals=True)

            msg_cls = TwistStamped if self.msg_type == "twist_stamped" else Twist
            self._pub = rospy.Publisher(self.topic, msg_cls, queue_size=10)
            self._extra_stop_pub = (
                rospy.Publisher(self.extra_stop_topic, Twist, queue_size=10)
                if self.extra_stop_topic
                else None
            )
            self._brake_pub = (
                rospy.Publisher(self.brake_topic, Bool, queue_size=10)
                if self.release_brake
                else None
            )
            self._asr_sub = rospy.Subscriber(ASR_TOPIC, String, self._on_asr, queue_size=10)
            self._asr_segment_sub = rospy.Subscriber(
                ASR_SEGMENT_TOPIC,
                String,
                self._on_asr_segment,
                queue_size=10,
            )
            self._rospy = rospy
            self._Twist = Twist
            self._TwistStamped = TwistStamped
            self._Bool = Bool
            self._String = String
            self._PointStamped = PointStamped
            self._tf = tf
            self._quaternion_from_euler = quaternion_from_euler
            self._euler_from_quaternion = euler_from_quaternion
            self._tf_listener = tf.TransformListener()
            self._ros_ready = True
            time.sleep(0.3)
            return True
        except Exception as exc:
            print(f"  🚶 [HostFollow] ROS 底盘/ASR 接口不可用: {exc}")
            self._ros_failed = True
            return False

    def _on_asr(self, msg):
        self._push_command_from_text(str(msg.data or ""))

    def _on_asr_segment(self, msg):
        raw = str(msg.data or "")
        try:
            data = json.loads(raw)
            text = str(data.get("text", "")).strip()
        except (json.JSONDecodeError, TypeError):
            text = raw.strip()
        self._push_command_from_text(text)

    def _push_command_from_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        kind = self._classify_command(text)
        if kind is None:
            if self.log_asr:
                print(f"  🚶 [HostFollow] 收到 ASR 但未匹配控制信号: {text!r}")
            return
        if self.log_asr:
            print(f"  🚶 [HostFollow] 收到 {kind} 信号: {text!r}")
        try:
            self._command_queue.put_nowait({"kind": kind, "text": text, "time": time.time()})
        except queue.Full:
            pass

    def _classify_command(self, text: str) -> Optional[str]:
        lowered = text.lower().strip()
        for pattern in self.STOP_PATTERNS:
            if pattern.search(lowered):
                return "stop"
        for pattern in self.FOLLOW_PATTERNS:
            if pattern.search(lowered):
                return "follow"
        return None

    def _drain_commands(self) -> None:
        while True:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break

    def _get_command_nowait(self) -> Optional[Dict[str, Any]]:
        while True:
            try:
                return self._command_queue.get_nowait()
            except queue.Empty:
                return None

    def _stdin_requested(self) -> bool:
        try:
            if not self._stdin_available():
                return False
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if readable:
                sys.stdin.readline()
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _stdin_available() -> bool:
        if not _interactive_enabled():
            return False
        try:
            return bool(sys.stdin and sys.stdin.isatty())
        except Exception:
            return False

    def _make_twist(self, vx: float, wz: float):
        msg = self._Twist()
        msg.linear.x = float(vx)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        return msg

    def _make_cmd_msg(self, vx: float, wz: float):
        twist = self._make_twist(vx, wz)
        if self.msg_type == "twist":
            return twist
        msg = self._TwistStamped()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.twist = twist
        return msg

    def _publish_velocity(self, vx: float, wz: float) -> None:
        if self.dry_run:
            return
        if self._brake_pub is not None:
            self._brake_pub.publish(False)
        self._pub.publish(self._make_cmd_msg(vx, wz))

    def _publish_stop(self, repeat: int = 1) -> None:
        if self.dry_run:
            return
        if not self._ensure_ros():
            return
        count = max(1, repeat)
        if self.stop_hold_sec > 0.0:
            count = max(count, int(math.ceil(self.stop_hold_sec / 0.04)))
        stop_twist = self._make_twist(0.0, 0.0)
        for _ in range(count):
            if self._brake_pub is not None:
                self._brake_pub.publish(False)
            self._pub.publish(self._make_cmd_msg(0.0, 0.0))
            if self._extra_stop_pub is not None:
                self._extra_stop_pub.publish(stop_twist)
            time.sleep(0.04)

    def _maybe_log(self, vx: float, wz: float, reason: str) -> None:
        if self.log_interval <= 0.0:
            return
        now = time.time()
        if now - self._last_log_time < self.log_interval:
            return
        self._last_log_time = now
        print(
            "  🚶 [HostFollow] "
            f"{reason}, vx={vx:.2f}m/s, wz={math.degrees(wz):.1f}deg/s"
        )


# ============================================================
# 语音模块
# ============================================================

class SpeechModule(BaseSubModule):
    """
    语音交互子模块

    负责所有需要 TTS + ASR 对话的状态:
      ASK_GUEST1_INFO, GUIDE_GUEST1 (TTS), PICK_UP_GUEST2 (语音部分),
      DESCRIBE_GUEST1 (TTS), SEAT_GUEST2 (语音部分),
      INTRODUCE_GUESTS, REQUEST_GUEST2_BAG (TTS),
      TASK_COMPLETE (TTS)

    依赖 SpeechInterface 注入 (Mock / ROS / Scripted)。
    """

    def __init__(self, speech=None, llm=None):
        super().__init__("speech")
        if speech is None:
            from task1_receptionist.sub_modules.speech_interaction import MockSpeechInterface
            speech = MockSpeechInterface()
        self._sp = speech

        if llm is None:
            from task1_receptionist.sub_modules.llm_interface import LocalLLMInterface
            llm = LocalLLMInterface()
        self._llm = llm
        self._door_gaze_tracker = DoorGuestGazeTracker()
        self._intro_gaze_settle_sec = max(0.0, _env_float("TASK1_INTRO_GAZE_SETTLE_SEC", 2.0))
        self._intro_speech_hold_sec = max(0.0, _env_float("TASK1_INTRO_SPEECH_HOLD_SEC", 4.0))

    @property
    def speech(self):
        return self._sp

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
            Task1StateID.ASK_GUEST1_INFO: self._ask_guest1_info,
            Task1StateID.GUIDE_GUEST1: self._guide_guest1,
            Task1StateID.PICK_UP_GUEST2: self._pick_up_guest2,
            Task1StateID.DESCRIBE_GUEST1: self._describe_guest1,
            Task1StateID.SEAT_GUEST2: self._seat_guest2,
            Task1StateID.INTRODUCE_GUESTS: self._introduce_guests,
            Task1StateID.REQUEST_GUEST2_BAG: self._request_guest2_bag,
            Task1StateID.FIND_HOST: self._ask_host_to_stand_front,
            Task1StateID.FOLLOW_HOST: self._prepare_to_follow_host,
            Task1StateID.PLACE_BAG: self._hand_bag_to_host,
            Task1StateID.TASK_COMPLETE: self._task_complete,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    # ---- 各状态具体实现 ----

    def _ask_guest1_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._with_door_gaze_tracking(
            "guest1",
            lambda: self._ask_guest1_info_dialog(context),
        )

    def _with_door_gaze_tracking(
        self,
        label: str,
        action: Callable[[], Any],
        target_appearance: Optional[Dict[str, Any]] = None,
        fallback_map_xyz: Optional[List[float]] = None,
        settle_sec: float = 0.0,
        track_during_action: bool = True,
        post_action_hold_sec: float = 0.0,
    ) -> Any:
        tracker_running = False
        self._door_gaze_tracker.start(
            label,
            target_appearance=target_appearance,
            fallback_map_xyz=fallback_map_xyz,
        )
        tracker_running = self._door_gaze_tracker.is_running()
        try:
            if settle_sec > 0.0:
                time.sleep(settle_sec)
            if not track_during_action and tracker_running:
                self._door_gaze_tracker.stop()
                tracker_running = False
            result = action()
            if post_action_hold_sec > 0.0:
                print(f"  💬 [Speech] 介绍播报后保持当前朝向 {post_action_hold_sec:.1f}s")
                time.sleep(post_action_hold_sec)
            return result
        finally:
            if tracker_running:
                self._door_gaze_tracker.stop()

    def stop_guest_gaze_tracking(self) -> None:
        self._door_gaze_tracker.stop()

    def _ask_guest1_info_dialog(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  💬 [Speech] 示例: 客人应该说 'My name is Alice' 或 'I am Alice'")
        raw_name = self._sp.ask("Welcome! May I have your name please?", timeout_sec=60.0)

        name = "Guest"
        try:
            if raw_name:
                extracted = self._llm.extract_name(raw_name, role="guest1")
                if extracted:
                    name = extracted
        except Exception as e:
            print(f"  ⚠ [Speech] LLM 提取姓名失败: {e}")

        if name == "Guest" and raw_name:
            name = raw_name

        print(f"  💬 [Speech] 示例: 客人应该说 'Orange juice please' 或 'I would like some cola'")
        raw_drink = self._sp.ask(f"{name}, what would you like to drink?", timeout_sec=60.0)

        drink = "water"
        try:
            if raw_drink:
                extracted = self._llm.extract_drink(raw_drink, role="guest1")
                if extracted:
                    drink = extracted
        except Exception as e:
            print(f"  ⚠ [Speech] LLM 提取饮品失败: {e}")

        if drink == "water" and raw_drink:
            drink = raw_drink

        return {"guest1_name": name, "guest1_drink": drink}

    def _pick_up_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._with_door_gaze_tracking(
            "guest2",
            lambda: self._pick_up_guest2_dialog(context),
        )

    def _pick_up_guest2_dialog(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  💬 [Speech] 示例: 客人应该说 'My name is Bob' 或 'I am Bob'")
        raw_name = self._sp.ask("Welcome! May I have your name please?", timeout_sec=60.0)

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

        print(f"  💬 [Speech] 示例: 客人应该说 'Coffee please' 或 'I would like some tea'")
        raw_drink = self._sp.ask(f"{name}, what would you like to drink?", timeout_sec=60.0)

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

        self._sp.say(f"Thank you, {name}. Please get ready to follow me to the living room.")
        return {"guest2_at_door": True, "guest2_name": name, "guest2_drink": drink}

    def _describe_guest1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        sentence, described = _guest1_description_sentence(context)
        self._sp.say(sentence)
        return {"guest1_described": described}

    def _guide_guest1(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guest = context.get("guest1_name", "guest")
        self._sp.say(f"{guest}, please follow me to the living room.")
        return None

    def _seat_guest2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        sentence, described = _guest1_description_sentence(context)
        self._sp.say(sentence)
        return {"guest1_described": described, "guest2_seated": True, "seat_number": 2}

    def _introduce_guests(self, context: Dict[str, Any]) -> Dict[str, Any]:
        g1 = context.get("guest1_name", "guest1")
        g1d = context.get("guest1_drink", "?")
        g2 = context.get("guest2_name", "guest2")
        g2d = context.get("guest2_drink", "?")
        g1_appearance = context.get("guest1_appearance")
        g2_appearance = context.get("guest2_appearance")
        if not isinstance(g1_appearance, dict):
            g1_appearance = {}
        if not isinstance(g2_appearance, dict):
            g2_appearance = {}
        guest1_seat_map_xyz = context.get("guest1_seat_map_xyz") or context.get("empty_seat_map_xyz")
        guest2_seat_map_xyz = context.get("guest2_seat_map_xyz") or context.get("guest2_empty_seat_map_xyz")

        self._with_door_gaze_tracking(
            "intro_guest1",
            lambda: self._sp.say(f"{g1}, this is {g2}, who would like {g2d}."),
            target_appearance=g1_appearance,
            fallback_map_xyz=guest1_seat_map_xyz,
            settle_sec=self._intro_gaze_settle_sec,
            track_during_action=False,
            post_action_hold_sec=self._intro_speech_hold_sec,
        )
        self._with_door_gaze_tracking(
            "intro_guest2",
            lambda: self._sp.say(f"{g2}, this is {g1}, who would like {g1d}."),
            target_appearance=g2_appearance,
            fallback_map_xyz=guest2_seat_map_xyz,
            settle_sec=self._intro_gaze_settle_sec,
            track_during_action=False,
            post_action_hold_sec=self._intro_speech_hold_sec,
        )
        return {"guests_introduced": True}

    def _request_guest2_bag(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guest = context.get("guest2_name", "guest2")
        self._sp.say(f"{guest}, please hold your bag in front of me. I will come closer and take it.")
        return None

    def _ask_host_to_stand_front(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if context.get("host_interaction_reached") is False:
            print("  💬 [Speech] 未到达 host 交互点，仍在当前位置请求 host 前方引导")
        self._sp.say("Hello. I have brought your bag.")
        self._sp.say("Please stand in front of me and lead the way.")
        return None

    def _prepare_to_follow_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._sp.say("I am ready to follow you.")
        self._sp.say("I will stop when we arrive.")
        return {"host_guidance_requested": True}

    def _hand_bag_to_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._sp.say("We have arrived. I am ready to hand your bag to you.")
        return {"host_handoff_announced": True}

    def _task_complete(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._sp.say("Task completed. Thank you all for your cooperation!")
        return {}

    def close(self):
        self._door_gaze_tracker.stop()
        self._sp.close()
        if hasattr(self._llm, 'close'):
            self._llm.close()


# ============================================================
# 视觉模块
# ============================================================

class VisionModule(BaseSubModule):
    """
    视觉子模块 — 读取已启动视觉服务的结果

    guest1 外貌默认只读取 start_all.sh 刷新的 latest_people.json,
    避免在主流程中重新触发相机或额外视觉进程。
    """

    def __init__(self):
        super().__init__("vision")
        self._bridge = ROSTopicBridge(VISION_COMMAND_TOPIC, VISION_RESULT_TOPIC, VISION_TIMEOUT)
        self._host_find_probe = HostFollowController()
        self._host_find_wait_sec = _env_float("TASK1_HOST_FIND_WAIT_SEC", 60.0)
        self._host_find_center_ratio = max(
            0.0,
            min(1.0, _env_float("TASK1_HOST_FIND_CENTER_RATIO", 0.45)),
        )
        self._host_find_min_distance = max(
            0.0,
            _env_float("TASK1_HOST_FIND_MIN_DISTANCE_M", 0.3),
        )
        self._host_find_max_distance = max(
            0.0,
            _env_float("TASK1_HOST_FIND_MAX_DISTANCE_M", 4.5),
        )
        self._host_find_log_interval = max(
            0.0,
            _env_float("TASK1_HOST_FIND_LOG_INTERVAL", 1.0),
        )
        self._guest_appearance_cache_max_age_sec = max(
            0.0,
            _env_float("TASK1_GUEST_APPEARANCE_CACHE_MAX_AGE_SEC", 60.0),
        )
        self._guest_appearance_active_query = _env_flag(
            "TASK1_GUEST_APPEARANCE_ACTIVE_QUERY",
            False,
        )
        self._guest_appearance_timeout_sec = max(
            0.0,
            _env_float("TASK1_GUEST_APPEARANCE_TIMEOUT_SEC", 3.0),
        )
        self._guest_appearance_people_json = Path(
            os.environ.get("TASK1_GUEST_APPEARANCE_PEOPLE_JSON", str(OBJECT_SEARCH_PEOPLE_JSON))
        )

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
            Task1StateID.ASK_GUEST1_INFO: self._capture_guest1_appearance,
            Task1StateID.PICK_UP_GUEST2: self._capture_guest2_appearance,
            Task1StateID.DESCRIBE_GUEST1: self._capture_guest1_appearance,
            Task1StateID.FIND_HOST: self._find_host,
            Task1StateID.FOLLOW_HOST: self._track_person,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    def _call_vision(self, command: str, timeout: Optional[float] = None) -> Optional[dict]:
        result = self._bridge.call(command, timeout=timeout)
        if result is None:
            print(f"  📷 [Vision] ⚠️ 未收到视觉结果: {command}")
            return None
        return result

    def _capture_guest1_appearance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._capture_guest_appearance("guest1")

    def _capture_guest2_appearance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._capture_guest_appearance("guest2")

    def _capture_guest_appearance(self, guest_key: str) -> Dict[str, Any]:
        result_key = f"{guest_key}_appearance"
        print(f"  📷 [Vision] 从后台视觉缓存读取 {guest_key} 衣着特征...")
        cache_appearance = self._guest_appearance_from_people_cache()
        if cache_appearance and _guest1_visual_attributes(cache_appearance):
            cache_appearance["guest_key"] = guest_key
            print(
                f"  📷 [Vision] 从人物缓存提取 {guest_key} 衣着: "
                f"{cache_appearance.get('clothing', 'unknown clothing')}"
            )
            return {result_key: cache_appearance}

        if self._guest_appearance_active_query:
            print("  📷 [Vision] 缓存无衣着字段，按配置主动请求视觉结果...")
            result = self._call_vision(
                VISION_CMD_DESCRIBE_PERSON,
                timeout=self._guest_appearance_timeout_sec,
            )
            if result and result.get("status") == "success":
                data = result.get("data", {})
                appearance = self._normalize_guest_appearance(data, source="vision_result")
                if _guest1_visual_attributes(appearance):
                    appearance["guest_key"] = guest_key
                    print(
                        f"  📷 [Vision] {guest_key} 衣着缓存完成: "
                        f"{appearance.get('clothing', 'unknown clothing')}"
                    )
                    return {result_key: appearance}

        print(f"  📷 [Vision] 未获得可靠 {guest_key} 衣着属性，缓存 unknown")
        return {
            result_key: {
                "clothing": "unknown clothing",
                "visual_attributes": [],
                "clothing_signature": {"attributes": [], "keys": []},
                "guest_key": guest_key,
                "source": "unavailable",
            }
        }

    def _normalize_guest_appearance(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        appearance = dict(data)
        appearance["source"] = source
        attrs = _guest1_visual_attributes(appearance)
        if attrs:
            appearance["visual_attributes"] = attrs
            appearance.setdefault("clothing", ", ".join(attrs))
        else:
            appearance.setdefault("visual_attributes", [])
            appearance.setdefault("clothing", "unknown clothing")
        appearance["clothing_signature"] = _guest_clothing_signature(appearance)
        return appearance

    def _guest_appearance_from_people_cache(self) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(self._guest_appearance_people_json.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            print(f"  📷 [Vision] 人物缓存 JSON 解析失败: {exc}")
            return None

        if payload.get("status") != "success":
            return None
        timestamp = payload.get("timestamp")
        if (
            self._guest_appearance_cache_max_age_sec > 0.0
            and isinstance(timestamp, (int, float))
            and time.time() - float(timestamp) > self._guest_appearance_cache_max_age_sec
        ):
            print(
                "  📷 [Vision] 人物缓存已过期: "
                f"{time.time() - float(timestamp):.1f}s > {self._guest_appearance_cache_max_age_sec:.1f}s"
            )
            return None
        person = payload.get("best_person") or {}
        if not isinstance(person, dict) or not person:
            return None

        data: Dict[str, Any] = {
            "source": "people_cache",
            "position": "near the door",
            "source_person": {
                "person_index": person.get("person_index") or person.get("index"),
                "bbox": person.get("bbox"),
                "bbox_center_px": person.get("bbox_center_px"),
                "depth_m": person.get("depth_m"),
                "confidence": person.get("confidence"),
            },
            "people_cache_timestamp": timestamp,
        }
        for key in (
            "clothing",
            "clothing_summary",
            "cloth_summary",
            "cloth_items",
            "cloth_color",
            "cloth_type",
            "has_glasses",
        ):
            if key in person:
                data[key] = person[key]
        return self._normalize_guest_appearance(data, source="people_cache")

    def _find_host(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("host_interaction_reached") is False:
            print("  📷 [Vision:Host] 未到达 host 交互点，仍在当前位置等待前方 host")

        PeopleVisionCacheManager.ensure_running(
            "  📷 [Vision:Host]",
            self._host_find_probe.cache_max_age,
        )
        print("  📷 [Vision:Host] 等待 host 站到机器人前方...")
        deadline = time.time() + max(0.0, self._host_find_wait_sec)
        last_log_time = 0.0
        last_reason = "not_started"

        while time.time() < deadline:
            payload, reject_reason = self._host_find_probe._load_people_payload_with_reason()
            if payload:
                person = payload.get("best_person") or {}
                ok, reason = self._host_person_is_front_candidate(payload, person)
                last_reason = reason
                if ok:
                    print(f"  📷 [Vision:Host] 已确认前方 host: {reason}")
                    return {
                        "host_found": True,
                        "host_location": "host_interaction_front",
                        "host_target": person,
                    }
            else:
                last_reason = reject_reason

            now = time.time()
            if self._host_find_log_interval > 0.0 and now - last_log_time >= self._host_find_log_interval:
                remaining = max(0.0, deadline - now)
                print(
                    "  📷 [Vision:Host] "
                    f"等待前方人物: {last_reason}, remaining={remaining:.1f}s"
                )
                last_log_time = now
            time.sleep(0.2)

        print(f"  📷 [Vision:Host] 未确认 host: {last_reason}")
        return {
            "host_found": False,
            "host_location": "unknown",
            "host_find_error": last_reason,
        }

    def _host_person_is_front_candidate(
        self,
        payload: Dict[str, Any],
        person: Dict[str, Any],
    ) -> tuple:
        if not isinstance(person, dict) or not person:
            return False, "missing_person"

        offset_ratio = HostFollowController._extract_offset_ratio(payload, person)
        if abs(offset_ratio) > self._host_find_center_ratio:
            return False, f"person_not_centered offset={offset_ratio:.2f}"

        depth_m = person.get("depth_m")
        if isinstance(depth_m, (int, float)):
            depth = float(depth_m)
            if self._host_find_min_distance > 0.0 and depth < self._host_find_min_distance:
                return False, f"person_too_close depth={depth:.2f}m"
            if self._host_find_max_distance > 0.0 and depth > self._host_find_max_distance:
                return False, f"person_too_far depth={depth:.2f}m"
            return True, f"offset={offset_ratio:.2f} depth={depth:.2f}m"

        return True, f"offset={offset_ratio:.2f} depth=unknown"

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
    操作子模块 — 直接调用 object_search 实现空座指向、接包和交包。
    """

    def __init__(
        self,
        speech=None,
        object_search_dry_run: bool = False,
        object_search_max_frames: int = 120,
        object_search_skip_vision: bool = False,
    ):
        super().__init__("manipulation")
        self._speech = speech
        self._seat_prompt = os.environ.get("TASK1_SEAT_PROMPT", "Please take a seat here.")
        self._object_search_dry_run = object_search_dry_run or _env_flag(
            "TASK1_OBJECT_SEARCH_DRY_RUN", False
        )
        self._object_search_max_frames = _env_int(
            "TASK1_OBJECT_SEARCH_MAX_FRAMES", object_search_max_frames
        )
        self._empty_seat_max_frames = _env_int(
            "TASK1_EMPTY_SEAT_MAX_FRAMES", DEFAULT_EMPTY_SEAT_MAX_FRAMES
        )
        self._object_search_skip_vision = object_search_skip_vision or _env_flag(
            "TASK1_OBJECT_SEARCH_SKIP_VISION", False
        )
        self._object_search_prefer_cache = _env_flag(
            "TASK1_OBJECT_SEARCH_PREFER_CACHE", True
        )
        self._object_search_cache_max_age = _env_float(
            "TASK1_OBJECT_SEARCH_CACHE_MAX_AGE_SEC",
            DEFAULT_OBJECT_SEARCH_CACHE_MAX_AGE_SEC,
        )
        self._hand_search_attempts = max(1, _env_int("TASK1_HAND_SEARCH_ATTEMPTS", 2))
        self._handover_recognition_delay_sec = _env_float(
            "TASK1_HANDOVER_RECOGNITION_DELAY_SEC",
            5.0,
        )
        self._handover_cache_wait_sec = max(
            0.0,
            _env_float("TASK1_HANDOVER_CACHE_WAIT_SEC", 6.0),
        )
        self._handover_require_approach_nav = _env_flag(
            "TASK1_HANDOVER_REQUIRE_APPROACH_NAV",
            True,
        )
        self._handover_hand_standoff = _env_float(
            "TASK1_HANDOVER_HAND_STANDOFF",
            0.14,
        )
        self._handover_max_arm_reach = _env_float(
            "TASK1_HANDOVER_MAX_ARM_REACH",
            0.70,
        )
        self._left_gripper_topic = os.environ.get(
            "TASK1_LEFT_GRIPPER_TOPIC", LEFT_GRIPPER_POSITION_TOPIC
        )
        self._open_gripper_value = _env_float("TASK1_OPEN_GRIPPER_VALUE", 100.0)
        self._close_gripper_value = _env_float("TASK1_CLOSE_GRIPPER_VALUE", 0.0)
        self._gripper_command_rate = _env_float("TASK1_GRIPPER_COMMAND_RATE", 10.0)
        self._gripper_command_duration = _env_float("TASK1_GRIPPER_COMMAND_DURATION", 1.0)

        print("  [Manip] 使用 object_search 操作后端")
        if self._object_search_dry_run:
            print("  [Manip] object_search dry-run: 只计算/打印目标，不发布机械臂或夹爪指令")

    def execute(self, state_id: Task1StateID, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dispatch: Dict[Task1StateID, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {
            Task1StateID.POINT_EMPTY_SEAT: self._point_empty_seat,
            Task1StateID.SEAT_GUEST2: self._point_guest2_seat,
            Task1StateID.REQUEST_GUEST2_BAG: self._wait_for_bag,
            Task1StateID.PLACE_BAG: self._place_bag,
        }
        handler = dispatch.get(state_id)
        return handler(context) if handler else None

    def _extra_args(self, env_name: str) -> List[str]:
        value = os.environ.get(env_name, "").strip()
        if not value:
            return []
        try:
            return shlex.split(value)
        except ValueError as exc:
            print(f"  [Manip] {env_name} 参数解析失败: {exc}")
            return []

    def _run_local_command(
        self,
        label: str,
        argv: List[str],
        timeout: Optional[float] = None,
    ) -> bool:
        printable = " ".join(shlex.quote(str(part)) for part in argv)
        print(f"  [Manip:ObjectSearch] {label}")
        print(f"    $ {printable}")
        try:
            completed = subprocess.run(
                [str(part) for part in argv],
                cwd=str(OBJECT_SEARCH_DIR),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"  [Manip:ObjectSearch] {label} 超时")
            return False
        except FileNotFoundError as exc:
            print(f"  [Manip:ObjectSearch] {label} 启动失败: {exc}")
            return False
        except Exception as exc:
            print(f"  [Manip:ObjectSearch] {label} 异常: {exc}")
            return False

        if completed.returncode != 0:
            print(f"  [Manip:ObjectSearch] {label} 失败, returncode={completed.returncode}")
            return False
        return True

    def _say_seat_prompt_async(self, label: str) -> None:
        if self._speech is None:
            print("  [Manip:ObjectSearch] 无语音接口，跳过座位提示")
            return

        text = self._seat_prompt.strip()
        if not text:
            return

        try:
            print(f"  [Manip:ObjectSearch] {label} 座位提示: {text}")
            self._speech.say(text)
        except Exception as exc:
            print(f"  [Manip:ObjectSearch] 座位提示失败: {exc}")

    def _read_json_file(self, path: Path, label: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"  [Manip] 未找到{label}结果: {path}")
            return None
        except json.JSONDecodeError as exc:
            print(f"  [Manip] {label}结果不是合法 JSON: {exc}")
            return None

    def _run_empty_seat_search(self) -> bool:
        cache_state = (
            _object_search_cache_state(
                OBJECT_SEARCH_OUTPUT_JSON,
                "空座识别",
                self._object_search_cache_max_age,
                "  [Manip:ObjectSearch]",
            )
            if self._object_search_prefer_cache
            else "miss"
        )
        if cache_state == "fresh":
            pass
        elif cache_state == "daemon_running":
            return False
        elif self._object_search_skip_vision:
            print("  [Manip:ObjectSearch] 跳过空座视觉刷新，复用 latest_empty_seat.json")
        else:
            argv = [
                "bash",
                str(OBJECT_SEARCH_RUN_EMPTY),
                "--no-display",
                "--max-frames",
                str(self._empty_seat_max_frames),
                "--stop-on-success",
            ]
            argv.extend(self._extra_args("TASK1_EMPTY_SEAT_SEARCH_ARGS"))
            if not self._run_local_command("运行空座识别", argv):
                return False

        payload = self._read_json_file(OBJECT_SEARCH_OUTPUT_JSON, "空座识别")
        if not payload:
            return False
        if payload.get("status") != "success":
            print(f"  [Manip:ObjectSearch] 空座识别未成功: {payload.get('status')}")
            return False
        return True

    def _handover_payload_success(
        self,
        payload: Optional[Dict[str, Any]],
        min_timestamp: Optional[float] = None,
    ) -> bool:
        if not payload or payload.get("status") != "success":
            return False
        if min_timestamp is None:
            return True
        timestamp = payload.get("timestamp")
        return isinstance(timestamp, (int, float)) and float(timestamp) >= min_timestamp - 0.05

    def _wait_for_handover_hand_cache(
        self,
        min_timestamp: Optional[float] = None,
    ) -> bool:
        if self._handover_cache_wait_sec <= 0.0:
            return False
        deadline = time.time() + self._handover_cache_wait_sec
        last_status = None
        print(
            "  [Manip:ObjectSearch] 等待后台接包目标刷新 "
            f"最多 {self._handover_cache_wait_sec:.1f}s"
        )
        while time.time() < deadline:
            payload = self._read_json_file(OBJECT_SEARCH_HAND_JSON, "拿包手腕识别")
            if self._handover_payload_success(payload, min_timestamp=min_timestamp):
                return True
            if payload:
                last_status = payload.get("status")
            time.sleep(0.2)
        print(f"  [Manip:ObjectSearch] 等待接包目标超时: last_status={last_status}")
        return False

    def _run_handover_hand_search(
        self,
        min_timestamp: Optional[float] = None,
    ) -> bool:
        cache_state = (
            _object_search_cache_state(
                OBJECT_SEARCH_HAND_JSON,
                "拿包手腕识别",
                self._object_search_cache_max_age,
                "  [Manip:ObjectSearch]",
            )
            if self._object_search_prefer_cache
            else "miss"
        )
        if cache_state == "fresh":
            payload = self._read_json_file(OBJECT_SEARCH_HAND_JSON, "拿包手腕识别")
            if self._handover_payload_success(payload, min_timestamp=min_timestamp):
                return True
            return self._wait_for_handover_hand_cache(min_timestamp=min_timestamp)
        if cache_state == "daemon_running":
            return self._wait_for_handover_hand_cache(min_timestamp=min_timestamp)

        if self._object_search_skip_vision:
            print("  [Manip:ObjectSearch] 跳过拿包手腕视觉刷新，复用 latest_handover_hand.json")
            payload = self._read_json_file(OBJECT_SEARCH_HAND_JSON, "拿包手腕识别")
            return self._handover_payload_success(payload, min_timestamp=min_timestamp)

        for attempt in range(1, self._hand_search_attempts + 1):
            if attempt > 1:
                print(f"  [Manip:ObjectSearch] 拿包手腕识别重试 {attempt}/{self._hand_search_attempts}")
            argv = [
                "bash",
                str(OBJECT_SEARCH_RUN_HAND),
                "--no-display",
                "--max-frames",
                str(self._object_search_max_frames),
            ]
            argv.extend(self._extra_args("TASK1_HAND_SEARCH_ARGS"))
            if not self._run_local_command("运行拿包手腕识别", argv):
                return False

            payload = self._read_json_file(OBJECT_SEARCH_HAND_JSON, "拿包手腕识别")
            if self._handover_payload_success(payload, min_timestamp=min_timestamp):
                return True

            status = None if payload is None else payload.get("status")
            print(f"  [Manip:ObjectSearch] 暂未识别到拿包的手: {status}")
            if attempt < self._hand_search_attempts:
                if _interactive_enabled():
                    try:
                        input("  [Manip:ObjectSearch] 请客人把包拿到相机前，按 Enter 重试识别: ")
                    except (EOFError, KeyboardInterrupt):
                        print("  [Manip:ObjectSearch] 输入中断，继续重试")
                else:
                    retry_delay = max(0.0, _env_float("TASK1_HAND_SEARCH_RETRY_DELAY_SEC", 1.0))
                    if retry_delay > 0.0:
                        print(
                            "  [Manip:ObjectSearch] 非交互模式: "
                            f"{retry_delay:.1f}s 后自动重试识别"
                        )
                        time.sleep(retry_delay)

        return False

    def _load_handover_target(self) -> Optional[Dict[str, Any]]:
        payload = self._read_json_file(OBJECT_SEARCH_HAND_JSON, "拿包手腕识别")
        if not payload or payload.get("status") != "success":
            return None
        target_record = payload.get("handover_target") or payload.get("best_hand") or {}
        position = target_record.get("position") or {}
        xyz = (
            position.get("leftbase_xyz_m")
            or position.get("left_arm_base_xyz_m")
            or position.get("target_frame_xyz_m")
            or position.get("calibrated_xyz_m")
        )
        if not isinstance(xyz, list) or len(xyz) != 3:
            print("  [Manip] 拿包手腕识别结果缺少可用目标坐标")
            return None
        return {
            "frame_id": payload.get("coordinate_frame") or POINTING_TARGET_FRAME,
            "xyz_m": [float(value) for value in xyz],
            "target_type": target_record.get("target_type"),
            "selection_label": target_record.get("selection_label"),
            "selection_priority": target_record.get("selection_priority"),
            "handover_position_source": target_record.get("handover_position_source"),
            "handover_depth_m": target_record.get("handover_depth_m"),
            "wrist_position": target_record.get("wrist_position"),
            "matched_object_position": target_record.get("matched_object_position"),
            "hand_side": target_record.get("hand_side"),
            "confidence": target_record.get("confidence"),
            "object_class_name": target_record.get("object_class_name"),
            "object_confidence": target_record.get("object_confidence"),
            "bag_class_name": target_record.get("bag_class_name"),
            "bag_confidence": target_record.get("bag_confidence"),
            "hand_object_distance_px": target_record.get("hand_object_distance_px"),
            "hand_bag_distance_px": target_record.get("hand_bag_distance_px"),
            "source_json": str(OBJECT_SEARCH_HAND_JSON),
            "timestamp": payload.get("timestamp"),
        }

    def _publish_gripper_position(self, value: float) -> bool:
        if self._object_search_dry_run:
            print(
                "  [Manip:ObjectSearch] dry-run: "
                f"不发布夹爪 value={value:.3f} 到 {self._left_gripper_topic}"
            )
            return True

        try:
            import rospy
            from std_msgs.msg import Float32

            if not rospy.core.is_initialized():
                rospy.init_node("task1_object_search_manipulation", anonymous=True, disable_signals=True)
            pub = rospy.Publisher(self._left_gripper_topic, Float32, queue_size=10)
            time.sleep(0.3)
            rate = rospy.Rate(self._gripper_command_rate)
            msg = Float32(data=float(value))
            end_time = time.time() + self._gripper_command_duration
            while not rospy.is_shutdown() and time.time() < end_time:
                pub.publish(msg)
                rate.sleep()
            print(
                "  [Manip:ObjectSearch] 已发布夹爪命令 "
                f"value={value:.3f} {self._gripper_command_duration:.1f}s "
                f"到 {self._left_gripper_topic}"
            )
            return True
        except Exception as exc:
            print(f"  [Manip:ObjectSearch] 发布夹爪命令失败: {exc}")
            return False

    def _load_empty_seat_target(self) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(OBJECT_SEARCH_OUTPUT_JSON.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"  🦾 [Manip] 未找到空座识别结果: {OBJECT_SEARCH_OUTPUT_JSON}")
            return None
        except json.JSONDecodeError as exc:
            print(f"  🦾 [Manip] 空座识别结果不是合法 JSON: {exc}")
            return None

        if payload.get("status") != "success":
            print(f"  🦾 [Manip] 空座识别状态不是 success: {payload.get('status')}")
            return None

        seat = payload.get("best_empty_seat") or {}
        position = seat.get("position") or {}
        xyz = (
            position.get("leftbase_xyz_m")
            or position.get("left_arm_base_xyz_m")
            or position.get("target_frame_xyz_m")
            or position.get("calibrated_xyz_m")
        )
        if not isinstance(xyz, list) or len(xyz) != 3:
            print("  🦾 [Manip] 空座识别结果缺少可用目标坐标")
            return None

        frame_id = payload.get("coordinate_frame") or POINTING_TARGET_FRAME
        target = {
            "frame_id": frame_id,
            "xyz_m": [float(value) for value in xyz],
            "class_name": seat.get("class_name"),
            "confidence": seat.get("confidence"),
            "depth_m": seat.get("depth_m"),
            "bbox": seat.get("bbox"),
            "source_json": str(OBJECT_SEARCH_OUTPUT_JSON),
            "timestamp": payload.get("timestamp"),
        }
        return target

    def _point_empty_seat_with_object_search(
        self,
        label: str,
        success_key: str,
        target_key: str,
    ) -> Dict[str, Any]:
        print(f"  [Manip:ObjectSearch] 为 {label} 刷新空座识别并指向")
        if not self._run_empty_seat_search():
            return {success_key: False, target_key: None}

        target = self._load_empty_seat_target()
        if target:
            print(
                "  [Manip:ObjectSearch] 空座目标 "
                f"{target['frame_id']}: {target['xyz_m']} m"
            )

        argv = [sys.executable, str(OBJECT_SEARCH_POINT_ARM)]
        if self._object_search_dry_run:
            argv.append("--dry-run")
        elif _assume_yes_enabled():
            argv.append("--yes")
        argv.extend(self._extra_args("TASK1_POINT_EMPTY_SEAT_ARM_ARGS"))

        self._say_seat_prompt_async(label)
        ok = self._run_local_command("发布左臂指向空座目标", argv)
        return {success_key: ok, target_key: target}

    def _receive_bag_with_object_search(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("  [Manip:ObjectSearch] 识别客人拿着包的手，并让左臂过去接包")
        handover_approach_reached = context.get("handover_approach_reached") is True
        if self._handover_require_approach_nav and not handover_approach_reached:
            print("  [Manip:ObjectSearch] 尚未到达接包接近点，跳过左臂接包动作")
            return {"bag_on_tray": False, "handover_hand_target": None}

        min_timestamp = None
        if handover_approach_reached:
            min_timestamp = context.get("handover_approach_completed_time")
            if not isinstance(min_timestamp, (int, float)):
                min_timestamp = time.time()
            print("  [Manip:ObjectSearch] 已到达接包接近点，等待移动后的新鲜接包目标")
        elif self._handover_recognition_delay_sec > 0.0:
            print(
                "  [Manip:ObjectSearch] 已提示客人递包，等待 "
                f"{self._handover_recognition_delay_sec:.1f}s 后开始识别"
            )
            time.sleep(self._handover_recognition_delay_sec)

        if not self._run_handover_hand_search(min_timestamp=min_timestamp):
            return {"bag_on_tray": False, "handover_hand_target": None}

        target = self._load_handover_target()
        if target:
            object_name = target.get("object_class_name") or target.get("bag_class_name") or "none"
            object_conf = target.get("object_confidence")
            if object_conf is None:
                object_conf = target.get("bag_confidence")
            print(
                "  [Manip:ObjectSearch] 接包目标 "
                f"{target['frame_id']}: {target['xyz_m']} m, "
                f"type={target.get('target_type')} "
                f"source={target.get('handover_position_source')} "
                f"object={object_name} "
                f"conf={object_conf}"
            )

        argv = [
            sys.executable,
            str(OBJECT_SEARCH_APPROACH_HANDOVER_ARM),
            "--gripper-topic",
            self._left_gripper_topic,
            "--open-gripper-value",
            str(self._open_gripper_value),
            "--close-gripper-value",
            str(self._close_gripper_value),
            "--gripper-command-rate",
            str(self._gripper_command_rate),
            "--gripper-command-duration",
            str(self._gripper_command_duration),
            "--hand-standoff",
            str(self._handover_hand_standoff),
            "--max-arm-reach",
            str(self._handover_max_arm_reach),
        ]
        if self._object_search_dry_run:
            argv.append("--dry-run")
        elif _assume_yes_enabled():
            argv.append("--yes")
        argv.extend(self._extra_args("TASK1_APPROACH_HANDOVER_ARM_ARGS"))

        ok = self._run_local_command("发布左臂接包目标", argv)
        return {
            "bag_on_tray": ok,
            "handover_hand_target": target,
            "handover_hand_target_after_nav": target,
        }

    def _place_bag_with_object_search(self) -> Dict[str, Any]:
        print("  [Manip:ObjectSearch] 到达 host 位置，准备打开左夹爪交包")
        print(f"  [Manip:ObjectSearch] 夹爪 topic: {self._left_gripper_topic}")
        print(f"  [Manip:ObjectSearch] 打开值: {self._open_gripper_value:.3f}")
        if not self._object_search_dry_run and not _assume_yes_enabled():
            try:
                if input("  [Manip:ObjectSearch] 输入 y 打开左夹爪交包，其它输入取消: ").strip().lower() != "y":
                    print("  [Manip:ObjectSearch] 已取消交包动作")
                    return {"bag_placed": False}
            except (EOFError, KeyboardInterrupt):
                print("  [Manip:ObjectSearch] 输入中断，取消交包动作")
                return {"bag_placed": False}
        elif not self._object_search_dry_run:
            print("  [Manip:ObjectSearch] 非交互模式: 自动打开左夹爪交包")

        ok = self._publish_gripper_position(self._open_gripper_value)
        return {"bag_placed": ok}

    def _point_empty_seat(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not context.get("empty_seat_approach_reached") and not context.get("empty_seat_target"):
            print("  [Manip:ObjectSearch] 未到达 guest1 空座接近点，且没有可用空座目标，无法指向")
            return {
                "seat_pointed": False,
                "empty_seat_target": context.get("empty_seat_target"),
            }
        if not context.get("empty_seat_approach_reached"):
            print("  [Manip:ObjectSearch] 未到达 guest1 空座接近点，但已有空座目标，继续尝试在当前位置指向")
        return self._point_empty_seat_with_object_search(
            "guest1", "seat_pointed", "empty_seat_target"
        )

    def _point_guest2_seat(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if (
            not context.get("guest2_empty_seat_approach_reached")
            and not context.get("guest2_empty_seat_target")
        ):
            print("  [Manip:ObjectSearch] 未到达 guest2 空座接近点，且没有可用空座目标，无法指向")
            return {
                "guest2_seat_pointed": False,
                "guest2_empty_seat_target": context.get("guest2_empty_seat_target"),
            }
        if not context.get("guest2_empty_seat_approach_reached"):
            print("  [Manip:ObjectSearch] 未到达 guest2 空座接近点，但已有空座目标，继续尝试在当前位置指向")
        return self._point_empty_seat_with_object_search(
            "guest2", "guest2_seat_pointed", "guest2_empty_seat_target"
        )

    def _wait_for_bag(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._receive_bag_with_object_search(context)

    def _place_bag(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._place_bag_with_object_search()

# ============================================================
# 兼容别名
# ============================================================

ASRModule = SpeechModule
