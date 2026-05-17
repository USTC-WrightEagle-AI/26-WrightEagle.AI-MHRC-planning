"""
Task1 — Receptionist 状态定义

RoboCup@Home 接待任务：机器人迎接两位客人，引导入座、介绍、拿包并跟随host放包。

=============================================================
状态机流程 (14 个执行状态)
=============================================================

IDLE
  │  /task1/start
  ▼
[1] WAIT_FOR_DOORBELL_1 ─ 等待门铃声 (guest1 到达)
  │
  ▼
[2] GO_TO_DOOR ───────── 机器人移动到门口
  │
  ▼
[3] ASK_GUEST1_INFO ──── 询问 guest1 姓名和想喝的饮料
  │                       产出: guest1_name, guest1_drink
  ▼
[4] GUIDE_GUEST1 ─────── 带领 guest1 到客厅
  │
  ▼
[5] POINT_EMPTY_SEAT ─── 指向预设空座，请 guest1 入座
  │
  ▼
[6] RETURN_TO_START ──── 返回起点
  │
  ▼
[7] WAIT_FOR_DOORBELL_2 ─ 等待门铃声 (guest2 到达)
  │
  ▼
[8] PICK_UP_GUEST2 ───── 到门口接 guest2
  │
  ▼
[9] DESCRIBE_GUEST1 ──── 向 guest2 描述 guest1 的外貌/衣着
  │                       需要: guest1_name
  ▼
[10] SEAT_GUEST2 ──────── 带领 guest2 到客厅入座
  │
  ▼
[11] INTRODUCE_GUESTS ─── 相互介绍两位客人
  │                       需要: guest1_name, guest2_name
  ▼
[12] REQUEST_GUEST2_BAG ─ 请求 guest2 把包放到机器人托盘上
  │
  ▼
[13] FIND_HOST ────────── 找到 host
  │
  ▼
[14] FOLLOW_HOST ─────── 跟随 host 去指定位置
  │
  ▼
[15] PLACE_BAG ────────── 将包放到指定位置
  │
  ▼
[16] TASK_COMPLETE ───── 任务完成，播报结束语
  │
  ▼
IDLE
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# 状态标识枚举
# ============================================================

class Task1StateID(Enum):
    """Task1 所有状态的唯一标识"""
    IDLE = "idle"

    # ---- 执行状态 (按流程顺序) ----
    WAIT_FOR_DOORBELL_1 = "wait_for_doorbell_1"
    GO_TO_DOOR = "go_to_door"
    ASK_GUEST1_INFO = "ask_guest1_info"
    GUIDE_GUEST1 = "guide_guest1"
    POINT_EMPTY_SEAT = "point_empty_seat"
    RETURN_TO_START = "return_to_start"
    WAIT_FOR_DOORBELL_2 = "wait_for_doorbell_2"
    PICK_UP_GUEST2 = "pick_up_guest2"
    DESCRIBE_GUEST1 = "describe_guest1"
    SEAT_GUEST2 = "seat_guest2"
    INTRODUCE_GUESTS = "introduce_guests"
    REQUEST_GUEST2_BAG = "request_guest2_bag"
    FIND_HOST = "find_host"
    FOLLOW_HOST = "follow_host"
    PLACE_BAG = "place_bag"
    TASK_COMPLETE = "task_complete"

    # ---- 异常状态 ----
    ERROR = "error"
    ABORTED = "aborted"


# ============================================================
# 状态定义数据类
# ============================================================

@dataclass
class StateDefinition:
    """
    单个状态的定义

    Attributes:
        state_id: 状态唯一标识
        index: 流程中的序号 (1-based, IDLE=0)
        description: 状态的业务含义
        next_state: 成功时的下一状态 (None 表示终端)
        ros_command_topic: 接收启动指令的 ROS topic
        ros_done_topic: 完成后发布结果的 ROS topic
        data_needed: 启动此状态需要的上下文数据键
        data_produced: 此状态完成后产生的上下文数据键
        timeout_sec: 超时时间 (秒)
        retry_on_failure: 失败后允许的最大重试次数
    """
    state_id: Task1StateID
    index: int
    description: str
    next_state: Optional[Task1StateID]
    data_needed: List[str] = field(default_factory=list)
    data_produced: List[str] = field(default_factory=list)
    timeout_sec: float = 120.0
    retry_on_failure: int = 0


# ============================================================
# 完整的任务状态表
# ============================================================

TASK1_STATES: Dict[Task1StateID, StateDefinition] = {
    # ── 起点 ──
    Task1StateID.IDLE: StateDefinition(
        state_id=Task1StateID.IDLE,
        index=0,
        description="空闲，等待任务启动指令",
        next_state=Task1StateID.WAIT_FOR_DOORBELL_1,
    ),

    # ── [1] 等待门铃 1 ──
    Task1StateID.WAIT_FOR_DOORBELL_1: StateDefinition(
        state_id=Task1StateID.WAIT_FOR_DOORBELL_1,
        index=1,
        description="等待门铃声 (guest1 到达门口)",
        next_state=Task1StateID.GO_TO_DOOR,
        data_produced=["doorbell_1_rang"],
        timeout_sec=300.0,
    ),

    # ── [2] 去门口 ──
    Task1StateID.GO_TO_DOOR: StateDefinition(
        state_id=Task1StateID.GO_TO_DOOR,
        index=2,
        description="机器人移动到门口接客位置",
        next_state=Task1StateID.ASK_GUEST1_INFO,
        data_produced=["robot_at_door"],
        timeout_sec=90.0,
    ),

    # ── [3] 询问 guest1 ──
    Task1StateID.ASK_GUEST1_INFO: StateDefinition(
        state_id=Task1StateID.ASK_GUEST1_INFO,
        index=3,
        description="询问第一位客人的姓名和想喝的饮料",
        next_state=Task1StateID.GUIDE_GUEST1,
        data_produced=["guest1_name", "guest1_drink"],
        timeout_sec=180.0,
        retry_on_failure=2,
    ),

    # ── [4] 带 guest1 到客厅 ──
    Task1StateID.GUIDE_GUEST1: StateDefinition(
        state_id=Task1StateID.GUIDE_GUEST1,
        index=4,
        description="带领 guest1 前往客厅",
        next_state=Task1StateID.POINT_EMPTY_SEAT,
        data_needed=["guest1_name"],
        timeout_sec=120.0,
    ),

    # ── [5] 指向空座 ──
    Task1StateID.POINT_EMPTY_SEAT: StateDefinition(
        state_id=Task1StateID.POINT_EMPTY_SEAT,
        index=5,
        description="指向预设空座位，请 guest1 入座",
        next_state=Task1StateID.RETURN_TO_START,
        data_needed=["guest1_name"],
        timeout_sec=60.0,
    ),

    # ── [6] 返回起点 ──
    Task1StateID.RETURN_TO_START: StateDefinition(
        state_id=Task1StateID.RETURN_TO_START,
        index=6,
        description="返回起点位置，准备接第二位客人",
        next_state=Task1StateID.WAIT_FOR_DOORBELL_2,
        timeout_sec=90.0,
    ),

    # ── [7] 等待门铃 2 ──
    Task1StateID.WAIT_FOR_DOORBELL_2: StateDefinition(
        state_id=Task1StateID.WAIT_FOR_DOORBELL_2,
        index=7,
        description="等待门铃声 (guest2 到达门口)",
        next_state=Task1StateID.PICK_UP_GUEST2,
        data_produced=["doorbell_2_rang"],
        timeout_sec=300.0,
    ),

    # ── [8] 接 guest2 ──
    Task1StateID.PICK_UP_GUEST2: StateDefinition(
        state_id=Task1StateID.PICK_UP_GUEST2,
        index=8,
        description="到门口迎接第二位客人，询问姓名",
        next_state=Task1StateID.DESCRIBE_GUEST1,
        data_produced=["guest2_at_door", "guest2_name"],
        timeout_sec=120.0,
    ),

    # ── [9] 描述 guest1 ──
    Task1StateID.DESCRIBE_GUEST1: StateDefinition(
        state_id=Task1StateID.DESCRIBE_GUEST1,
        index=9,
        description="向 guest2 描述 guest1 的外貌特征（衣服颜色、位置等）",
        next_state=Task1StateID.SEAT_GUEST2,
        data_needed=["guest1_name"],
        timeout_sec=90.0,
    ),

    # ── [10] 带 guest2 入座 ──
    Task1StateID.SEAT_GUEST2: StateDefinition(
        state_id=Task1StateID.SEAT_GUEST2,
        index=10,
        description="带领 guest2 到客厅，询问饮料偏好，安排入座",
        next_state=Task1StateID.INTRODUCE_GUESTS,
        data_needed=["guest2_name"],
        data_produced=["guest2_seated", "guest2_drink", "seat_number"],
        timeout_sec=120.0,
    ),

    # ── [11] 介绍两位客人 ──
    Task1StateID.INTRODUCE_GUESTS: StateDefinition(
        state_id=Task1StateID.INTRODUCE_GUESTS,
        index=11,
        description="向两位客人相互介绍彼此的姓名和饮料偏好",
        next_state=Task1StateID.REQUEST_GUEST2_BAG,
        data_needed=["guest1_name", "guest1_drink", "guest2_name", "guest2_drink"],
        timeout_sec=90.0,
    ),

    # ── [12] 请求 guest2 放包 ──
    Task1StateID.REQUEST_GUEST2_BAG: StateDefinition(
        state_id=Task1StateID.REQUEST_GUEST2_BAG,
        index=12,
        description="请求 guest2 将随身包放到机器人托盘上",
        next_state=Task1StateID.FIND_HOST,
        data_needed=["guest2_name"],
        data_produced=["bag_on_tray"],
        timeout_sec=120.0,
        retry_on_failure=1,
    ),

    # ── [13] 找 host ──
    Task1StateID.FIND_HOST: StateDefinition(
        state_id=Task1StateID.FIND_HOST,
        index=13,
        description="在环境中寻找 host",
        next_state=Task1StateID.FOLLOW_HOST,
        data_produced=["host_found", "host_location"],
        timeout_sec=180.0,
        retry_on_failure=2,
    ),

    # ── [14] 跟随 host ──
    Task1StateID.FOLLOW_HOST: StateDefinition(
        state_id=Task1StateID.FOLLOW_HOST,
        index=14,
        description="跟随 host 走到指定位置",
        next_state=Task1StateID.PLACE_BAG,
        data_needed=["host_found"],
        timeout_sec=120.0,
    ),

    # ── [15] 放包 ──
    Task1StateID.PLACE_BAG: StateDefinition(
        state_id=Task1StateID.PLACE_BAG,
        index=15,
        description="将托盘上的包放到 host 指定的位置",
        next_state=Task1StateID.TASK_COMPLETE,
        data_needed=["bag_on_tray"],
        data_produced=["bag_placed"],
        timeout_sec=60.0,
    ),

    # ── [16] 完成 ──
    Task1StateID.TASK_COMPLETE: StateDefinition(
        state_id=Task1StateID.TASK_COMPLETE,
        index=16,
        description="任务完成，播报结束语，回到 IDLE",
        next_state=Task1StateID.IDLE,
        timeout_sec=30.0,
    ),

    # ── 异常 ──
    Task1StateID.ERROR: StateDefinition(
        state_id=Task1StateID.ERROR,
        index=-1,
        description="发生错误，等待人工干预或重试",
        next_state=None,
    ),

    Task1StateID.ABORTED: StateDefinition(
        state_id=Task1StateID.ABORTED,
        index=-2,
        description="任务被取消，回到 IDLE",
        next_state=Task1StateID.IDLE,
    ),
}

# ============================================================
# ROS 话题命名规范
# ============================================================
# 每个执行状态有自己专属的一对 topic：
#
#   启动指令:  /task1/{state_name}/start   (总控 → 子模块)
#   完成报告:  /task1/{state_name}/done    (子模块 → 总控)
#
# 全局控制 topic (总控专属)：
#   /task1/start   — 外部触发整个任务
#   /task1/abort   — 外部取消任务
#   /task1/status  — 广播当前进度 (plain text)
#   /task1/result  — 广播最终结果 (JSON)

# ============================================================
# 消息格式定义
# ============================================================

# --- /task1/{state}/start 消息格式 ---
# 总控 → 子模块：告知"现在请执行此状态"
# {
#   "seq": 3,
#   "context": {
#     "robot_position": "home",
#     "guest1_name": "Alice"
#   }
# }

# --- /task1/{state}/done 消息格式 ---
# 子模块 → 总控：告知"当前状态已完成"
# {
#   "seq": 3,
#   "result": "success",     # success | failed | timeout | skipped
#   "data": {
#     "robot_at_door": true
#   },
#   "message": "已到达门口"
# }

# --- /task1/status 消息 ---
# 总控 → 外部：广播当前任务状态（plain string）
# "当前状态: [2/14] ask_guest1_info — 正在询问 guest1 姓名和饮料"

# --- /task1/start 消息 ---
# 外部 → 总控：触发任务启动
# 可以是 std_msgs/Empty 或 std_msgs/String("start")

# --- /task1/result 消息 ---
# 总控 → 外部：任务最终结果
# {
#   "result": "success",
#   "states_completed": 14,
#   "total_duration_sec": 245.3,
#   "guest1_name": "Alice",
#   "guest2_name": "Bob",
#   "error": null
# }


# ============================================================
# ROS Topic 名称生成
# ============================================================

def get_start_topic(state_id: Task1StateID) -> str:
    """每个状态的启动指令 topic: /task1/{state}/start"""
    return f"/task1/{state_id.value}/start"


def get_done_topic(state_id: Task1StateID) -> str:
    """每个状态的完成报告 topic: /task1/{state}/done"""
    return f"/task1/{state_id.value}/done"


# ============================================================
# 辅助函数
# ============================================================

def get_state(state_id: Task1StateID) -> StateDefinition:
    """根据 ID 获取状态定义"""
    return TASK1_STATES[state_id]


def get_state_chain() -> List[Task1StateID]:
    """返回执行状态的顺序列表 (不含 IDLE/ERROR/ABORTED)"""
    chain = []
    current = TASK1_STATES[Task1StateID.IDLE]
    while current.next_state is not None:
        next_sid = current.next_state
        if TASK1_STATES[next_sid].index <= 0:  # 回到 IDLE/ERROR/ABORTED, 停止
            break
        chain.append(next_sid)
        current = TASK1_STATES[next_sid]
    return chain


def get_execution_states() -> List[StateDefinition]:
    """返回所有执行状态的 StateDefinition 列表 (按流程顺序)"""
    states = []
    for sid in get_state_chain():
        s = TASK1_STATES[sid]
        if s.index > 0:  # 排除 IDLE
            states.append(s)
    return states


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=== Task1 状态机定义 ===\n")
    print(f"总状态数: {len(TASK1_STATES)}")

    print("\n执行流程 + ROS Topic:")
    for i, sid in enumerate(get_state_chain()):
        s = TASK1_STATES[sid]
        arrow = " │ " if i < len(get_state_chain()) - 1 else " ▼ "
        print(f"  [{s.index:2d}] {s.state_id.value:25s} → {s.description}")
        print(f"       start: {get_start_topic(sid)}")
        print(f"       done:  {get_done_topic(sid)}")
        if s.data_needed:
            print(f"       需要: {s.data_needed}")
        if s.data_produced:
            print(f"       产出: {s.data_produced}")

    print(f"\n执行状态数: {len(get_execution_states())}")
    print("✓ 状态定义测试通过")
