"""
数据模型定义 - Data Schemas

使用 Pydantic 严格定义机器人的动作空间和决策输出格式
"""

from typing import Literal, Union, Optional, List
from pydantic import BaseModel, Field


class Position3D(BaseModel):
    """3D position in the robot/world coordinate frame."""
    x: float = Field(..., description="X 坐标")
    y: float = Field(..., description="Y 坐标")
    z: float = Field(..., description="Z 坐标")


# ==================== 状态机动作类型 (基于你的15+状态定义) ====================

# ==================== 1. 监听状态 ====================

class ListeningAction(BaseModel):
    """
    监听状态 - 持续监听音频流
    
    功能：实时语音活动检测(VAD)、接收音频信号、将音频传输给LLM
    输出：唤醒信号/触发信号，启动处理流程
    """
    type: Literal["listening"] = "listening"
    continuous: bool = Field(default=True, description="是否持续监听")
    vad_enabled: bool = Field(default=True, description="是否启用语音活动检测")
    wake_word: Optional[str] = Field(None, description="唤醒词（可选）")


# ==================== 2. 对话状态 ====================

class TalkingAction(BaseModel):
    """
    对话状态 - 响应用户对话
    
    输入：人类语音
    功能：响应用户对话
    输出：机器人语音回复
    """
    type: Literal["talking"] = "talking"
    text: str = Field(..., description="要说的文本内容")
    wait_for_response: bool = Field(default=False, description="是否等待用户响应")


# ==================== 3. 人名识别状态 ====================

class NameRecognitionAction(BaseModel):
    """
    人名识别状态 - 通过姓名识别人物
    
    功能：识别特定名字的人物，并将名字与视觉外观绑定
    输出：人物ID、位置
    """
    type: Literal["name_recognition"] = "name_recognition"
    name: str = Field(..., description="要识别的人物名称")
    bind_to_appearance: bool = Field(default=True, description="是否将名字与视觉外观绑定")


# ==================== 4. 导航状态 ====================

class NavigationAction(BaseModel):
    """
    导航状态 - 驱动机器人到指定位置
    
    输入：地图上的3D位置
    功能：驱动机器人到目标位置
    输出：到达确认
    """
    type: Literal["navigation"] = "navigation"
    position: str = Field(..., description="目标位置（语义标签或坐标）")



# ==================== 5. 姿态/手势识别状态 ====================

class GestureRecognitionAction(BaseModel):
    """
    姿态/手势识别状态 - 识别特定姿态/手势的人
    
    功能：观察并识别具有特定姿态/手势的人物
    输出：人物位置（3D坐标）
    """
    type: Literal["gesture_recognition"] = "gesture_recognition"
    gesture: str = Field(..., description="要识别的姿态/手势（如 waving, sitting, standing, pointing）")
    room: Optional[str] = Field(None, description="限定搜索的房间（可选）")
    return_position: bool = Field(default=True, description="是否返回人物位置")


# ==================== 6. 人物追踪状态 ====================

class PersonTrackingAction(BaseModel):
    """
    人物追踪状态 - 持续追踪特定人物
    
    功能：保持对人物的追踪，持续输出实时3D位置
    输出：人物的实时3D位置流
    """
    type: Literal["person_tracking"] = "person_tracking"
    person_id: Optional[str] = Field(None, description="人物ID（可选，如不指定则追踪最近发现的人）")
    person_pos: Optional[str] = Field(None, description="人物位置（可选）")
    duration: Optional[float] = Field(None, description="追踪持续时间（秒，None表示持续直到中断）")
    continuous_output: bool = Field(default=True, description="是否持续输出位置")


# ==================== 7. 衣服识别状态 ====================

class ClothesRecognitionAction(BaseModel):
    """
    衣服识别状态 - 通过衣服颜色识别人物
    
    功能：观察并通过衣服颜色识别特定人物
    输出：人物位置（3D坐标）
    """
    type: Literal["clothes_recognition"] = "clothes_recognition"
    cloth_color: str = Field(..., description="衣服颜色（如 red, blue, green, yellow, black, white）")
    room: Optional[str] = Field(None, description="限定搜索的房间（可选）")
    return_position: bool = Field(default=True, description="是否返回人物位置")


# ==================== 8. 物体搜索状态 ====================

class ObjectSearchAction(BaseModel):
    """
    物体搜索状态 - 按名称搜索物体
    
    功能：通过物体名称寻找物体
    输出：物体的实时3D位置
    """
    type: Literal["object_search"] = "object_search"
    object_name: str = Field(..., description="要搜索的物体名称")
    continuous_output: bool = Field(default=False, description="是否持续输出位置（追踪移动物体）")


# ==================== 9. 物体抓取状态 ====================

class ObjectGraspAction(BaseModel):
    """
    物体抓取状态 - 抓取指定物体
    
    功能：通过物体名称抓取物体
    输入：物体名称
    输出：抓取成功/失败状态
    """
    type: Literal["object_grasp"] = "object_grasp"
    object_name: str = Field(..., description="要抓取的物体名称")
    object_position: Optional[Union[Position3D, List[float]]] = Field(
        None, 
        description="物体位置（如果不提供，则先搜索）"
    )
    grasp_force: Optional[float] = Field(None, description="抓取力度（0-1，可选）")


# ==================== 10. 物体放置状态 ====================

class ObjectDumpAction(BaseModel):
    """
    物体放置状态 - 释放当前抓取的物体
    
    功能：机器人释放正在抓取的物体
    输入：3D放置位置
    输出：放置完成确认
    """
    type: Literal["object_dump"] = "object_dump"
    target_position: Union[str, Position3D, List[float]] = Field(
        ..., 
        description="放置位置（语义标签如'table'，或3D坐标[x,y,z]）"
    )
    release_safe: bool = Field(default=True, description="是否安全释放（先检测下方是否有支撑面）")


# ==================== 11. 姿态/手势计数状态 ====================

class GestureCountingAction(BaseModel):
    """
    姿态/手势计数状态 - 统计特定姿态/手势的人数
    
    功能：观察并统计具有特定姿态/手势的人数
    输出：总人数
    """
    type: Literal["gesture_counting"] = "gesture_counting"
    gesture: str = Field(..., description="要计数的姿态/手势（如 sitting, standing, raising_hand）")
    room: str = Field(..., description="要统计的房间名称")
    return_count_only: bool = Field(default=True, description="是否只返回计数（而非位置列表）")


# ==================== 12. 衣服计数状态 ====================

class ClothesCountingAction(BaseModel):
    """
    衣服计数状态 - 统计穿特定颜色衣服的人数
    
    功能：观察并通过衣服颜色统计人数
    输出：总人数
    """
    type: Literal["clothes_counting"] = "clothes_counting"
    cloth_color: str = Field(..., description="衣服颜色")
    room: str = Field(..., description="要统计的房间名称")
    return_count_only: bool = Field(default=True, description="是否只返回计数")


# ==================== 辅助动作（扩展） ====================

class WaitAction(BaseModel):
    """等待状态 - 等待指定时间或事件"""
    type: Literal["wait"] = "wait"
    duration: float = Field(..., description="等待时间（秒）")
    interruptible: bool = Field(default=True, description="是否可被中断")


class IdleAction(BaseModel):
    """空闲状态 - 机器人待机，等待指令"""
    type: Literal["idle"] = "idle"
    low_power_mode: bool = Field(default=False, description="是否进入低功耗模式")
    summary: str = Field(..., description="任务总结")

# ==================== atom动作类型 ====================

RobotAction = Union[
    ListeningAction,           # 1. 监听状态
    TalkingAction,             # 2. 对话状态
    NameRecognitionAction,     # 3. 人名识别状态
    NavigationAction,          # 4. 导航状态
    GestureRecognitionAction,  # 5. 姿态/手势识别状态
    PersonTrackingAction,      # 6. 人物追踪状态
    ClothesRecognitionAction,  # 7. 衣服识别状态
    ObjectSearchAction,        # 8. 物体搜索状态
    ObjectGraspAction,         # 9. 物体抓取状态
    ObjectDumpAction,          # 10. 物体放置状态
    GestureCountingAction,     # 11. 姿态/手势计数状态
    ClothesCountingAction,     # 12. 衣服计数状态
    WaitAction,                # 等待状态（扩展）
    IdleAction,                # 空闲状态（扩展）
]

# ==================== 决策输出格式 ====================

class RobotDecision(BaseModel):
    """
    机器人决策输出格式（LLM的输出结构）
    
    LLM 需要将复杂任务分解为原子动作的序列（SequentialAction）
    """
    thought: Optional[str] = Field(
        None,
        description="内部思考过程（CoT - Chain of Thought）- 解释如何分解任务"
    )
    reply: Optional[str] = Field(
        None,
        description="给用户的自然语言回复（可选，如'好的，我正在处理'）"
    )
    action: Optional[RobotAction] = Field(
        None,
        description="要执行的顺序动作序列（多个原子动作按顺序执行）"
    )

    class Config:
        arbitrary_types_allowed = True


# ==================== 辅助函数 ====================

def parse_action(action_dict: dict) -> RobotAction:
    """
    根据 type 字段解析动作
    """
    action_dict = dict(action_dict)
    action_type = action_dict.get("type")

    type_aliases = {
        "ListeningAction": "listening",
        "TalkingAction": "talking",
        "NameRecognitionAction": "name_recognition",
        "NavigationAction": "navigation",
        "GestureRecognitionAction": "gesture_recognition",
        "PersonTrackingAction": "person_tracking",
        "ClothesRecognitionAction": "clothes_recognition",
        "ObjectSearchAction": "object_search",
        "ObjectGraspAction": "object_grasp",
        "ObjectDumpAction": "object_dump",
        "GestureCountingAction": "gesture_counting",
        "ClothesCountingAction": "clothes_counting",
        "TaskCompleteAction": "idle",
        "IdleAction": "idle",
    }
    action_type = type_aliases.get(action_type, action_type)
    action_dict["type"] = action_type

    field_aliases = {
        "navigation": {"target": "position"},
        "name_recognition": {"person_name": "name"},
        "gesture_recognition": {"gesture_type": "gesture"},
        "gesture_counting": {"gesture_type": "gesture"},
        "person_tracking": {"target_person": "person_id", "person_position": "person_pos"},
        "object_grasp": {"placement": "object_position"},
        "object_dump": {"position": "target_position"},
        "idle": {"TaskCompleteAction": "idle"},
    }
    for old_key, new_key in field_aliases.get(action_type, {}).items():
        if old_key in action_dict and new_key not in action_dict:
            action_dict[new_key] = action_dict.pop(old_key)

    action_map = {
        # 基础状态
        "listening": ListeningAction,
        "talking": TalkingAction,
        "name_recognition": NameRecognitionAction,
        "navigation": NavigationAction,
        "gesture_recognition": GestureRecognitionAction,
        "person_tracking": PersonTrackingAction,
        "clothes_recognition": ClothesRecognitionAction,
        "object_search": ObjectSearchAction,
        "object_grasp": ObjectGraspAction,
        "object_dump": ObjectDumpAction,
        "gesture_counting": GestureCountingAction,
        "clothes_counting": ClothesCountingAction,
        # 扩展
        "wait": WaitAction,
        "idle": IdleAction,
    }

    if action_type not in action_map:
        raise ValueError(
            f"未知的动作类型: {action_type}. "
            f"可用动作: {list(action_map.keys())}"
        )

    return action_map[action_type](**action_dict)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 状态机动作定义测试 ===\n")

    # 1. 监听状态
    listening = ListeningAction()
    print(f"1. 监听状态: {listening.model_dump_json(indent=2)}\n")

    # 2. 对话状态
    talking = TalkingAction(text="你好，我是机器人助手")
    print(f"2. 对话状态: {talking.model_dump_json(indent=2)}\n")

    # 3. 人名识别
    name_rec = NameRecognitionAction(name="张三")
    print(f"3. 人名识别: {name_rec.model_dump_json(indent=2)}\n")

    # 4. 导航状态（坐标）
    nav = NavigationAction(position="[1.5, 2.3, 0.0]")
    print(f"4. 导航（坐标）: {nav.model_dump_json(indent=2)}\n")

    # 5. 导航状态（语义）
    nav_semantic = NavigationAction(position="kitchen")
    print(f"5. 导航（语义）: {nav_semantic.model_dump_json(indent=2)}\n")

    # 6. 姿态识别
    gesture_rec = GestureRecognitionAction(gesture="waving", room="living_room")
    print(f"6. 姿态识别: {gesture_rec.model_dump_json(indent=2)}\n")

    # 7. 衣服识别
    clothes_rec = ClothesRecognitionAction(cloth_color="red")
    print(f"7. 衣服识别: {clothes_rec.model_dump_json(indent=2)}\n")

    # 8. 物体搜索
    obj_search = ObjectSearchAction(object_name="apple")
    print(f"8. 物体搜索: {obj_search.model_dump_json(indent=2)}\n")

    # 9. 物体抓取
    obj_grasp = ObjectGraspAction(object_name="apple")
    print(f"9. 物体抓取: {obj_grasp.model_dump_json(indent=2)}\n")

    # 10. 物体放置
    obj_dump = ObjectDumpAction(target_position="table")
    print(f"10. 物体放置: {obj_dump.model_dump_json(indent=2)}\n")

    # 15. 姿态计数
    gesture_count = GestureCountingAction(gesture="sitting", room="classroom")
    print(f"11. 姿态计数（15）: {gesture_count.model_dump_json(indent=2)}\n")

    # 17. 衣服计数
    clothes_count = ClothesCountingAction(cloth_color="blue", room="office")
    print(f"12. 衣服计数（17）: {clothes_count.model_dump_json(indent=2)}\n")

    # 完整决策示例
    decision = RobotDecision(
        thought="用户想要红色的苹果，我需要先找到并抓取它",
        reply="好的，我正在为您寻找红色的苹果",
        action=ObjectSearchAction(object_name="apple")
    )
    print(f"13. 完整决策: {decision.model_dump_json(indent=2)}\n")

    print("✓ 所有测试通过！")
