"""
Task1 ROS 话题名称和指令常量

所有模块与外部 ROS 节点通信的话题名称和指令字符串集中定义在此,
方便统一管理和修改。对接外部模块时只需修改此文件。

结果消息格式约定 (std_msgs/String, JSON):
    成功: {"status": "success", "data": {...}}
    失败: {"status": "failed",  "error": "error message"}

data 字段为可选, 具体内容因模块而异。
如果外部节点返回非 JSON 字符串 (如 "done"), 将被解析为:
    {"status": "success", "raw": "done"}
"""

# ============================================================
# 门铃
# ============================================================
DOORBELL_TOPIC = "/doorbell/detected"
DOORBELL_TIMEOUT = 300.0

# ============================================================
# 导航
# ============================================================
NAV_COMMAND_TOPIC = "/navigation/command"
NAV_RESULT_TOPIC = "/navigation/result"
NAV_TIMEOUT = 120.0

NAV_CMD_GO_TO_DOOR = "导航到门口"
NAV_CMD_GO_TO_LIVING_ROOM = "导航到客厅"
NAV_CMD_GO_TO_START = "导航到起点"
NAV_CMD_FOLLOW_PERSON = "跟随人物"
NAV_CMD_TURN_TO_ANGLE = "转向角度"

# ============================================================
# 视觉
# ============================================================
VISION_COMMAND_TOPIC = "/vision/command"
VISION_RESULT_TOPIC = "/vision/result"
VISION_TIMEOUT = 120.0

VISION_CMD_DESCRIBE_PERSON = "识别人物外貌"
VISION_CMD_FIND_HOST = "寻找host"
VISION_CMD_TRACK_PERSON = "跟踪人物"

# ============================================================
# 操作 (手臂/夹爪/托盘)
# ============================================================
MANIP_COMMAND_TOPIC = "/manipulation/command"
MANIP_RESULT_TOPIC = "/manipulation/result"
MANIP_TIMEOUT = 60.0

MANIP_CMD_POINT_SEAT = "指向空座"
MANIP_CMD_WAIT_FOR_BAG = "等待放包"
MANIP_CMD_PLACE_BAG = "放包到指定位置"

# ============================================================
# 说话人识别 + 声源定位
# ============================================================
SPEAKER_DOA_ENROLL_TOPIC = "/speaker_doa/enroll"
SPEAKER_DOA_COMMAND_TOPIC = "/speaker_doa/command"
SPEAKER_DOA_RESULT_TOPIC = "/speaker_doa/result"
SPEAKER_DOA_TIMEOUT = 60.0
SPEAKER_DOA_FACE_TIMEOUT = 10.0

SPEAKER_DOA_CMD_LOCATE = "声源定位"
SPEAKER_DOA_CMD_TRACK = "声源跟踪"
SPEAKER_DOA_CMD_FACE_SPEAKER = "面向说话人"

# ============================================================
# 语音 (TTS + ASR)
# ============================================================
TTS_TOPIC = "/tts"
ASR_TOPIC = "/asr"
TTS_PLAYING_TOPIC = "/tts/playing"

# ============================================================
# LLM
# ============================================================
LLM_REQUEST_TOPIC = "/llm/request"
LLM_RESPONSE_TOPIC = "/llm/response"

# ============================================================
# Task1 总控
# ============================================================
TASK1_START_TOPIC = "/task1/start"
TASK1_ABORT_TOPIC = "/task1/abort"
TASK1_STATUS_TOPIC = "/task1/status"
TASK1_RESULT_TOPIC = "/task1/result"
