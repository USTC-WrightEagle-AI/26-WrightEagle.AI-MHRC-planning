"""
测试文本集：通用日常对话 + RoboCup@Home 指令混合（英文）
用于TTS生成音频，再用于ASR模型评估
"""

# 通用日常对话句子 (8条)
GENERAL_SENTENCES = [
    "Hello, how are you doing today? I hope everything is going well for you.",
    "The weather is really nice today, perfect for a walk outside.",
    "I had a great lunch with my friends at the new restaurant downtown.",
    "Could you please help me find my keys? I think I left them on the table.",
    "The meeting has been rescheduled to three o'clock tomorrow afternoon.",
    "I am learning about artificial intelligence and machine learning these days.",
    "My favorite color is blue, but I also like green and yellow quite a lot.",
    "She went to the supermarket to buy some milk, bread, and fresh fruit.",
]

# RoboCup@Home 场景指令 (8条)
ROBOCUP_COMMANDS = [
    "Bring me a coke from the kitchen.",
    "Go to the living room and wait there.",
    "Point to the person wearing a red shirt.",
    "Follow me to the bedroom please.",
    "What objects do you see on the dining table?",
    "Pick up the green cup and put it on the shelf.",
    "Tell me the name of this object you are holding.",
    "Navigate to the door and open it for me.",
]

# 所有测试文本
ALL_TEST_TEXTS = GENERAL_SENTENCES + ROBOCUP_COMMANDS


def get_test_texts():
    """返回所有测试文本列表"""
    return ALL_TEST_TEXTS


def get_general_sentences():
    """返回通用对话句子"""
    return GENERAL_SENTENCES


def get_robocup_commands():
    """返回RoboCup指令"""
    return ROBOCUP_COMMANDS
