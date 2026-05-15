"""
Task1 — Receptionist 接待任务模块

RoboCup@Home 任务: 机器人接待两位客人，引导入座、介绍、拿包、跟随host放包。

模块结构:
    task1_receptionist/
    ├── state_definitions.py  — 状态定义、消息格式、流转表
    ├── task1_controller.py   — 总控执行器 (纯 Python, 直接运行)
    ├── sub_modules/          — 各子模块基类和骨架实现
    │   ├── base_module.py    — BaseSubModule 抽象类 + 示例骨架
    │   └── __init__.py

使用:
    python task1_receptionist/task1_controller.py
    python task1_receptionist/task1_controller.py --delay 0.5
"""

from task1_receptionist.state_definitions import (
    Task1StateID,
    StateDefinition,
    TASK1_STATES,
    get_state,
    get_state_chain,
    get_execution_states,
    get_start_topic,
    get_done_topic,
)

__all__ = [
    "Task1StateID",
    "StateDefinition",
    "TASK1_STATES",
    "get_state",
    "get_state_chain",
    "get_execution_states",
    "get_start_topic",
    "get_done_topic",
]
