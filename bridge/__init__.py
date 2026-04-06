"""
Bridge Module - ROS 桥接模块

提供 ROS 话题与 CADE 控制器之间的桥接功能
"""

from .ros_voice_bridge import RosVoiceBridge

__all__ = ['RosVoiceBridge']
