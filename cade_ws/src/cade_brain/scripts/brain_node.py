#!/usr/bin/env python3
"""
Brain Node - CADE 大脑主启动脚本

启动方式：
    rosrun cade_brain brain_node.py
或
    python brain_node.py

功能：
    初始化 ROS 节点、LLM 客户端、控制器，
    监听 /asr 话题并处理语音输入，发布回复到 /tts。
"""

import sys
import argparse
import threading
import json

try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available, running in standalone mode")


from cade_brain.controller import RobotController, RobotState


def print_banner(config_mode, model, robot_name):
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   CADE - Cognitive Agent for Domestic Environment        ║
║   Architecture: Modular ROS Pub/Sub (Refactored)          ║
║                                                           ║
║   Project: Project LARA                                  ║
║   Version: 0.2.0 (Refactored)                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"Mode: {config_mode}")
    print(f"Model: {model}")
    print(f"Robot: {robot_name}")
    print(f"Input: ROS /asr topic")
    print(f"Output: ROS /tts topic")
    print(f"Task Cmd: ROS /cade/task_cmd")
    print(f"Task Status: ROS /cade/task_status")
    print(f"Vision Detections: ROS /vision/detections_3d")
    print()


class BrainNode:
    """
    Brain ROS Node - 大脑节点

    职责：
    1. 订阅 /asr 接收语音识别结果
    2. 订阅 /vision/detections_3d 接收视觉检测结果（更新世界模型）
    3. 调用 Controller 进行 LLM 推理
    4. 发布回复到 /tts
    """

    def __init__(
        self,
        prompt_mode: str = "default",
        show_thought: bool = True,
        environment_context: str = ""
    ):
        if ROS_AVAILABLE:
            rospy.init_node('cade_brain', anonymous=True)

        # 创建控制器
        self.controller = RobotController(
            prompt_mode=prompt_mode,
            show_thought=show_thought,
            environment_context=environment_context
        )

        # 状态锁
        self._state_lock = threading.Lock()

        if ROS_AVAILABLE:
            # ROS 通信
            self.tts_publisher = rospy.Publisher('/tts', String, queue_size=10)
            self.asr_subscriber = rospy.Subscriber(
                '/asr', String, self._on_asr_message, queue_size=10
            )
            self.vision_subscriber = rospy.Subscriber(
                '/vision/detections_3d', String, self._on_vision_message, queue_size=10
            )

            rospy.loginfo("=" * 60)
            rospy.loginfo("Brain Node initialized")
            rospy.loginfo(f"  Subscribing: /asr, /vision/detections_3d")
            rospy.loginfo(f"  Publishing: /tts")
            rospy.loginfo("=" * 60)

        # 统计
        self.total_inputs = 0
        self.ignored_inputs = 0
        self.successful_replies = 0

    def _on_asr_message(self, msg: String):
        """ASR 消息回调"""
        text = msg.data.strip()
        if not text:
            return

        self.total_inputs += 1
        print(f"\n{'='*60}")
        print(f"[ASR] Received: \"{text}\"")
        print(f"{'='*60}")

        # 状态拦截（回声抑制）
        with self._state_lock:
            if self.controller.is_busy():
                self.ignored_inputs += 1
                print(f"[IGNORED] Controller busy ({self.controller.state.value})")
                return

        # 异步处理
        thread = threading.Thread(
            target=self._process_input_async,
            args=(text,),
            daemon=True
        )
        thread.start()

    def _on_vision_message(self, msg: String):
        """
        视觉检测结果回调 - 动态更新世界模型

        消息格式：
        {
            "type": "object_detection" | "person_detection",
            "name": "apple",
            "position_3d": [x, y, z],
            "location": "kitchen",
            ...
        }
        """
        try:
            data = json.loads(msg.data)
            det_type = data.get("type", "unknown")

            if det_type == "object_detection":
                name = data.get("name", "unknown_object")
                self.controller.world.add_object(
                    name=name,
                    category=data.get("category"),
                    location=data.get("location"),
                    placement=data.get("placement", data.get("location")),
                    position=data.get("position_3d"),
                )
                print(f"[VISION] Object updated: {name}")

            elif det_type == "person_detection":
                name = data.get("name", "unknown_person")
                self.controller.world.add_person(
                    name=name,
                    cloth_color=data.get("cloth_color"),
                    gesture=data.get("gesture"),
                    location=data.get("location"),
                    beacon=data.get("beacon"),
                    info=data.get("info"),
                )
                print(f"[VISION] Person updated: {name}")

            # 注入观测到对话上下文
            observation = data.get("description", json.dumps(data))
            self.controller.observe(f"Vision: {observation}")

        except json.JSONDecodeError as e:
            print(f"[VISION] Invalid JSON: {e}")

    def _process_input_async(self, text: str):
        """异步处理用户输入"""
        try:
            with self._state_lock:
                self.controller.set_state(RobotState.THINKING)

            print(f"[LLM] Thinking...")

            decision = self.controller.process_input(text)

            reply = decision.reply if decision.reply else "OK, I got it."

            print(f"[REPLY] {reply}")

            self._publish_tts(reply)
            self.successful_replies += 1

            print(f"[DONE] Round complete, waiting for next input...\n")

        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()

            with self._state_lock:
                self.controller.set_state(RobotState.IDLE)

            self._publish_tts("Sorry, I encountered a problem.")

    def _publish_tts(self, text: str):
        """发布 TTS 文本"""
        if not ROS_AVAILABLE:
            print(f"[TTS] (no ROS): {text}")
            return

        with self._state_lock:
            self.controller.set_state(RobotState.SPEAKING)

        rospy.loginfo(f"[TTS] Publishing: \"{text}\"")
        msg = String()
        msg.data = text
        self.tts_publisher.publish(msg)

        # 粗略估计语音播放时长
        import time
        estimated_duration = max(1.0, len(text) * 0.1)
        time.sleep(estimated_duration)

        with self._state_lock:
            self.controller.set_state(RobotState.IDLE)

    def spin(self):
        """进入主循环"""
        if ROS_AVAILABLE:
            rospy.loginfo("Brain node spinning...")
            rospy.spin()
        else:
            # 无 ROS 时的交互模式
            print("Running in interactive mode (no ROS)")
            while True:
                try:
                    text = input("You: ").strip()
                    if not text:
                        continue
                    if text.lower() in ('quit', 'exit', 'q'):
                        break
                    self._process_input_async(text)
                except KeyboardInterrupt:
                    break

    def print_statistics(self):
        print(f"\n{'='*60}")
        print("Statistics:")
        print(f"  Total inputs: {self.total_inputs}")
        print(f"  Ignored: {self.ignored_inputs}")
        print(f"  Successful replies: {self.successful_replies}")
        if self.total_inputs > 0:
            success_rate = (self.successful_replies / self.total_inputs) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        print(f"{'='*60}")
        self.controller.print_statistics()


# ==================== Main ====================

def main():
    from cade_brain.llm_core.config import Config

    parser = argparse.ArgumentParser(description="CADE Brain Node")
    parser.add_argument('--mode', type=str, choices=['default', 'simple', 'compact', 'debug'],
                       default='default', help='Prompt mode')
    parser.add_argument('--no-thought', action='store_true', help='Hide LLM thought process')
    parser.add_argument('--env', type=str,
                       default="You are sitting in a Fedora lab, communicating via voice.",
                       help='Environment context')

    args = parser.parse_args()

    print_banner(
        "Cloud" if Config.is_cloud_mode() else "Local",
        Config.get_llm_config()['model'],
        Config.ROBOT_NAME
    )

    node = BrainNode(
        prompt_mode=args.mode,
        show_thought=not args.no_thought,
        environment_context=args.env
    )

    print("\nCADE Brain is ready, waiting for voice input...\n")
    print("Tips:")
    print("  - Speak into the microphone (ASR->/asr)")
    print("  - Vision detections will auto-update world model")
    print("  - Press Ctrl+C to exit")
    print()

    try:
        node.spin()
    except KeyboardInterrupt:
        print("\n\nProgram exited")
        node.print_statistics()
        sys.exit(0)
    except Exception as e:
        print(f"\nStartup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
