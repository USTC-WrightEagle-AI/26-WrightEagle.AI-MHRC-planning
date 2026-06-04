#!/home/nvidia/.conda/envs/task3/bin/python3
"""
Brain Node - CADE 大脑主启动脚本

启动方式：
    rosrun cade_brain brain_node.py
或
    python brain_node.py

功能：
    初始化 ROS 节点、LLM 客户端、控制器，
    监听 /asr 话题并处理语音输入，发布回复到 /tts。

注意：
    技能函数通过 @register_nav_tool / @register_vision_tool 装饰器
    在模块导入时自动注册到工具箱字典，Controller 直接使用字典分发。
"""

import argparse
import json
import sys
import threading

try:
    import rospy
    from std_msgs.msg import String

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available, running in standalone mode")

# ★ 导入 skills 包以触发装饰器注册（nav_skills / vision_skills 被自动加载）
import cade_brain.skills.nav_skills  # noqa: F401 — 触发 @register_nav_tool
import cade_brain.skills.vision_skills  # noqa: F401 — 触发 @register_vision_tool
from cade_brain.controller import RobotController


def print_banner(config_mode, model, robot_name):
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   CADE - Cognitive Agent for Domestic Environment        ║
║   Architecture: Function-as-Tool (Refactored v2)         ║
║                                                           ║
║   Project: Project LARA                                  ║
║   Version: 0.3.0 (Modular Tools)                         ║
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
    2. 调用 Controller 进行 LLM 推理
    3. 发布回复到 /tts
    """

    def __init__(
        self,
        prompt_mode: str = "default",
        show_thought: bool = True,
        environment_context: str = "",
    ):
        if ROS_AVAILABLE:
            rospy.init_node("cade_brain", anonymous=True)

        # 创建控制器（工具箱字典由装饰器自动填充，无需手动传入技能实例）
        self.controller = RobotController(
            prompt_mode=prompt_mode,
            show_thought=show_thought,
            environment_context=environment_context,
        )

        # 状态锁
        self._state_lock = threading.Lock()

        if ROS_AVAILABLE:
            # ROS 通信
            self.tts_publisher = rospy.Publisher("/tts", String, queue_size=10)
            self.asr_subscriber = rospy.Subscriber(
                "/asr", String, self._on_asr_message, queue_size=10
            )

            rospy.loginfo("=" * 60)
            rospy.loginfo("Brain Node initialized")
            rospy.loginfo(f"  Subscribing: /asr")
            rospy.loginfo(f"  Publishing: /tts")
            rospy.loginfo(f"  Tools: {len(self.controller.skills_registry)} registered")
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
        print(f"\n{'=' * 60}")
        print(f'[ASR] Received: "{text}"')
        print(f"{'=' * 60}")

        # 异步处理
        thread = threading.Thread(
            target=self._process_input_async, args=(text,), daemon=True
        )
        thread.start()

    def _process_input_async(self, text: str):
        """异步处理用户输入"""
        try:
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

            self._publish_tts("Sorry, I encountered a problem.")

    def _publish_tts(self, text: str):
        """发布 TTS 文本"""
        if not ROS_AVAILABLE:
            print(f"[TTS] (no ROS): {text}")
            return

        rospy.loginfo(f'[TTS] Publishing: "{text}"')
        msg = String()
        msg.data = text
        self.tts_publisher.publish(msg)

        # 粗略估计语音播放时长
        import time

        estimated_duration = max(1.0, len(text) * 0.1)
        time.sleep(estimated_duration)

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
                    if text.lower() in ("quit", "exit", "q"):
                        break
                    self._process_input_async(text)
                except KeyboardInterrupt:
                    break

    def print_statistics(self):
        print(f"\n{'=' * 60}")
        print("Statistics:")
        print(f"  Total inputs: {self.total_inputs}")
        print(f"  Ignored: {self.ignored_inputs}")
        print(f"  Successful replies: {self.successful_replies}")
        if self.total_inputs > 0:
            success_rate = (self.successful_replies / self.total_inputs) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        print(f"{'=' * 60}")
        self.controller.print_statistics()


# ==================== Main ====================


def main():
    from cade_brain.llm_core.config import Config

    parser = argparse.ArgumentParser(description="CADE Brain Node")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "simple", "compact", "debug"],
        default="default",
        help="Prompt mode",
    )
    parser.add_argument(
        "--no-thought", action="store_true", help="Hide LLM thought process"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="You are sitting in a Fedora lab, communicating via voice.",
        help="Environment context",
    )

    args = parser.parse_args()

    print_banner(
        "Cloud" if Config.is_cloud_mode() else "Local",
        Config.get_llm_config()["model"],
        Config.ROBOT_NAME,
    )

    node = BrainNode(
        prompt_mode=args.mode,
        show_thought=not args.no_thought,
        environment_context=args.env,
    )

    print("\nCADE Brain is ready, waiting for voice input...\n")
    print("Tips:")
    print("  - Speak into the microphone (ASR->/asr)")
    print("  - Press Ctrl+C to exit")
    print()
    print(f"\n🚀 {'=' * 25} 最终组装的 SYSTEM PROMPT {'=' * 25}")
    # 此时 controller.system_prompt 已经是静态身份 + 动态 Pydantic Tools 的合体了
    print(node.controller.system_prompt)
    print(f"{'=' * 75}\n")

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
