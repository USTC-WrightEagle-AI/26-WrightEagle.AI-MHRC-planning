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
import re
import sys
import threading
import time

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
from cade_brain.skills import VisionHardwareContext


ASR_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
ASR_NOISE_ONLY_PHRASES = {
    "a",
    "ah",
    "an",
    "and",
    "eh",
    "er",
    "hmm",
    "hm",
    "huh",
    "mm",
    "mmm",
    "oh",
    "or",
    "so",
    "the",
    "uh",
    "um",
}


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
    print(f"Vision Cmd: ROS /cade/task_cmd_task3")
    print(f"Vision Status: ROS /cade/task_status_task3")
    print(f"Vision Detections: ROS /vision/detections_3d_task3")
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
        max_react_loops: int = 30,
    ):
        if ROS_AVAILABLE:
            rospy.init_node("cade_brain", anonymous=True)
            max_react_loops = int(rospy.get_param("~max_react_loops", max_react_loops))

        # 创建控制器（工具箱字典由装饰器自动填充，无需手动传入技能实例）
        self.controller = RobotController(
            prompt_mode=prompt_mode,
            show_thought=show_thought,
            environment_context=environment_context,
            max_react_loops=max_react_loops,
            live_reply_callback=self._publish_live_reply,
        )
        if ROS_AVAILABLE:
            VisionHardwareContext().warmup()

        # 状态锁
        self._state_lock = threading.Lock()
        self._worker_active = False
        self._pending_user_inputs = []
        self._tts_busy_until = 0.0
        self._last_live_reply = ""

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
            rospy.loginfo(f"  Max ReAct loops: {self.controller.max_react_loops}")
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

        if self._should_ignore_asr(text):
            self.ignored_inputs += 1
            print(f'[ASR] Ignored: "{text}"')
            return

        with self._state_lock:
            if time.time() < self._tts_busy_until:
                self.ignored_inputs += 1
                print(f'[ASR] Ignored during TTS playback window: "{text}"')
                return

            if self._worker_active:
                self._pending_user_inputs.append(text)
                print(f'[ASR] Buffered while task is running: "{text}"')
                return

            self._worker_active = True

        thread = threading.Thread(
            target=self._input_worker, args=(text,), daemon=True
        )
        thread.start()

    def _input_worker(self, text: str):
        """串行处理 ASR 输入；运行中收到的短句在当前任务后合并处理。"""
        current_text = text
        while current_text:
            self._process_single_input(current_text)
            with self._state_lock:
                if self._pending_user_inputs:
                    current_text = self._merge_pending_inputs_locked()
                    print(f'[ASR] Processing buffered input: "{current_text}"')
                else:
                    self._worker_active = False
                    return

    def _process_single_input(self, text: str):
        """处理单条用户输入"""
        try:
            self._last_live_reply = ""
            print(f"[LLM] Thinking...")

            decision = self.controller.process_input(text)

            reply = decision.reply if decision.reply else "OK, I got it."

            print(f"[REPLY] {reply}")

            if not self._is_duplicate_live_reply(reply):
                self._publish_tts(reply)
            self.successful_replies += 1

            print(f"[DONE] Round complete, waiting for next input...\n")

        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            import traceback

            traceback.print_exc()

            self._publish_tts("Sorry, I encountered a problem.")

    def _merge_pending_inputs_locked(self) -> str:
        merged = " ".join(self._pending_user_inputs)
        self._pending_user_inputs.clear()
        return merged

    def _should_ignore_asr(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return True
        if normalized.startswith("(") or normalized.endswith(")"):
            return True
        tokens = ASR_TOKEN_PATTERN.findall(normalized)
        if len(tokens) == 1 and tokens[0] in ASR_NOISE_ONLY_PHRASES:
            return True
        noise_markers = (
            "loud rumbling",
            "background noise",
            "noise",
            "inaudible",
            "music",
            "static",
        )
        if any(marker in normalized for marker in noise_markers):
            return True
        return False

    def _publish_live_reply(self, text: str):
        """发布执行前的即时计划/状态回复。"""
        self._last_live_reply = (text or "").strip()
        if self._last_live_reply:
            print(f"[LIVE REPLY] {self._last_live_reply}")
            self._publish_tts(self._last_live_reply)

    def _is_duplicate_live_reply(self, text: str) -> bool:
        return bool(
            text
            and self._last_live_reply
            and text.strip() == self._last_live_reply
        )

    def _publish_tts(self, text: str):
        """发布 TTS 文本"""
        if not ROS_AVAILABLE:
            print(f"[TTS] (no ROS): {text}")
            return

        estimated_duration = max(5.0, min(35.0, len(text) * 0.35))
        with self._state_lock:
            self._tts_busy_until = max(
                self._tts_busy_until,
                time.time() + estimated_duration,
            )

        rospy.loginfo(f'[TTS] Publishing: "{text}"')
        msg = String()
        msg.data = text
        self.tts_publisher.publish(msg)

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
                    self._process_single_input(text)
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
    parser.add_argument(
        "--max-react-loops",
        type=int,
        default=30,
        help="Maximum ReAct loops per user command",
    )

    args, unknown = parser.parse_known_args()
    other_unknown = [item for item in unknown if not item.startswith("__")]
    if other_unknown:
        parser.error("unrecognized arguments: %s" % " ".join(other_unknown))

    Config.validate_or_raise()

    print_banner(
        "Cloud" if Config.is_cloud_mode() else "Local",
        Config.get_llm_config()["model"],
        Config.ROBOT_NAME,
    )

    node = BrainNode(
        prompt_mode=args.mode,
        show_thought=not args.no_thought,
        environment_context=args.env,
        max_react_loops=args.max_react_loops,
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
