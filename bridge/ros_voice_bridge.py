"""
ROS Voice Bridge - ROS 语音桥接器

作为 CADE 系统的 ROS 入口，负责：
1. 订阅 /asr 话题接收语音识别结果
2. 发布 /tts 话题发送语音合成请求
3. 状态管理和回声抑制
4. 异步调用 LLM 避免阻塞 ROS 回调
"""

import threading
from typing import Optional
import rospy
from std_msgs.msg import String

from robot_controller import RobotController
from body.robot import Robot
from body.robot_interface import RobotState
from config import Config


class RosVoiceBridge:
    """
    ROS 语音桥接器

    架构：
    - Pull Mode (旧): stdin -> RobotController -> print
    - Push Mode (新): /asr -> RosVoiceBridge -> RobotController -> /tts

    核心职责：
    1. 感知：监听 /asr，拦截无效输入
    2. 认知：调用 RobotController 进行 LLM 推理
    3. 执行：发布回复到 /tts
    """

    def __init__(
        self,
        prompt_mode: str = "default",
        show_thought: bool = True,
        environment_context: Optional[str] = None
    ):
        """
        初始化 ROS 语音桥接器

        Args:
            prompt_mode: 提示词模式
            show_thought: 是否显示思考过程
            environment_context: 环境上下文信息（如"你正坐在实验室的桌子上"）
        """
        # 初始化 ROS 节点
        rospy.init_node('cade_voice_bridge', anonymous=True)

        # 创建语音机器人
        self.robot = Robot(name=Config.ROBOT_NAME)

        # 创建控制器
        self.controller = RobotController(
            robot=self.robot,
            prompt_mode=prompt_mode,
            show_thought=show_thought
        )

        # 注入环境上下文
        if environment_context:
            self._inject_environment_context(environment_context)

        # 状态锁（用于线程安全）
        self._state_lock = threading.Lock()

        # ROS 话题
        self.tts_publisher = rospy.Publisher('/tts', String, queue_size=10)
        self.asr_subscriber = rospy.Subscriber(
            '/asr',
            String,
            self._on_asr_message,
            queue_size=10
        )

        # 统计信息
        self.total_inputs = 0
        self.ignored_inputs = 0
        self.successful_replies = 0

        rospy.loginfo("=" * 60)
        rospy.loginfo("ROS Voice Bridge 初始化成功")
        rospy.loginfo(f"  机器人: {Config.ROBOT_NAME}")
        rospy.loginfo(f"  运行模式: {'云端' if Config.is_cloud_mode() else '本地'}")
        rospy.loginfo(f"  模型: {Config.get_llm_config()['model']}")
        rospy.loginfo(f"  订阅: /asr")
        rospy.loginfo(f"  发布: /tts")
        rospy.loginfo("=" * 60)

    def _inject_environment_context(self, context: str):
        """
        注入环境上下文到系统提示词

        Args:
            context: 环境描述
        """
        # 在系统提示词末尾添加环境信息
        self.controller.system_prompt += f"\n\n## 当前环境\n{context}"
        rospy.loginfo(f"已注入环境上下文: {context}")

    def _on_asr_message(self, msg: String):
        """
        ASR 消息回调（由 ROS 在后台线程调用）

        Args:
            msg: ASR 识别结果消息
        """
        text = msg.data.strip()
        if not text:
            return

        self.total_inputs += 1

        # ========== 关键输出：用户说的内容 ==========
        print("\n" + "=" * 60)
        print(f"🎤 [ASR] 收到语音: \"{text}\"")
        print("=" * 60)

        # 阶段 A：状态拦截（回声抑制）
        with self._state_lock:
            if self.robot.is_busy():
                self.ignored_inputs += 1
                print(f"⏭️  [IGNORED] 机器人忙碌 ({self.robot.state.value})，忽略此输入")
                return

        # 在新线程中处理，避免阻塞 ROS 回调
        thread = threading.Thread(
            target=self._process_input_async,
            args=(text,),
            daemon=True
        )
        thread.start()

    def _process_input_async(self, text: str):
        """
        异步处理用户输入

        Args:
            text: 用户输入文本
        """
        try:
            # 阶段 B：认知与决策
            with self._state_lock:
                self.robot.set_state(RobotState.THINKING)

            print(f"🧠 [LLM] 正在思考...")

            # 调用 LLM 进行决策
            decision = self.controller.process_input(text)

            # 阶段 C：执行与反馈
            # 提取回复内容
            reply = decision.reply if decision.reply else "好的，我收到了。"

            print(f"💬 [REPLY] {reply}")

            # 发布到 TTS
            self._publish_tts(reply)

            self.successful_replies += 1

            print(f"✅ [DONE] 本轮对话完成，等待下一轮输入...")
            print()

        except Exception as e:
            print(f"❌ [ERROR] 处理输入失败: {e}")
            import traceback
            traceback.print_exc()

            # 错误恢复
            with self._state_lock:
                self.robot.set_state(RobotState.IDLE)

            # 发送错误提示
            self._publish_tts("抱歉，我遇到了一些问题。")

    def _publish_tts(self, text: str):
        """
        发布文本到 TTS 话题

        Args:
            text: 要合成的文本
        """
        with self._state_lock:
            self.robot.set_state(RobotState.SPEAKING)

        rospy.loginfo(f"[TTS] 发布语音: \"{text}\"")

        msg = String()
        msg.data = text
        self.tts_publisher.publish(msg)

        # 注意：TTS 播放完成后状态由外部重置
        # 这里我们设置一个延迟来模拟语音播放时间
        # 实际应用中应该由 TTS 节点反馈播放完成
        import time
        estimated_duration = max(1.0, len(text) * 0.1)  # 粗略估计
        time.sleep(estimated_duration)

        with self._state_lock:
            self.robot.set_state(RobotState.IDLE)

    def spin(self):
        """
        进入 ROS 主循环
        """
        rospy.loginfo("开始监听 /asr 话题...")
        rospy.spin()

    def print_statistics(self):
        """打印统计信息"""
        rospy.loginfo("=" * 60)
        rospy.loginfo("统计信息:")
        rospy.loginfo(f"  总输入数: {self.total_inputs}")
        rospy.loginfo(f"  忽略输入: {self.ignored_inputs}")
        rospy.loginfo(f"  成功回复: {self.successful_replies}")
        if self.total_inputs > 0:
            success_rate = (self.successful_replies / self.total_inputs) * 100
            rospy.loginfo(f"  成功率: {success_rate:.1f}%")
        rospy.loginfo("=" * 60)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CADE ROS 语音桥接器")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'simple', 'compact', 'debug'],
        default='default',
        help='提示词模式'
    )
    parser.add_argument(
        '--no-thought',
        action='store_true',
        help='不显示 LLM 思考过程'
    )
    parser.add_argument(
        '--env',
        type=str,
        default="你正坐在 Fedora 实验室的桌子上，目前只能通过语音与人交流。",
        help='环境上下文信息'
    )

    args = parser.parse_args()

    try:
        bridge = RosVoiceBridge(
            prompt_mode=args.mode,
            show_thought=not args.no_thought,
            environment_context=args.env
        )
        bridge.spin()

    except rospy.ROSInterruptException:
        pass

    except Exception as e:
        rospy.logerr(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
