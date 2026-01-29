"""
Robot Controller - 机器人控制器

整合大脑(LLM)和躯体(Robot)，实现完整的感知-决策-执行循环
"""

from typing import List, Dict, Optional
from config import Config
from brain.llm_client import LLMClient
from brain.prompts import get_system_prompt
from brain.schemas import RobotDecision, RobotAction
from body.mock_robot import MockRobot
from body.robot_interface import RobotInterface, RobotState


class RobotController:
    """
    机器人主控制器

    负责：
    1. 接收用户输入
    2. 调用LLM进行决策
    3. 执行动作
    4. 管理对话历史
    """

    def __init__(
        self,
        robot: Optional[RobotInterface] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_mode: str = "default",
        show_thought: bool = True  # ← 新增参数
    ):
        """
        初始化控制器

        Args:
            robot: 机器人实例（如果为None则创建MockRobot）
            llm_client: LLM客户端（如果为None则创建默认客户端）
            prompt_mode: 提示词模式（default/simple/debug）
            show_thought: 是否显示 LLM 的思考过程（默认 True）
        """
        # 初始化机器人
        self.robot = robot or MockRobot(name=Config.ROBOT_NAME)

        # 初始化LLM客户端
        self.llm_client = llm_client or LLMClient()

        # 系统提示词
        self.system_prompt = get_system_prompt(prompt_mode)

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

        # 显示选项
        self.show_thought = show_thought  # ← 保存参数

        # 统计信息
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0

        print(f"\n{'='*60}")
        print(f"🚀 机器人控制器初始化成功")
        print(f"{'='*60}")

    def process_input(self, user_input: str) -> RobotDecision:
        """
        处理用户输入（完整流程）

        Args:
            user_input: 用户输入文本

        Returns:
            RobotDecision: 决策对象
        """
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*60}")

        self.total_interactions += 1

        # 1. 让LLM思考
        self.robot.set_state(RobotState.THINKING)
        print(f"\n🧠 [大脑思考中...]")

        try:
            decision = self.llm_client.get_decision(
                user_input=user_input,
                system_prompt=self.system_prompt,
                conversation_history=self.conversation_history
            )

            # 打印决策
            if self.show_thought and decision.thought:  # ← 检查是否存在且需要显示
                print(f"\n💭 思考过程: {decision.thought}")
            if decision.reply:
                print(f"💬 回复: {decision.reply}")
            if decision.action:
                print(f"⚡ 计划动作: {decision.action.type}")

            # 2. 执行动作
            if decision.action:
                success = self._execute_action(decision.action)
                if success:
                    self.successful_actions += 1
                else:
                    self.failed_actions += 1

            # 3. 更新对话历史
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })

            # 构建助手回复（包含思考、回复和动作）
            assistant_response_parts = []
            if decision.thought:  # ← 只在有 thought 时添加
                assistant_response_parts.append(f"思考: {decision.thought}")
            if decision.reply:
                assistant_response_parts.append(f"回复: {decision.reply}")
            if decision.action:
                assistant_response_parts.append(f"动作: {decision.action.model_dump_json()}")

            assistant_response = "\n".join(assistant_response_parts)

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })

            return decision

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            self.robot.set_state(RobotState.ERROR)
            raise

    def _execute_action(self, action: RobotAction) -> bool:
        """
        执行具体动作

        Args:
            action: 动作对象

        Returns:
            bool: 是否成功
        """
        action_type = action.type

        try:
            if action_type == "navigate":
                return self.robot.navigate(action.target)

            elif action_type == "search":
                result = self.robot.search(action.object_name)
                return result is not None

            elif action_type == "pick":
                return self.robot.pick(action.object_name, action.object_id)

            elif action_type == "place":
                return self.robot.place(action.location)

            elif action_type == "speak":
                return self.robot.speak(action.content)

            elif action_type == "wait":
                return self.robot.wait(action.reason)

            else:
                print(f"⚠ 未知动作类型: {action_type}")
                return False

        except Exception as e:
            print(f"❌ 动作执行失败: {e}")
            return False

    def interactive_mode(self):
        """
        交互模式（命令行对话）
        """
        print(f"\n{'='*60}")
        print(f"🤖 进入交互模式")
        print(f"提示：输入 'quit' 或 'exit' 退出")
        print(f"{'='*60}\n")

        while True:
            try:
                user_input = input("👤 你: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再见！")
                    self.print_statistics()
                    break

                if user_input.lower() == 'status':
                    self.robot.print_status()
                    continue

                if user_input.lower() == 'stats':
                    self.print_statistics()
                    continue

                # 处理输入
                self.process_input(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                self.print_statistics()
                break

            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()

    def run_test_scenario(self, scenarios: List[str]):
        """
        运行测试场景

        Args:
            scenarios: 测试用例列表
        """
        print(f"\n{'='*60}")
        print(f"🧪 开始测试场景 (共 {len(scenarios)} 个)")
        print(f"{'='*60}")

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n\n{'─'*60}")
            print(f"测试 {i}/{len(scenarios)}")
            print(f"{'─'*60}")

            try:
                self.process_input(scenario)
                # 等待一下，模拟真实交互
                import time
                time.sleep(1)

            except Exception as e:
                print(f"❌ 测试失败: {e}")

        print(f"\n\n{'='*60}")
        print(f"🏁 测试完成")
        print(f"{'='*60}")
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        print(f"\n📊 统计信息:")
        print(f"   总交互次数: {self.total_interactions}")
        print(f"   成功动作: {self.successful_actions}")
        print(f"   失败动作: {self.failed_actions}")
        if self.total_interactions > 0:
            success_rate = (self.successful_actions / self.total_interactions) * 100
            print(f"   成功率: {success_rate:.1f}%")

    def reset(self):
        """重置控制器状态"""
        self.conversation_history.clear()
        self.total_interactions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        print("✓ 控制器已重置")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=== 测试机器人控制器 ===\n")

    # 创建控制器
    controller = RobotController()

    # 测试场景
    test_scenarios = [
        "你好",                          # 闲聊
        "你叫什么名字？",                # 闲聊
        "去厨房",                        # 简单导航
        "帮我找苹果",                    # 搜索
        "把苹果拿过来",                  # 复合任务
    ]

    # 运行测试
    controller.run_test_scenario(test_scenarios)
