#!/usr/bin/env python3
"""
CADE - 具身智能机器人主程序

运行模式：
1. 交互模式: python main.py
2. 测试模式: python main.py --test
3. 演示模式: python main.py --demo
"""

import sys
import argparse
from robot_controller import RobotController
from config import Config


def print_banner():
    """打印欢迎信息"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 CADE - 具身智能机器人系统                            ║
║   Cognitive Agent for Domestic Environment                ║
║                                                           ║
║   项目: Project LARA                                      ║
║   版本: 0.1.0                                             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"运行模式: {'☁️  云端' if Config.is_cloud_mode() else '💻 本地'}")
    print(f"模型: {Config.get_llm_config()['model']}")
    print(f"机器人: {Config.ROBOT_NAME}")
    print(f"Mock模式: {'✓' if Config.ENABLE_MOCK else '✗'}")
    print()


def interactive_mode(args):
    """交互模式"""
    print_banner()
    controller = RobotController(
        prompt_mode=args.mode,
        show_thought=not args.no_thought
    )
    controller.interactive_mode()


def test_mode(args):
    """测试模式"""
    print_banner()
    print("🧪 运行测试场景\n")

    controller = RobotController(
        prompt_mode=args.mode,
        show_thought=not args.no_thought
    )

    # 测试用例
    test_cases = [
        # 1. 闲聊测试
        "你好呀",
        "你叫什么名字？",
        "你能做什么？",

        # 2. 简单导航
        "去厨房",
        "回到起点",

        # 3. 搜索任务
        "帮我找苹果",
        "找到水杯",

        # 4. 抓取任务
        "拿起苹果",

        # 5. 复合任务
        "把苹果放到桌子上",

        # 6. 边界情况
        "帮我订个外卖",  # 无法完成的任务
    ]

    controller.run_test_scenario(test_cases)


def demo_mode(args):
    """演示模式 - 展示一个完整的服务流程"""
    print_banner()
    print("🎬 演示模式：展示完整服务流程\n")

    controller = RobotController(
        prompt_mode=args.mode,
        show_thought=not args.no_thought
    )

    demo_scenario = [
        "你好",
        "我想要桌子上的苹果",
        "去桌子那里",
        "找到苹果",
        "拿起苹果",
        "回到起点",
        "谢谢你",
    ]

    print("📋 演示场景：用户请求获取桌子上的苹果\n")
    controller.run_test_scenario(demo_scenario)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CADE 具身智能机器人系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py              # 交互模式
  python main.py --test       # 测试模式
  python main.py --demo       # 演示模式
  python main.py --mode debug # 使用调试提示词

提示:
  - 首次运行请先配置 config.py 中的 API 密钥
  - 交互模式下输入 'quit' 退出
  - 输入 'status' 查看机器人状态
  - 输入 'stats' 查看统计信息
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='运行测试模式'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行演示模式'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'simple', 'compact', 'debug'],
        default='default',
        help='提示词模式（default=标准, compact=精简无思考, debug=调试）'
    )

    parser.add_argument(
        '--no-thought',
        action='store_true',
        help='不显示 LLM 的思考过程（测试纯执行性能）'
    )

    args = parser.parse_args()

    try:
        if args.test:
            test_mode(args)
        elif args.demo:
            demo_mode(args)
        else:
            interactive_mode(args)

    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
