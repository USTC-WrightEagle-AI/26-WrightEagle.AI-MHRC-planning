#!/usr/bin/env python3
"""
CADE ROS Entry Point - ROS 环境下的启动入口

运行方式：
    rosrun cade main_ros.py
或
    python main_ros.py

功能：
    启动 ROS 语音桥接器，实现语音交互循环
"""

import sys
import argparse
from bridge.ros_voice_bridge import RosVoiceBridge
from config import Config


def print_banner():
    """打印欢迎信息"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖 CADE - 具身智能机器人系统 (ROS Voice Mode)           ║
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
    print(f"输入源: ROS /asr 话题")
    print(f"输出源: ROS /tts 话题")
    print()


def print_test_cases():
    """打印测试案例"""
    test_cases = """
╔═══════════════════════════════════════════════════════════╗
║                    📋 测试案例 (可以照着念)                ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  【基础对话】                                              ║
║   1. "你好"                                               ║
║   2. "你叫什么名字"                                        ║
║   3. "你能做什么"                                          ║
║                                                           ║
║  【导航任务】                                              ║
║   4. "去厨房"                                             ║
║   5. "回到起点"                                           ║
║                                                           ║
║  【搜索任务】                                              ║
║   6. "帮我找苹果"                                         ║
║   7. "找到水杯"                                           ║
║                                                           ║
║  【复合任务】                                              ║
║   8. "把苹果拿到桌子上"                                    ║
║   9. "我渴了，帮我拿瓶水"                                  ║
║                                                           ║
║  【闲聊】                                                  ║
║  10. "今天天气怎么样"                                      ║
║  11. "给我讲个笑话"                                        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(test_cases)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CADE 具身智能机器人系统 - ROS 语音模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main_ros.py                    # 默认模式
    python main_ros.py --mode debug       # 调试模式
    python main_ros.py --no-thought       # 不显示思考过程
    python main_ros.py --env "你在家中"   # 自定义环境上下文

前提条件:
    1. ROS master 已启动 (roscore)
    2. ASR 节点已启动，发布到 /asr 话题
    3. TTS 节点已启动，订阅 /tts 话题

启动顺序:
    1. roscore
    2. roslaunch asr_tts speech.launch   # 启动 ASR 和 TTS 节点
    3. python main_ros.py                 # 启动 CADE 控制器
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['default', 'simple', 'compact', 'debug'],
        default='default',
        help='提示词模式（default=标准, compact=精简, debug=调试）'
    )

    parser.add_argument(
        '--no-thought',
        action='store_true',
        help='不显示 LLM 的思考过程'
    )

    parser.add_argument(
        '--env',
        type=str,
        default="你正坐在 Fedora 实验室的桌子上，目前只能通过语音与人交流。",
        help='环境上下文信息，注入到系统提示词中'
    )

    args = parser.parse_args()

    print_banner()
    print_test_cases()

    try:
        # 创建并启动桥接器
        bridge = RosVoiceBridge(
            prompt_mode=args.mode,
            show_thought=not args.no_thought,
            environment_context=args.env
        )

        print("\n✓ CADE 已就绪，等待语音输入...\n")
        print("提示：")
        print("  - 对着麦克风说话，可以说上面的测试案例")
        print("  - 观察终端输出，确认语音是否被识别")
        print("  - 按 Ctrl+C 退出")
        print()

        # 进入 ROS 主循环
        bridge.spin()

    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
