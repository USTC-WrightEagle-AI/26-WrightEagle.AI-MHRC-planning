#!/usr/bin/env python3
"""
测试 DeepSeek 的"无思考"模式性能

对比显示/不显示思考过程的输出差异
"""

from robot_controller import RobotController


def test_with_thought():
    """标准模式 - 显示思考过程"""
    print("="*70)
    print("🧠 标准模式：显示思考过程")
    print("="*70)

    controller = RobotController(show_thought=True)

    test_inputs = [
        "你好",
        "去厨房",
        "找到苹果"
    ]

    for user_input in test_inputs:
        controller.process_input(user_input)
        print()


def test_without_thought():
    """无思考模式 - 不显示思考过程"""
    print("\n\n" + "="*70)
    print("⚡ 无思考模式：不显示思考过程")
    print("="*70)

    controller = RobotController(show_thought=False)

    test_inputs = [
        "你好",
        "去厨房",
        "找到苹果"
    ]

    for user_input in test_inputs:
        controller.process_input(user_input)
        print()


def compare():
    """对比说明"""
    print("\n\n" + "="*70)
    print("📊 对比说明")
    print("="*70)

    print("""
标准模式（show_thought=True）输出：
  ✅ 💭 思考过程: ...
  ✅ 💬 回复: ...
  ✅ ⚡ 计划动作: ...
  ✅ 🤖 执行动作

无思考模式（show_thought=False）输出：
  ❌ 💭 思考过程: ...（不显示）
  ✅ 💬 回复: ...
  ✅ ⚡ 计划动作: ...
  ✅ 🤖 执行动作

注意：
  • LLM 仍然会生成 thought 字段（遵循 Prompt）
  • 只是在显示时被隐藏了
  • 适合：
    - 生产环境（用户不需要看到机器人的"内心独白"）
    - 演示展示（更简洁的输出）
    - 性能测试（减少终端输出）

如果想让 LLM 完全不生成 thought，需要修改 Schema 和 Prompt（见方案2）
    """)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "with":
            test_with_thought()
        elif sys.argv[1] == "without":
            test_without_thought()
        else:
            print("用法: python test_no_thought.py [with|without]")
    else:
        print("🎬 完整对比测试\n")
        compare()
        print("\n运行示例:")
        print("  python test_no_thought.py with      # 只测试标准模式")
        print("  python test_no_thought.py without   # 只测试无思考模式")
