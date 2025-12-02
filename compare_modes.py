#!/usr/bin/env python3
"""
对比测试：标准模式 vs 精简模式

测试 DeepSeek 是否真的不生成 thought 字段
"""

import time
from robot_controller import RobotController


def test_mode(mode_name: str, prompt_mode: str, show_thought: bool = True):
    """测试指定模式"""

    print("="*70)
    print(f"🧪 测试模式: {mode_name}")
    print(f"   Prompt: {prompt_mode}")
    print(f"   显示思考: {show_thought}")
    print("="*70)

    controller = RobotController(
        prompt_mode=prompt_mode,
        show_thought=show_thought
    )

    test_inputs = [
        "你好",
        "去厨房",
        "找到苹果",
    ]

    total_time = 0
    for user_input in test_inputs:
        start = time.time()
        decision = controller.process_input(user_input)
        elapsed = time.time() - start

        total_time += elapsed

        # 检查是否生成了 thought
        has_thought = decision.thought is not None
        thought_info = "✅ 有" if has_thought else "❌ 无"

        print(f"\n📊 结果分析:")
        print(f"   thought 字段: {thought_info}")
        if has_thought:
            print(f"   thought 长度: {len(decision.thought)} 字符")
        print(f"   耗时: {elapsed:.2f}s")
        print()

    print(f"\n⏱️  总耗时: {total_time:.2f}s")
    print(f"📊 平均耗时: {total_time/len(test_inputs):.2f}s/次")

    return total_time


def main():
    """主函数 - 对比测试"""

    print("\n" + "🎯"*35)
    print("对比测试：标准模式 vs 精简模式")
    print("🎯"*35 + "\n")

    # 测试1：标准模式（default）
    time1 = test_mode(
        mode_name="标准模式（包含思考）",
        prompt_mode="default",
        show_thought=True
    )

    input("\n按 Enter 继续下一个测试...")

    # 测试2：精简模式（compact）
    time2 = test_mode(
        mode_name="精简模式（无思考）",
        prompt_mode="compact",
        show_thought=False
    )

    # 对比结果
    print("\n\n" + "="*70)
    print("📊 对比结果")
    print("="*70)

    print(f"\n标准模式总耗时: {time1:.2f}s")
    print(f"精简模式总耗时: {time2:.2f}s")
    print(f"时间差: {abs(time1-time2):.2f}s")

    if time2 < time1:
        speedup = ((time1 - time2) / time1) * 100
        print(f"⚡ 精简模式快了 {speedup:.1f}%")
    else:
        slowdown = ((time2 - time1) / time1) * 100
        print(f"⚠️  精简模式慢了 {slowdown:.1f}%")

    print("\n💡 观察重点:")
    print("   1. 精简模式的 LLM 输出中是否真的没有 thought 字段？")
    print("   2. 精简模式是否节省了推理时间？")
    print("   3. 精简模式的动作执行是否同样准确？")

    print("\n📖 查看详细分析:")
    print("   - LLM 原始输出在 '📄 LLM 原始输出' 部分")
    print("   - thought 字段状态在 '📊 结果分析' 部分")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "default":
            test_mode("标准模式", "default", True)
        elif mode == "compact":
            test_mode("精简模式", "compact", False)
        else:
            print(f"未知模式: {mode}")
            print("用法: python compare_modes.py [default|compact]")
    else:
        main()
