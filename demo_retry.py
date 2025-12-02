#!/usr/bin/env python3
"""
演示 LLM 重试机制

模拟 LLM 第一次输出错误，第二次输出正确的过程
"""

import json
from brain.schemas import RobotDecision, parse_action


def simulate_llm_retry():
    """模拟 LLM 的重试过程"""

    print("="*70)
    print("🎬 模拟场景：用户说 '去桌子那里'")
    print("="*70)

    # ==================== 第1次尝试 ====================
    print("\n【第1次尝试】")
    print("─"*70)

    # LLM 第一次可能的错误输出
    response_1 = "好的，我这就去桌子那里。我会立即执行导航动作。"

    print(f"\n📄 LLM 原始输出:")
    print(f"   {response_1}")

    print(f"\n🔍 尝试解析 JSON...")

    try:
        # 尝试解析
        data = json.loads(response_1)
        print(f"   ✅ 解析成功")
    except json.JSONDecodeError as e:
        print(f"   ❌ 解析失败: {e}")
        print(f"   ⚠ 警告: 无法从文本中提取JSON")

        # 系统会把错误反馈给 LLM
        print(f"\n💬 系统反馈给 LLM:")
        print(f"   '你的输出格式有误，错误信息：{e}'")
        print(f"   '请严格按照JSON格式重新输出。'")

    # ==================== 第2次尝试 ====================
    print("\n\n【第2次尝试】")
    print("─"*70)

    # LLM 第二次的正确输出
    response_2 = '''```json
{
  "thought": "用户明确指示我去桌子那里。这是一个直接的导航指令。我应该执行导航动作。",
  "reply": "好的，我这就去桌子那里。",
  "action": {
    "type": "navigate",
    "target": "table"
  }
}
```'''

    print(f"\n📄 LLM 原始输出:")
    print(response_2)

    print(f"\n🔍 尝试解析 JSON...")

    try:
        # 提取 JSON（支持 markdown 代码块）
        if "```json" in response_2:
            start = response_2.find("```json") + 7
            end = response_2.find("```", start)
            json_str = response_2[start:end].strip()
        else:
            json_str = response_2

        # 解析 JSON
        data = json.loads(json_str)
        print(f"   ✅ JSON 解析成功！")

        # 打印解析结果
        print(f"\n📋 解析结果:")
        print(f"   thought: {data['thought'][:50]}...")
        print(f"   reply: {data['reply']}")
        print(f"   action.type: {data['action']['type']}")
        print(f"   action.target: {data['action']['target']}")

        # 验证并创建 Pydantic 模型
        action = parse_action(data['action'])
        decision = RobotDecision(**data)

        print(f"\n✅ Pydantic 验证通过！")

        # 模拟执行动作
        print(f"\n🤖 执行动作:")
        print(f"   类型: {action.type}")
        print(f"   目标: {action.target}")
        print(f"   → 调用 robot.navigate('table')")
        print(f"   ✅ 导航成功！")

    except Exception as e:
        print(f"   ❌ 失败: {e}")

    # ==================== 总结 ====================
    print("\n\n" + "="*70)
    print("📊 总结")
    print("="*70)
    print("""
流程：
  1️⃣  用户输入 → "去桌子那里"
  2️⃣  LLM 第1次尝试 → 输出纯文本（失败）
  3️⃣  系统检测到错误 → 反馈给 LLM
  4️⃣  LLM 第2次尝试 → 输出正确 JSON（成功）
  5️⃣  解析 JSON → 提取 action 字段
  6️⃣  调用对应函数 → robot.navigate('table')
  7️⃣  机器人执行动作 → 移动到桌子 ✅

关键点：
  • JSON 是大脑和躯体的"通信协议"
  • action 字段决定了调用哪个函数
  • 重试机制保证了即使第一次失败也能成功
  • 最终只要有一次成功，动作就会执行
    """)

    print("="*70)


def show_json_structure():
    """展示 JSON 结构的重要性"""

    print("\n\n" + "="*70)
    print("🧩 JSON 输出和机器人动作的映射关系")
    print("="*70)

    mappings = [
        {
            "json": {
                "action": {"type": "navigate", "target": "kitchen"}
            },
            "function": "robot.navigate('kitchen')",
            "result": "🚗 机器人移动到厨房"
        },
        {
            "json": {
                "action": {"type": "search", "object_name": "apple"}
            },
            "function": "robot.search('apple')",
            "result": "🔍 机器人搜索苹果"
        },
        {
            "json": {
                "action": {"type": "pick", "object_name": "bottle"}
            },
            "function": "robot.pick('bottle')",
            "result": "🤏 机器人抓取瓶子"
        },
        {
            "json": {
                "action": {"type": "speak", "content": "任务完成"}
            },
            "function": "robot.speak('任务完成')",
            "result": "💬 机器人说话：任务完成"
        },
    ]

    for i, mapping in enumerate(mappings, 1):
        print(f"\n示例 {i}:")
        print(f"  JSON: {json.dumps(mapping['json'], ensure_ascii=False)}")
        print(f"  ↓")
        print(f"  调用: {mapping['function']}")
        print(f"  ↓")
        print(f"  结果: {mapping['result']}")
        print()

    print("="*70)


if __name__ == "__main__":
    simulate_llm_retry()
    show_json_structure()

    print("\n💡 提示:")
    print("   查看详细文档: docs/LLM_RETRY_MECHANISM.md")
    print()
