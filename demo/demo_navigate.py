#!/usr/bin/env python3
"""
navigate 指令完整流程演示

展示从用户输入到机器人执行的每一步
"""

import sys
import os

# 添加项目根目录和 src 目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import json
from brain.schemas import NavigateAction, parse_action
from body.mock_robot import MockRobot


def demo_flow():
    """完整流程演示"""

    print("="*70)
    print("🎬 navigate 指令完整流程演示")
    print("="*70)

    # ==================== 阶段1：用户输入 ====================
    print("\n【阶段1：用户输入（自然语言）】")
    print("─"*70)
    user_input = "去厨房"
    print(f"👤 用户说: \"{user_input}\"")
    print(f"\n💭 这只是普通的人类语言，机器人不能直接理解")

    input("\n按 Enter 继续...")

    # ==================== 阶段2：System Prompt ====================
    print("\n\n【阶段2：System Prompt 提供\"说明书\"】")
    print("─"*70)

    prompt_snippet = """
你拥有以下动作能力：

1. **navigate** - 导航到指定位置
   - 参数: target (如"kitchen"或[x,y,z])
   - 示例: {"type": "navigate", "target": "kitchen"}

输出格式：
{
  "thought": "思考过程",
  "reply": "回复",
  "action": {"type": "动作类型", "参数": "值"}
}
    """
    print(f"📖 System Prompt（片段）:\n{prompt_snippet}")

    print(f"\n💡 这告诉 LLM:")
    print(f"   ✅ 有哪些动作可以用")
    print(f"   ✅ 每个动作需要什么参数")
    print(f"   ✅ 如何输出 JSON 格式")

    input("\n按 Enter 继续...")

    # ==================== 阶段3：LLM 推理 ====================
    print("\n\n【阶段3：LLM 的推理过程】")
    print("─"*70)

    print(f"\n🧠 LLM 的内心独白:")
    print(f"   1. '用户说\"去厨房\"'")
    print(f"   2. '这是一个位置移动的需求'")
    print(f"   3. '查看动作列表... navigate 可以用！'")
    print(f"   4. 'navigate 需要 target 参数'")
    print(f"   5. '\"厨房\"就是 target'")
    print(f"   6. '输出 JSON'")

    llm_output = {
        "thought": "用户要求我移动到厨房，这是一个明确的导航指令",
        "reply": "好的，我这就去厨房",
        "action": {
            "type": "navigate",
            "target": "kitchen"
        }
    }

    print(f"\n📤 LLM 输出:")
    print(json.dumps(llm_output, indent=2, ensure_ascii=False))

    input("\n按 Enter 继续...")

    # ==================== 阶段4：Schema 验证 ====================
    print("\n\n【阶段4：Pydantic Schema 验证】")
    print("─"*70)

    print(f"\n📋 Schema 定义 (brain/schemas.py):")
    schema_code = '''
class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    target: Union[str, List[float]]

    @field_validator('target')
    def validate_target(cls, v):
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError("坐标必须是 [x, y, z]")
        return v
    '''
    print(schema_code)

    print(f"\n🔍 验证过程:")

    try:
        action_data = llm_output["action"]
        print(f"   输入: {action_data}")

        # 验证
        action = parse_action(action_data)
        print(f"   ✅ type 检查: '{action.type}' == 'navigate'")
        print(f"   ✅ target 检查: '{action.target}' 是字符串")
        print(f"   ✅ 验证通过！")

        print(f"\n📦 创建的对象:")
        print(f"   action.type = '{action.type}'")
        print(f"   action.target = '{action.target}'")

    except Exception as e:
        print(f"   ❌ 验证失败: {e}")

    input("\n按 Enter 继续...")

    # ==================== 阶段5：动作派发 ====================
    print("\n\n【阶段5：动作派发（Dispatch）】")
    print("─"*70)

    dispatch_code = '''
def _execute_action(self, action: RobotAction) -> bool:
    action_type = action.type  # "navigate"

    if action_type == "navigate":
        return self.robot.navigate(action.target)
                                    ↑
                            传入 "kitchen"
    '''
    print(f"📝 代码逻辑 (robot_controller.py):")
    print(dispatch_code)

    print(f"\n🔀 派发流程:")
    print(f"   1. 读取 action.type = '{action.type}'")
    print(f"   2. 匹配到 if action_type == \"navigate\"")
    print(f"   3. 调用 robot.navigate('{action.target}')")

    input("\n按 Enter 继续...")

    # ==================== 阶段6：机器人执行 ====================
    print("\n\n【阶段6：机器人执行】")
    print("─"*70)

    print(f"\n🤖 创建机器人实例:")
    robot = MockRobot(name="DEMO")

    print(f"\n📍 当前位置: {robot.current_position}")
    print(f"🗺️  已知位置: {list(robot.known_locations.keys())[:6]}...")

    print(f"\n⚡ 执行 robot.navigate('kitchen'):")
    success = robot.navigate("kitchen")

    if success:
        print(f"\n✅ 执行成功！")
        print(f"📍 新位置: {robot.current_position}")
    else:
        print(f"\n❌ 执行失败")

    # ==================== 完整流程图 ====================
    print("\n\n" + "="*70)
    print("📊 完整流程总结")
    print("="*70)

    flow_diagram = '''
┌────────────────────────────────────────────────────┐
│  用户: "去厨房"                                     │
│  (自然语言)                                        │
└─────────────┬──────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  System Prompt                                      │
│  告诉 LLM: navigate 是什么，怎么用                  │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  LLM 推理                                           │
│  "用户要移动" → navigate 动作                       │
│  输出: {"type": "navigate", "target": "kitchen"}    │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  Pydantic Schema 验证                               │
│  ✓ type 是 "navigate"                              │
│  ✓ target 是字符串 "kitchen"                       │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  代码派发                                           │
│  if type == "navigate":                             │
│      robot.navigate(target)                         │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  机器人执行                                         │
│  1. 查找 "kitchen" 的坐标: [5.0, 2.0, 0.0]         │
│  2. 移动到目标位置                                  │
│  3. 更新当前位置                                    │
└─────────────┬───────────────────────────────────────┘
              ↓
         ✅ 完成！
    '''
    print(flow_diagram)


def demo_variations():
    """演示不同的 navigate 变体"""

    print("\n\n" + "="*70)
    print("🔬 navigate 的不同变体")
    print("="*70)

    robot = MockRobot(name="DEMO")

    variations = [
        {
            "name": "语义导航（推荐）",
            "json": {"type": "navigate", "target": "kitchen"},
            "description": "使用语义标签，LLM 容易理解"
        },
        {
            "name": "中文语义",
            "json": {"type": "navigate", "target": "厨房"},
            "description": "支持中文地点名"
        },
        {
            "name": "坐标导航",
            "json": {"type": "navigate", "target": [1.5, 2.3, 0.0]},
            "description": "直接使用坐标（精确但 LLM 难生成）"
        },
    ]

    for i, var in enumerate(variations, 1):
        print(f"\n变体 {i}: {var['name']}")
        print(f"  说明: {var['description']}")
        print(f"  JSON: {json.dumps(var['json'], ensure_ascii=False)}")

        try:
            action = NavigateAction(**var['json'])
            print(f"  ✅ Schema 验证通过")

            # 模拟执行（不真的移动）
            target = action.target
            if isinstance(target, str):
                if target in robot.known_locations:
                    coords = robot.known_locations[target]
                    print(f"  🎯 目标坐标: {coords}")
                else:
                    print(f"  ⚠️  未知位置: {target}")
            else:
                print(f"  🎯 直接坐标: {target}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")


def demo_invalid_cases():
    """演示非法输入的处理"""

    print("\n\n" + "="*70)
    print("❌ 非法输入示例（Schema 会拦截）")
    print("="*70)

    invalid_cases = [
        {
            "name": "缺少 target 参数",
            "json": {"type": "navigate"},
            "expected_error": "field required"
        },
        {
            "name": "错误的 type",
            "json": {"type": "fly", "target": "sky"},
            "expected_error": "Input should be 'navigate'"
        },
        {
            "name": "坐标不是3个",
            "json": {"type": "navigate", "target": [1.0, 2.0]},
            "expected_error": "坐标必须是 [x, y, z]"
        },
        {
            "name": "坐标包含字符串",
            "json": {"type": "navigate", "target": [1, 2, "three"]},
            "expected_error": "Input should be a valid number"
        },
    ]

    for i, case in enumerate(invalid_cases, 1):
        print(f"\n案例 {i}: {case['name']}")
        print(f"  输入: {json.dumps(case['json'], ensure_ascii=False)}")

        try:
            action = parse_action(case['json'])
            print(f"  ⚠️  意外：验证通过了（不应该发生）")
        except Exception as e:
            error_msg = str(e)
            print(f"  ✅ 正确拦截: {error_msg[:60]}...")


if __name__ == "__main__":
    demo_flow()

    print("\n\n" + "="*70)
    input("按 Enter 查看更多示例...")

    demo_variations()
    demo_invalid_cases()

    print("\n\n" + "="*70)
    print("✅ 演示完成！")
    print("\n💡 查看详细文档: docs/HOW_ROBOT_UNDERSTANDS.md")
    print("="*70)
