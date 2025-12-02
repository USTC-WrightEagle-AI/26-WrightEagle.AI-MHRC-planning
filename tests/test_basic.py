"""
基础测试用例

测试各个模块的基本功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    """测试配置模块"""
    print("=== 测试 Config ===")
    from config import Config

    config = Config.get_llm_config()
    assert "base_url" in config
    assert "api_key" in config
    assert "model" in config

    print(f"✓ 配置模块正常")
    print(f"  模式: {Config.MODE}")
    print(f"  模型: {config['model']}\n")


def test_schemas():
    """测试数据模型"""
    print("=== 测试 Schemas ===")
    from brain.schemas import (
        NavigateAction, SearchAction, RobotDecision, parse_action
    )

    # 测试导航动作
    nav = NavigateAction(target="kitchen")
    assert nav.type == "navigate"
    assert nav.target == "kitchen"

    # 测试搜索动作
    search = SearchAction(object_name="apple")
    assert search.type == "search"
    assert search.object_name == "apple"

    # 测试决策
    decision = RobotDecision(
        thought="测试思考",
        reply="测试回复",
        action=nav
    )
    assert decision.thought == "测试思考"

    # 测试解析
    action_dict = {"type": "navigate", "target": "kitchen"}
    action = parse_action(action_dict)
    assert action.type == "navigate"

    print("✓ 数据模型正常\n")


def test_mock_robot():
    """测试Mock Robot"""
    print("=== 测试 Mock Robot ===")
    from body.mock_robot import MockRobot
    from body.robot_interface import RobotState

    robot = MockRobot(name="TestBot")

    # 测试导航
    success = robot.navigate("kitchen")
    assert success is True
    assert robot.current_position == "kitchen"
    assert robot.get_state() == RobotState.IDLE

    # 测试搜索
    result = robot.search("apple")
    assert result is not None
    assert "name" in result

    # 测试语音
    success = robot.speak("测试语音")
    assert success is True

    print("✓ Mock Robot 正常\n")


def test_prompts():
    """测试提示词"""
    print("=== 测试 Prompts ===")
    from brain.prompts import get_system_prompt, add_context

    # 测试获取提示词
    prompt = get_system_prompt("default")
    assert len(prompt) > 0
    assert "LARA" in prompt or "机器人" in prompt

    # 测试简化提示词
    simple = get_system_prompt("simple")
    assert len(simple) > 0

    # 测试添加上下文
    enhanced = add_context(prompt, "当前位置: 厨房")
    assert "当前位置" in enhanced

    print("✓ 提示词模块正常\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 开始运行基础测试")
    print("="*60 + "\n")

    tests = [
        test_config,
        test_schemas,
        test_mock_robot,
        test_prompts,
    ]

    failed = []

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ {test_func.__name__} 失败: {e}\n")
            import traceback
            traceback.print_exc()
            failed.append(test_func.__name__)

    print("="*60)
    if not failed:
        print("✅ 所有测试通过！")
    else:
        print(f"❌ {len(failed)} 个测试失败: {', '.join(failed)}")
    print("="*60)

    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
