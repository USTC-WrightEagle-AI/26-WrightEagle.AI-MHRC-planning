# src/cade_brain/llm_core/providers/tools_manifest.py
import json
from typing import List, Dict, Any
from cade_brain.schemas import RobotAction
# from pydantic import tpc  # 或者是直接使用 Union 的特性

def generate_tools_manifest() -> str:
    """
    动态生成供大模型阅读的 13 个原子动作说明书。
    
    利用 Pydantic 的 schema 导出功能，直接将 Python 代码转化为
    完美的、带字段描述的工具清单文本。
    """
    tools_list: List[Dict[str, Any]] = []
    
    # RobotAction 是一个 Union[NavigationModel, PersonTrackingModel, ...]
    # 在 Pydantic 中，我们可以通过 __args__ 拿到 Union 内部包含的所有独立类
    action_classes = RobotAction.__args__
    
    for cls in action_classes:
        # ⚡ 核心魔法：让 Pydantic 自动导出这个类的 JSON Schema 规范
        # 它会自动抓取你的类注释、字段名、字段类型、以及 Field(description="...") 里的文本！
        schema = cls.model_json_schema()  # Pydantic v2 用法；v1 请用 cls.schema()
        
        tool_info = {
            "name": schema.get("properties", {}).get("type", {}).get("default", cls.__name__.lower()),
            "description": cls.__doc__ or "未提供描述",
            "parameters": {
                "type": "object",
                "properties": {
                    k: v for k, v in schema.get("properties", {}).items() if k != "type"
                },
                "required": [r for r in schema.get("required", []) if r != "type"]
            }
        }
        tools_list.append(tool_info)
        
    # 格式化为漂亮的 Markdown/JSON 文本，未来直接拼进 System Prompt
    manifest_text = "## 可调用的原子动作工具箱 (Available Tools)\n"
    manifest_text += "你必须且只能从以下工具中选择 Action 输出。输出格式严格遵循 JSON Schema 规范：\n\n"
    manifest_text += json.dumps(tools_list, ensure_ascii=False, indent=2)
    
    return manifest_text