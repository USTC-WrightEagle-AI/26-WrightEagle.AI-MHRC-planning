"""
Analyzer - 命令解析 + 按需调度子模块

接收 brain 发来的指令 JSON，解析出需要检索的人物属性，
按代价排序后逐层过滤检测到的人。

代价顺序: posture (~5ms) → gesture (~5ms) → clothing (后续实现 ~100ms)
"""

import re
from typing import Optional


class Analyzer:
    """属性过滤管线，按代价排序串联过滤"""

    def __init__(self):
        self._cost_order = [
            "posture",
            "gesture",
            "cloth_color",
            "cloth_type",
            "identity",
        ]

    def filter_by_attributes(
        self, people: list, attributes: Optional[dict]
    ) -> list:
        """
        Args:
            people: 当前帧检测到的所有人（含 bbox, landmarks, posture, gesture 等）
            attributes: 要找的特征，如 {"posture": "sitting", "gesture": "waving"}
                       为 None 或空则原样返回

        Returns:
            过滤后的人列表（符合全部特征）
        """
        if not attributes:
            return people

        ordered_names = list(self._cost_order)
        ordered_names.extend(
            name for name in attributes.keys() if name not in self._cost_order
        )

        matched = list(people)
        for attr_name in ordered_names:
            if attr_name not in attributes or not matched:
                continue
            expected = attributes[attr_name]
            if expected is None or expected == "" or expected == "unknown":
                continue
            matched = [
                p for p in matched
                if self._value_matches(p.get(attr_name), expected)
            ]

        return matched

    @staticmethod
    def _value_matches(actual, expected) -> bool:
        """Match scalar or list-like attributes, including slash-separated values."""
        if isinstance(expected, (list, tuple, set)):
            return any(Analyzer._value_matches(actual, item) for item in expected)
        if isinstance(actual, (list, tuple, set)):
            return any(Analyzer._value_matches(item, expected) for item in actual)

        actual_lower = str(actual or "").strip().lower()
        expected_lower = str(expected or "").strip().lower()
        if not actual_lower or actual_lower == "unknown":
            return False
        if actual_lower == expected_lower:
            return True

        tokens = [
            token.strip()
            for token in re.split(r"[/,|;]+", actual_lower)
            if token.strip()
        ]
        return expected_lower in tokens
