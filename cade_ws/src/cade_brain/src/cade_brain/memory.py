"""
Episodic Memory - 任务级情景记忆

在整个 brain 节点生命周期内持久。启动时为空，关闭时自然销毁。
只存外观文字，不存位置，不存 track_id。
"""

from typing import Optional


class EpisodicMemory:
    """任务级情景记忆 — 在整个 brain 节点生命周期内持久"""

    def __init__(self):
        # people: {"Charlie": {"appearance": {...}, "info": {...}}}
        self.people = {}

    def bind_person(self, name: str, appearance: dict) -> bool:
        """绑定人名到外观属性"""
        if name not in self.people:
            self.people[name] = {"appearance": {}, "info": {}}
        self.people[name]["appearance"] = appearance
        print(f"[Episodic] Bind '{name}' -> {appearance}")
        return True

    def remember(self, name: str, key: str, value: str) -> bool:
        """存储人物的额外信息"""
        if name not in self.people:
            self.people[name] = {"appearance": {}, "info": {}}
        self.people[name]["info"][key] = value
        print(f"[Episodic] Remember '{name}'.{key} = '{value}'")
        return True

    def recall_appearance(self, name: str) -> Optional[dict]:
        """按名字回忆外观属性"""
        entry = self.people.get(name)
        if entry and entry.get("appearance"):
            return entry["appearance"]
        return None

    def recall_info(self, name: str) -> dict:
        """按名字回忆额外信息"""
        entry = self.people.get(name)
        if entry:
            return entry.get("info", {})
        return {}
