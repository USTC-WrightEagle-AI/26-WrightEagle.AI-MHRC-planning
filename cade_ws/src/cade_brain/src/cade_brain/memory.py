"""
Dynamic World Model - 动态世界模型

开机为空，通过视觉回调等外部感知动态填补。
提供统一的 CRUD 接口，取代旧 robot.py 中的硬编码数据。
"""

from typing import Optional, Dict, Any


class WorldModel:
    """
    动态世界模型

    职责：
    - 存储人物、物体、位置的动态状态
    - 提供查询和更新接口
    - 生成供 LLM 注入的文本摘要

    关键原则：开机为空，所有数据由外部感知节点（vision, nav）动态写入。
    """

    def __init__(self):
        self.current_position: str = "home"
        self.holding_object: Optional[str] = None

        # 动态数据（开机为空）
        self.known_locations: Dict[str, list] = {}
        self.known_objects: Dict[str, Dict[str, Any]] = {}
        self.known_people: Dict[str, Dict[str, Any]] = {}

    # ==================== Location ====================

    def add_location(self, name: str, coordinates: list) -> bool:
        """添加/更新已知位置"""
        self.known_locations[name] = coordinates
        return True

    # ==================== Person ====================

    def add_person(self, name: str, cloth_color: Optional[str] = None,
                   gesture: Optional[str] = None, location: Optional[str] = None,
                   beacon: Optional[str] = None, info: Optional[str] = None) -> bool:
        """向世界模型添加新人物（供视觉模块等外部感知系统调用）"""
        if name in self.known_people:
            print(f"[WorldModel] {name} 已存在，将更新信息")
        self.known_people[name] = {
            "name": name,
            "cloth_color": cloth_color or "unknown",
            "gesture": gesture or "unknown",
            "location": location or "unknown",
            "beacon": beacon or "unknown",
            "info": info or f"This is {name}.",
        }
        print(f"[WorldModel] +人物: {name} (location={location}, cloth={cloth_color}, gesture={gesture})")
        return True

    def update_person(self, name: str, **kwargs) -> bool:
        """更新世界模型中人物的属性（位置、手势、衣服等），不存在则自动创建"""
        if name not in self.known_people:
            print(f"[WorldModel] 未知人物 {name}，自动创建")
            self.add_person(name)
        for key, value in kwargs.items():
            old_val = self.known_people[name].get(key)
            self.known_people[name][key] = value
            print(f"[WorldModel] ~人物 {name}.{key}: {old_val} -> {value}")
        return True

    def remove_person(self, name: str) -> bool:
        """从世界模型中移除人物"""
        if name in self.known_people:
            del self.known_people[name]
            print(f"[WorldModel] -人物: {name}")
            return True
        return False

    # ==================== Object ====================

    def add_object(self, name: str, category: Optional[str] = None,
                   location: Optional[str] = None, placement: Optional[str] = None,
                   position: Optional[list] = None, **extra_props) -> bool:
        """向世界模型添加新物体（供视觉模块等外部感知系统调用）"""
        if name in self.known_objects:
            print(f"[WorldModel] {name} 已存在，将更新信息")
        obj = {"name": name, "category": category or "unknown"}
        if location:
            obj["location"] = location
        obj["placement"] = placement or location or "unknown"
        if position:
            obj["position"] = position
        obj.update(extra_props)
        self.known_objects[name] = obj
        print(f"[WorldModel] +物体: {name} (category={category}, placement={obj['placement']})")
        return True

    def update_object(self, name: str, **kwargs) -> bool:
        """更新世界模型中物体的属性，不存在则自动创建"""
        if name not in self.known_objects:
            print(f"[WorldModel] 未知物体 {name}，自动创建")
            self.add_object(name)
        for key, value in kwargs.items():
            old_val = self.known_objects[name].get(key)
            self.known_objects[name][key] = value
            print(f"[WorldModel] ~物体 {name}.{key}: {old_val} -> {value}")
        return True

    def remove_object(self, name: str) -> bool:
        """从世界模型中移除物体"""
        if name in self.known_objects:
            del self.known_objects[name]
            print(f"[WorldModel] -物体: {name}")
            return True
        return False

    # ==================== Query ====================

    def get_person(self, name: str) -> Optional[Dict[str, Any]]:
        """按名字获取人物信息"""
        return self.known_people.get(name)

    def get_object(self, name: str) -> Optional[Dict[str, Any]]:
        """按名字获取物体信息"""
        return self.known_objects.get(name)

    def get_people_in_location(self, location: str) -> list:
        """获取指定位置的所有人物"""
        return [p for p in self.known_people.values() if p.get("location") == location]

    def get_objects_in_location(self, location: str) -> list:
        """获取指定位置的所有物体"""
        return [o for o in self.known_objects.values()
                if o.get("placement") == location or o.get("location") == location]

    def get_world_state(self) -> str:
        """获取当前世界模型的文本摘要（用于注入 LLM 上下文）"""
        lines = ["--- Current World State ---"]
        lines.append(f"Position: {self.current_position}")
        lines.append(f"Holding: {self.holding_object or 'nothing'}")
        lines.append("")

        lines.append("Known People:")
        if self.known_people:
            for p in self.known_people.values():
                lines.append(f"  {p['name']}: at {p['location']}, beacon {p['beacon']}, "
                            f"cloth={p['cloth_color']}, gesture={p['gesture']}")
        else:
            lines.append("  (no people detected yet)")

        lines.append("")
        lines.append("Known Objects:")
        if self.known_objects:
            for obj in self.known_objects.values():
                loc = obj.get('placement', obj.get('location', 'unknown'))
                pos = obj.get('position', None)
                pos_str = f", position={pos}" if pos else ""
                lines.append(f"  {obj['name']}: category={obj.get('category', '-')}, at {loc}{pos_str}")
        else:
            lines.append("  (no objects detected yet)")

        lines.append("")
        lines.append("Known Locations:")
        if self.known_locations:
            for name, coords in self.known_locations.items():
                lines.append(f"  {name}: {coords}")
        else:
            lines.append("  (no locations mapped yet)")

        return "\n".join(lines)
