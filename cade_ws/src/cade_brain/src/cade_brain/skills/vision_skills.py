"""
Vision Skills - small perception and geometry tools.

Each function name is the action.type exposed to the LLM.
"""

import math
from typing import Any, Dict, List, Optional

from cade_brain.skills import VisionHardwareContext, register_vision_tool


def _execute(action_type: str, wait_timeout: float = 30.0, **params: Any) -> Dict[str, Any]:
    """Publish a vision command and wait for its task status."""
    return VisionHardwareContext().execute(
        action_type,
        wait_timeout=wait_timeout,
        **params,
    )


def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if value is not None and value != "" and value != "unknown"
    }


@register_vision_tool
def observe_people(
    include: Optional[List[str]] = None,
    position: Optional[Any] = None,
    timeout: float = 10.0,
    **kwargs,
) -> Dict[str, Any]:
    """Observe visible people; include may request gesture/posture/clothing fields."""
    observe_timeout = float(timeout or 10.0)
    params = _clean_params(
        {
            "include": include,
            "position": position,
            "timeout": observe_timeout,
        }
    )
    return _execute("observe_people", wait_timeout=observe_timeout + 2.0, **params)


@register_vision_tool
def find_people(
    gesture: Optional[str] = None,
    posture: Optional[str] = None,
    clothing: Optional[Dict[str, Any]] = None,
    position: Optional[Any] = None,
    timeout: float = 10.0,
    **kwargs,
) -> Dict[str, Any]:
    """Find people matching gesture/posture/clothing filters."""
    observe_timeout = float(timeout or 10.0)
    params = _clean_params(
        {
            "gesture": gesture,
            "posture": posture,
            "clothing": clothing,
            "position": position,
            "timeout": observe_timeout,
        }
    )
    return _execute("find_people", wait_timeout=observe_timeout + 2.0, **params)


@register_vision_tool
def count_people(
    gesture: Optional[str] = None,
    posture: Optional[str] = None,
    clothing: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    **kwargs,
) -> Dict[str, Any]:
    """Count people matching gesture/posture/clothing filters."""
    observe_timeout = float(timeout or 10.0)
    params = _clean_params(
        {
            "gesture": gesture,
            "posture": posture,
            "clothing": clothing,
            "timeout": observe_timeout,
        }
    )
    return _execute("count_people", wait_timeout=observe_timeout + 2.0, **params)


@register_vision_tool
def observe_objects(timeout: float = 10.0, **kwargs) -> Dict[str, Any]:
    """Return all visible non-person detections."""
    observe_timeout = float(timeout or 10.0)
    return _execute(
        "observe_objects",
        wait_timeout=observe_timeout + 2.0,
        timeout=observe_timeout,
    )


@register_vision_tool
def calculate_distance(
    a: Any,
    b: Optional[Any] = None,
    people: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Deterministic distance calculator for positions or people lists."""
    origin = _parse_position(a)
    if origin is None:
        return {"status": "FAILED", "error": "Invalid position a"}
    if b is not None:
        target = _parse_position(b)
        if target is None:
            return {"status": "FAILED", "error": "Invalid position b"}
        return {"status": "SUCCESS", "distance": _distance(origin, target)}
    if people is None:
        return {"status": "FAILED", "error": "Either b or people is required"}

    distances = []
    nearest = None
    for person in people:
        position = _parse_position(person.get("position_3d") if isinstance(person, dict) else None)
        if position is None:
            continue
        item = {
            "track_id": person.get("track_id"),
            "distance": _distance(origin, position),
            "person": person,
        }
        distances.append(item)
        if nearest is None or item["distance"] < nearest["distance"]:
            nearest = item
    if nearest is None:
        return {"status": "FAILED", "error": "No people with valid position_3d"}
    return {
        "status": "SUCCESS",
        "distances": distances,
        "nearest_track_id": nearest.get("track_id"),
        "nearest_distance": nearest.get("distance"),
        "nearest_person": nearest.get("person"),
    }


def _parse_position(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "position_3d" in value:
            value = value["position_3d"]
        else:
            value = [value.get("x"), value.get("y"), value.get("z")]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        if len(parts) != 3:
            return None
        value = parts
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _distance(a, b):
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )
