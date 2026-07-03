"""
Vision Actions - observation, filtering, counting, and geometry helpers.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field

from .base_action import BaseAction


PositionInput = Union[str, List[float], Dict[str, Any]]


class ObservePeopleAction(BaseAction):
    """
    Observe visible people.

    Use include to request extra fields. Use position to return only the person
    nearest to that vision/camera 3D point.
    """

    type: Literal["observe_people"] = "observe_people"
    include: Optional[List[str]] = Field(
        default=None,
        description="Optional fields to include: gesture, posture, clothing.",
    )
    position: Optional[PositionInput] = Field(
        default=None,
        description=(
            "Optional vision/camera 3D point [x,y,z]. If provided, vision returns "
            "only the visible person nearest to this point."
        ),
    )
    timeout: float = Field(default=10.0, description="Observation timeout in seconds")


class FindPeopleAction(BaseAction):
    """
    Find people matching simple visual filters.
    """

    type: Literal["find_people"] = "find_people"
    gesture: Optional[str] = Field(
        default=None,
        description="Gesture filter, e.g. waving, raising_left_arm, pointing_right.",
    )
    posture: Optional[str] = Field(
        default=None,
        description="Posture filter: standing, sitting, lying.",
    )
    clothing: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Clothing filter, e.g. {'color':'blue'}, {'category':'top','color':'blue'}, "
            "{'category':'glasses'}, {'category':'watch'}."
        ),
    )
    position: Optional[PositionInput] = Field(
        default=None,
        description="Optional [x,y,z]; after filtering, return only the nearest matching person.",
    )
    timeout: float = Field(default=10.0, description="Search timeout in seconds")


class CountPeopleAction(BaseAction):
    """
    Count people matching simple visual filters.
    """

    type: Literal["count_people"] = "count_people"
    gesture: Optional[str] = Field(default=None, description="Gesture filter")
    posture: Optional[str] = Field(default=None, description="Posture filter")
    clothing: Optional[Dict[str, Any]] = Field(default=None, description="Clothing filter")
    timeout: float = Field(default=10.0, description="Count timeout in seconds")


class ObserveObjectsAction(BaseAction):
    """
    Observe visible non-person objects.
    """

    type: Literal["observe_objects"] = "observe_objects"
    timeout: float = Field(default=10.0, description="Observation timeout in seconds")


class CalculateDistanceAction(BaseAction):
    """
    Calculate 3D distance deterministically.

    Use either b for point-to-point distance, or people for distances from a
    point to every person and nearest-person selection.
    """

    type: Literal["calculate_distance"] = "calculate_distance"
    a: PositionInput = Field(..., description="First vision/camera 3D point [x,y,z]")
    b: Optional[PositionInput] = Field(default=None, description="Optional second 3D point")
    people: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional people list returned by observe_people/find_people",
    )
