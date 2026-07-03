"""Vision people-track cache used by continuous follow_person."""

import json
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

import rospy
from std_msgs.msg import String


class VisionTrackCache:
    """Thread-safe cache for /vision/people_tracks_task3 JSON snapshots."""

    def __init__(self, topic: str):
        self.topic = topic
        self._lock = threading.Lock()
        self._stamp = 0.0
        self._people = []
        self._last_error = None
        self._last_positions = {}
        self._last_position_max_age = 3.0
        self._sub = rospy.Subscriber(topic, String, self._on_msg, queue_size=5)

    def _on_msg(self, msg: String):
        try:
            payload = json.loads(msg.data)
            people = payload.get("people", [])
            stamp = float(payload.get("stamp") or time.time())
            if not isinstance(people, list):
                raise ValueError("people must be a list")
            people = self._fill_missing_positions(people, stamp)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return

        with self._lock:
            self._stamp = stamp
            self._people = people
            self._last_error = None

    def _fill_missing_positions(self, people, stamp):
        filled_people = []
        for person in people:
            if not isinstance(person, dict):
                continue
            item = dict(person)
            track_id = self._as_int(item.get("track_id"))
            position = self._position_tuple(item.get("position_3d"))
            if track_id is not None and position is not None:
                self._last_positions[track_id] = {
                    "position": list(position),
                    "stamp": float(stamp),
                }
            elif track_id is not None:
                cached = self._last_positions.get(track_id)
                if cached is not None:
                    age = float(stamp) - float(cached.get("stamp", 0.0) or 0.0)
                    if 0.0 <= age <= self._last_position_max_age:
                        item["position_3d"] = list(cached["position"])
                        item["position_3d_filled"] = True
                        item["position_3d_fill_age_sec"] = float(age)
            filled_people.append(item)
        return filled_people

    def latest(self, max_age: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        with self._lock:
            stamp = self._stamp
            people = list(self._people)
            last_error = self._last_error

        if stamp <= 0.0:
            return None, last_error or "No vision people track message received"

        age = time.time() - stamp
        if age > float(max_age):
            return None, "Vision people track data is stale: %.2fs" % age

        return {"stamp": stamp, "age": age, "people": people}, None

    def find_target(
        self,
        track_id: Optional[int],
        initial_position: Optional[Tuple[float, float, float]],
        max_age: float,
        max_initial_distance: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
        snapshot, error = self.latest(max_age)
        if snapshot is None:
            return None, error, None

        people = [
            person
            for person in snapshot.get("people", [])
            if person.get("position_3d") is not None
        ]
        if not people:
            return None, "No tracked person has position_3d", snapshot

        if track_id is not None:
            for person in people:
                if self._as_int(person.get("track_id")) == int(track_id):
                    return person, None, snapshot
            return None, "Track id %s is not visible" % track_id, snapshot

        if initial_position is None:
            return None, "Missing track_id or initial person_pos", snapshot

        best_person = None
        best_distance = float("inf")
        for person in people:
            position = self._position_tuple(person.get("position_3d"))
            if position is None:
                continue
            distance = self._distance(position, initial_position)
            if distance < best_distance:
                best_person = person
                best_distance = distance

        if best_person is None:
            return None, "No valid person position for initial matching", snapshot
        if best_distance > float(max_initial_distance):
            return (
                None,
                "Closest person is %.2fm from initial target, over %.2fm"
                % (best_distance, float(max_initial_distance)),
                snapshot,
            )

        return best_person, None, snapshot

    @staticmethod
    def _as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _position_tuple(value):
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _distance(a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2
        )
