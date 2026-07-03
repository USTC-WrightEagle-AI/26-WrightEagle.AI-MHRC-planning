"""Local obstacle diagnostics for navigation recovery decisions."""

import math
import threading

import rospy
from sensor_msgs.msg import LaserScan


class LocalNavigationDiagnostics:
    """Summarize nearby scan clearances and suggest bounded recovery moves."""

    SECTOR_CENTERS_DEG = {
        "front": 0.0,
        "front_left": 45.0,
        "front_right": -45.0,
        "left": 90.0,
        "right": -90.0,
        "rear": 180.0,
    }

    def __init__(
        self,
        scan_topic="/scan",
        stale_timeout=1.0,
        min_valid_range=0.05,
        blocked_clearance=0.6,
        caution_clearance=0.9,
        sector_half_width_deg=20.0,
        rear_clearance_required=0.75,
        suggested_distance=0.3,
        suggested_angle_deg=30.0,
    ):
        self.scan_topic = str(scan_topic)
        self.stale_timeout = float(stale_timeout)
        self.min_valid_range = float(min_valid_range)
        self.blocked_clearance = float(blocked_clearance)
        self.caution_clearance = float(caution_clearance)
        self.sector_half_width_deg = float(sector_half_width_deg)
        self.rear_clearance_required = float(rear_clearance_required)
        self.suggested_distance = float(suggested_distance)
        self.suggested_angle_deg = float(suggested_angle_deg)

        self._lock = threading.Lock()
        self._scan = None
        self._received_time = None
        self._sub = rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self._on_scan,
            queue_size=1,
        )

    def _on_scan(self, msg):
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else 0.0
        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()

        scan = {
            "stamp": stamp,
            "angle_min": float(msg.angle_min),
            "angle_increment": float(msg.angle_increment),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
            "ranges": list(msg.ranges),
        }
        with self._lock:
            self._scan = scan
            self._received_time = rospy.Time.now().to_sec()

    def snapshot(self, context=None, compact=False):
        with self._lock:
            scan = self._scan
            received_time = self._received_time

        if scan is None:
            summary = self._unavailable_summary("no_scan", context)
            payload = {
                "obstacle_summary": summary,
                "suggested_reposition": None,
            }
            return self._compact_payload(payload) if compact else payload

        now = rospy.Time.now().to_sec()
        age = max(0.0, now - float(received_time or scan["stamp"]))
        if age > self.stale_timeout:
            summary = self._unavailable_summary("stale_scan", context, age)
            payload = {
                "obstacle_summary": summary,
                "suggested_reposition": None,
            }
            return self._compact_payload(payload) if compact else payload

        sectors = {}
        blocked_sectors = []
        caution_sectors = []
        for name, center_deg in self.SECTOR_CENTERS_DEG.items():
            clearance = self._sector_min_range(scan, center_deg)
            state = self._clearance_state(clearance)
            sectors[name] = {
                "clearance_m": self._round_or_none(clearance),
                "state": state,
            }
            if state == "blocked":
                blocked_sectors.append(name)
            elif state == "caution":
                caution_sectors.append(name)

        summary = {
            "source": "scan",
            "topic": self.scan_topic,
            "available": True,
            "context": context,
            "age_sec": round(age, 3),
            "blocked_clearance_m": round(self.blocked_clearance, 3),
            "caution_clearance_m": round(self.caution_clearance, 3),
            "sectors": sectors,
            "blocked_sectors": blocked_sectors,
            "caution_sectors": caution_sectors,
        }
        payload = {
            "obstacle_summary": summary,
            "suggested_reposition": self._suggest_reposition(summary),
        }
        return self._compact_payload(payload) if compact else payload

    def _unavailable_summary(self, reason, context=None, age=None):
        summary = {
            "source": "scan",
            "topic": self.scan_topic,
            "available": False,
            "context": context,
            "reason": reason,
        }
        if age is not None:
            summary["age_sec"] = round(float(age), 3)
        return summary

    def _sector_min_range(self, scan, center_deg):
        half_width = math.radians(self.sector_half_width_deg)
        center_rad = math.radians(center_deg)
        best = None
        angle = scan["angle_min"]
        range_min = max(self.min_valid_range, scan["range_min"])
        range_max = scan["range_max"]

        for raw_range in scan["ranges"]:
            if self._angle_distance(angle, center_rad) <= half_width:
                try:
                    value = float(raw_range)
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value) and range_min <= value <= range_max:
                    if best is None or value < best:
                        best = value
            angle += scan["angle_increment"]
        return best

    def _clearance_state(self, clearance):
        if clearance is None:
            return "unknown"
        if clearance < self.blocked_clearance:
            return "blocked"
        if clearance < self.caution_clearance:
            return "caution"
        return "clear"

    def _suggest_reposition(self, summary):
        sectors = summary.get("sectors") or {}
        front_state = self._sector_state(sectors, "front")
        front_left_clearance = self._sector_clearance(sectors, "front_left")
        front_right_clearance = self._sector_clearance(sectors, "front_right")
        left_clearance = self._sector_clearance(sectors, "left")
        right_clearance = self._sector_clearance(sectors, "right")
        rear_clearance = self._sector_clearance(sectors, "rear")

        if front_state in ("blocked", "caution"):
            if self._is_clear_for_backward(rear_clearance):
                return self._motion(
                    "backward",
                    distance=self.suggested_distance,
                    reason="front_blocked_rear_clear",
                )
            return self._turn_toward_clearer_side(
                left_clearance,
                right_clearance,
                reason="front_blocked_turn_to_clearer_side",
            )

        if self._clearance_value(front_left_clearance) < self.blocked_clearance:
            return self._motion(
                "turn_right",
                angle_deg=self.suggested_angle_deg,
                reason="front_left_blocked",
            )

        if self._clearance_value(front_right_clearance) < self.blocked_clearance:
            return self._motion(
                "turn_left",
                angle_deg=self.suggested_angle_deg,
                reason="front_right_blocked",
            )

        return self._turn_toward_clearer_side(
            left_clearance,
            right_clearance,
            reason="improve_view_toward_clearer_side",
        )

    def _turn_toward_clearer_side(self, left_clearance, right_clearance, reason):
        left_value = self._clearance_value(left_clearance)
        right_value = self._clearance_value(right_clearance)
        motion = "turn_left" if left_value >= right_value else "turn_right"
        return self._motion(
            motion,
            angle_deg=self.suggested_angle_deg,
            reason=reason,
        )

    def _motion(self, motion, distance=None, angle_deg=None, reason=None):
        payload = {
            "motion": motion,
            "reason": reason,
        }
        if distance is not None:
            payload["distance"] = round(float(distance), 3)
        if angle_deg is not None:
            payload["angle_deg"] = round(float(angle_deg), 1)
        return payload

    def _compact_payload(self, payload):
        summary = payload.get("obstacle_summary") or {}
        compact = {
            "obstacle_summary": self._compact_summary(summary),
            "suggested_reposition": self._compact_motion(
                payload.get("suggested_reposition")
            ),
        }
        return compact

    def _compact_summary(self, summary):
        if not summary.get("available", False):
            reason = summary.get("reason", "unavailable")
            scan_state = "stale" if reason == "stale_scan" else "unavailable"
            compact = {
                "scan": scan_state,
                "reason": reason,
            }
            if "age_sec" in summary:
                compact["age_sec"] = summary["age_sec"]
            return compact

        blocked = list(summary.get("blocked_sectors") or [])
        caution = list(summary.get("caution_sectors") or [])
        if blocked:
            scan_state = "blocked"
        elif caution:
            scan_state = "caution"
        else:
            scan_state = "clear"

        compact = {
            "scan": scan_state,
            "blocked": blocked,
            "caution": caution,
        }
        nearest = self._compact_nearest(summary.get("sectors") or {})
        if nearest:
            compact["nearest"] = nearest
        return compact

    @staticmethod
    def _compact_nearest(sectors):
        wanted = ("front", "front_left", "front_right", "left", "right")
        nearest = {}
        for name in wanted:
            clearance = (sectors.get(name) or {}).get("clearance_m")
            if clearance is not None:
                nearest[name] = clearance
        return nearest

    @staticmethod
    def _compact_motion(motion):
        if not motion:
            return None
        compact = {"motion": motion.get("motion")}
        if motion.get("distance") is not None:
            compact["distance"] = motion.get("distance")
        if motion.get("angle_deg") is not None:
            compact["angle_deg"] = motion.get("angle_deg")
        return compact

    def _is_clear_for_backward(self, rear_clearance):
        if rear_clearance is None:
            return False
        return float(rear_clearance) >= self.rear_clearance_required

    @staticmethod
    def _sector_state(sectors, name):
        return (sectors.get(name) or {}).get("state", "unknown")

    @staticmethod
    def _sector_clearance(sectors, name):
        return (sectors.get(name) or {}).get("clearance_m")

    @staticmethod
    def _clearance_value(clearance):
        if clearance is None:
            return -1.0
        return float(clearance)

    @staticmethod
    def _round_or_none(value):
        if value is None:
            return None
        return round(float(value), 3)

    @staticmethod
    def _angle_distance(a, b):
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))
