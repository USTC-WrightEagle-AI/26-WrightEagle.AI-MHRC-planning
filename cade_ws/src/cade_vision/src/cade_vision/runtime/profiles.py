"""Task-driven perception profile selection."""

import time


VISION_PROFILES = {"idle", "person", "gesture", "posture", "cloth", "full"}
GESTURE_ATTRIBUTE_KEYS = {
    "gesture",
    "gesture_raw",
    "gesture_static",
    "gesture_waving",
    "waving",
    "hand_gesture",
}
GESTURE_ATTRIBUTE_VALUES = {
    "waving",
    "pointing_left",
    "pointing_right",
    "raising_left_arm",
    "raising_right_arm",
    "raising_left",
    "raising_right",
    "raise_left",
    "raise_right",
}
POSTURE_ATTRIBUTE_KEYS = {"posture", "pose"}
POSTURE_ATTRIBUTE_VALUES = {"standing", "sitting", "lying"}
CLOTH_ATTRIBUTE_KEYS = {
    "cloth_color",
    "cloth_type",
    "cloth_summary",
    "cloth_items",
    "clothing",
    "clothes",
    "cloth",
    "wearing",
}


def profile_has_pose(profile):
    return profile in {"gesture", "posture", "full"}


def profile_has_cloth(profile):
    return profile in {"cloth", "full"}


def iter_command_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_command_values(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iter_command_values(item)
        return
    if value is not None:
        yield str(value).strip().lower()


def profile_from_command(action, cmd, attributes):
    explicit = cmd.get("vision_profile") or cmd.get("perception_profile")
    if explicit is not None:
        explicit = str(explicit).strip().lower()
        if explicit in VISION_PROFILES:
            return explicit

    attr_keys = set(attributes.keys()) if isinstance(attributes, dict) else set()
    keys = {str(key).lower() for key in cmd.keys()} | {
        str(key).lower() for key in attr_keys
    }
    values = set(iter_command_values(cmd))

    needs_gesture = bool(
        keys & GESTURE_ATTRIBUTE_KEYS
        or values & GESTURE_ATTRIBUTE_VALUES
        or "gesture" in values
    )
    needs_posture = bool(
        keys & POSTURE_ATTRIBUTE_KEYS
        or values & POSTURE_ATTRIBUTE_VALUES
        or "posture" in values
    )
    needs_cloth = bool(
        keys & CLOTH_ATTRIBUTE_KEYS
        or values & {"clothing", "clothes", "cloth", "wearing"}
    )

    if needs_cloth and (needs_gesture or needs_posture):
        return "full"
    if needs_gesture:
        return "gesture"
    if needs_posture:
        return "posture"
    if needs_cloth:
        return "cloth"
    if action == "observe_objects":
        return "cloth"
    return "person"


def profile_ttl_for_command(cmd, default_ttl=3.0):
    raw_timeout = cmd.get("timeout", default_ttl)
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        timeout = default_ttl
    return max(default_ttl, min(60.0, timeout + 0.5))


class ProfileState:
    """Small state machine owned by the gateway."""

    def __init__(self, default_ttl=3.0):
        self.profile = "idle"
        self.until_time = 0.0
        self.default_ttl = float(default_ttl)

    def set(self, profile, ttl=None):
        if profile not in VISION_PROFILES:
            profile = "person"
        previous = self.profile
        self.profile = profile
        self.until_time = time.time() + max(0.1, self.default_ttl if ttl is None else ttl)
        return previous, profile, previous != profile

    def expire_if_needed(self, task_active=False):
        if (
            self.profile != "idle"
            and not task_active
            and self.until_time > 0.0
            and time.time() > self.until_time
        ):
            previous = self.profile
            self.profile = "idle"
            self.until_time = 0.0
            return previous, "idle", True
        return self.profile, self.profile, False
