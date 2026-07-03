"""Observation, filtering, and counting task executors."""

import json
import math
import time

import numpy as np

from .base_task import BaseTask


INCLUDE_ALIASES = {
    "cloth": "clothing",
    "clothes": "clothing",
    "wearing": "clothing",
    "pose": "posture",
    "hand_gesture": "gesture",
}
VALID_INCLUDES = {"gesture", "posture", "clothing"}


class ObserveTask(BaseTask):
    """Return current perception candidates with optional filtering."""

    POLL_INTERVAL = 0.1

    def execute(self, cmd_dict):
        action = cmd_dict.get("action", "")
        if action in ("observe_people", "find_people", "count_people"):
            self._execute_people(cmd_dict, action)
            return
        if action == "observe_objects":
            self._execute_objects(cmd_dict)
            return
        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status("FAILED", error=f"Unsupported vision action: {action}")

    def _execute_people(self, cmd_dict, action):
        filters = self._filters_from_command(cmd_dict) if action in ("find_people", "count_people") else {}
        include = self._include_from_command(cmd_dict, filters)
        timeout = self._timeout_for_people(cmd_dict, include)
        query_position = self._parse_position(cmd_dict.get("position"))
        if action in ("find_people", "count_people") and filters:
            predicate = (
                lambda items: self._people_ready(items, include)
                and bool(self._select_people(items, filters, query_position))
            )
        else:
            predicate = lambda items: self._people_ready(items, include)
        detections = self._latest_snapshot(
            cmd_dict,
            predicate,
            timeout=timeout,
        )
        people = self._select_people(detections, filters, query_position)

        query = self._query(cmd_dict, include=include, filters=filters, position=query_position)
        if action == "count_people":
            result = {
                "type": "people_count",
                "query": query,
                "count": len(people),
            }
        else:
            result = {
                "type": "people_search" if action == "find_people" else "people_observation",
                "query": query,
                "count": len(people),
                "people": [
                    self._person_entry(obj, include, filters)
                    for obj in people
                ],
            }

        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status("SUCCESS", result=result)

    def _execute_objects(self, cmd_dict):
        target = str(cmd_dict.get("target") or cmd_dict.get("category") or "").lower()
        detections = self._latest_snapshot(
            cmd_dict,
            lambda items: any(
                item.get("class_name", "").lower() != "person"
                and (not target or target in item.get("class_name", "").lower())
                for item in items
            ),
        )
        objects = []
        for obj in detections:
            class_name = obj.get("class_name", "")
            if class_name.lower() == "person":
                continue
            if target and target not in class_name.lower():
                continue
            objects.append(self._object_entry(obj))

        result = {
            "type": "objects_observation",
            "query": self._query(cmd_dict),
            "count": len(objects),
            "objects": objects,
        }
        if self.should_continue() and self.node.finish_task(self):
            self.node._publish_status("SUCCESS", result=result)

    def _latest_snapshot(self, cmd_dict, predicate=None, timeout=None):
        timeout = float(cmd_dict.get("timeout", 1.0) if timeout is None else timeout)
        start_time = time.time()
        detections = []
        while self.should_continue() and time.time() - start_time < timeout:
            detections = self.node.get_latest_detections()
            if detections and (predicate is None or predicate(detections)):
                return detections
            time.sleep(self.POLL_INTERVAL)
        return detections

    def _query(self, cmd_dict, include=None, filters=None, position=None):
        query = {}
        if include:
            query["include"] = sorted(include)
        if filters:
            query.update(self._clean_value(filters))
        if position is not None:
            query["position"] = self._clean_value(position)
        if "timeout" in cmd_dict:
            query["timeout"] = self._clean_value(cmd_dict.get("timeout"))
        target = cmd_dict.get("target") or cmd_dict.get("category")
        if target:
            query["target"] = self._clean_value(target)
        return query

    def _person_entry(self, obj, include, filters=None):
        entry = {
            "track_id": obj.get("track_id"),
            "confidence": self._clean_value(obj.get("confidence")),
            "bbox": self._clean_value(obj.get("bbox")),
            "center": self._clean_value(obj.get("center")),
            "position_3d": self._clean_value(obj.get("position_3d")),
        }
        if "_distance_to_query" in obj:
            entry["distance_to_query"] = self._clean_value(obj.get("_distance_to_query"))
        if "posture" in include:
            entry["posture"] = obj.get("posture", "unknown")
            entry["posture_confidence"] = self._clean_value(obj.get("posture_confidence", 0.0))
        if "gesture" in include:
            expected = (filters or {}).get("gesture")
            entry["gesture"] = self._public_gesture(obj, expected=expected)
            entry["gesture_confidence"] = self._gesture_confidence(obj, expected=expected)
        if "clothing" in include:
            clothing = self._clothing_items(obj)
            entry["clothing_summary"] = self._clothing_summary(obj, clothing)
            entry["clothing"] = self._clean_value(clothing)
        return entry

    def _object_entry(self, obj):
        return {
            "track_id": obj.get("track_id"),
            "index": obj.get("index"),
            "name": obj.get("class_name"),
            "source_model": obj.get("source_model"),
            "confidence": self._clean_value(obj.get("confidence")),
            "bbox": self._clean_value(obj.get("bbox")),
            "center": self._clean_value(obj.get("center")),
            "position_3d": self._clean_value(obj.get("position_3d")),
            "color": obj.get("color"),
            "associated_person_index": obj.get("associated_person_index"),
            "association_region": obj.get("association_region"),
        }

    def _timeout_for_people(self, cmd_dict, include):
        default_timeout = 1.0
        if "gesture" in include:
            default_timeout = max(default_timeout, 2.0)
        if "clothing" in include:
            default_timeout = max(default_timeout, 3.0)
        try:
            timeout = float(cmd_dict.get("timeout", default_timeout))
        except (TypeError, ValueError):
            timeout = default_timeout
        return max(default_timeout, timeout)

    def _people_ready(self, detections, include):
        people = [
            obj
            for obj in detections
            if obj.get("class_name", "").lower() == "person"
        ]
        if not people:
            return False
        if ("posture" in include or "gesture" in include) and not self.node.has_fresh_worker_result("pose_gesture"):
            return False
        if "clothing" in include and not self.node.has_fresh_worker_result("cloth", max_age=2.0):
            return False
        return True

    def _filters_from_command(self, cmd_dict):
        filters = {}
        for key in ("gesture", "posture"):
            value = cmd_dict.get(key)
            if value not in (None, "", "unknown"):
                filters[key] = value
        clothing = cmd_dict.get("clothing")
        clothing_filter = {}
        if isinstance(clothing, dict):
            clothing_filter.update(
                {k: v for k, v in clothing.items() if v not in (None, "", "unknown")}
            )
        elif clothing not in (None, "", "unknown"):
            clothing_filter["category"] = clothing
        for old_key, new_key in (("cloth_color", "color"), ("cloth_type", "type")):
            value = cmd_dict.get(old_key)
            if value not in (None, "", "unknown"):
                clothing_filter[new_key] = value
        if clothing_filter:
            filters["clothing"] = clothing_filter
        return filters

    def _include_from_command(self, cmd_dict, filters):
        include = set()
        raw_include = cmd_dict.get("include")
        if raw_include in ("all", "*"):
            include.update(VALID_INCLUDES)
        elif raw_include is not None:
            if isinstance(raw_include, str):
                raw_values = [raw_include]
            else:
                raw_values = list(raw_include) if isinstance(raw_include, (list, tuple, set)) else []
            for item in raw_values:
                name = INCLUDE_ALIASES.get(str(item).strip().lower(), str(item).strip().lower())
                if name in VALID_INCLUDES:
                    include.add(name)
        if filters.get("gesture"):
            include.add("gesture")
        if filters.get("posture"):
            include.add("posture")
        if filters.get("clothing"):
            include.add("clothing")
        return include

    def _matches_person(self, obj, filters):
        if "gesture" in filters and not self._matches_gesture(obj, filters["gesture"]):
            return False
        if "posture" in filters and not self._matches_label(obj.get("posture"), filters["posture"], "posture"):
            return False
        if "clothing" in filters and not self._matches_clothing(obj, filters["clothing"]):
            return False
        return True

    def _select_people(self, detections, filters=None, position=None):
        filters = filters or {}
        people = [
            obj
            for obj in detections
            if obj.get("class_name", "").lower() == "person"
        ]
        if filters:
            people = [obj for obj in people if self._matches_person(obj, filters)]
        if position is not None:
            people = self._nearest_people(people, position)
        return people

    def _matches_gesture(self, obj, expected):
        if isinstance(expected, (list, tuple, set)):
            return any(self._matches_gesture(obj, item) for item in expected)
        expected = self._normalize_label(expected, "gesture")
        candidates = [
            obj.get("gesture"),
            obj.get("gesture_raw"),
            obj.get("gesture_static"),
            obj.get("gesture_static_voted"),
            obj.get("gesture_waving"),
            obj.get("gesture_waving_voted"),
        ]
        return any(
            self._normalize_label(candidate, "gesture") == expected
            for candidate in candidates
        )

    def _matches_label(self, actual, expected, kind):
        actual = self._normalize_label(actual, kind)
        if isinstance(expected, (list, tuple, set)):
            return any(actual == self._normalize_label(item, kind) for item in expected)
        return actual == self._normalize_label(expected, kind)

    @staticmethod
    def _normalize_label(value, kind):
        text = str(value or "").strip().lower()
        if kind == "gesture":
            aliases = {
                "raising_left": "raising_left_arm",
                "raise_left": "raising_left_arm",
                "raising_right": "raising_right_arm",
                "raise_right": "raising_right_arm",
            }
            return aliases.get(text, text)
        return text

    def _matches_clothing(self, obj, condition):
        if not isinstance(condition, dict):
            condition = {"category": condition}
        items = self._clothing_items(obj)
        if not items:
            return False
        for item in items:
            if self._clothing_item_matches(item, condition):
                return True
        return False

    def _clothing_item_matches(self, item, condition):
        checks = {
            "category": item.get("category"),
            "type": item.get("type"),
            "color": item.get("color"),
        }
        for key, expected in condition.items():
            if expected in (None, "", "unknown"):
                continue
            if key == "category":
                if not self._matches_category(checks.get("category"), expected):
                    return False
            elif key in checks:
                if not self._matches_token(checks.get(key), expected):
                    return False
        return True

    def _matches_category(self, actual, expected):
        actual = self._normalize_category(actual)
        if isinstance(expected, (list, tuple, set)):
            return any(self._matches_category(actual, item) for item in expected)
        expected = self._normalize_category(expected)
        if expected == actual:
            return True
        if expected == "bottom" and actual in {"pants", "shorts", "skirt", "bottom"}:
            return True
        return False

    @staticmethod
    def _matches_token(actual, expected):
        if isinstance(expected, (list, tuple, set)):
            return any(ObserveTask._matches_token(actual, item) for item in expected)
        actual_tokens = {
            token.strip().lower()
            for token in str(actual or "").replace("/", ",").replace("|", ",").split(",")
            if token.strip()
        }
        return str(expected or "").strip().lower() in actual_tokens

    def _clothing_items(self, obj):
        items = []
        raw_items = obj.get("cloth_items") or []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            ctype = item.get("type", "unknown")
            region = item.get("region", "")
            items.append(
                {
                    "category": self._category_from_type_region(ctype, region),
                    "type": ctype,
                    "color": item.get("color", "unknown"),
                    "confidence": self._clean_value(item.get("confidence", item.get("score", 0.0))),
                    "bbox": self._clean_value(item.get("bbox")),
                    "source_model": item.get("source_model"),
                }
            )
        if items:
            return items
        color = obj.get("cloth_color")
        ctype = obj.get("cloth_type")
        if color and ctype and color != "unknown" and ctype != "unknown":
            for color_part, type_part in zip(str(color).split("/"), str(ctype).split("/")):
                items.append(
                    {
                        "category": self._category_from_type_region(type_part, ""),
                        "type": type_part,
                        "color": color_part,
                        "confidence": 0.0,
                    }
                )
        return items

    def _clothing_summary(self, obj, clothing):
        summary = obj.get("cloth_summary")
        if summary and summary != "unknown":
            return summary
        parts = []
        for item in clothing:
            color = item.get("color", "unknown")
            ctype = item.get("type", item.get("category", "unknown"))
            parts.append(f"{color} {ctype}" if color != "unknown" else str(ctype))
        return ", ".join(parts) if parts else "unknown"

    def _category_from_type_region(self, ctype, region):
        text = str(ctype or "").lower()
        region = str(region or "").lower()
        accessories = {
            "glasses",
            "watch",
            "hat",
            "cap",
            "bag",
            "backpack",
            "shoe",
            "shoes",
        }
        for name in accessories:
            if name in text:
                return "shoes" if name in {"shoe", "shoes"} else name
        if any(token in text for token in ("pants", "trouser", "jeans")):
            return "pants"
        if "shorts" in text:
            return "shorts"
        if "skirt" in text:
            return "skirt"
        if any(token in text for token in ("shirt", "top", "jacket", "coat", "sweater", "blouse", "hoodie")):
            return "top"
        if region in {"upper", "torso"}:
            return "top"
        if region in {"lower", "legs"}:
            return "bottom"
        if region in {"full", "whole"}:
            return "full_body"
        return "unknown"

    @staticmethod
    def _normalize_category(value):
        text = str(value or "").strip().lower()
        aliases = {
            "cloth": "clothing",
            "clothes": "clothing",
            "upper": "top",
            "shirt": "top",
            "t-shirt": "top",
            "tee": "top",
            "lower": "bottom",
            "trousers": "pants",
            "jeans": "pants",
            "shoe": "shoes",
        }
        return aliases.get(text, text)

    def _nearest_people(self, people, position):
        scored = []
        for obj in people:
            point = obj.get("position_3d")
            distance = self._distance(position, point)
            if distance is None:
                continue
            candidate = dict(obj)
            candidate["_distance_to_query"] = distance
            scored.append(candidate)
        if not scored:
            return []
        return [min(scored, key=lambda item: item["_distance_to_query"])]

    def _parse_position(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"nearest", "robot", "camera", "origin"}:
                return [0.0, 0.0, 0.0]
            try:
                value = json.loads(value)
            except Exception:
                parts = [part.strip() for part in text.replace(";", ",").split(",")]
                if len(parts) != 3:
                    return None
                try:
                    return [float(part) for part in parts]
                except ValueError:
                    return None
        if isinstance(value, dict):
            if "position_3d" in value:
                value = value["position_3d"]
            else:
                value = [value.get("x"), value.get("y"), value.get("z")]
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return [float(value[0]), float(value[1]), float(value[2])]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _distance(a, b):
        if a is None or b is None or len(a) < 3 or len(b) < 3:
            return None
        try:
            return math.sqrt(
                (float(a[0]) - float(b[0])) ** 2
                + (float(a[1]) - float(b[1])) ** 2
                + (float(a[2]) - float(b[2])) ** 2
            )
        except (TypeError, ValueError):
            return None

    def _public_gesture(self, obj, expected=None):
        label, _ = self._public_gesture_candidate(obj, expected=expected)
        return label

    def _gesture_confidence(self, obj, expected=None):
        label, key = self._public_gesture_candidate(obj, expected=expected)
        if label == "waving":
            candidates = [
                obj.get("waving_probability"),
                obj.get("gesture_waving_confidence"),
                obj.get("gesture_confidence"),
            ]
            values = []
            for value in candidates:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
            return self._clean_value(max(values) if values else 0.0)
        if key in ("gesture_static", "gesture_static_voted"):
            value = obj.get("gesture_static_confidence")
        elif key == "gesture_raw":
            value = obj.get("gesture_raw_confidence")
        else:
            value = obj.get("gesture_confidence")
        return self._clean_value(0.0 if value is None else value)

    def _public_gesture_candidate(self, obj, expected=None):
        expected_label = None
        if expected is not None and not isinstance(expected, (list, tuple, set)):
            expected_label = self._normalize_label(expected, "gesture")

        candidates = [
            ("gesture", obj.get("gesture")),
            ("gesture_waving", obj.get("gesture_waving")),
            ("gesture_waving_voted", obj.get("gesture_waving_voted")),
            ("gesture_static_voted", obj.get("gesture_static_voted")),
            ("gesture_static", obj.get("gesture_static")),
            ("gesture_raw", obj.get("gesture_raw")),
        ]
        normalized = []
        for key, value in candidates:
            label = self._normalize_label(value, "gesture")
            if label in ("", "unknown", "none", "no_waving"):
                continue
            normalized.append((label, key))

        if expected_label is not None:
            for label, key in normalized:
                if label == expected_label:
                    return label, key
        for label, key in normalized:
            if label == "waving":
                return label, key
        if normalized:
            return normalized[0]
        return "unknown", None

    @staticmethod
    def _clean_value(value):
        if isinstance(value, dict):
            return {
                str(key): ObserveTask._clean_value(val)
                for key, val in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [ObserveTask._clean_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return ObserveTask._clean_value(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value
