#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect Task1 people cache clothing fields for field testing."""

import argparse
import json
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PEOPLE_JSON = SCRIPT_DIR / "latest_people.json"
DEFAULT_STATUS_JSON = SCRIPT_DIR / "latest_vision_cache_status.json"
CLOTHING_KEYS = (
    "clothing_summary",
    "cloth_summary",
    "cloth_items",
    "cloth_color",
    "cloth_type",
)


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"_error": f"missing: {path}"}
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid json: {exc}"}


def age_text(timestamp):
    if not isinstance(timestamp, (int, float)):
        return "unknown"
    return f"{time.time() - float(timestamp):.2f}s"


def known_text(value):
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"unknown", "unknown clothing", "none", "null", "n/a"}


def clothing_attributes(person):
    attrs = []
    for key in ("clothing_summary", "cloth_summary", "clothing"):
        value = person.get(key)
        if known_text(value):
            attrs.append(str(value))
    color = person.get("cloth_color")
    ctype = person.get("cloth_type")
    if known_text(color) and known_text(ctype):
        for color_part, type_part in zip(str(color).split("/"), str(ctype).split("/")):
            color_part = color_part.strip()
            type_part = type_part.strip()
            if color_part and type_part and color_part != "unknown" and type_part != "unknown":
                attrs.append(f"{color_part} {type_part}")
    for item in person.get("cloth_items") or []:
        if not isinstance(item, dict):
            continue
        color = item.get("color")
        ctype = item.get("type") or item.get("category")
        if known_text(color) and known_text(ctype):
            attrs.append(f"{str(color).strip()} {str(ctype).strip()}")
        elif known_text(ctype):
            attrs.append(str(ctype).strip())
    if person.get("has_glasses") is True:
        attrs.append("glasses")

    deduped = []
    seen = set()
    for attr in attrs:
        key = attr.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(attr)
    return deduped


def print_report(people_path, status_path, as_json=False):
    people = load_json(people_path)
    status = load_json(status_path)
    person = people.get("best_person") or {}
    candidates = people.get("people") or []
    attrs = clothing_attributes(person)
    missing = [] if attrs else [key for key in CLOTHING_KEYS if key not in person]
    candidate_reports = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_reports.append(
            {
                "list_index": index,
                "person_index": candidate.get("person_index") or candidate.get("index"),
                "bbox": candidate.get("bbox"),
                "depth_m": candidate.get("depth_m"),
                "confidence": candidate.get("confidence"),
                "cloth_summary": candidate.get("clothing_summary") or candidate.get("cloth_summary"),
                "cloth_color": candidate.get("cloth_color"),
                "cloth_type": candidate.get("cloth_type"),
                "attributes": clothing_attributes(candidate),
            }
        )
    report = {
        "people_json": str(people_path),
        "people_status": people.get("status"),
        "people_age_sec": None,
        "person_count": people.get("person_count"),
        "selection_method": people.get("selection_method"),
        "vision_pid": status.get("pid"),
        "vision_pid_alive": False,
        "vision_status": status.get("status"),
        "vision_age_sec": None,
        "clothing_enabled": (status.get("enabled") or {}).get("clothing"),
        "clothing_available": (status.get("clothing") or {}).get("available"),
        "clothing_error": (status.get("clothing") or {}).get("error"),
        "best_person_bbox": person.get("bbox"),
        "best_person_depth_m": person.get("depth_m"),
        "cloth_color": person.get("cloth_color"),
        "cloth_type": person.get("cloth_type"),
        "cloth_summary": person.get("cloth_summary"),
        "clothing_summary": person.get("clothing_summary"),
        "cloth_items": person.get("cloth_items"),
        "attributes": attrs,
        "people": candidate_reports,
        "missing_clothing_keys": missing,
    }
    if isinstance(people.get("timestamp"), (int, float)):
        report["people_age_sec"] = round(time.time() - float(people["timestamp"]), 3)
    if isinstance(status.get("timestamp"), (int, float)):
        report["vision_age_sec"] = round(time.time() - float(status["timestamp"]), 3)
    if status.get("pid") is not None:
        report["vision_pid_alive"] = Path("/proc", str(status["pid"])).exists()

    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print("=" * 64)
    print(
        f"people: status={report['people_status']} age={age_text(people.get('timestamp'))} "
        f"count={report['person_count']} selection={report['selection_method']}"
    )
    print(
        f"vision: status={report['vision_status']} pid={report['vision_pid']} "
        f"alive={report['vision_pid_alive']} age={age_text(status.get('timestamp'))}"
    )
    print(
        f"clothing: enabled={report['clothing_enabled']} "
        f"available={report['clothing_available']}"
    )
    if report["clothing_error"]:
        print(f"clothing_error: {report['clothing_error']}")
    print(f"best_person: bbox={report['best_person_bbox']} depth_m={report['best_person_depth_m']}")
    print(f"cloth_color: {report['cloth_color']}")
    print(f"cloth_type: {report['cloth_type']}")
    print(f"cloth_summary: {report['cloth_summary']}")
    print(f"clothing_summary: {report['clothing_summary']}")
    print(f"cloth_items: {json.dumps(report['cloth_items'], ensure_ascii=False)}")
    print(f"attributes: {', '.join(attrs) if attrs else '<none>'}")
    if candidate_reports:
        print("people_candidates:")
        for candidate in candidate_reports:
            candidate_attrs = candidate["attributes"]
            print(
                "  "
                f"#{candidate['list_index']} person_index={candidate['person_index']} "
                f"bbox={candidate['bbox']} depth_m={candidate['depth_m']} "
                f"conf={candidate['confidence']} summary={candidate['cloth_summary']} "
                f"attrs={', '.join(candidate_attrs) if candidate_attrs else '<none>'}"
            )
    if missing:
        print(f"missing_clothing_keys: {', '.join(missing)}")
        if report["clothing_enabled"] is not True:
            print("hint: restart vision_cache_daemon.py if clothing_enabled is not true.")
        elif report["clothing_available"] is not True:
            print("hint: check clothing_error in latest_vision_cache_status.json.")
        else:
            print("hint: keep the guest centered and visible, then wait for the next clothing refresh.")
    elif not report["vision_pid_alive"]:
        print("hint: clothing is from cache, but the vision cache process is not alive.")


def main():
    parser = argparse.ArgumentParser(description="Inspect latest_people.json clothing fields")
    parser.add_argument("--people-json", default=str(DEFAULT_PEOPLE_JSON))
    parser.add_argument("--status-json", default=str(DEFAULT_STATUS_JSON))
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--watch", type=float, default=0.0, help="repeat every N seconds")
    args = parser.parse_args()

    people_path = Path(args.people_json)
    status_path = Path(args.status_json)
    if args.watch and args.watch > 0:
        while True:
            print_report(people_path, status_path, as_json=args.json)
            time.sleep(args.watch)
    else:
        print_report(people_path, status_path, as_json=args.json)


if __name__ == "__main__":
    main()
