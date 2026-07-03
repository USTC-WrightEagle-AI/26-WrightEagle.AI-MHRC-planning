"""OpenCV drawing helpers kept out of the gateway control flow."""

import cv2
import numpy as np


def pose_landmark_visibility(landmark):
    if landmark is None or len(landmark) < 3:
        return 0.0
    return float(landmark[2])


def pose_landmark_pixel(landmark, bbox):
    x1, y1, x2, y2 = bbox
    px = int(landmark[0] * (x2 - x1) + x1)
    py = int(landmark[1] * (y2 - y1) + y1)
    return px, py


def draw_pose_overlay(image, landmarks, bbox, connections=None):
    if not landmarks:
        return
    if connections is None:
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
            (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
            (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
            (30, 32), (27, 31), (28, 32),
        )
    image_h, image_w = image.shape[:2]
    points = {}
    for idx, landmark in enumerate(landmarks):
        px, py = pose_landmark_pixel(landmark, bbox)
        px = int(np.clip(px, 0, image_w - 1))
        py = int(np.clip(py, 0, image_h - 1))
        visibility = pose_landmark_visibility(landmark)
        color = (0, 255, 0) if visibility > 0.8 else (0, 0, 255)
        points[idx] = (px, py, visibility, color)

    for start_idx, end_idx in connections:
        if start_idx not in points or end_idx not in points:
            continue
        sx, sy, sv, _ = points[start_idx]
        ex, ey, ev, _ = points[end_idx]
        line_color = (0, 255, 0) if sv > 0.8 and ev > 0.8 else (0, 0, 255)
        cv2.line(image, (sx, sy), (ex, ey), line_color, 1, cv2.LINE_AA)

    for _, (px, py, visibility, color) in points.items():
        cv2.circle(image, (px, py), 3, color, -1, cv2.LINE_AA)
        label = f"{visibility:.1f}"
        tx = int(np.clip(px + 4, 0, image_w - 18))
        ty = int(np.clip(py - 4, 8, image_h - 1))
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)


def draw_perf_overlay(image, perf_snapshot):
    if not perf_snapshot:
        return
    workers = perf_snapshot.get("workers", {})
    person_worker = workers.get("person", {})
    pose_worker = workers.get("pose_gesture", {})
    cloth_worker = workers.get("cloth", {})
    lines = [
        (
            f"FPS {perf_snapshot.get('loop_fps', 0.0):.1f} "
            f"{perf_snapshot.get('vision_profile', 'idle')} "
            f"G {perf_snapshot.get('gesture_fps', 0.0):.1f} "
            f"age {perf_snapshot.get('frame_age_sec', 0.0):.2f}s"
        ),
        (
            f"P {person_worker.get('fps', 0.0):.1f}/"
            f"{person_worker.get('result_age_sec', 0.0):.2f}s "
            f"Pose {pose_worker.get('fps', 0.0):.1f}/"
            f"{pose_worker.get('result_age_sec', 0.0):.2f}s "
            f"Cloth {cloth_worker.get('fps', 0.0):.1f}/"
            f"{cloth_worker.get('result_age_sec', 0.0):.2f}s"
        ),
        (
            f"Wave {perf_snapshot.get('waving_status', 'unknown')} "
            f"h {perf_snapshot.get('waving_history_frames', 0)}/"
            f"{perf_snapshot.get('waving_history_span_sec', 0.0):.1f}s "
            f"p {perf_snapshot.get('waving_probability', 0.0):.2f}"
        ),
    ]
    if perf_snapshot.get("waving_under_sampled"):
        lines.append("WARNING waving window under-sampled")
    if perf_snapshot.get("pose_result_stale"):
        lines.append("WARNING pose result stale")

    x, y = 10, 24
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def draw_detections(image, detections, target_class=None):
    if not detections:
        cv2.putText(image, "No objects detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return
    target_cls = target_class.lower() if target_class else None
    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]
        c_name = obj["class_name"]
        is_target = target_cls is not None and target_cls in c_name.lower()
        box_color = (0, 0, 255) if is_target else (0, 255, 0)
        thickness = 3 if is_target else 1
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
        if c_name == "person":
            draw_pose_overlay(image, obj.get("landmarks", []), obj.get("pose_bbox", obj["bbox"]))
        lines = [f"#{obj['index']} {c_name} {obj['confidence']:.2f}"]
        if "track_id" in obj:
            lines[0] += f" [ID: {obj['track_id']}]"
        if c_name == "person":
            lines.append(f"PG: {obj.get('posture', 'unk')}/{obj.get('gesture', 'unk')}")
            lines.append(
                "Wave: "
                f"p={float(obj.get('waving_probability', 0.0) or 0.0):.2f} "
                f"{obj.get('waving_status', 'unk')} "
                f"h={obj.get('waving_history_frames', 0)}/"
                f"{float(obj.get('waving_history_span_sec', 0.0) or 0.0):.1f}s "
                f"v={obj.get('gesture_waving_voted', 'unk')}"
            )
            lines.append(f"Cloth: {obj.get('cloth_summary', 'unk')}")
        if obj.get("position_3d") is not None:
            cx, cy, cz = obj["position_3d"]
            lines.append(f"XYZ: ({cx:.2f}, {cy:.2f}, {cz:.2f})m")
        line_height = 18
        curr_y = y1 - 10 - (len(lines) - 1) * line_height
        for line in lines:
            cv2.putText(image, line, (x1, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(image, line, (x1, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
            curr_y += line_height
