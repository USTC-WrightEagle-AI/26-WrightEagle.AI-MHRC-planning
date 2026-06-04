#!/usr/bin/env python3
"""挥手视频测试：YOLO-World + MediaPipe Pose 逐帧检测 + 可视化输出"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.insert(0, "/home/huyanshen/HysProjects/CADE/cade_ws/src/cade_vision/src/cade_vision")


MODEL_PATH = "/home/huyanshen/HysProjects/CADE/models/yolov8x-worldv2.pt"
VIDEO_PATH = "/home/huyanshen/HysProjects/CADE/tests/data/videos/test_video_waving.mp4"
OUTPUT_PATH = "/home/huyanshen/HysProjects/CADE/tests/outputs/videos/test_video_waving_out.mp4"

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 21), (19, 15),
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 25),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 26),
]

# MediaPipe Hands 21 关键点连线
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),             # MCP cross-connections
]


def iou(box_a, box_b):
    """两个 bbox 的 IoU"""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def assign_ids(prev_tracks, current_boxes):
    """简单 IOU 匹配分配 track ID"""
    if not prev_tracks:
        return {i: i for i in range(len(current_boxes))}
    assigned = {}
    used = set()
    cur_boxes = list(current_boxes)
    for pid, pbox in prev_tracks.items():
        best_iou, best_idx = 0, -1
        for i, cb in enumerate(cur_boxes):
            if i in used:
                continue
            v = iou(pbox, cb)
            if v > best_iou and v > 0.3:
                best_iou, best_idx = v, i
        if best_idx >= 0:
            assigned[best_idx] = pid
            used.add(best_idx)
    next_id = max(prev_tracks.keys()) + 1 if prev_tracks else 0
    for i in range(len(cur_boxes)):
        if i not in assigned:
            assigned[i] = next_id
            next_id += 1
    return assigned


def draw_skeleton(image, landmarks, bbox, color=(128, 255, 128), thickness=2):
    """绘制关键点和骨架连线"""
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1
    if crop_h <= 0 or crop_w <= 0:
        return
    img_h, img_w = image.shape[:2]

    for lm in landmarks:
        if len(lm) < 3:
            continue
        px = int(x1 + lm[0] * crop_w)
        py = int(y1 + lm[1] * crop_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            cv2.circle(image, (px, py), 2, (0, 0, 255), -1)

    for i1, i2 in POSE_CONNECTIONS:
        if i1 >= len(landmarks) or i2 >= len(landmarks):
            continue
        lm1, lm2 = landmarks[i1], landmarks[i2]
        if len(lm1) < 3 or len(lm2) < 3:
            continue
        if lm1[2] < 0.5 or lm2[2] < 0.5:
            continue
        px1 = int(x1 + lm1[0] * crop_w)
        py1 = int(y1 + lm1[1] * crop_h)
        px2 = int(x1 + lm2[0] * crop_w)
        py2 = int(y1 + lm2[1] * crop_h)
        cv2.line(image, (px1, py1), (px2, py2), color, thickness)


def draw_hand(image, landmarks, bbox, color=(255, 128, 128), thickness=1):
    """绘制 Hands 模型 21 个关键点和连线"""
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1
    if crop_h <= 0 or crop_w <= 0:
        return
    img_h, img_w = image.shape[:2]

    # 关键点
    for lm in landmarks:
        if len(lm) < 2:
            continue
        px = int(x1 + lm[0] * crop_w)
        py = int(y1 + lm[1] * crop_h)
        if 0 <= px < img_w and 0 <= py < img_h:
            cv2.circle(image, (px, py), 1, color, -1)

    # 连线
    for i1, i2 in HAND_CONNECTIONS:
        if i1 >= len(landmarks) or i2 >= len(landmarks):
            continue
        lm1, lm2 = landmarks[i1], landmarks[i2]
        if len(lm1) < 2 or len(lm2) < 2:
            continue
        px1 = int(x1 + lm1[0] * crop_w)
        py1 = int(y1 + lm1[1] * crop_h)
        px2 = int(x1 + lm2[0] * crop_w)
        py2 = int(y1 + lm2[1] * crop_h)
        cv2.line(image, (px1, py1), (px2, py2), color, thickness)


def main():
    print(f"Loading YOLO-World: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    if hasattr(model, 'set_classes'):
        model.set_classes(["person"])

    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {w_in}x{h_in}, {fps:.0f}fps, {total} frames")

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w_in, h_in))

    from posture_gesture import PostureGestureAnalyzer

    pga = PostureGestureAnalyzer()
    prev_tracks = {}  # track_id -> bbox
    frame_idx = 0
    waving_history = {}  # track_id -> consecutive waving frames

    print("\nProcessing frames...")

    # 挥手帧方差日志 CSV (仅 waving 帧)
    csv_path = os.path.join(os.path.dirname(OUTPUT_PATH), "waving_variance_log.csv")
    csv_f = open(csv_path, "w")
    csv_f.write("frame,pid,var_fa_L,var_fa_R,var_fa_max,var_palm_L,var_palm_R,var_palm_max,shoulder_ok,T_FOREARM,T_WRIST\n")

    # 逐帧 debug CSV (所有帧，含原始角度值)
    dbg_path = os.path.join(os.path.dirname(OUTPUT_PATH), "per_frame_debug.csv")
    dbg_f = open(dbg_path, "w")
    dbg_f.write("frame,pid,angle_elbow_L,angle_elbow_R,angle_palm_L,angle_palm_R,"
                "var_fa_L,var_fa_R,var_fa_max,var_palm_L,var_palm_R,var_palm_max,"
                "shoulder_ok,gesture,waving\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO 推理
        results = model.predict(frame, conf=0.3, verbose=False)
        boxes = results[0].boxes

        current_boxes = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append((x1, y1, x2, y2))

        # ID 分配
        id_map = assign_ids(prev_tracks, current_boxes)
        new_tracks = {}

        display = frame.copy()
        detections_this_frame = []

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = current_boxes[i]
                conf = float(box.conf[0])
                pid = id_map[i]
                new_tracks[pid] = (x1, y1, x2, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Hands + Pose 并行：用 Hands 精细关键点替代 Pose 粗手部点
                hands_data = pga.process_hands(crop)
                result = pga.analyze_with_temporal(pid, crop, hands_data=hands_data)
                posture = result["posture"]
                gesture = result["gesture"]
                landmarks = result.get("landmarks", [])

                # 每帧 debug 日志（所有帧）
                dbg_f.write(f"{frame_idx},{pid},"
                            f"{result.get('elbow_angle_left', 0)},{result.get('elbow_angle_right', 0)},"
                            f"{result.get('wrist_angle_left', 0)},{result.get('wrist_angle_right', 0)},"
                            f"{result.get('var_fa_l', 0):.6f},{result.get('var_fa_r', 0):.6f},{result.get('max_fa', 0):.6f},"
                            f"{result.get('var_wr_l', 0):.6f},{result.get('var_wr_r', 0):.6f},{result.get('max_wr', 0):.6f},"
                            f"{result.get('shoulder_ok', False)},{gesture},"
                            f"{1 if gesture == 'waving' else 0}\n")

                # 跟踪连续挥手帧数
                if gesture == "waving":
                    waving_history[pid] = waving_history.get(pid, 0) + 1
                    csv_f.write(f"{frame_idx},{pid},"
                                f"{result.get('var_fa_l', 0):.6f},{result.get('var_fa_r', 0):.6f},{result.get('max_fa', 0):.6f},"
                                f"{result.get('var_wr_l', 0):.6f},{result.get('var_wr_r', 0):.6f},{result.get('max_wr', 0):.6f},"
                                f"{result.get('shoulder_ok', False)},{pga.T_FOREARM},{pga.T_WRIST}\n")
                else:
                    waving_history[pid] = 0

                detections_this_frame.append((pid, x1, y1, x2, y2, conf, posture, gesture, landmarks))

                # 颜色：挥手为红色框，否则绿色
                is_waving = gesture == "waving"
                box_color = (0, 0, 255) if is_waving else (0, 255, 0)
                thickness = 3 if is_waving else 2
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, thickness)

                # 左上角标签
                label = f"#{pid} {posture}/{gesture} {conf:.2f}"
                cv2.putText(display, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                # 连续挥手帧数
                if waving_history.get(pid, 0) > 0:
                    wave_label = f"WAVING x{waving_history[pid]}"
                    cv2.putText(display, wave_label, (x1, max(y1 - 28, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # Pose 骨架
                if landmarks:
                    draw_skeleton(display, landmarks, (x1, y1, x2, y2))

                # Hands 模型 21 点精细手部关键点
                if hands_data:
                    for side_label in ("Left", "Right"):
                        hd = hands_data.get(side_label)
                        if hd and hd.get("landmarks"):
                            h_color = (255, 80, 80) if side_label == "Left" else (80, 255, 80)
                            draw_hand(display, hd["landmarks"], (x1, y1, x2, y2), h_color)

        prev_tracks = new_tracks

        # 顶部信息栏
        cv2.putText(display, f"Frame: {frame_idx}/{total}  Persons: {len(detections_this_frame)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 列出当前帧检测
        y_offset = 50
        for pid, x1, y1, x2, y2, conf, posture, gesture, lm in detections_this_frame:
            wave_info = f"  WAVE x{waving_history.get(pid, 0)}" if gesture == "waving" else ""
            text = f"#{pid}: {posture}/{gesture} ({conf:.2f}){wave_info}"
            cv2.putText(display, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255) if gesture == "waving" else (200, 200, 200), 1)
            y_offset += 18

        out.write(display)

        if frame_idx % 30 == 0:
            # tuple: (pid, x1, y1, x2, y2, conf, posture, gesture, landmarks)
            waving_now = sum(1 for _ in detections_this_frame if _[7] == "waving")
            gs = [(_[0], _[7]) for _ in detections_this_frame]
            print(f"  Frame {frame_idx}/{total}  "
                  f"persons={len(detections_this_frame)}  waving={waving_now}  "
                  f"gestures={gs}")

    cap.release()
    out.release()
    csv_f.close()
    dbg_f.close()

    # 统计
    print(f"\nDone.  Processed {frame_idx} frames")
    total_waving_frames = sum(1 for v in waving_history.values() if v > 0)
    print(f"  Persons tracked: {len(waving_history)}")
    print(f"  Persons with waving: {total_waving_frames}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
