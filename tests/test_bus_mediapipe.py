#!/usr/bin/env python3
"""端到端测试：bus.jpg 上 YOLO-World + MediaPipe Pose 姿态手势识别 + 可视化"""
import sys
import os
import cv2
import numpy as np

# 确保 cade_vision 包在路径中
sys.path.insert(0, "/home/huyanshen/HysProjects/CADE/cade_ws/src/cade_vision/src/cade_vision")

from posture_gesture import PostureGestureAnalyzer
from ultralytics import YOLO

# 路径
MODEL_PATH = "/home/huyanshen/HysProjects/CADE/models/yolov8x-worldv2.pt"
IMAGE_PATH = "/home/huyanshen/HysProjects/CADE/test_images/bus.jpg"
OUTPUT_PATH = "/home/huyanshen/HysProjects/CADE/test_images_result/bus_mediapipe_test.jpg"

# MediaPipe Pose 骨架连线定义（33 个关键点之间的连接）
POSE_CONNECTIONS = [
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    # 躯干
    (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
    # 左臂
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 21), (19, 15),
    # 右臂
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 22),
    # 左腿
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 25),
    # 右腿
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 26),
]


def draw_skeleton(image, landmarks, bbox, color=(255, 128, 64), thickness=2):
    """在图像上绘制 33 个关键点和骨架连线"""
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1
    if crop_h <= 0 or crop_w <= 0:
        return

    img_h, img_w = image.shape[:2]

    # 画关键点（红色小圆点）
    for lm in landmarks:
        if len(lm) < 3:
            continue
        # 归一化坐标 → 原图坐标
        px = int(x1 + lm[0] * crop_w)
        py = int(y1 + lm[1] * crop_h)
        cv2.circle(image, (px, py), 3, (0, 0, 255), -1)

    # 画骨架连线（淡蓝色）
    for i1, i2 in POSE_CONNECTIONS:
        if i1 >= len(landmarks) or i2 >= len(landmarks):
            continue
        lm1 = landmarks[i1]
        lm2 = landmarks[i2]
        if len(lm1) < 3 or len(lm2) < 3:
            continue
        # 任一关键点不可见则跳过
        if lm1[2] < 0.5 or lm2[2] < 0.5:
            continue
        px1 = int(x1 + lm1[0] * crop_w)
        py1 = int(y1 + lm1[1] * crop_h)
        px2 = int(x1 + lm2[0] * crop_w)
        py2 = int(y1 + lm2[1] * crop_h)
        cv2.line(image, (px1, py1), (px2, py2), color, thickness)


def main():
    print(f"Loading YOLO-World: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # YOLO-World: 设置检测类别为 person
    if hasattr(model, 'set_classes'):
        model.set_classes(["person"])
        print("YOLO-World set_classes: ['person']")

    print(f"Loading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"ERROR: Cannot read {IMAGE_PATH}")
        sys.exit(1)
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    print("Initializing PostureGestureAnalyzer...")
    pga = PostureGestureAnalyzer()

    # YOLO 推理
    print("Running YOLO inference...")
    results = model.predict(img, conf=0.3, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("No persons detected")
        sys.exit(1)

    N = len(boxes)
    print(f"\n检测到 {N} 个人\n")

    output_img = img.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # MediaPipe 分析
        result = pga.analyze(crop)
        bbox = (x1, y1, x2, y2)

        # 终端输出
        print(f"人 #{i} (置信度 {conf:.2f}) bbox=({x1},{y1},{x2},{y2})")
        print(f"  姿态: {result['posture']}")
        print(f"  手势: {result['gesture']}")
        print(f"  关键点: {len(result.get('landmarks', []))} 个\n")

        # 绘制人框（绿色矩形）
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 左上角标注
        label = f"#{i} {result['posture']}/{result['gesture']}"
        cv2.putText(output_img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制骨架
        landmarks = result.get("landmarks", [])
        if landmarks:
            draw_skeleton(output_img, landmarks, bbox)

    # 保存可视化结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, output_img)
    print(f"Visualization saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
