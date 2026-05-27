#!/usr/bin/env python3
"""快速测试：bus.jpg 上的人体姿态+手势识别"""
import sys, cv2
sys.path.insert(0, "/home/huyanshen/HysProjects/CADE/cade_ws/src/cade_vision")

from src.cade_vision.posture_gesture import PostureGestureAnalyzer
from ultralytics import YOLO

# 加载 YOLO 和 MediaPipe
img = cv2.imread("/home/huyanshen/HysProjects/CADE/test_images/bus.jpg")
model = YOLO("yolo11x.pt")  # 或者 yolo11x-seg.pt
pga = PostureGestureAnalyzer()

# YOLO 检测人
results = model.predict(img, conf=0.3, verbose=False)
boxes = results[0].boxes
if boxes is None:
    print("没有检测到人")
    sys.exit(1)

print(f"检测到 {len(boxes)} 个人\n")
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # MediaPipe 分析
    result = pga.analyze(crop)
    print(f"人 #{i}  (置信度 {conf:.2f})  bbox=({x1},{y1},{x2},{y2})")
    print(f"  姿态: {result['posture']}")
    print(f"  手势: {result['gesture']}")
    print(f"  关键点: {len(result.get('landmarks',[]))} 个")
