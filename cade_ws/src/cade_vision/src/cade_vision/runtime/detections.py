"""Detection conversion and geometry helpers."""

import cv2
import numpy as np


def class_name(model_names, class_id):
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, class_id))
    if 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    return str(class_id)


def mask_polygon(result, box_index):
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return None
    if box_index >= len(masks.xy):
        return None
    polygon = masks.xy[box_index]
    if polygon is None or len(polygon) < 3:
        return None
    return np.round(polygon).astype(int).tolist()


def median_depth(depth_m, x, y, roi_size=20, min_depth=0.1, max_depth=2.0):
    if depth_m is None:
        return None
    height, width = depth_m.shape[:2]
    x1 = max(0, int(x - roi_size // 2))
    y1 = max(0, int(y - roi_size // 2))
    x2 = min(width - 1, int(x + roi_size // 2))
    y2 = min(height - 1, int(y + roi_size // 2))
    roi = depth_m[y1:y2, x1:x2]
    valid_depths = roi[(roi > min_depth) & (roi < max_depth)]
    if len(valid_depths) == 0:
        return None
    return float(np.median(valid_depths))


def deproject(pixel_x, pixel_y, depth, camera_info):
    if depth is None or depth <= 0 or camera_info is None:
        return None
    width = int(camera_info.get("width", 0) or 0)
    height = int(camera_info.get("height", 0) or 0)
    if pixel_x < 0 or pixel_y < 0 or pixel_x >= width or pixel_y >= height:
        return None
    fx = float(camera_info["fx"])
    fy = float(camera_info["fy"])
    cx = float(camera_info["cx"])
    cy = float(camera_info["cy"])
    x = (float(pixel_x) - cx) * depth / fx
    y = (float(pixel_y) - cy) * depth / fy
    return [x, y, depth]


def append_yolo_detections(
    detections,
    results,
    model_names,
    source_model,
    depth_m=None,
    camera_info=None,
    allowed_names=None,
    min_area_ratio=None,
):
    min_depth = 0.2
    max_depth = 4.0

    for result in results:
        orig_shape = getattr(result, "orig_shape", None) or (0, 0)
        image_h, image_w = orig_shape[:2]
        frame_area = float(image_w * image_h)
        boxes = result.boxes
        if boxes is None:
            continue
        for box_index, box in enumerate(boxes):
            class_id = int(box.cls[0])
            name = class_name(model_names, class_id)
            if allowed_names is not None and name not in allowed_names:
                continue

            conf = box.conf[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            area_ratio = 0.0
            if frame_area > 0.0:
                area_ratio = (max(0, x2 - x1) * max(0, y2 - y1)) / frame_area
            if min_area_ratio is not None and area_ratio < min_area_ratio:
                continue
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            point_3d = None
            depth = median_depth(depth_m, center_x, center_y)
            if depth is not None:
                point_3d = deproject(center_x, center_y, depth, camera_info)
                if point_3d is not None:
                    distance = float(np.linalg.norm(point_3d))
                    if distance < min_depth or distance > max_depth:
                        point_3d = None

            obj = {
                "index": len(detections),
                "class_id": class_id,
                "class_name": name,
                "source_model": source_model,
                "confidence": float(conf),
                "bbox": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "area_ratio": float(area_ratio),
                "position_3d": point_3d,
            }
            polygon = mask_polygon(result, box_index)
            if polygon:
                obj["mask_polygon"] = polygon
                obj["mask_area"] = float(
                    cv2.contourArea(np.asarray(polygon, dtype=np.float32))
                )
            detections.append(obj)
    return detections


def bbox_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def set_default_cloth_fields(person):
    person.setdefault("cloth_color", "unknown")
    person.setdefault("cloth_type", "unknown")
    person.setdefault("cloth_items", [])
    person.setdefault("cloth_summary", "unknown")

