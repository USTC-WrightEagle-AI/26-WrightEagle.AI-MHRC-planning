"""
Cloth Analyzer - 衣物颜色提取与人物关联

无状态，每帧独立调用。将 YOLO 检出的衣物/配饰匹配到人，
优先用 segmentation mask 内像素提取颜色，写入 person 对象的
cloth_color / cloth_type 字段。

不依赖 ROS。颜色分类使用包内 OpenCV 颜色模型：
models/color_svm/color_model.xml。
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .color_features import FEATURE_DIM, extract_color_features

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


__all__ = ["associate_clothing", "classify_color_pixels", "extract_color_from_region"]


# OpenCV 模型标签来自 models/color_svm/label_names.json，当前包含 purple：
# black / white / gray / red / orange / yellow / green / blue / purple
SCRIPT_DIR = Path(__file__).resolve().parent
COLOR_SVM_DIR = SCRIPT_DIR / "models" / "color_svm"
COLOR_MODEL_META_PATH = COLOR_SVM_DIR / "color_model_meta.json"
COLOR_MODEL_PATH = COLOR_SVM_DIR / "color_model.xml"
COLOR_MODEL_NORM_PATH = COLOR_SVM_DIR / "color_model_norm.npz"
COLOR_SVM_LABEL_PATH = COLOR_SVM_DIR / "label_names.json"
COLOR_MIN_PIXELS = 30
_COLOR_MODEL_BUNDLE = None
_COLOR_MODEL_LOAD_ERROR = None

UPPER_LABEL_KEYWORDS = {
    "t-shirt",
    "shirt",
    "sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "coat",
    "blouse",
    "hoodie",
    "vest",
    "top",
    "cape",
}
LOWER_LABEL_KEYWORDS = {
    "pants",
    "trousers",
    "shorts",
    "skirt",
    "tights",
    "stockings",
}
WHOLE_BODY_LABEL_KEYWORDS = {
    "dress",
    "jumpsuit",
}
FOOTWEAR_LABEL_KEYWORDS = {
    "shoe",
    "sock",
    "leg warmer",
}
ACCESSORY_LABEL_KEYWORDS = {
    "glasses",
    "hat",
    "headband",
    "head covering",
    "tie",
    "glove",
    "watch",
    "belt",
    "scarf",
}
REGION_ORDER = ("whole", "upper", "lower", "footwear", "accessory")
CLOTHING_REGION_ORDER = ("whole", "upper", "lower", "footwear")


def _iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _overlap_metrics(person_box, cloth_box):
    xa, ya = max(person_box[0], cloth_box[0]), max(person_box[1], cloth_box[1])
    xb, yb = min(person_box[2], cloth_box[2]), min(person_box[3], cloth_box[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    person_area = max(0, person_box[2] - person_box[0]) * max(
        0, person_box[3] - person_box[1]
    )
    cloth_area = max(0, cloth_box[2] - cloth_box[0]) * max(
        0, cloth_box[3] - cloth_box[1]
    )
    iou = inter / (person_area + cloth_area - inter + 1e-6)
    cloth_coverage = inter / (cloth_area + 1e-6)
    cx = (cloth_box[0] + cloth_box[2]) / 2.0
    cy = (cloth_box[1] + cloth_box[3]) / 2.0
    center_inside = (
        person_box[0] <= cx <= person_box[2]
        and person_box[1] <= cy <= person_box[3]
    )
    return inter, iou, cloth_coverage, center_inside


def _association_score(person_box, cloth_box):
    inter, iou, cloth_coverage, center_inside = _overlap_metrics(
        person_box, cloth_box
    )
    if inter <= 0:
        return 0.0, iou, cloth_coverage, inter
    center_bonus = 0.05 if center_inside else 0.0
    return cloth_coverage + iou + center_bonus, iou, cloth_coverage, inter


def _is_clothing(class_name):
    """只保留主体衣物类。"""
    return _clothing_region(class_name) is not None


def _is_wearable(class_name):
    """保留用于人物描述的衣物和配饰类。"""
    return _clothing_region(class_name) is not None or _accessory_region(class_name)


def _clothing_region(class_name):
    name = class_name.lower()
    if "person" in name:
        return None
    if any(keyword in name for keyword in WHOLE_BODY_LABEL_KEYWORDS):
        return "whole"
    if any(keyword in name for keyword in UPPER_LABEL_KEYWORDS):
        return "upper"
    if any(keyword in name for keyword in LOWER_LABEL_KEYWORDS):
        return "lower"
    if any(keyword in name for keyword in FOOTWEAR_LABEL_KEYWORDS):
        return "footwear"
    return None


def _accessory_region(class_name):
    name = class_name.lower()
    if any(keyword in name for keyword in ACCESSORY_LABEL_KEYWORDS):
        return "accessory"
    return None


def _item_region(class_name):
    return _clothing_region(class_name) or _accessory_region(class_name)


def _load_opencv_model(model_type, model_path):
    if model_type == "svm":
        return cv2.ml.SVM_load(str(model_path))
    if model_type == "rtrees":
        return cv2.ml.RTrees_load(str(model_path))
    if model_type == "boost":
        return cv2.ml.Boost_load(str(model_path))
    raise RuntimeError(f"unknown OpenCV model type: {model_type}")


def _resolve_model_asset(path_value, default_name):
    """Resolve training-time absolute paths to package-local runtime assets."""
    env_dir = os.environ.get("CADE_CLOTH_COLOR_MODEL_DIR")
    search_dir = Path(env_dir).expanduser() if env_dir else COLOR_SVM_DIR
    path = Path(path_value) if path_value else Path(default_name)
    if path.exists():
        return path

    package_path = search_dir / path.name
    if package_path.exists():
        return package_path
    return search_dir / default_name


def _load_color_model_bundle():
    global _COLOR_MODEL_BUNDLE, _COLOR_MODEL_LOAD_ERROR
    if _COLOR_MODEL_BUNDLE is not None:
        return _COLOR_MODEL_BUNDLE, None
    if _COLOR_MODEL_LOAD_ERROR is not None:
        return None, _COLOR_MODEL_LOAD_ERROR

    try:
        if COLOR_MODEL_META_PATH.exists():
            meta = json.loads(COLOR_MODEL_META_PATH.read_text(encoding="utf-8"))
            model_type = meta["model_type"]
            model_path = _resolve_model_asset(meta.get("model_path"), "color_model.xml")
            norm_path = _resolve_model_asset(
                meta.get("norm_path"), "color_model_norm.npz"
            )
            label_path = _resolve_model_asset(
                meta.get("label_path"), "label_names.json"
            )
        else:
            model_type = "rtrees"
            model_path = COLOR_MODEL_PATH
            norm_path = COLOR_MODEL_NORM_PATH
            label_path = COLOR_SVM_LABEL_PATH
            meta = {
                "model_type": model_type,
                "feature_version": "lab_hsv_v2",
                "feature_dim": FEATURE_DIM,
            }

        missing = [str(path) for path in (model_path, norm_path, label_path) if not path.exists()]
        if missing:
            raise FileNotFoundError("missing color model files: " + ", ".join(missing))

        model = _load_opencv_model(model_type, model_path)
        if model is None or not model.isTrained():
            raise RuntimeError(f"failed to load trained color model: {model_path}")

        norm = np.load(str(norm_path))
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)
        expected_dim = int(meta.get("feature_dim", FEATURE_DIM))
        if mean.shape[0] != expected_dim or std.shape[0] != expected_dim:
            raise RuntimeError(
                f"unexpected SVM norm shape: mean={mean.shape}, std={std.shape}"
            )
        if not labels:
            raise RuntimeError(f"empty color label file: {label_path}")

        _COLOR_MODEL_BUNDLE = {
            "model": model,
            "model_type": model_type,
            "model_path": str(model_path),
            "norm_path": str(norm_path),
            "label_path": str(label_path),
            "mean": mean,
            "std": std,
            "labels": labels,
            "meta": meta,
        }
        return _COLOR_MODEL_BUNDLE, None
    except Exception as exc:
        _COLOR_MODEL_LOAD_ERROR = str(exc)
        return None, _COLOR_MODEL_LOAD_ERROR


def _classify_color(bgr_values):
    bgr_pixels = np.asarray(bgr_values, dtype=np.uint8).reshape(-1, 3)
    if len(bgr_pixels) == 0:
        return "unknown", {"mode": "opencv_color_model", "reason": "empty_crop"}

    bundle, error = _load_color_model_bundle()
    if error:
        return "unknown", {
            "mode": "opencv_color_model",
            "reason": "model_load_failed",
            "error": error,
        }

    try:
        features, feature_debug = extract_color_features(bgr_pixels, return_debug=True)
        features = features.reshape(1, -1)
    except Exception as exc:
        return "unknown", {
            "mode": "opencv_color_model",
            "reason": "feature_failed",
            "error": str(exc),
            "total_pixels": int(len(bgr_pixels)),
        }

    x = (features - bundle["mean"]) / bundle["std"]
    _, raw_pred = bundle["model"].predict(x.astype(np.float32))
    label_index = int(raw_pred.reshape(-1)[0])
    labels = bundle["labels"]
    color = labels[label_index] if 0 <= label_index < len(labels) else "unknown"

    debug = {
        "mode": "opencv_color_model",
        "model_type": bundle["model_type"],
        "model_path": bundle["model_path"],
        "norm_path": bundle["norm_path"],
        "label_path": bundle["label_path"],
        "model_feature_version": bundle["meta"].get("feature_version", "unknown"),
        "labels": labels,
        "raw_prediction": label_index,
        "selected_color": color,
        "feature_dim": int(features.shape[1]),
    }
    debug.update(feature_debug)
    return color, debug


def classify_color_pixels(bgr_values, with_debug=False):
    """Classify a BGR pixel array with the trained OpenCV color model."""
    color, debug = _classify_color(bgr_values)
    return (color, debug) if with_debug else color


def _polygon_mask(shape, polygon):
    if not polygon:
        return None
    points = np.asarray(polygon, dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return None
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def _extract_color(
    color_image,
    bbox,
    mask_polygon=None,
    class_name="",
    source_model="",
    with_debug=False,
):
    """优先从 segmentation mask 内提取颜色，鞋子无 mask 时使用完整 bbox。"""
    x1, y1, x2, y2 = bbox
    h, w = color_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        debug = {"reason": "invalid_bbox", "bbox": [int(x1), int(y1), int(x2), int(y2)]}
        return ("unknown", debug) if with_debug else "unknown"

    crop = color_image[y1:y2, x1:x2]
    if mask_polygon:
        mask = _polygon_mask(color_image.shape, mask_polygon)
        if mask is not None:
            crop_mask = mask[y1:y2, x1:x2] > 0
            mask_pixels = crop[crop_mask]
            if len(mask_pixels) >= COLOR_MIN_PIXELS:
                color, debug = _classify_color(mask_pixels)
                debug.update(
                    {
                        "sample_mode": "mask_polygon",
                        "class_name": class_name,
                        "source_model": source_model,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "crop_shape": [int(crop.shape[0]), int(crop.shape[1])],
                        "mask_pixels": int(len(mask_pixels)),
                    }
                )
                return (color, debug) if with_debug else color

    ch, cw = crop.shape[:2]
    if _clothing_region(class_name) == "footwear":
        center = crop
        sample_mode = "bbox_full"
        inset_x = inset_y = 0
    else:
        inset_x = int(cw * 0.12)
        inset_y = int(ch * 0.12)
        if inset_x * 2 >= cw or inset_y * 2 >= ch:
            inset_x = inset_y = 0
        center = crop[inset_y:ch - inset_y, inset_x:cw - inset_x]
        sample_mode = "bbox_center"
    if center.size == 0:
        debug = {"reason": "empty_center_crop", "crop_shape": [int(ch), int(cw)]}
        return ("unknown", debug) if with_debug else "unknown"

    color, debug = _classify_color(center)
    debug.update(
        {
            "sample_mode": sample_mode,
            "class_name": class_name,
            "source_model": source_model,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "crop_shape": [int(ch), int(cw)],
            "center_shape": [int(center.shape[0]), int(center.shape[1])],
        }
    )
    return (color, debug) if with_debug else color


def extract_color_from_region(
    color_image,
    bbox,
    mask_polygon=None,
    class_name="",
    source_model="",
    with_debug=False,
):
    """Classify color from an image region using the runtime extraction policy."""
    return _extract_color(
        color_image,
        bbox,
        mask_polygon=mask_polygon,
        class_name=class_name,
        source_model=source_model,
        with_debug=with_debug,
    )


def _greedy_assignment(score_matrix):
    candidates = []
    for row_idx in range(score_matrix.shape[0]):
        for col_idx in range(score_matrix.shape[1]):
            score = score_matrix[row_idx, col_idx]
            if score > 0:
                candidates.append((score, row_idx, col_idx))

    assigned_rows = set()
    assigned_cols = set()
    rows = []
    cols = []
    for _, row_idx, col_idx in sorted(candidates, reverse=True):
        if row_idx in assigned_rows or col_idx in assigned_cols:
            continue
        assigned_rows.add(row_idx)
        assigned_cols.add(col_idx)
        rows.append(row_idx)
        cols.append(col_idx)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _hungarian_assignment(score_matrix):
    if score_matrix.size == 0 or np.max(score_matrix) <= 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    if linear_sum_assignment is None:
        return _greedy_assignment(score_matrix)

    # linear_sum_assignment minimizes cost, so negate the positive match score.
    return linear_sum_assignment(-score_matrix)


def _build_cloth_summary(items):
    if not items:
        return "unknown"
    parts = []
    for item in items:
        color = item.get("color", "unknown")
        ctype = item.get("type", "unknown")
        if color and color != "unknown":
            parts.append(f"{color} {ctype}")
        else:
            parts.append(ctype)
    return "; ".join(parts)


def _cached_extract_color(color_image, cloth, color_cache=None, max_cache_size=256):
    bbox = cloth["bbox"]
    # Quantize the key so small frame-to-frame bbox jitter reuses the same
    # color decision for the same garment.
    quantized_bbox = tuple(int(round(float(value) / 12.0) * 12) for value in bbox)
    mask_area = int(round(float(cloth.get("mask_area", 0.0) or 0.0) / 256.0) * 256)
    cache_key = (
        cloth.get("source_model", ""),
        cloth.get("class_name", ""),
        quantized_bbox,
        mask_area,
    )

    if color_cache is not None and cache_key in color_cache:
        color, debug = color_cache[cache_key]
        if hasattr(color_cache, "move_to_end"):
            color_cache.move_to_end(cache_key)
        debug = dict(debug)
        debug["cache_hit"] = True
        return color, debug

    color, debug = _extract_color(
        color_image,
        bbox,
        mask_polygon=cloth.get("mask_polygon"),
        class_name=cloth.get("class_name", ""),
        source_model=cloth.get("source_model", ""),
        with_debug=True,
    )
    debug = dict(debug)
    debug["cache_hit"] = False

    if color_cache is not None:
        color_cache[cache_key] = (color, dict(debug))
        if hasattr(color_cache, "move_to_end"):
            color_cache.move_to_end(cache_key)
        while len(color_cache) > max_cache_size:
            if hasattr(color_cache, "popitem"):
                try:
                    color_cache.popitem(last=False)
                    continue
                except TypeError:
                    pass
            first_key = next(iter(color_cache))
            color_cache.pop(first_key, None)
    return color, debug


def associate_clothing(
    detections,
    color_image,
    color_cache=None,
    max_color_cache_size=256,
    perf=None,
):
    """
    主入口。将衣物框匹配到人框，提取颜色和类型。

    Args:
        detections: [obj_info, ...]  含 class_name, bbox，可选 mask_polygon
        color_image: BGR numpy array（原始帧）

    修改 detections 中的 person 对象，增加 cloth_color / cloth_type 字段。
    """
    persons = [d for d in detections if d["class_name"] == "person"]
    cloths = [d for d in detections if _is_wearable(d["class_name"])]

    if not persons or not cloths:
        for p in persons:
            p.setdefault("cloth_color", "unknown")
            p.setdefault("cloth_type", "unknown")
            p.setdefault("cloth_items", [])
            p.setdefault("cloth_summary", "unknown")
        return

    # 主体衣物按区域做匈牙利匹配：同一件衣物最多匹配到一个人，
    # 同一个人可以同时匹配上衣、下装、鞋等不同区域。
    matches = []  # [(person_idx, cloth_idx, score), ...]
    for region in CLOTHING_REGION_ORDER:
        region_cloth_indices = [
            ci
            for ci, cloth in enumerate(cloths)
            if _clothing_region(cloth["class_name"]) == region
        ]
        if not region_cloth_indices:
            continue

        score_matrix = np.zeros((len(region_cloth_indices), len(persons)), dtype=float)
        metric_map = {}
        for row_idx, ci in enumerate(region_cloth_indices):
            cloth = cloths[ci]
            for col_idx, person in enumerate(persons):
                score, iou, coverage, inter = _association_score(
                    person["bbox"], cloth["bbox"]
                )
                score_matrix[row_idx, col_idx] = score
                metric_map[(row_idx, col_idx)] = (iou, coverage, inter)

        row_indices, col_indices = _hungarian_assignment(score_matrix)
        for row_idx, col_idx in zip(row_indices, col_indices):
            score = score_matrix[row_idx, col_idx]
            if score <= 0:
                continue

            ci = region_cloth_indices[row_idx]
            cloth = cloths[ci]
            iou, coverage, inter = metric_map[(row_idx, col_idx)]
            cloth["associated_person_index"] = int(col_idx)
            cloth["association_region"] = region
            cloth["association_score"] = float(score)
            cloth["association_iou"] = float(iou)
            cloth["association_coverage"] = float(coverage)
            cloth["association_intersection"] = float(inter)
            matches.append((int(col_idx), ci, float(score)))

    # 配饰可以有多个，逐个挂到重叠分数最高的人。
    accessory_indices = [
        ci for ci, cloth in enumerate(cloths) if _accessory_region(cloth["class_name"])
    ]
    for ci in accessory_indices:
        cloth = cloths[ci]
        best = None
        for pi, person in enumerate(persons):
            score, iou, coverage, inter = _association_score(
                person["bbox"], cloth["bbox"]
            )
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, pi, iou, coverage, inter)
        if best is None:
            continue
        score, pi, iou, coverage, inter = best
        cloth["associated_person_index"] = int(pi)
        cloth["association_region"] = "accessory"
        cloth["association_score"] = float(score)
        cloth["association_iou"] = float(iou)
        cloth["association_coverage"] = float(coverage)
        cloth["association_intersection"] = float(inter)
        matches.append((int(pi), ci, float(score)))

    person_items = {pi: [] for pi in range(len(persons))}
    for pi, ci, score in matches:
        cloth = cloths[ci]
        region = _item_region(cloth["class_name"])
        color_start = time.perf_counter()
        color, color_debug = _cached_extract_color(
            color_image,
            cloth,
            color_cache=color_cache,
            max_cache_size=max_color_cache_size,
        )
        if perf is not None:
            perf["cloth_color_ms"] = perf.get("cloth_color_ms", 0.0) + (
                time.perf_counter() - color_start
            ) * 1000.0
        cloth["color"] = color
        cloth["color_debug"] = color_debug
        person_items[pi].append(
            {
                "region": region,
                "type": cloth["class_name"],
                "color": color,
                "color_debug": color_debug,
                "confidence": cloth.get("confidence"),
                "bbox": cloth.get("bbox"),
                "source_model": cloth.get("source_model"),
                "score": score,
            }
        )

    # 写入 person 对象
    for pi, person in enumerate(persons):
        items = sorted(
            person_items.get(pi, []),
            key=lambda item: REGION_ORDER.index(item["region"]),
        )
        person["cloth_items"] = items
        person["cloth_summary"] = _build_cloth_summary(items)

        if not items:
            person.setdefault("cloth_color", "unknown")
            person.setdefault("cloth_type", "unknown")
            continue

        person["cloth_color"] = "/".join(item["color"] for item in items)
        person["cloth_type"] = "/".join(item["type"] for item in items)
