"""Fusion helpers for Fashionpedia detect + segmentation outputs."""

PRIMARY_UPPER_KEYWORDS = {
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
PRIMARY_LOWER_KEYWORDS = {
    "pants",
    "trousers",
    "shorts",
    "skirt",
    "tights",
    "stockings",
}
PRIMARY_WHOLE_KEYWORDS = {
    "dress",
    "jumpsuit",
}
PRIMARY_FOOTWEAR_KEYWORDS = {
    "shoe",
    "sock",
    "leg warmer",
}

EXCLUDED_KEYWORDS = {
    "bag",
    "wallet",
    "umbrella",
}


def is_excluded_fashion_class(class_name):
    name = class_name.lower()
    return any(keyword in name for keyword in EXCLUDED_KEYWORDS)


def fusion_region(class_name):
    """Return the coarse body region used only for detect-vs-seg de-duplication."""
    name = class_name.lower()
    if any(keyword in name for keyword in PRIMARY_WHOLE_KEYWORDS):
        return "whole"
    if any(keyword in name for keyword in PRIMARY_UPPER_KEYWORDS):
        return "upper"
    if any(keyword in name for keyword in PRIMARY_LOWER_KEYWORDS):
        return "lower"
    if any(keyword in name for keyword in PRIMARY_FOOTWEAR_KEYWORDS):
        return "footwear"
    return None


def _box_overlap(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return {
        "iou": inter / (union + 1e-6),
        "a_coverage": inter / (area_a + 1e-6),
        "b_coverage": inter / (area_b + 1e-6),
    }


def _regions_compatible(region_a, region_b):
    if region_a == region_b:
        return True
    if region_a == "whole" and region_b in {"upper", "lower"}:
        return True
    if region_b == "whole" and region_a in {"upper", "lower"}:
        return True
    return False


def _is_overridden_by_seg(detect_obj, seg_objects, iou_threshold):
    detect_region = fusion_region(detect_obj["class_name"])
    if detect_region is None:
        return False

    for seg_obj in seg_objects:
        seg_region = fusion_region(seg_obj["class_name"])
        if seg_region is None or not _regions_compatible(detect_region, seg_region):
            continue
        overlap = _box_overlap(detect_obj["bbox"], seg_obj["bbox"])
        if (
            overlap["iou"] >= iou_threshold
            or overlap["a_coverage"] >= iou_threshold
            or overlap["b_coverage"] >= iou_threshold
        ):
            return True
    return False


def filter_fashion_detections(detections):
    return [
        obj
        for obj in detections
        if not is_excluded_fashion_class(obj.get("class_name", ""))
    ]


def fuse_fashion_detections(seg_detections, detect_detections, iou_threshold=0.5):
    """Prefer seg for overlapping primary garments and keep detect-only details."""
    seg_kept = filter_fashion_detections(seg_detections)
    detect_kept = []
    for obj in filter_fashion_detections(detect_detections):
        if _is_overridden_by_seg(obj, seg_kept, iou_threshold):
            continue
        detect_kept.append(obj)
    return seg_kept + detect_kept


def renumber_detections(detections):
    for index, obj in enumerate(detections):
        obj["index"] = index
    return detections
