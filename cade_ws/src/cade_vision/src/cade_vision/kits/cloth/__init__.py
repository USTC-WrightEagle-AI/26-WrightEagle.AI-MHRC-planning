"""Cloth recognition and person association for CADE vision."""

from .cloth_analyzer import (
    associate_clothing,
    classify_color_pixels,
    extract_color_from_region,
)
from .cloth_fusion import (
    filter_fashion_detections,
    fuse_fashion_detections,
    renumber_detections,
)

__all__ = [
    "associate_clothing",
    "classify_color_pixels",
    "extract_color_from_region",
    "filter_fashion_detections",
    "fuse_fashion_detections",
    "renumber_detections",
]
