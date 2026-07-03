"""Shared CLI for open vision gateway and worker scripts."""

import argparse

from cade_vision.runtime.model_paths import preferred_model_path


def build_arg_parser(description="CADE Open Vision Node"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        type=str,
        default=preferred_model_path("yolo11n.pt"),
        help="Person/general YOLO model path",
    )
    parser.add_argument(
        "--cloth-model",
        type=str,
        default=None,
        help="Deprecated alias for --cloth-detect-model",
    )
    parser.add_argument(
        "--cloth-detect-model",
        type=str,
        default=None,
        help="Optional Fashionpedia detect model path; disabled by default",
    )
    parser.add_argument(
        "--cloth-seg-model",
        type=str,
        default=preferred_model_path("yolov8s-seg-fashionpedia-best.pt"),
        help="Fashionpedia segmentation model path",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=320,
        help="Person YOLO inference size",
    )
    parser.add_argument(
        "--person-min-box-area-ratio",
        type=float,
        default=0.015,
        help="Drop person boxes smaller than this frame area ratio",
    )
    parser.add_argument(
        "--cloth-fusion-iou",
        type=float,
        default=0.5,
        help="Detect/seg clothing fusion overlap threshold",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--image-source",
        type=str,
        default="realsense",
        choices=["realsense", "file", "usb_cam"],
        help="Image source",
    )
    parser.add_argument("--image-path", type=str, default="", help="Image/video path")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop file video")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="File video playback speed multiplier",
    )
    parser.add_argument("--usb-cam-id", type=int, default=0, help="USB camera id")
    parser.add_argument("--width", type=int, default=640, help="USB camera width")
    parser.add_argument("--height", type=int, default=480, help="USB camera height")
    parser.add_argument(
        "--serial-number",
        type=str,
        default="333422301212",
        help="RealSense serial number",
    )
    parser.add_argument(
        "--people-tracks-rate",
        type=float,
        default=10.0,
        help="Publish rate in Hz for /vision/people_tracks_task3; <=0 disables it",
    )
    parser.add_argument(
        "--gesture-track-iou-threshold",
        type=float,
        default=0.25,
        help="Person IoU threshold for gesture temporal tracking",
    )
    parser.add_argument(
        "--gesture-max-track-missing-frames",
        type=int,
        default=15,
        help="Missing person frames kept for gesture temporal tracking",
    )
    parser.add_argument(
        "--gesture-min-crop-size",
        type=int,
        default=64,
        help="Minimum expanded person crop size for MediaPipe gesture inference",
    )
    parser.add_argument("--gesture-debug", action="store_true", help="Print gesture diagnostics")
    parser.add_argument("--gesture-debug-interval", type=int, default=15)
    parser.add_argument("--perf-debug", action="store_true", help="Print timing diagnostics")
    parser.add_argument("--perf-debug-interval", type=int, default=15)
    parser.add_argument(
        "--idle-person-rate",
        type=float,
        default=0.0,
        help="Person YOLO rate in idle profile; 0 disables idle person detection",
    )
    parser.add_argument(
        "--worker-warm-ttl",
        type=float,
        default=20.0,
        help="Seconds to keep inactive model workers warm after a task profile ends",
    )
    parser.add_argument(
        "--prewarm-workers",
        type=str,
        default="all",
        help="Comma-separated workers to keep loaded while idle: none, person, pose_gesture, cloth, all",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=20.0,
        help="Maximum OpenCV display refresh FPS",
    )
    parser.add_argument("--display", action="store_true", default=True, help="Show detection window")
    parser.add_argument("--no-display", action="store_false", dest="display", help="Hide detection window")
    parser.add_argument(
        "--no-launch-workers",
        action="store_true",
        help="Run only the gateway; useful when workers are launched manually",
    )
    return parser
