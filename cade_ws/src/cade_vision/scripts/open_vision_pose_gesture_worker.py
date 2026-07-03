#!/usr/bin/env python3

import sys
from pathlib import Path

_SOURCE_PYTHON = Path(__file__).resolve().parents[1] / "src"
if _SOURCE_PYTHON.exists():
    sys.path.insert(0, str(_SOURCE_PYTHON))

from cade_vision.runtime.cli import build_arg_parser


def main():
    args = build_arg_parser("CADE Vision Pose/Gesture Worker").parse_args()
    from cade_vision.workers.pose_gesture import PoseGestureWorker

    PoseGestureWorker(args).run()


if __name__ == "__main__":
    main()
