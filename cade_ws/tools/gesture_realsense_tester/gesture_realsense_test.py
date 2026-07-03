#!/usr/bin/env python3
"""
Standalone RealSense gesture tester.

This file intentionally vendors the gesture rule logic from
cade_vision.kits.posture_gesture so the module can be edited and tested
independently from ROS and the main CADE vision pipeline.
"""

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:
    raise SystemExit(f"mediapipe import failed: {exc}") from exc

try:
    import pyrealsense2 as rs
except Exception as exc:
    raise SystemExit(f"pyrealsense2 import failed: {exc}") from exc


LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20


def get_index_pose(landmarks, side):
    idx = LEFT_INDEX if side == "left" else RIGHT_INDEX
    if idx < len(landmarks) and landmarks[idx][2] > 0.3:
        return get_landmark(landmarks, idx)
    return None


def get_landmark(landmarks, idx):
    return np.array([landmarks[idx][0], landmarks[idx][1]])


def is_visible(landmarks, idx):
    return idx < len(landmarks) and landmarks[idx][2] > 0.5


def is_visible_relaxed(landmarks, idx):
    return idx < len(landmarks) and landmarks[idx][2] > 0.3


def compute_side_angles(landmarks, side, elbow_idx, wrist_idx, fine_wrist, fine_index):
    if not is_visible_relaxed(landmarks, elbow_idx) or not is_visible_relaxed(
        landmarks, wrist_idx
    ):
        return None, None

    elbow = get_landmark(landmarks, elbow_idx)
    wrist = get_landmark(landmarks, wrist_idx)

    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    elbow_angle = math.degrees(math.atan2(dy, dx))

    wrist_angle = None
    if fine_wrist[side] is not None and fine_index[side] is not None:
        dx = fine_index[side][0] - fine_wrist[side][0]
        dy = fine_index[side][1] - fine_wrist[side][1]
        wrist_angle = math.degrees(math.atan2(dy, dx))

    return elbow_angle, wrist_angle


def is_arm_raised(wrist, elbow, shoulder, shoulder_dist):
    margin = shoulder_dist * 0.2
    if wrist[1] > shoulder[1] + margin:
        return False

    horiz_ok = abs(wrist[0] - elbow[0]) < shoulder_dist * 0.3
    vert_ok = wrist[1] < elbow[1]
    return horiz_ok and vert_ok


def is_pointing(wrist, elbow, index_pose, side, hands_data, shoulder_dist):
    arrow_y = shoulder_dist * 0.25
    aligned = (
        abs(elbow[1] - wrist[1]) < arrow_y
        and abs(wrist[1] - index_pose[1]) < arrow_y
    )
    if not aligned:
        return False, None

    if not hands_data:
        return False, None

    hd = hands_data.get("Left" if side == "left" else "Right")
    if not hd or not hd.get("landmarks") or len(hd["landmarks"]) < 21:
        return False, None

    landmarks = hd["landmarks"]

    def tip_mcp_dist(mcp_idx):
        tip = np.array(landmarks[mcp_idx + 3])
        mcp = np.array(landmarks[mcp_idx])
        return np.linalg.norm(tip - mcp)

    ref_len = np.linalg.norm(np.array(landmarks[6]) - np.array(landmarks[5])) * 2.5
    if ref_len < 1e-6:
        ref_len = 0.1

    index_ext = tip_mcp_dist(5) > ref_len * 0.75
    middle_ret = tip_mcp_dist(9) < ref_len * 0.50
    ring_ret = tip_mcp_dist(13) < ref_len * 0.50
    pinky_ret = tip_mcp_dist(17) < ref_len * 0.50
    thumb_ret = (
        np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[2]))
        < ref_len * 0.50
    )

    pattern_a = index_ext and middle_ret and ring_ret and pinky_ret and thumb_ret
    pattern_b = index_ext and ring_ret and pinky_ret and thumb_ret
    if not (pattern_a or pattern_b):
        return False, None

    direction = "right" if elbow[0] < wrist[0] else "left"
    return True, direction


def classify_static_gesture(landmarks, h, w, hands_data=None):
    fine_wrist = {"left": None, "right": None}
    fine_index = {"left": None, "right": None}
    if hands_data:
        for side_key, side_lower in [("Left", "left"), ("Right", "right")]:
            hd = hands_data.get(side_key)
            if hd and hd.get("wrist") and hd.get("index_tip"):
                fine_wrist[side_lower] = np.array(hd["wrist"])
                fine_index[side_lower] = np.array(hd["index_tip"])

    elbow_l, wrist_l = compute_side_angles(
        landmarks, "left", LEFT_ELBOW, LEFT_WRIST, fine_wrist, fine_index
    )
    elbow_r, wrist_r = compute_side_angles(
        landmarks, "right", RIGHT_ELBOW, RIGHT_WRIST, fine_wrist, fine_index
    )

    key_points = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    ]
    if not all(is_visible(landmarks, i) for i in key_points):
        return "unknown", elbow_l, elbow_r, wrist_l, wrist_r

    left_shoulder = get_landmark(landmarks, LEFT_SHOULDER)
    right_shoulder = get_landmark(landmarks, RIGHT_SHOULDER)
    left_elbow = get_landmark(landmarks, LEFT_ELBOW)
    right_elbow = get_landmark(landmarks, RIGHT_ELBOW)
    left_wrist = get_landmark(landmarks, LEFT_WRIST)
    right_wrist = get_landmark(landmarks, RIGHT_WRIST)

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist < 1e-6:
        return "unknown", elbow_l, elbow_r, wrist_l, wrist_r

    left_raised = is_arm_raised(left_wrist, left_elbow, left_shoulder, shoulder_dist)
    right_raised = is_arm_raised(
        right_wrist, right_elbow, right_shoulder, shoulder_dist
    )

    left_index_mcp = get_index_pose(landmarks, "left")
    right_index_mcp = get_index_pose(landmarks, "right")

    left_pointing, left_dir = (
        is_pointing(
            left_wrist,
            left_elbow,
            left_index_mcp,
            "left",
            hands_data,
            shoulder_dist,
        )
        if left_index_mcp is not None
        else (False, None)
    )
    right_pointing, right_dir = (
        is_pointing(
            right_wrist,
            right_elbow,
            right_index_mcp,
            "right",
            hands_data,
            shoulder_dist,
        )
        if right_index_mcp is not None
        else (False, None)
    )

    if left_raised and right_raised:
        return "maybe_raising_both_arms", elbow_l, elbow_r, wrist_l, wrist_r
    if left_raised and not right_raised:
        return "maybe_raising_left_arm", elbow_l, elbow_r, wrist_l, wrist_r
    if right_raised and not left_raised:
        return "maybe_raising_right_arm", elbow_l, elbow_r, wrist_l, wrist_r

    if left_pointing and right_pointing:
        return "maybe_pointing_both", elbow_l, elbow_r, wrist_l, wrist_r
    if left_pointing and not right_pointing:
        return f"maybe_pointing_{left_dir}", elbow_l, elbow_r, wrist_l, wrist_r
    if right_pointing and not left_pointing:
        return f"maybe_pointing_{right_dir}", elbow_l, elbow_r, wrist_l, wrist_r

    return "none", elbow_l, elbow_r, wrist_l, wrist_r


def check_shoulder_proximity(landmarks, shoulder_offset):
    if len(landmarks) <= RIGHT_INDEX:
        return False

    def visible(idx):
        return landmarks[idx][2] > 0.5

    ok_left = False
    if visible(LEFT_SHOULDER) and visible(LEFT_INDEX):
        ok_left = landmarks[LEFT_INDEX][1] <= landmarks[LEFT_SHOULDER][1] + shoulder_offset

    ok_right = False
    if visible(RIGHT_SHOULDER) and visible(RIGHT_INDEX):
        ok_right = landmarks[RIGHT_INDEX][1] <= landmarks[RIGHT_SHOULDER][1] + shoulder_offset

    return ok_left or ok_right


def judge_all_temporal_gestures(raw_gesture, variances, landmarks, thresholds):
    max_fa = max(variances["fa_l"], variances["fa_r"])
    max_wr = max(variances["wr_l"], variances["wr_r"])

    static_threshold = 50.0
    left_is_static = variances["fa_l"] < static_threshold and variances["wr_l"] < static_threshold
    right_is_static = variances["fa_r"] < static_threshold and variances["wr_r"] < static_threshold

    rule_a_waving = max_fa > thresholds["T_FOREARM"]
    rule_b_waving = max_wr > thresholds["T_WRIST"]
    shoulder_ok = check_shoulder_proximity(landmarks, thresholds["SHOULDER_OFFSET"])

    if (rule_a_waving or rule_b_waving) and shoulder_ok:
        return "waving", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "waving_override",
        }

    if raw_gesture == "maybe_raising_left_arm" and left_is_static:
        return "raising_left_arm", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "stable_raising",
        }
    if raw_gesture == "maybe_raising_right_arm" and right_is_static:
        return "raising_right_arm", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "stable_raising",
        }
    if raw_gesture == "maybe_raising_both_arms" and left_is_static and right_is_static:
        return "raising_both_arms", {
            "max_fa": max_fa,
            "max_wr": max_wr,
            "decision": "stable_raising",
        }

    if raw_gesture == "maybe_pointing_left" and left_is_static:
        return "pointing_left", {"decision": "stable_pointing"}
    if raw_gesture == "maybe_pointing_right" and right_is_static:
        return "pointing_right", {"decision": "stable_pointing"}
    if raw_gesture == "maybe_pointing_both" and left_is_static and right_is_static:
        return "pointing_both", {"decision": "stable_pointing"}

    maybe_gestures = {
        "maybe_raising_left_arm",
        "maybe_raising_right_arm",
        "maybe_raising_both_arms",
        "maybe_pointing_left",
        "maybe_pointing_right",
        "maybe_pointing_both",
    }
    if raw_gesture in maybe_gestures:
        return "none", {"decision": "filtered_by_shaking"}

    return "none", {"decision": "static_none"}


class RingBuffer:
    MIN_POST_JUMP = 15
    MAX_CONSECUTIVE_NONE = 5

    def __init__(self, capacity=30):
        self.capacity = capacity
        self.forearm = defaultdict(lambda: {"left": [], "right": []})
        self.wrist_rot = defaultdict(lambda: {"left": [], "right": []})
        self._none = defaultdict(lambda: defaultdict(lambda: {"left": 0, "right": 0}))

    def _push_side(self, source_name, store, person_id, side, angle):
        if angle is not None:
            store[person_id][side].append(angle)
            if len(store[person_id][side]) > self.capacity:
                store[person_id][side].pop(0)
            self._none[person_id][source_name][side] = 0
        else:
            self._none[person_id][source_name][side] += 1
            if self._none[person_id][source_name][side] >= self.MAX_CONSECUTIVE_NONE:
                store[person_id][side].clear()

    def push_forearm(self, person_id, angle_left=None, angle_right=None):
        self._push_side("forearm", self.forearm, person_id, "left", angle_left)
        self._push_side("forearm", self.forearm, person_id, "right", angle_right)

    def push_wrist(self, person_id, angle_left=None, angle_right=None):
        self._push_side("wrist", self.wrist_rot, person_id, "left", angle_left)
        self._push_side("wrist", self.wrist_rot, person_id, "right", angle_right)

    @staticmethod
    def _jump_aware_variance(angles, jump_threshold, window=30):
        if len(angles) < 2:
            return 0.0, 0

        recent = angles[-min(window, len(angles)) :]
        cut_idx = 0
        for i in range(len(recent) - 1, 0, -1):
            diff = abs(recent[i] - recent[i - 1])
            if diff > jump_threshold:
                cut_idx = i
                break

        post_jump = recent[cut_idx:]
        if len(post_jump) < RingBuffer.MIN_POST_JUMP:
            return 0.0, len(post_jump)

        return float(np.var(post_jump)), len(post_jump)

    def get_forearm_variance(self, person_id, side="left", window=30, jump_threshold=None):
        angles = self.forearm[person_id][side]
        if not angles:
            return 0.0
        if jump_threshold is not None:
            variance, _ = self._jump_aware_variance(angles, jump_threshold, window)
            return variance
        if len(angles) < max(5, window // 2):
            return 0.0
        recent = angles[-min(window, len(angles)) :]
        return float(np.var(recent))

    def get_wrist_variance(self, person_id, side="left", window=30, jump_threshold=None):
        angles = self.wrist_rot[person_id][side]
        if not angles:
            return 0.0
        if jump_threshold is not None:
            variance, _ = self._jump_aware_variance(angles, jump_threshold, window)
            return variance
        if len(angles) < max(5, window // 2):
            return 0.0
        recent = angles[-min(window, len(angles)) :]
        return float(np.var(recent))

    def clear(self):
        self.forearm.clear()
        self.wrist_rot.clear()
        self._none.clear()


class GestureTester:
    def __init__(self, args):
        self.args = args
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        self.ring_buffer = RingBuffer(capacity=args.buffer_size)

        self.pose = self.mp_pose.Pose(
            static_image_mode=args.pose_static,
            model_complexity=args.pose_complexity,
            enable_segmentation=False,
            smooth_landmarks=not args.pose_static,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=args.hand_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

    def close(self):
        self.pose.close()
        self.hands.close()

    def process_pose(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks is None:
            return None, results
        landmarks = [
            (lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark
        ]
        return {
            "landmarks": landmarks,
            "img_h": image.shape[0],
            "img_w": image.shape[1],
        }, results

    def process_hands(self, image):
        output = {"Left": None, "Right": None}
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks is None or results.multi_handedness is None:
            return output, results

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx]
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            if score < self.args.min_detection_confidence:
                continue
            output[label] = {
                "wrist": (hand_lms.landmark[0].x, hand_lms.landmark[0].y),
                "index_tip": (hand_lms.landmark[8].x, hand_lms.landmark[8].y),
                "landmarks": [(lm.x, lm.y) for lm in hand_lms.landmark],
                "score": float(score),
            }
        return output, results

    def analyze(self, image):
        pose_data, pose_results = self.process_pose(image)
        hands_data, hands_results = self.process_hands(image)
        if pose_data is None:
            return {
                "raw_gesture": "unknown",
                "gesture": "unknown",
                "decision": "no_pose",
                "pose_results": pose_results,
                "hands_results": hands_results,
                "hands_data": hands_data,
            }

        landmarks = pose_data["landmarks"]
        raw, elbow_l, elbow_r, wrist_l, wrist_r = classify_static_gesture(
            landmarks,
            pose_data["img_h"],
            pose_data["img_w"],
            hands_data,
        )

        person_id = 0
        self.ring_buffer.push_forearm(person_id, elbow_l, elbow_r)
        self.ring_buffer.push_wrist(person_id, wrist_l, wrist_r)
        variances = {
            "fa_l": self.ring_buffer.get_forearm_variance(
                person_id, "left", jump_threshold=self.args.jump_threshold
            ),
            "fa_r": self.ring_buffer.get_forearm_variance(
                person_id, "right", jump_threshold=self.args.jump_threshold
            ),
            "wr_l": self.ring_buffer.get_wrist_variance(
                person_id, "left", jump_threshold=self.args.jump_threshold
            ),
            "wr_r": self.ring_buffer.get_wrist_variance(
                person_id, "right", jump_threshold=self.args.jump_threshold
            ),
        }
        thresholds = {
            "T_FOREARM": self.args.t_forearm,
            "T_WRIST": self.args.t_wrist,
            "SHOULDER_OFFSET": self.args.shoulder_offset,
        }
        final, debug = judge_all_temporal_gestures(
            raw_gesture=raw,
            variances=variances,
            landmarks=landmarks,
            thresholds=thresholds,
        )

        return {
            "raw_gesture": raw,
            "gesture": final,
            "decision": debug.get("decision", ""),
            "elbow_l": elbow_l,
            "elbow_r": elbow_r,
            "wrist_l": wrist_l,
            "wrist_r": wrist_r,
            "variances": variances,
            "pose_data": pose_data,
            "pose_results": pose_results,
            "hands_results": hands_results,
            "hands_data": hands_data,
            **debug,
        }


def draw_pose_overlay(image, pose_data, connections):
    if not pose_data:
        return
    h, w = image.shape[:2]
    points = {}
    for idx, lm in enumerate(pose_data["landmarks"]):
        x = int(np.clip(lm[0] * w, 0, w - 1))
        y = int(np.clip(lm[1] * h, 0, h - 1))
        visibility = lm[2]
        color = (0, 255, 0) if visibility > 0.8 else (0, 0, 255)
        points[idx] = (x, y, visibility, color)

    for start, end in connections:
        if start not in points or end not in points:
            continue
        sx, sy, sv, _ = points[start]
        ex, ey, ev, _ = points[end]
        color = (0, 255, 0) if sv > 0.8 and ev > 0.8 else (0, 0, 255)
        cv2.line(image, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)

    for _, (x, y, visibility, color) in points.items():
        cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{visibility:.1f}",
            (x + 4, max(8, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{visibility:.1f}",
            (x + 4, max(8, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_hands_overlay(image, hands_data):
    if not hands_data:
        return
    h, w = image.shape[:2]
    for label, data in hands_data.items():
        if not data:
            continue
        color = (255, 180, 0) if label == "Left" else (255, 0, 180)
        for x_norm, y_norm in data.get("landmarks", []):
            x = int(np.clip(x_norm * w, 0, w - 1))
            y = int(np.clip(y_norm * h, 0, h - 1))
            cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)
        wx, wy = data["wrist"]
        ix, iy = data["index_tip"]
        cv2.line(
            image,
            (int(wx * w), int(wy * h)),
            (int(ix * w), int(iy * h)),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (int(wx * w), int(wy * h) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_status(image, result, fps):
    variances = result.get("variances", {})
    lines = [
        f"final: {result.get('gesture', 'unknown')}",
        f"raw: {result.get('raw_gesture', 'unknown')}  decision: {result.get('decision', '')}",
        (
            "var fa L/R: "
            f"{variances.get('fa_l', 0.0):.1f}/{variances.get('fa_r', 0.0):.1f}  "
            "wr L/R: "
            f"{variances.get('wr_l', 0.0):.1f}/{variances.get('wr_r', 0.0):.1f}"
        ),
        f"fps: {fps:.1f}   q: quit   r: reset buffer   s: screenshot",
    ]
    y = 26
    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def open_realsense(args):
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    return pipeline, profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="", help="RealSense serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", action="store_true", help="Mirror the image before inference")
    parser.add_argument("--pose-static", action="store_true", help="Use MediaPipe static image mode for Pose")
    parser.add_argument("--pose-complexity", type=int, default=1)
    parser.add_argument("--hand-complexity", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument("--t-forearm", type=float, default=300.0)
    parser.add_argument("--t-wrist", type=float, default=500.0)
    parser.add_argument("--shoulder-offset", type=float, default=0.15)
    parser.add_argument("--jump-threshold", type=float, default=50.0)
    parser.add_argument("--window-name", default="Gesture RealSense Tester")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline, _ = open_realsense(args)
    tester = GestureTester(args)
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, args.width, args.height)

    last_time = time.time()
    fps = 0.0
    screenshot_dir = Path.home() / "Desktop" / "gesture_realsense_tester" / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            if args.flip:
                image = cv2.flip(image, 1)

            result = tester.analyze(image)
            display = image.copy()
            draw_pose_overlay(display, result.get("pose_data"), tester.pose_connections)
            draw_hands_overlay(display, result.get("hands_data"))

            now = time.time()
            dt = max(1e-6, now - last_time)
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            draw_status(display, result, fps)

            cv2.imshow(args.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                tester.ring_buffer.clear()
                print("Ring buffer cleared")
            if key == ord("s"):
                path = screenshot_dir / f"gesture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), display)
                print(f"Saved screenshot: {path}")
    finally:
        tester.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
