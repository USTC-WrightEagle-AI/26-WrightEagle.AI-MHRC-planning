"""Vision gateway: task/profile control, fusion, display, and public topics."""

import copy
import json
import threading
import time

import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from cade_vision.analyzer import Analyzer

    ANALYZER_AVAILABLE = True
except Exception:
    Analyzer = None
    ANALYZER_AVAILABLE = False

from cade_vision.debug.overlays import draw_detections, draw_perf_overlay
from cade_vision.kits.cloth import renumber_detections
from cade_vision.runtime.detections import bbox_iou, set_default_cloth_fields
from cade_vision.runtime.json_utils import json_clean
from cade_vision.runtime.profiles import (
    ProfileState,
    profile_from_command,
    profile_has_cloth,
    profile_has_pose,
    profile_ttl_for_command,
)
from cade_vision.runtime.ros_frames import image_to_numpy
from cade_vision.tasks import ObserveTask
from cade_vision.tasks.base_task import BaseTask


class VisionGateway:
    def __init__(self, args, worker_manager=None):
        rospy.init_node("cade_open_vision", anonymous=True)
        self.args = args
        self.worker_manager = worker_manager
        self.image_source = getattr(args, "image_source", "realsense")
        self.people_tracks_rate = float(getattr(args, "people_tracks_rate", 10.0) or 0.0)
        self._last_people_tracks_pub_time = 0.0
        self.display_fps = max(0.0, float(getattr(args, "display_fps", 20.0) or 0.0))
        self._last_display_time = 0.0
        self._last_display_sequence = 0
        self._profile = ProfileState(default_ttl=3.0)
        self._lock = threading.Lock()
        self._result_lock = threading.Lock()
        self.detected_objects = []
        self.current_task_executor = None
        self.task_active = False
        self.task_id = None
        self.task_attributes = None
        self._last_finished_request_id = None
        self.target_class = None
        self._task_thread = None
        self._latest_color_msg = None
        self._results = {}
        self._worker_last_end = {}
        self._processed_frame_index = 0
        self._loop_fps_smooth = 0.0
        self._last_perf_snapshot = {}
        self._result_max_age_sec = 1.0
        self._cloth_result_max_age_sec = 2.0
        self._last_nonempty_people = None
        self._last_nonempty_people_updated_at = 0.0
        self._last_nonempty_people_max_age_sec = 0.45

        self.analyzer = None
        if ANALYZER_AVAILABLE:
            try:
                self.analyzer = Analyzer()
            except Exception as exc:
                print(f"Analyzer init failed: {exc}")

        self._task_mapping = {
            "observe_people": ObserveTask,
            "find_people": ObserveTask,
            "count_people": ObserveTask,
            "observe_objects": ObserveTask,
        }

        self.profile_pub = rospy.Publisher("/vision/profile_task3", String, queue_size=1, latch=True)
        self.detections_3d_pub = rospy.Publisher("/vision/detections_3d_task3", String, queue_size=10)
        self.people_tracks_pub = rospy.Publisher("/vision/people_tracks_task3", String, queue_size=10)
        self.task_status_pub = rospy.Publisher("/cade/task_status_task3", String, queue_size=10)
        rospy.Subscriber("/cade/task_cmd_task3", String, self._on_task_cmd, queue_size=10)
        rospy.Subscriber("/vision/frame/color_task3", Image, self._on_color, queue_size=1)
        rospy.Subscriber("/vision/person_raw_task3", String, self._on_worker_result("person"), queue_size=1)
        rospy.Subscriber("/vision/pose_gesture_raw_task3", String, self._on_worker_result("pose_gesture"), queue_size=1)
        rospy.Subscriber("/vision/cloth_raw_task3", String, self._on_worker_result("cloth"), queue_size=1)

        if args.display:
            cv2.namedWindow("CADE Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CADE Vision", 800, 600)

        self._publish_profile("idle", reason="startup")
        self._print_startup()

    def _print_startup(self):
        print("OpenVision gateway initialized")
        print(f"  Image Source: {self.image_source}")
        print("  Inference: multi-process task-driven workers")
        print(f"  Prewarm workers: {getattr(self.args, 'prewarm_workers', 'all')}")
        print(f"  Idle person rate: {float(getattr(self.args, 'idle_person_rate', 0.0) or 0.0):.2f} Hz")
        print(f"  Display FPS cap: {self.display_fps if self.args.display else 0.0:.1f}")
        print(f"  Display: {self.args.display}")
        print("  Listening: /cade/task_cmd_task3")
        print("  Publishing: /vision/detections_3d_task3, /vision/people_tracks_task3")

    def _on_color(self, msg):
        with self._result_lock:
            self._latest_color_msg = msg

    def _on_worker_result(self, name):
        def callback(msg):
            try:
                payload = json.loads(msg.data)
            except Exception as exc:
                print(f"[Vision] invalid {name} result: {exc}")
                return
            now = time.time()
            with self._result_lock:
                last_end = self._worker_last_end.get(name)
                fps = 0.0 if last_end is None else 1.0 / max(1e-6, now - last_end)
                self._worker_last_end[name] = now
                payload["worker"] = name
                payload["updated_at"] = now
                payload["fps"] = fps
                self._results[name] = payload

        return callback

    def _publish_profile(self, profile=None, reason=""):
        if profile is None:
            profile = self._profile.profile
        msg = String()
        msg.data = json.dumps(
            {"profile": profile, "stamp": time.time(), "reason": reason},
            ensure_ascii=False,
        )
        self.profile_pub.publish(msg)

    def _clear_perception_cache(self):
        with self._lock:
            self.detected_objects = []
        with self._result_lock:
            self._results.clear()
        self._last_nonempty_people = None
        self._last_nonempty_people_updated_at = 0.0

    def _set_profile(self, profile, reason="", ttl=None, clear_cache=True):
        previous, current, changed = self._profile.set(profile, ttl=ttl)
        if clear_cache:
            self._clear_perception_cache()
        self._publish_profile(current, reason=reason)
        if self.worker_manager is not None:
            self.worker_manager.set_profile(current)
        if changed:
            print(f"[Vision] profile {previous} -> {current}{f' ({reason})' if reason else ''}")
        elif reason:
            print(f"[Vision] profile {current} refreshed ({reason})")

    def _expire_profile_if_needed(self):
        previous, current, changed = self._profile.expire_if_needed(self.task_active)
        if changed:
            self._clear_perception_cache()
            self._publish_profile(current, reason="expired")
            if self.worker_manager is not None:
                self.worker_manager.set_profile(current)
            print(f"[Vision] profile {previous} -> {current} (expired)")

    def _on_task_cmd(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            print(f"[Vision] Invalid task cmd JSON: {exc}")
            return
        action = cmd.get("action", "")
        request_id = cmd.get("request_id")
        task_cls = self._task_mapping.get(action)
        if task_cls is None:
            print(f"[Vision] Unknown action: {action}")
            self._publish_status(
                "FAILED",
                error=f"Unknown action: {action}",
                request_id=request_id,
            )
            return

        executor = task_cls(self)
        executor.request_id = request_id
        with self._lock:
            old_executor = self.current_task_executor
            if old_executor is not None and old_executor.should_continue():
                old_executor.cancel()
            self.current_task_executor = executor
            self.task_active = True
            self.task_id = action
            self.task_attributes = self._extract_task_attributes(action, cmd)
            self.target_class = self._target_for_command(action, cmd)
            attrs = self.task_attributes
            target = self.target_class

        profile = profile_from_command(action, cmd, attrs)
        self._set_profile(
            profile,
            reason=action,
            ttl=profile_ttl_for_command(cmd, self._profile.default_ttl),
            clear_cache=True,
        )
        print(f"\n[Vision Task] {action}: target='{target}' attrs={attrs} profile={profile}")
        self._task_thread = threading.Thread(
            target=self._run_task_executor,
            args=(executor, cmd),
            daemon=True,
        )
        self._task_thread.start()

    def _run_task_executor(self, executor, cmd):
        try:
            executor.execute(cmd)
        except Exception as exc:
            if self.finish_task(executor):
                self._publish_status(
                    "FAILED",
                    error=str(exc),
                    request_id=getattr(executor, "request_id", None),
                )

    def _extract_task_attributes(self, action, cmd):
        attributes = {}
        raw_attributes = cmd.get("attributes")
        if isinstance(raw_attributes, dict):
            attributes.update(raw_attributes)
        keys = [
            key
            for key in BaseTask.PERSON_ATTRIBUTE_KEYS
            if key in cmd and key not in attributes
        ]
        for key in keys:
            attributes.setdefault(key, cmd.get(key))
        normalized = {
            key: value
            for key, value in attributes.items()
            if value is not None and value != "" and value != "unknown"
        }
        return normalized or None

    @staticmethod
    def _target_for_command(action, cmd):
        if action in ("observe_people", "find_people", "count_people"):
            return "person"
        if action == "observe_objects":
            target = cmd.get("target") or cmd.get("category")
            return str(target).lower() if target else None
        return None

    def get_latest_detections(self):
        with self._lock:
            return copy.deepcopy(self.detected_objects)

    def has_fresh_worker_result(self, name, max_age=None):
        now = time.time()
        with self._result_lock:
            result = self._results.get(name)
            if result is None:
                return False
            updated_at = float(result.get("updated_at", 0.0) or 0.0)
        if updated_at <= 0.0:
            return False
        max_age = self._result_max_age_sec if max_age is None else float(max_age)
        return now - updated_at <= max_age

    def finish_task(self, executor):
        with self._lock:
            if self.current_task_executor is not executor:
                return False
            self._last_finished_request_id = getattr(executor, "request_id", None)
            self.task_active = False
            self.target_class = None
            self.task_id = None
            self.task_attributes = None
            self.current_task_executor = None
        self._set_profile("idle", reason="task_done", clear_cache=False)
        return True

    def _reset_task(self):
        with self._lock:
            executor = self.current_task_executor
            if executor is not None:
                executor.cancel()
            self.task_active = False
            self.target_class = None
            self.task_id = None
            self.task_attributes = None
            self.current_task_executor = None
        self._set_profile("idle", reason="reset", clear_cache=True)

    def _publish_detection(self, detection):
        msg = String()
        msg.data = json.dumps(detection, ensure_ascii=False)
        self.detections_3d_pub.publish(msg)

    def _publish_status(self, status, result=None, error=None, request_id="__auto__"):
        if request_id == "__auto__":
            request_id = self._last_finished_request_id
        status_msg = {"status": status}
        if request_id:
            status_msg["request_id"] = request_id
        if result:
            status_msg["result"] = result
        if error:
            status_msg["error"] = error
        print(f"[Vision] Task status: {status_msg}")
        msg = String()
        msg.data = json.dumps(status_msg, ensure_ascii=False)
        self.task_status_pub.publish(msg)
        if request_id == self._last_finished_request_id:
            self._last_finished_request_id = None

    def _fresh_result(self, name, now, max_age=None):
        result = self._results.get(name)
        if result is None:
            return None
        age = now - float(result.get("updated_at", 0.0) or 0.0)
        if age > (self._result_max_age_sec if max_age is None else max_age):
            return None
        return result

    def _stabilize_people(self, people, now):
        if people:
            self._last_nonempty_people = copy.deepcopy(people)
            self._last_nonempty_people_updated_at = now
            return people
        if self._last_nonempty_people is None:
            return people
        age = now - float(self._last_nonempty_people_updated_at or 0.0)
        if age > self._last_nonempty_people_max_age_sec:
            return people
        stabilized = copy.deepcopy(self._last_nonempty_people)
        for obj in stabilized:
            obj["person_hold_age_sec"] = float(age)
            obj["person_hold"] = True
        return stabilized

    def _apply_cloth_result(self, people, cloth_result):
        if cloth_result is None:
            for person in people:
                set_default_cloth_fields(person)
            return []
        person_cloth = copy.deepcopy(cloth_result.get("person_cloth", []) or [])
        by_track_id = {
            item.get("track_id"): item
            for item in person_cloth
            if item.get("track_id") is not None
        }
        unmatched = list(person_cloth)
        for person in people:
            match = None
            track_id = person.get("track_id")
            if track_id is not None:
                match = by_track_id.get(track_id)
            if match is None:
                best_iou = 0.0
                best_item = None
                for item in unmatched:
                    iou = bbox_iou(person.get("bbox"), item.get("bbox"))
                    if iou > best_iou:
                        best_iou = iou
                        best_item = item
                if best_iou >= 0.2:
                    match = best_item
            if match is None:
                set_default_cloth_fields(person)
                continue
            for key in ("cloth_color", "cloth_type", "cloth_items", "cloth_summary"):
                person[key] = match.get(key, [] if key == "cloth_items" else "unknown")
        return copy.deepcopy(cloth_result.get("cloth_detections", []) or [])

    def _worker_perf(self, now):
        perf = {}
        for name, result in self._results.items():
            updated_at = float(result.get("updated_at", 0.0) or 0.0)
            stamp = float(result.get("stamp", 0.0) or 0.0)
            count_key = "detections" if name != "cloth" else "cloth_detections"
            perf[name] = {
                "fps": float(result.get("fps", 0.0) or 0.0),
                "duration_ms": float(result.get("duration_ms", 0.0) or 0.0),
                "result_age_sec": max(0.0, now - updated_at) if updated_at else 0.0,
                "frame_age_sec": max(0.0, now - stamp) if stamp else 0.0,
                "sequence": int(result.get("sequence", 0) or 0),
                "count": len(result.get(count_key, []) or []),
                "stage_ms": dict(result.get("stage_ms", {}) or {}),
                "cache_size": int(result.get("cache_size", 0) or 0),
            }
        return perf

    def _update_perf_snapshot(self, loop_ms, frame_stamp, people, commit=True):
        now = time.time()
        frame_age_sec = max(0.0, now - frame_stamp) if frame_stamp else 0.0
        loop_fps = 1000.0 / loop_ms if loop_ms > 1e-6 else 0.0
        if not commit:
            loop_fps_smooth = self._loop_fps_smooth or loop_fps
        elif self._loop_fps_smooth <= 0.0:
            self._loop_fps_smooth = loop_fps
            loop_fps_smooth = self._loop_fps_smooth
        else:
            self._loop_fps_smooth = 0.85 * self._loop_fps_smooth + 0.15 * loop_fps
            loop_fps_smooth = self._loop_fps_smooth
        workers = self._worker_perf(now)
        gesture_fps = workers.get("pose_gesture", {}).get("fps", 0.0)
        pose_age = workers.get("pose_gesture", {}).get("result_age_sec", 0.0)
        max_history_frames = 0
        max_history_span = 0.0
        max_waving_probability = 0.0
        waving_statuses = []
        for obj in people:
            max_history_frames = max(max_history_frames, int(obj.get("waving_history_frames", 0) or 0))
            max_history_span = max(max_history_span, float(obj.get("waving_history_span_sec", 0.0) or 0.0))
            max_waving_probability = max(max_waving_probability, float(obj.get("waving_probability", 0.0) or 0.0))
            status = obj.get("waving_status")
            if status:
                waving_statuses.append(str(status))
        profile = self._profile.profile
        snapshot = {
            "frame_index": int(self._processed_frame_index),
            "vision_profile": profile,
            "frame_age_sec": float(frame_age_sec),
            "loop_fps": float(loop_fps_smooth),
            "gesture_fps": float(gesture_fps),
            "person_count": int(len(people)),
            "waving_history_frames": int(max_history_frames),
            "waving_history_span_sec": float(max_history_span),
            "waving_probability": float(max_waving_probability),
            "waving_status": ",".join(sorted(set(waving_statuses))) or "no_person",
            "waving_under_sampled": bool(profile_has_pose(profile) and people and gesture_fps > 0.0 and gesture_fps < 7.0),
            "pose_result_stale": bool(profile_has_pose(profile) and people and pose_age > 0.35),
            "workers": workers,
            "worker_processes": (
                self.worker_manager.states()
                if self.worker_manager is not None
                else {}
            ),
        }
        if commit:
            self._last_perf_snapshot = snapshot
        return snapshot

    def _maybe_print_gesture_debug(self, people):
        if not getattr(self.args, "gesture_debug", False):
            return
        interval = max(1, int(getattr(self.args, "gesture_debug_interval", 15) or 15))
        if self._processed_frame_index % interval:
            return
        for obj in people:
            print(
                "[GestureDebug] "
                f"frame={self._processed_frame_index} "
                f"id={obj.get('track_id')} "
                f"gesture={obj.get('gesture', 'unknown')} "
                f"static={obj.get('gesture_static', 'unknown')} "
                f"wave={obj.get('gesture_waving', 'unknown')}/"
                f"{obj.get('gesture_waving_voted', 'unknown')} "
                f"p={float(obj.get('waving_probability', 0.0) or 0.0):.3f} "
                f"status={obj.get('waving_status', 'unknown')} "
                f"hist={obj.get('waving_history_frames', 0)}/"
                f"{float(obj.get('waving_history_span_sec', 0.0) or 0.0):.2f}s "
                f"vote={obj.get('gesture_vote_count', 0)}/"
                f"{obj.get('gesture_vote_total', 0)} "
                f"vis={float(obj.get('gesture_mean_visibility', 0.0) or 0.0):.2f}"
            )

    def _maybe_print_perf_debug(self, snapshot):
        if not getattr(self.args, "perf_debug", False):
            return
        interval = max(1, int(getattr(self.args, "perf_debug_interval", 15) or 15))
        if self._processed_frame_index % interval:
            return
        worker_parts = []
        for name in ("person", "pose_gesture", "cloth"):
            item = snapshot.get("workers", {}).get(name, {})
            proc = snapshot.get("worker_processes", {}).get(name, {})
            worker_parts.append(
                f"{name}=fps{item.get('fps', 0.0):.1f}/"
                f"{item.get('duration_ms', 0.0):.1f}ms/"
                f"age{item.get('result_age_sec', 0.0):.2f}s/"
                f"n{item.get('count', 0)}/"
                f"{proc.get('state', 'unknown')}"
            )
        warnings = []
        if snapshot.get("waving_under_sampled"):
            warnings.append("waving_window_under_sampled")
        if snapshot.get("pose_result_stale"):
            warnings.append("pose_result_stale")
        warning = " WARNING=" + ",".join(warnings) if warnings else ""
        print(
            "[Perf] "
            f"frame={snapshot.get('frame_index')} "
            f"profile={snapshot.get('vision_profile', 'idle')} "
            f"fps={snapshot.get('loop_fps', 0.0):.1f} "
            f"gesture_fps={snapshot.get('gesture_fps', 0.0):.1f} "
            f"age={snapshot.get('frame_age_sec', 0.0):.2f}s "
            f"wave={snapshot.get('waving_status')} "
            f"hist={snapshot.get('waving_history_frames')}/"
            f"{snapshot.get('waving_history_span_sec', 0.0):.2f}s "
            f"p={snapshot.get('waving_probability', 0.0):.3f} "
            f"{' '.join(worker_parts)}{warning}"
        )

    def _publish_people_tracks(self, detections, snapshot=None):
        if self.people_tracks_rate <= 0.0:
            return
        now = time.time()
        min_interval = 1.0 / self.people_tracks_rate
        if now - self._last_people_tracks_pub_time < min_interval:
            return
        self._last_people_tracks_pub_time = now
        people = []
        for obj in detections:
            if obj.get("class_name", "").lower() != "person":
                continue
            people.append(
                {
                    "track_id": json_clean(obj.get("track_id")),
                    "index": json_clean(obj.get("index")),
                    "confidence": json_clean(obj.get("confidence")),
                    "bbox": json_clean(obj.get("bbox")),
                    "center": json_clean(obj.get("center")),
                    "position_3d": json_clean(obj.get("position_3d")),
                    "posture": obj.get("posture", "unknown"),
                    "gesture": obj.get("gesture", "unknown"),
                    "gesture_raw": obj.get("gesture_raw", "unknown"),
                    "gesture_static": obj.get("gesture_static", "unknown"),
                    "gesture_static_voted": obj.get("gesture_static_voted", "unknown"),
                    "gesture_waving": obj.get("gesture_waving", "unknown"),
                    "gesture_waving_voted": obj.get("gesture_waving_voted", "unknown"),
                    "gesture_waving_confidence": json_clean(obj.get("gesture_waving_confidence", 0.0)),
                    "waving_probability": json_clean(obj.get("waving_probability", 0.0)),
                    "waving_status": obj.get("waving_status", "unknown"),
                    "waving_history_frames": json_clean(obj.get("waving_history_frames", 0)),
                    "waving_history_span_sec": json_clean(obj.get("waving_history_span_sec", 0.0)),
                    "waving_sampled": bool(obj.get("waving_sampled", False)),
                    "gesture_vote_count": json_clean(obj.get("gesture_vote_count", 0)),
                    "gesture_vote_total": json_clean(obj.get("gesture_vote_total", 0)),
                    "gesture_vote_ratio": json_clean(obj.get("gesture_vote_ratio", 0.0)),
                    "gesture_vote_stable": bool(obj.get("gesture_vote_stable", False)),
                    "gesture_status": obj.get("gesture_status", "unknown"),
                    "gesture_missing_landmark_count": json_clean(obj.get("gesture_missing_landmark_count", 0)),
                    "gesture_mean_visibility": json_clean(obj.get("gesture_mean_visibility", 0.0)),
                    "cloth_summary": obj.get("cloth_summary", "unknown"),
                    "cloth_color": obj.get("cloth_color", "unknown"),
                    "cloth_type": obj.get("cloth_type", "unknown"),
                    "cloth_items": json_clean(obj.get("cloth_items", [])),
                }
            )
        msg = String()
        msg.data = json.dumps(
            {
                "stamp": now,
                "count": len(people),
                "people": people,
                "perf": json_clean(snapshot or self._last_perf_snapshot),
            },
            ensure_ascii=False,
        )
        self.people_tracks_pub.publish(msg)

    def _compose_once(self):
        self._expire_profile_if_needed()
        if self.worker_manager is not None:
            self.worker_manager.maintain()
        loop_start = time.perf_counter()
        now = time.time()
        profile = self._profile.profile
        display_due = (
            self.args.display
            and self.display_fps > 0.0
            and now - self._last_display_time >= 1.0 / self.display_fps
        )
        with self._result_lock:
            results = copy.deepcopy(self._results)
            color_msg = self._latest_color_msg if display_due else None
            color_stamp = (
                self._latest_color_msg.header.stamp.to_sec()
                if self._latest_color_msg is not None
                else now
            )
        pose_result = self._fresh_result_from(results, "pose_gesture", now) if profile_has_pose(profile) else None
        person_result = self._fresh_result_from(results, "person", now)
        if pose_result is not None:
            people = copy.deepcopy(pose_result.get("detections", []) or [])
        elif person_result is not None:
            people = copy.deepcopy(person_result.get("detections", []) or [])
        else:
            people = []
        people = self._stabilize_people(people, now)
        cloth_result = (
            self._fresh_result_from(results, "cloth", now, self._cloth_result_max_age_sec)
            if profile_has_cloth(profile)
            else None
        )
        cloth_detections = self._apply_cloth_result(people, cloth_result)
        new_detections = renumber_detections(people + cloth_detections)
        people = [obj for obj in new_detections if obj.get("class_name") == "person"]
        self._maybe_print_gesture_debug(people)
        with self._lock:
            self.detected_objects = copy.deepcopy(new_detections)
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        snapshot = self._update_perf_snapshot(loop_ms, color_stamp, people, commit=True)
        self._publish_people_tracks(new_detections, snapshot=snapshot)
        if display_due and color_msg is not None:
            try:
                color_image = image_to_numpy(color_msg)
            except Exception:
                color_image = None
        else:
            color_image = None
        if color_image is not None:
            with self._lock:
                target = self.target_class
            draw_detections(color_image, new_detections, target_class=target)
            draw_perf_overlay(color_image, snapshot)
            cv2.imshow("CADE Vision", color_image)
            self._last_display_time = now
            self._last_display_sequence = int(color_msg.header.seq)
        self._maybe_print_perf_debug(snapshot)
        self._processed_frame_index += 1

    def _fresh_result_from(self, results, name, now, max_age=None):
        result = results.get(name)
        if result is None:
            return None
        age = now - float(result.get("updated_at", 0.0) or 0.0)
        if age > (self._result_max_age_sec if max_age is None else max_age):
            return None
        return result

    def run(self):
        print("CADE Vision gateway running...")
        print("Listening on /cade/task_cmd_task3")
        print("Publishing detections to /vision/detections_3d_task3")
        rate_hz = max(
            5.0,
            min(
                60.0,
                max(
                    self.display_fps if self.args.display else 0.0,
                    self.people_tracks_rate,
                    10.0,
                ),
            ),
        )
        rate = rospy.Rate(rate_hz)
        try:
            while not rospy.is_shutdown():
                self._compose_once()
                if self.args.display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                rate.sleep()
        finally:
            self._reset_task()
            if self.args.display:
                cv2.destroyAllWindows()
