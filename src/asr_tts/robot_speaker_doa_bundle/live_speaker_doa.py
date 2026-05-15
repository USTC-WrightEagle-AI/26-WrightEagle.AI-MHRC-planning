import argparse
import json
import os
import queue
import shutil
import select
import sys
import threading
import time
import wave
from contextlib import contextmanager

from m2_mic_array import M2MicArray
from speaker_manager import SpeakerManager


def pcm_to_wav(pcm_path, wav_path, sample_rate=16000, channels=1, sample_width=2):
    with open(pcm_path, "rb") as src:
        data = src.read()
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(int(sample_width))
        wf.setframerate(int(sample_rate))
        wf.writeframes(data)
    return wav_path


def format_result(payload):
    return json.dumps(payload, ensure_ascii=False)


@contextmanager
def terminal_cbreak():
    if not sys.stdin.isatty():
        yield
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def poll_key(timeout=0.0):
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


class KeyMonitor:
    def __init__(self):
        self.events = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if not sys.stdin.isatty():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ch = poll_key(timeout=0.2)
            if ch is not None:
                self.events.put(ch)

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.events.get_nowait())
            except Exception:
                return items

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


def next_capture_paths(base_dir, prefix):
    os.makedirs(base_dir, exist_ok=True)
    token = f"{prefix}_{int(time.time() * 1000)}"
    out_dir = os.path.join(base_dir, token)
    os.makedirs(out_dir, exist_ok=True)
    return {
        "out_dir": out_dir,
        "origin_pcm": os.path.join(out_dir, "origin.pcm"),
        "iat_pcm": os.path.join(out_dir, "iat.pcm"),
        "iat_wav": os.path.join(out_dir, "iat.wav"),
    }


def build_session_db_path(work_dir, db_arg):
    if db_arg:
        return os.path.abspath(db_arg), True
    os.makedirs(work_dir, exist_ok=True)
    return os.path.join(os.path.abspath(work_dir), "session_speakers_db.pkl"), False


def default_work_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir")


def cleanup_capture_dir(capture_kind, out_dir):
    if capture_kind == "detect" and os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Robot speaker recognition + DOA")
    parser.add_argument("--serial-device", default="auto")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--db", default=None, help="optional persistent speaker db path")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--work-dir", default=default_work_dir())
    parser.add_argument("--detect-seconds", type=float, default=0.5)
    parser.add_argument("--enroll-seconds", type=float, default=5.0)
    parser.add_argument("--active-channels", default="0,1,2,3,4,5")
    parser.add_argument("--geometry", choices=["circle6", "line6"], default="circle6")
    parser.add_argument("--num-mics", type=int, default=6)
    parser.add_argument("--radius-m", type=float, default=0.035)
    parser.add_argument("--spacing-m", type=float, default=0.028)
    parser.add_argument("--segment-seconds", type=float, default=0.5)
    parser.add_argument("--angle-step-deg", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-enroll-speech-seconds", type=float, default=0.8)
    parser.add_argument("--min-detect-speech-seconds", type=float, default=0.15)
    args = parser.parse_args()

    db_path, persistent_db = build_session_db_path(args.work_dir, args.db)
    if os.path.exists(db_path) and not persistent_db:
        os.remove(db_path)

    model_dir = os.path.join(os.path.abspath(args.work_dir), "pretrained_models", "spkrec-ecapa-voxceleb")
    mic = M2MicArray(serial_device=args.serial_device, baudrate=args.baud)
    mgr = SpeakerManager(db_path=db_path, threshold=args.threshold, model_dir=model_dir)

    print(
        format_result(
            {
                "status": "ready",
                "db": db_path,
                "persistent_db": persistent_db,
                "detect_seconds": args.detect_seconds,
                "enroll_seconds": args.enroll_seconds,
                "threshold": args.threshold,
                "model_dir": model_dir,
            }
        ),
        flush=True,
    )

    with terminal_cbreak():
        key_monitor = KeyMonitor()
        key_monitor.start()
        try:
            while True:
                keys = key_monitor.drain()
                if any(k in ("q", "Q") for k in keys):
                    print(format_result({"status": "stop"}), flush=True)
                    break

                enroll_mode = any(k in ("t", "T") for k in keys)
                capture_seconds = args.enroll_seconds if enroll_mode else args.detect_seconds
                capture_kind = "enroll" if enroll_mode else "detect"
                paths = next_capture_paths(args.work_dir, capture_kind)

                if enroll_mode:
                    print(
                        format_result(
                            {"status": "enroll_requested", "message": "收到按键 T，立即开始录入 5 秒说话人样本。"}
                        ),
                        flush=True,
                    )

                print(
                    format_result(
                        {
                            "status": "capturing",
                            "mode": capture_kind,
                            "seconds": capture_seconds,
                            "out_dir": paths["out_dir"],
                        }
                    ),
                    flush=True,
                )

                try:
                    try:
                        dumped = mic.dump_origin_audio(paths["out_dir"], seconds=capture_seconds)
                        pcm_to_wav(dumped["iat_pcm"], paths["iat_wav"], sample_rate=16000, channels=1, sample_width=2)
                    except Exception as exc:
                        print(format_result({"status": "error", "mode": capture_kind, "detail": str(exc)}), flush=True)
                        continue

                    speech_check = mgr.detect_speech(
                        paths["iat_wav"],
                        min_speech_seconds=args.min_enroll_speech_seconds if enroll_mode else args.min_detect_speech_seconds,
                    )

                    if not speech_check["speech_like"]:
                        status = "enroll_rejected_non_speech" if enroll_mode else "skip_non_speech"
                        print(
                            format_result(
                                {
                                    "status": status,
                                    "mode": capture_kind,
                                    "reason": speech_check["reason"],
                                    "speech_stats": speech_check,
                                }
                            ),
                            flush=True,
                        )
                        continue

                    if enroll_mode:
                        try:
                            res = mgr.enroll(paths["iat_wav"])
                        except Exception as exc:
                            print(format_result({"status": "error", "mode": "enroll", "detail": str(exc)}), flush=True)
                            continue
                        print(
                            format_result(
                                {
                                    "status": "enrolled",
                                    "speaker_id": res.get("speaker_id"),
                                    "action": res.get("action"),
                                    "score": res.get("score"),
                                }
                            ),
                            flush=True,
                        )
                        continue

                    try:
                        cls = mgr.classify(paths["iat_wav"])
                    except Exception as exc:
                        print(format_result({"status": "error", "mode": "classify", "detail": str(exc)}), flush=True)
                        continue

                    if cls.get("error") == "empty_db":
                        print(format_result({"status": "empty_db", "message": "当前还没有录入说话人。按 T 开始录入 5 秒样本。"}), flush=True)
                        continue

                    speaker_id = cls.get("speaker_id")
                    score = cls.get("score")
                    if not speaker_id:
                        print(format_result({"status": "no_match", "score": score}), flush=True)
                        continue

                    if args.score_threshold is not None and score is not None and score < args.score_threshold:
                        print(format_result({"status": "below_score_threshold", "speaker_id": speaker_id, "score": score}), flush=True)
                        continue

                    try:
                        doa = mic.estimate_doa_from_origin(
                            dumped["origin_pcm"],
                            active_channels=args.active_channels,
                            geometry=args.geometry,
                            num_mics=args.num_mics,
                            radius_m=args.radius_m,
                            spacing_m=args.spacing_m,
                            angle_offset_deg=0.0,
                            segment_seconds=args.segment_seconds,
                            angle_step_deg=args.angle_step_deg,
                        )
                    except Exception as exc:
                        print(
                            format_result(
                                {
                                    "status": "recognized_no_doa",
                                    "speaker_id": speaker_id,
                                    "score": score,
                                    "detail": str(exc),
                                }
                            ),
                            flush=True,
                        )
                        continue

                    print(
                        format_result(
                            {
                                "status": "recognized",
                                "speaker_id": speaker_id,
                                "speaker_name": speaker_id,
                                "score": score,
                                "angle_deg": doa.get("estimated_angle_deg"),
                            }
                        ),
                        flush=True,
                    )
                finally:
                    cleanup_capture_dir(capture_kind, paths["out_dir"])
        finally:
            key_monitor.stop()


if __name__ == "__main__":
    main()
