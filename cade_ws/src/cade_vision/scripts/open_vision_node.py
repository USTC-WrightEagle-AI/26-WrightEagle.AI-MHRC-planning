#!/usr/bin/env python3
"""CADE vision gateway launcher.

This script intentionally stays thin. Runtime logic lives under
``cade_vision.workers`` and shared helpers under ``cade_vision.runtime``.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_SOURCE_PYTHON = Path(__file__).resolve().parents[1] / "src"
if _SOURCE_PYTHON.exists():
    sys.path.insert(0, str(_SOURCE_PYTHON))

from cade_vision.runtime.cli import build_arg_parser


WORKER_SCRIPTS = {
    "frame_source": "open_vision_frame_source.py",
    "person": "open_vision_person_worker.py",
    "pose_gesture": "open_vision_pose_gesture_worker.py",
    "cloth": "open_vision_cloth_worker.py",
}
PROFILE_WORKERS = {
    "idle": set(),
    "person": {"person"},
    "gesture": {"person", "pose_gesture"},
    "posture": {"person", "pose_gesture"},
    "cloth": {"person", "cloth"},
    "full": {"person", "pose_gesture", "cloth"},
}
MODEL_WORKERS = {"person", "pose_gesture", "cloth"}


def _parse_worker_set(value):
    raw = str(value or "all").strip().lower()
    if raw in {"", "none", "false", "0", "off"}:
        return set()
    if raw == "all":
        return set(MODEL_WORKERS)
    aliases = {
        "gesture": "pose_gesture",
        "pose": "pose_gesture",
        "posture": "pose_gesture",
    }
    workers = set()
    for item in raw.replace(";", ",").replace("|", ",").split(","):
        name = aliases.get(item.strip(), item.strip())
        if name in MODEL_WORKERS:
            workers.add(name)
    return workers


class WorkerLauncher:
    def __init__(self, args, argv):
        self.args = args
        self.argv = [arg for arg in argv if arg != "--no-launch-workers"]
        self.processes = {}
        self.last_start = {}
        self.last_needed = {}
        self.active_profile = "idle"
        self.script_dir = Path(__file__).resolve().parent
        self.warm_ttl = max(0.0, float(getattr(args, "worker_warm_ttl", 20.0) or 0.0))
        self.restart_backoff = 2.0
        self.prewarm_workers = _parse_worker_set(getattr(args, "prewarm_workers", "all"))

    def start(self):
        self._start_worker("frame_source")
        for name in sorted(self.prewarm_workers):
            self._start_worker(name)
        self.set_profile("idle")

    def _worker_env(self):
        env = os.environ.copy()
        libgomp = "/lib/aarch64-linux-gnu/libgomp.so.1"
        if Path(libgomp).exists():
            current_preload = env.get("LD_PRELOAD", "")
            preload_parts = [libgomp]
            if current_preload:
                preload_parts.append(current_preload)
            env["LD_PRELOAD"] = ":".join(preload_parts)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        return env

    def _start_worker(self, name):
        proc = self.processes.get(name)
        if proc is not None and proc.poll() is None:
            return
        now = time.time()
        if now - float(self.last_start.get(name, 0.0) or 0.0) < self.restart_backoff:
            return
        script_name = WORKER_SCRIPTS[name]
        script_path = self.script_dir / script_name
        cmd = [sys.executable, str(script_path)] + self.argv
        proc = subprocess.Popen(cmd, env=self._worker_env())
        self.processes[name] = proc
        self.last_start[name] = now
        print(f"[VisionLauncher] started {script_name} pid={proc.pid}")

    def _stop_worker(self, name, kill_timeout=2.0):
        proc = self.processes.get(name)
        if proc is None:
            return
        script_name = WORKER_SCRIPTS[name]
        if proc.poll() is None:
            print(f"[VisionLauncher] stopping {script_name} pid={proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=kill_timeout)
            except subprocess.TimeoutExpired:
                print(f"[VisionLauncher] killing {script_name} pid={proc.pid}")
                proc.kill()
        self.processes.pop(name, None)

    def _required_workers(self, profile):
        required = set(PROFILE_WORKERS.get(profile, set()))
        if profile == "idle" and float(getattr(self.args, "idle_person_rate", 0.0) or 0.0) > 0.0:
            required.add("person")
        return required

    def set_profile(self, profile):
        self.active_profile = profile
        self._start_worker("frame_source")
        required = self._required_workers(profile)
        now = time.time()
        for name in required:
            self.last_needed[name] = now
            self._start_worker(name)
        self.maintain()

    def maintain(self):
        required = self._required_workers(self.active_profile)
        now = time.time()
        for name, proc in list(self.processes.items()):
            if proc.poll() is not None:
                self.processes.pop(name, None)
        keepalive = required | self.prewarm_workers | {"frame_source"}
        for name in keepalive:
            self._start_worker(name)
        for name in MODEL_WORKERS:
            if name in required:
                self.last_needed[name] = now
                continue
            if name in self.prewarm_workers:
                continue
            if name not in self.processes:
                continue
            inactive_for = now - float(self.last_needed.get(name, 0.0) or 0.0)
            if inactive_for >= self.warm_ttl:
                self._stop_worker(name)

    def states(self):
        states = {}
        for name in WORKER_SCRIPTS:
            proc = self.processes.get(name)
            if proc is None:
                states[name] = {"state": "stopped", "pid": None}
            elif proc.poll() is None:
                state = "running" if name in self._required_workers(self.active_profile) else "warm"
                if name == "frame_source":
                    state = "running"
                states[name] = {"state": state, "pid": proc.pid}
            else:
                states[name] = {"state": "exited", "pid": proc.pid, "returncode": proc.returncode}
        return states

    def stop(self):
        for name in list(self.processes.keys()):
            self._stop_worker(name)


def main():
    parser = build_arg_parser("CADE Open Vision Node")
    args, unknown = parser.parse_known_args()
    ros_args = [item for item in unknown if item.startswith("__")]
    other_unknown = [item for item in unknown if not item.startswith("__")]
    if other_unknown:
        parser.error("unrecognized arguments: %s" % " ".join(other_unknown))
    launcher = None
    if not args.no_launch_workers:
        worker_argv = [
            item for item in sys.argv[1:]
            if not item.startswith("__") and item not in ros_args
        ]
        launcher = WorkerLauncher(args, worker_argv)
        launcher.start()

    def _stop_children(signum, frame):
        if launcher is not None:
            launcher.stop()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _stop_children)
    signal.signal(signal.SIGINT, _stop_children)
    try:
        from cade_vision.workers.gateway import VisionGateway

        VisionGateway(args, worker_manager=launcher).run()
    finally:
        if launcher is not None:
            launcher.stop()


if __name__ == "__main__":
    main()
