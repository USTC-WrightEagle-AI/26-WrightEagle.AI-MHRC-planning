#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speaker DOA Node — 说话人识别 + 声源定位 ROS 节点

封装 robot_speaker_doa_bundle 的 SpeakerManager + M2MicArray,
持续采集麦克风阵列音频, 检测说话人并估计到达角 (DOA)。

发布话题:
  /speaker_doa/result  (std_msgs/String, JSON)  每次检测的结果

订阅话题:
  /speaker_doa/enroll   (std_msgs/String, JSON)  录入新说话人指令

参数:
  ~serial_device       (str,  default: "auto")    串口设备路径
  ~baud                (int,  default: 115200)    波特率
  ~threshold           (float, default: 0.3)      说话人识别阈值
  ~detect_seconds      (float, default: 0.5)      每次检测采集秒数
  ~enroll_seconds      (float, default: 5.0)      录入采集秒数
  ~active_channels     (str,  default: "0,1,2,3,4,5")
  ~geometry            (str,  default: "circle6") 阵列几何
  ~num_mics            (int,  default: 6)         麦克风数量
  ~radius_m            (float, default: 0.035)    圆形阵列半径
  ~spacing_m           (float, default: 0.028)    线形阵列间距
  ~score_threshold     (float, default: None)     输出结果的最低置信度

Task1 使用方式:
  1. 启动本节点 (随 speech.launch)
  2. ASK_GUEST1_INFO 完成后 → /speaker_doa/enroll {"command":"enroll","speaker_name":"guest1","duration":5.0}
  3. PICK_UP_GUEST2 完成后 → /speaker_doa/enroll {"command":"enroll","speaker_name":"guest2","duration":5.0}
  4. FIND_HOST 阶段 → 订阅 /speaker_doa/result, 用 angle_deg 导航; 找到 host 后录入
  5. 之后任何阶段都能通过 /speaker_doa/result 确认当前说话人身份
"""

import json
import os
import queue
import shutil
import sys
import threading
import time
import wave

import rospy
from std_msgs.msg import String

_bundle_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "robot_speaker_doa_bundle"
)
if _bundle_dir not in sys.path:
    sys.path.insert(0, _bundle_dir)

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


class SpeakerDOANode:
    def __init__(self):
        rospy.init_node("speaker_doa_node")

        # ---- 参数 ----
        serial_device = rospy.get_param("~serial_device", "auto")
        baud = rospy.get_param("~baud", 115200)
        self.threshold = rospy.get_param("~threshold", 0.3)
        self.detect_seconds = rospy.get_param("~detect_seconds", 0.5)
        self.enroll_seconds = rospy.get_param("~enroll_seconds", 5.0)
        self.active_channels = rospy.get_param("~active_channels", "0,1,2,3,4,5")
        self.geometry = rospy.get_param("~geometry", "circle6")
        self.num_mics = rospy.get_param("~num_mics", 6)
        self.radius_m = rospy.get_param("~radius_m", 0.035)
        self.spacing_m = rospy.get_param("~spacing_m", 0.028)
        self.segment_seconds = rospy.get_param("~segment_seconds", 0.5)
        self.angle_step_deg = rospy.get_param("~angle_step_deg", 2.0)
        self.score_threshold = rospy.get_param("~score_threshold", None)
        self.min_enroll_speech = rospy.get_param("~min_enroll_speech_seconds", 0.8)
        self.min_detect_speech = rospy.get_param("~min_detect_speech_seconds", 0.15)

        # ---- 工作目录 ----
        self.work_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "robot_speaker_doa_bundle", "workdir"
        )

        # ---- 模型和数据库 ----
        model_dir = os.path.join(os.path.abspath(self.work_dir), "pretrained_models", "spkrec-ecapa-voxceleb")
        self.db_path = os.path.join(os.path.abspath(self.work_dir), "session_speakers_db.pkl")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)  # 每次启动清空临时库

        # ---- 设备初始化 ----
        rospy.loginfo(f"SpeakerDOANode: 初始化麦克风阵列 (device={serial_device})")
        self.mic = M2MicArray(serial_device=serial_device, baudrate=baud)
        self.mgr = SpeakerManager(db_path=self.db_path, threshold=self.threshold, model_dir=model_dir)

        # ---- 说话人名字映射: speaker_id → human_name ----
        self._name_map = {}
        self._lock = threading.Lock()

        # ---- 录入指令队列 ----
        self._enroll_queue = queue.Queue()

        # ---- 停止信号 ----
        self._stop_event = threading.Event()

        # ---- ROS 接口 ----
        self._pub_result = rospy.Publisher("/speaker_doa/result", String, queue_size=10)
        self._sub_enroll = rospy.Subscriber("/speaker_doa/enroll", String, self._on_enroll)

        rospy.loginfo(f"SpeakerDOANode 初始化完成")
        rospy.loginfo(f"  串口: {self.mic.serial_device}")
        rospy.loginfo(f"  阈值: {self.threshold}")
        rospy.loginfo(f"  检测时长: {self.detect_seconds}s, 录入时长: {self.enroll_seconds}s")

    # ============================================================
    # ROS 回调
    # ============================================================

    def _on_enroll(self, msg: String):
        """接收录入指令"""
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logwarn(f"SpeakerDOANode: 无效的 enroll 消息: {msg.data}")
            return

        command = cmd.get("command", "")
        if command == "enroll":
            duration = float(cmd.get("duration", self.enroll_seconds))
            speaker_name = cmd.get("speaker_name", None)
            self._enroll_queue.put({"type": "enroll", "duration": duration, "speaker_name": speaker_name})
            rospy.loginfo(f"SpeakerDOANode: 收到录入指令 name={speaker_name}, duration={duration}s")
        elif command == "forget_all":
            self._enroll_queue.put({"type": "forget_all"})
            rospy.loginfo("SpeakerDOANode: 收到清除指令 — 清空说话人库")
        else:
            rospy.logwarn(f"SpeakerDOANode: 未知指令: {command}")

    # ============================================================
    # 检测主循环 (后台线程)
    # ============================================================

    def run(self):
        thread = threading.Thread(target=self._detection_loop, daemon=True)
        thread.start()
        rospy.loginfo("SpeakerDOANode: 检测循环已启动, 进入 ROS spin")
        rospy.spin()
        self._stop_event.set()
        thread.join(timeout=3.0)

    def _detection_loop(self):
        while not self._stop_event.is_set():
            try:
                # 检查录入队列
                try:
                    task = self._enroll_queue.get_nowait()
                    if task["type"] == "enroll":
                        self._do_enroll(task["duration"], task.get("speaker_name"))
                    elif task["type"] == "forget_all":
                        self._do_forget_all()
                except queue.Empty:
                    pass

                # 执行一次检测
                self._do_detect()

            except Exception as exc:
                rospy.logerr(f"SpeakerDOANode 检测循环异常: {exc}")
                time.sleep(0.5)

    # ============================================================
    # 检测
    # ============================================================

    def _do_detect(self):
        paths = self._next_capture_paths("detect")
        try:
            dumped = self.mic.dump_origin_audio(paths["out_dir"], seconds=self.detect_seconds)
            pcm_to_wav(dumped["iat_pcm"], paths["iat_wav"], sample_rate=16000, channels=1, sample_width=2)
        except Exception as exc:
            self._publish({"status": "error", "mode": "detect", "detail": str(exc)})
            return

        speech_check = self.mgr.detect_speech(paths["iat_wav"], min_speech_seconds=self.min_detect_speech)

        if not speech_check["speech_like"]:
            self._publish({"status": "silence", "speech_stats": speech_check})
            self._cleanup(paths["out_dir"])
            return

        try:
            cls = self.mgr.classify(paths["iat_wav"])
        except Exception as exc:
            self._publish({"status": "error", "mode": "classify", "detail": str(exc)})
            return

        if cls.get("error") == "empty_db":
            self._publish({"status": "empty_db", "message": "说话人库为空, 请先录入"})
            self._cleanup(paths["out_dir"])
            return

        speaker_id = cls.get("speaker_id")
        score = cls.get("score", 0.0)

        if not speaker_id:
            self._publish({"status": "no_match", "score": score, "speech_detected": True})
            self._cleanup(paths["out_dir"])
            return

        if self.score_threshold is not None and score is not None and score < self.score_threshold:
            self._publish({"status": "below_threshold", "speaker_id": speaker_id, "score": score})
            self._cleanup(paths["out_dir"])
            return

        # 估计 DOA
        try:
            doa = self.mic.estimate_doa_from_origin(
                dumped["origin_pcm"],
                active_channels=self.active_channels,
                geometry=self.geometry,
                num_mics=self.num_mics,
                radius_m=self.radius_m,
                spacing_m=self.spacing_m,
                angle_offset_deg=0.0,
                segment_seconds=self.segment_seconds,
                angle_step_deg=self.angle_step_deg,
            )
            angle = doa.get("estimated_angle_deg")
        except Exception as exc:
            self._publish({
                "status": "recognized_no_doa",
                "speaker_id": speaker_id,
                "speaker_name": self._name_map.get(speaker_id),
                "score": score,
                "detail": str(exc),
            })
            self._cleanup(paths["out_dir"])
            return

        self._publish({
            "status": "recognized",
            "speaker_id": speaker_id,
            "speaker_name": self._name_map.get(speaker_id),
            "score": score,
            "angle_deg": angle,
        })
        self._cleanup(paths["out_dir"])

    # ============================================================
    # 录入
    # ============================================================

    def _do_enroll(self, duration: float, speaker_name=None):
        paths = self._next_capture_paths("enroll")
        self._publish({"status": "enrolling", "speaker_name": speaker_name, "duration": duration})

        try:
            dumped = self.mic.dump_origin_audio(paths["out_dir"], seconds=duration)
            pcm_to_wav(dumped["iat_pcm"], paths["iat_wav"], sample_rate=16000, channels=1, sample_width=2)
        except Exception as exc:
            self._publish({"status": "error", "mode": "enroll", "detail": str(exc)})
            return

        speech_check = self.mgr.detect_speech(paths["iat_wav"], min_speech_seconds=self.min_enroll_speech)

        if not speech_check["speech_like"]:
            self._publish({
                "status": "enroll_rejected",
                "reason": speech_check["reason"],
                "speech_stats": speech_check,
            })
            return

        try:
            res = self.mgr.enroll(paths["iat_wav"])
        except Exception as exc:
            self._publish({"status": "error", "mode": "enroll", "detail": str(exc)})
            return

        speaker_id = res.get("speaker_id")
        if speaker_id and speaker_name:
            self._name_map[speaker_id] = speaker_name

        self._publish({
            "status": "enrolled",
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "action": res.get("action"),
            "score": res.get("score"),
        })
        rospy.loginfo(f"SpeakerDOANode: 录入完成 id={speaker_id} name={speaker_name}")

    def _do_forget_all(self):
        self.mgr.db.clear()
        self.mgr._save_db()
        self._name_map.clear()
        self._publish({"status": "db_cleared", "message": "说话人库已清空"})

    # ============================================================
    # 辅助方法
    # ============================================================

    def _next_capture_paths(self, prefix):
        os.makedirs(self.work_dir, exist_ok=True)
        token = f"{prefix}_{int(time.time() * 1000)}"
        out_dir = os.path.join(self.work_dir, token)
        os.makedirs(out_dir, exist_ok=True)
        return {
            "out_dir": out_dir,
            "origin_pcm": os.path.join(out_dir, "origin.pcm"),
            "iat_pcm": os.path.join(out_dir, "iat.pcm"),
            "iat_wav": os.path.join(out_dir, "iat.wav"),
        }

    def _cleanup(self, out_dir):
        """删除临时检测目录"""
        try:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass

    def _publish(self, payload: dict):
        msg = String(data=json.dumps(payload, ensure_ascii=False))
        self._pub_result.publish(msg)


def main():
    node = SpeakerDOANode()
    node.run()


if __name__ == "__main__":
    main()
