#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doorbell Node — 门铃检测 ROS 节点

订阅 /audio/raw (Float32MultiArray), 使用 sherpa-onnx AudioTagging (CED)
检测门铃声 (Doorbell / Ding / Ding-dong) 并通过 ROS 话题发布。

发布话题:
  /doorbell/detected  (std_msgs/String, JSON)  门铃检测结果

订阅话题:
  /audio/raw          (std_msgs/Float32MultiArray)  来自 audio_capture_node

参数:
  ~model_dir        (str,  default: 包内模型路径)
  ~threshold        (float, default: 0.25)      门铃检测概率阈值
  ~buffer_duration  (float, default: 2.0)       每次检测的音频窗口秒数
  ~overlap_duration (float, default: 1.0)       相邻窗口重叠秒数
  ~cooldown_sec     (float, default: 3.0)       两次检测触发的最小间隔
  ~save_audio       (bool,  default: true)      保存有声音的分析窗口
  ~save_dir         (str,   default: recordings/doorbell_debug)
  ~save_min_peak    (float, default: 0.01)      保存音频窗口的峰值阈值
  ~save_min_rms     (float, default: 0.002)     保存音频窗口的 RMS 阈值
  ~log_top_labels   (bool,  default: true)      打印每个有声音窗口的 top 标签
"""

import json
import os
import sys
import threading
import time
import wave

import numpy as np
import rospy
import sherpa_onnx
from std_msgs.msg import Float32MultiArray, String

try:
    import rospkg

    _pkg_path = rospkg.RosPack().get_path("asr_tts")
except Exception:
    _pkg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _default_model_dir():
    return os.path.join(_pkg_path, "models", "sherpa-onnx-ced-mini-audio-tagging-2024-04-19")


def _default_save_dir():
    return os.path.join(_pkg_path, "recordings", "doorbell_debug")


DOORBELL_LABELS = {"Doorbell", "Ding", "Ding-dong", "Music", "Chime"}


class DoorbellNode:
    def __init__(self):
        rospy.init_node("doorbell_node")

        rospy.loginfo("[Doorbell] 初始化门铃检测节点")

        # ---- 参数 ----
        model_dir = rospy.get_param("~model_dir", _default_model_dir())
        self.threshold = rospy.get_param("~threshold", 0.25)
        buffer_duration = rospy.get_param("~buffer_duration", 2.0)
        overlap_duration = rospy.get_param("~overlap_duration", 1.0)
        self.cooldown_sec = rospy.get_param("~cooldown_sec", 3.0)
        self.save_audio = rospy.get_param("~save_audio", True)
        self.save_dir = rospy.get_param("~save_dir", _default_save_dir())
        self.save_min_peak = rospy.get_param("~save_min_peak", 0.01)
        self.save_min_rms = rospy.get_param("~save_min_rms", 0.002)
        self.log_top_labels = rospy.get_param("~log_top_labels", True)

        rospy.loginfo(f"[Doorbell] 参数: threshold={self.threshold}, buffer={buffer_duration}s, overlap={overlap_duration}s, cooldown={self.cooldown_sec}s")
        if self.save_audio:
            os.makedirs(self.save_dir, exist_ok=True)
            rospy.loginfo(
                f"[Doorbell] 调试录音保存目录: {self.save_dir} "
                f"(peak>={self.save_min_peak}, rms>={self.save_min_rms})"
            )

        sample_rate = 16000
        self.samples_per_read = int(0.2 * sample_rate)
        self.buffer_read_count = int(buffer_duration / 0.2)
        self.overlap_read_count = int(overlap_duration / 0.2)
        self.overlap_read_count = max(0, min(self.overlap_read_count, self.buffer_read_count - 1))

        # ---- 加载模型 ----
        model_file = os.path.join(model_dir, "model.int8.onnx")
        label_file = os.path.join(model_dir, "class_labels_indices.csv")

        if not os.path.isfile(model_file):
            rospy.logerr(f"[Doorbell] 模型文件不存在: {model_file}")
            sys.exit(1)

        rospy.loginfo(f"[Doorbell] 模型路径: {model_file}")
        rospy.loginfo(f"[Doorbell] 正在加载 AudioTagging 模型 (CED)...")
        config = sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                ced=model_file, num_threads=1, debug=False, provider="cpu"
            ),
            labels=label_file,
            top_k=5,
        )
        if not config.validate():
            rospy.logerr("[Doorbell] AudioTagging 配置无效")
            sys.exit(1)

        self.audio_tagger = sherpa_onnx.AudioTagging(config)
        rospy.loginfo(f"[Doorbell] 模型加载完成, 检测标签: {DOORBELL_LABELS}")

        # ---- ROS ----
        self._pub = rospy.Publisher("/doorbell/detected", String, queue_size=10)
        self._last_trigger_time = 0.0
        self._sample_rate = sample_rate
        self._window_index = 0
        self._last_silent_candidate_log_time = 0.0

        rospy.loginfo(f"[Doorbell] 已发布 /doorbell/detected 话题")

        self._buffer = []
        self._lock = threading.Lock()

        rospy.Subscriber("/audio/raw", Float32MultiArray, self._audio_callback, queue_size=20)
        rospy.loginfo("[Doorbell] 已订阅 /audio/raw 话题")
        rospy.loginfo("[Doorbell] 节点就绪, 开始监听门铃...")

    # ============================================================
    # 音频回调
    # ============================================================

    def _audio_callback(self, msg):
        samples = np.array(msg.data, dtype=np.float32)

        process_now = False
        with self._lock:
            self._buffer.append(samples)
            if len(self._buffer) >= self.buffer_read_count:
                self._window = np.concatenate(self._buffer)
                self._buffer = self._buffer[-self.overlap_read_count:] if self.overlap_read_count > 0 else []
                process_now = True

        if process_now:
            self._process_window(self._window)

    # ============================================================
    # 主循环
    # ============================================================

    def run(self):
        rospy.loginfo("[Doorbell] 进入主循环")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("[Doorbell] Ctrl+C, 退出")

    # ============================================================
    # 检测逻辑
    # ============================================================

    def _process_window(self, audio: np.ndarray):
        self._window_index += 1
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

        s = self.audio_tagger.create_stream()
        s.accept_waveform(self._sample_rate, audio)
        result = self.audio_tagger.compute(s)

        prob = 0.0
        matched_label = None
        top_labels = [
            {"name": event.name, "probability": round(float(event.prob), 6)}
            for event in result[:5]
        ]
        for event in result:
            if event.name in DOORBELL_LABELS:
                prob += event.prob
                if matched_label is None:
                    matched_label = event.name

        prob = float(prob)
        active_audio = peak >= self.save_min_peak or rms >= self.save_min_rms
        detected = active_audio and prob >= self.threshold
        if prob >= self.threshold and not active_audio:
            now = time.time()
            if now - self._last_silent_candidate_log_time >= 10.0:
                self._last_silent_candidate_log_time = now
                rospy.logwarn(
                    "[Doorbell] 忽略静音窗口候选: "
                    f"score={prob:.4f}, threshold={self.threshold}, "
                    f"matched={matched_label}, peak={peak:.6f}, rms={rms:.6f}"
                )
        wav_path = self._save_debug_window(
            audio=audio,
            top_labels=top_labels,
            matched_label=matched_label,
            probability=prob,
            detected=detected,
            peak=peak,
            rms=rms,
        )

        if self.log_top_labels and self._is_debug_audio_window(peak, rms, detected):
            rospy.loginfo(
                "[Doorbell] 窗口分析: "
                f"score={prob:.4f}, threshold={self.threshold}, "
                f"matched={matched_label}, peak={peak:.4f}, rms={rms:.4f}, "
                f"top={top_labels}, wav={wav_path}"
            )

        if detected:
            now = time.time()
            if now - self._last_trigger_time >= self.cooldown_sec:
                self._last_trigger_time = now
                self._publish(
                    True,
                    matched_label or "Doorbell",
                    prob,
                    wav_path=wav_path,
                    peak=peak,
                    rms=rms,
                    top_labels=top_labels,
                )
            else:
                rospy.logdebug(f"DoorbellNode: 冷却中, 跳过 (prob={prob:.2f})")

    def _is_debug_audio_window(self, peak: float, rms: float, detected: bool) -> bool:
        return detected or peak >= self.save_min_peak or rms >= self.save_min_rms

    def _safe_token(self, value: str) -> str:
        token = "".join(ch if ch.isalnum() else "_" for ch in value)
        return token.strip("_") or "none"

    def _save_debug_window(
        self,
        audio: np.ndarray,
        top_labels: list,
        matched_label: str,
        probability: float,
        detected: bool,
        peak: float,
        rms: float,
    ):
        if not self.save_audio or not self._is_debug_audio_window(peak, rms, detected):
            return None

        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        millis = int((now - int(now)) * 1000)
        status = "detected" if detected else "candidate"
        label = self._safe_token(matched_label or "none")
        basename = (
            f"{stamp}_{millis:03d}_{self._window_index:06d}_"
            f"{status}_{label}_{probability:.3f}"
        )
        wav_path = os.path.join(self.save_dir, f"{basename}.wav")
        json_path = os.path.join(self.save_dir, f"{basename}.json")

        clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm.tobytes())

        metadata = {
            "timestamp": now,
            "sample_rate": self._sample_rate,
            "duration_sec": round(float(len(audio)) / float(self._sample_rate), 4),
            "detected": detected,
            "threshold": self.threshold,
            "doorbell_score": round(probability, 6),
            "matched_label": matched_label,
            "peak": round(peak, 6),
            "rms": round(rms, 6),
            "top_labels": top_labels,
            "wav_path": wav_path,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return wav_path

    # ============================================================
    # 发布
    # ============================================================

    def _publish(
        self,
        detected: bool,
        label: str,
        probability: float,
        wav_path=None,
        peak=None,
        rms=None,
        top_labels=None,
    ):
        msg_data = {
            "detected": detected,
            "label": label,
            "probability": round(probability, 4),
            "timestamp": time.time(),
        }
        if wav_path:
            msg_data["wav_path"] = wav_path
        if peak is not None:
            msg_data["peak"] = round(float(peak), 6)
        if rms is not None:
            msg_data["rms"] = round(float(rms), 6)
        if top_labels is not None:
            msg_data["top_labels"] = top_labels
        self._pub.publish(String(data=json.dumps(msg_data, ensure_ascii=False)))
        rospy.loginfo(
            f"DoorbellNode: 检测到门铃! label={label} prob={probability:.3f} "
            f"wav={wav_path}"
        )


def main():
    node = DoorbellNode()
    node.run()


if __name__ == "__main__":
    main()
