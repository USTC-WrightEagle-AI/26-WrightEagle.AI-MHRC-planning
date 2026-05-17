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
  ~threshold        (float, default: 0.5)       门铃检测概率阈值
  ~buffer_duration  (float, default: 2.0)       每次检测的音频窗口秒数
  ~overlap_duration (float, default: 1.0)       相邻窗口重叠秒数
  ~cooldown_sec     (float, default: 3.0)       两次检测触发的最小间隔
"""

import json
import os
import sys
import threading
import time

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


DOORBELL_LABELS = {"Doorbell", "Ding", "Ding-dong"}


class DoorbellNode:
    def __init__(self):
        rospy.init_node("doorbell_node")

        rospy.loginfo("[Doorbell] 初始化门铃检测节点")

        # ---- 参数 ----
        model_dir = rospy.get_param("~model_dir", _default_model_dir())
        self.threshold = rospy.get_param("~threshold", 0.5)
        buffer_duration = rospy.get_param("~buffer_duration", 2.0)
        overlap_duration = rospy.get_param("~overlap_duration", 1.0)
        self.cooldown_sec = rospy.get_param("~cooldown_sec", 3.0)

        rospy.loginfo(f"[Doorbell] 参数: threshold={self.threshold}, buffer={buffer_duration}s, overlap={overlap_duration}s, cooldown={self.cooldown_sec}s")

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
        s = self.audio_tagger.create_stream()
        s.accept_waveform(self._sample_rate, audio)
        result = self.audio_tagger.compute(s)

        prob = 0.0
        matched_label = None
        for event in result:
            if event.name in DOORBELL_LABELS:
                prob += event.prob
                if matched_label is None:
                    matched_label = event.name

        if prob >= self.threshold:
            now = time.time()
            if now - self._last_trigger_time >= self.cooldown_sec:
                self._last_trigger_time = now
                self._publish(True, matched_label or "Doorbell", float(prob))
            else:
                rospy.logdebug(f"DoorbellNode: 冷却中, 跳过 (prob={prob:.2f})")

    # ============================================================
    # 发布
    # ============================================================

    def _publish(self, detected: bool, label: str, probability: float):
        msg_data = {
            "detected": detected,
            "label": label,
            "probability": round(probability, 4),
            "timestamp": time.time(),
        }
        self._pub.publish(String(data=json.dumps(msg_data, ensure_ascii=False)))
        rospy.loginfo(f"DoorbellNode: 检测到门铃! label={label} prob={probability:.3f}")


def main():
    node = DoorbellNode()
    node.run()


if __name__ == "__main__":
    main()
