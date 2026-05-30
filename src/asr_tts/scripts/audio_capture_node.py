#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Capture Node — 麦克风音频采集 ROS 节点

唯一打开物理麦克风的节点, 将音频块发布到 /audio/raw 话题,
供 asr_node / doorbell_node 等订阅者使用, 解决 PortAudio 设备独占冲突。

发布话题:
  /audio/raw  (std_msgs/Float32MultiArray)  音频采样块 (float32, 单声道)

参数:
  ~device_name    (str,  default: "default")  音频输入设备名
  ~sample_rate    (int,  default: 16000)      采样率 Hz
  ~chunk_duration (float, default: 0.2)       每块时长秒
"""

import argparse
import queue
import sys

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

try:
    import sounddevice as sd
except ImportError:
    rospy.logerr("请安装 sounddevice: pip install sounddevice")
    sys.exit(-1)


# ============================================================
# 命令行参数
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device-name", type=str, default="default",
                        help="音频输入设备名")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="采样率 Hz")
    parser.add_argument("--chunk-duration", type=float, default=0.2,
                        help="每块时长秒")
    return parser.parse_known_args()


# ============================================================
# 设备发现
# ============================================================

def find_input_device(device_name: str) -> int:
    """
    在系统中查找可用的音频输入设备。

    查找策略:
      1. device_name == "default" → 使用系统默认输入设备
      2. 按名称匹配 (max_input_channels > 0)
      3. Fallback: 名称含 "pulse" 的设备
      4. Fallback: 系统默认设备

    Returns:
        设备索引号

    Raises:
        SystemExit: 找不到任何可用输入设备
    """
    devices = sd.query_devices()
    if len(devices) == 0:
        rospy.logerr("[AudioCapture] 未找到任何音频设备")
        sys.exit(1)

    rospy.loginfo(f"[AudioCapture] 系统可用音频设备数: {len(devices)}")

    if device_name == "default":
        device_idx = sd.default.device[0]
        rospy.loginfo(f"[AudioCapture] 使用默认输入设备 (index={device_idx})")
        return device_idx

    device_idx = None

    for i, d in enumerate(devices):
        if device_name in d["name"] and d["max_input_channels"] > 0:
            device_idx = i
            break

    if device_idx is None:
        rospy.logwarn(f"[AudioCapture] 找不到 \"{device_name}\"，尝试 fallback 到 pulse")
        for i, d in enumerate(devices):
            if "pulse" in d["name"].lower() and d["max_input_channels"] > 0:
                device_idx = i
                rospy.loginfo(f"[AudioCapture] Fallback 成功: 使用 [{i}] {d['name']}")
                break

    if device_idx is None:
        rospy.logwarn("[AudioCapture] pulse 不可用，尝试 fallback 到 default")
        default_idx = sd.default.device[0]
        if default_idx is not None and default_idx < len(devices):
            d = devices[int(default_idx)]
            if d["max_input_channels"] > 0:
                device_idx = int(default_idx)
                rospy.loginfo(f"[AudioCapture] Fallback 成功: 使用 default [{device_idx}] {d['name']}")

    if device_idx is None:
        rospy.logerr("[AudioCapture] 找不到任何可用的输入设备")
        rospy.loginfo("[AudioCapture] 可用输入设备列表:")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                rospy.loginfo(f"[AudioCapture]   [{i}] {d['name']}")
        sys.exit(1)

    rospy.loginfo(f"[AudioCapture] 输入设备已锁定: [{device_idx}] {devices[device_idx]['name']}")
    return device_idx


# ============================================================
# ROS 节点
# ============================================================

class AudioCaptureNode:
    """ROS node for microphone audio capture and publishing."""

    def __init__(self, node_name: str = "audio_capture_node"):
        rospy.init_node(node_name)

        self._device_idx = None
        self._device_name = None
        self._sample_rate = 16000
        self._samples_per_chunk = 0

        self._pub = None
        self._queue = queue.Queue(maxsize=100)

    # ---- 初始化 ----

    def _init_params(self, args):
        self._device_name = rospy.get_param("~device_name", args.device_name)
        self._sample_rate = rospy.get_param("~sample_rate", args.sample_rate)
        chunk_duration = rospy.get_param("~chunk_duration", args.chunk_duration)
        self._samples_per_chunk = int(self._sample_rate * chunk_duration)

        rospy.loginfo("[AudioCapture] 初始化音频采集节点")
        rospy.loginfo(f"[AudioCapture] 参数: device_name={self._device_name}, sample_rate={self._sample_rate}, chunk_duration={chunk_duration}s")

    def _init_device(self):
        self._device_idx = find_input_device(self._device_name)
        rospy.loginfo(f"[AudioCapture] 采集参数: {self._sample_rate} Hz, 块大小={self._samples_per_chunk} samples")

    def _init_publisher(self):
        self._pub = rospy.Publisher("/audio/raw", Float32MultiArray, queue_size=20)
        rospy.set_param("/audio/raw/sample_rate", self._sample_rate)
        rospy.loginfo(f"[AudioCapture] 已发布 /audio/raw/sample_rate 参数 = {self._sample_rate}")

    # ---- 音频回调 ----

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            rospy.logwarn(f"音频流状态: {status}")
        try:
            self._queue.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass

    # ---- 发布 ----

    def _publish_chunk(self, samples: np.ndarray):
        msg = Float32MultiArray(data=samples.tolist())
        self._pub.publish(msg)

    # ---- 主循环 ----

    def run(self):
        args, _ = get_args()
        self._init_params(args)
        self._init_device()
        self._init_publisher()

        try:
            with sd.InputStream(
                device=self._device_idx,
                channels=1,
                callback=self._audio_callback,
                samplerate=self._sample_rate,
                dtype="float32",
                blocksize=self._samples_per_chunk,
            ):
                rospy.loginfo("[AudioCapture] 音频流已启动, 开始发布到 /audio/raw")
                while not rospy.is_shutdown():
                    try:
                        samples = self._queue.get(timeout=1.0)
                        self._publish_chunk(samples)
                    except queue.Empty:
                        continue
        except KeyboardInterrupt:
            rospy.loginfo("[AudioCapture] Ctrl+C, 退出")


# ============================================================
# 入口
# ============================================================

def main():
    node = AudioCaptureNode()
    node.run()


if __name__ == "__main__":
    main()
