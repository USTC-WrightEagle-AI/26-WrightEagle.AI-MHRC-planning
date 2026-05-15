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


class AudioCaptureNode:
    def __init__(self):
        rospy.init_node("audio_capture_node")

        self.device_name = rospy.get_param("~device_name", "default")
        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        chunk_duration = rospy.get_param("~chunk_duration", 0.2)

        self.samples_per_chunk = int(self.sample_rate * chunk_duration)

        devices = sd.query_devices()
        if len(devices) == 0:
            rospy.logerr("未找到音频设备")
            sys.exit(1)

        if self.device_name == "default":
            self.device_idx = sd.default.device[0]
        else:
            self.device_idx = None
            for i, d in enumerate(devices):
                if self.device_name in d["name"] and d["max_input_channels"] > 0:
                    self.device_idx = i
                    break

        if self.device_idx is None:
            rospy.logerr(f"找不到设备: \"{self.device_name}\"")
            sys.exit(1)

        rospy.loginfo(
            f"AudioCapture: 设备 [{self.device_idx}] {devices[self.device_idx]['name']}"
        )
        rospy.loginfo(
            f"AudioCapture: {self.sample_rate} Hz, chunks={self.samples_per_chunk} samples"
        )

        self._pub = rospy.Publisher("/audio/raw", Float32MultiArray, queue_size=20)
        self._queue = queue.Queue(maxsize=100)

        # 发布音频采样率作为参数, 供订阅者查询
        rospy.set_param("/audio/raw/sample_rate", self.sample_rate)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            rospy.logwarn(f"音频流状态: {status}")
        try:
            self._queue.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass

    def run(self):
        try:
            with sd.InputStream(
                device=self.device_idx,
                channels=1,
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                dtype="float32",
                blocksize=self.samples_per_chunk,
            ):
                rospy.loginfo("AudioCapture: 音频流已启动, 发布到 /audio/raw")
                while not rospy.is_shutdown():
                    try:
                        samples = self._queue.get(timeout=1.0)
                        msg = Float32MultiArray(data=samples.tolist())
                        self._pub.publish(msg)
                    except queue.Empty:
                        continue
        except KeyboardInterrupt:
            rospy.loginfo("AudioCapture: Ctrl+C, 退出")


def main():
    node = AudioCaptureNode()
    node.run()


if __name__ == "__main__":
    main()
