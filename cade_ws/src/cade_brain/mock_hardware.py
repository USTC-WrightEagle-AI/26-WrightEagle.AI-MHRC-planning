#!/usr/bin/env python3
import json

import rospy
from std_msgs.msg import String


class MockHardware:
    def __init__(self):
        rospy.init_node("mock_hardware_node")
        self.pub = rospy.Publisher("/cade/task_status", String, queue_size=10)
        self.sub = rospy.Subscriber("/cade/task_cmd", String, self.callback)
        rospy.loginfo("🚀 硬件 Mock 节点已启动，正在自动化回应所有大脑指令...")

    def callback(self, msg):
        try:
            cmd = json.loads(msg.data)
            action = cmd.get("action")
            rospy.loginfo(f"📥 拦截到大脑动作: {action}")

            # 根据不同的动作，自动伪造精细的返回结果
            result_data = "operation executed"
            if action == "goToLoc":
                result_data = "arrived at target"
            elif action == "gesture_recognition":
                result_data = {"person_pos": "near table", "gesture": "waving"}
            elif action == "follow_person":
                result_data = "following the target person"

            response = {"status": "SUCCESS", "result": result_data}

            # 延迟 0.5 秒模拟硬件动作耗时，然后秒回
            rospy.sleep(0.5)
            self.pub.publish(String(data=json.dumps(response, ensure_ascii=False)))
            rospy.loginfo(f"📤 已自动秒回 SUCCESS 给大脑")

        except Exception as e:
            rospy.logerr(f"Mock 解析失败: {e}")


if __name__ == "__main__":
    MockHardware()
    rospy.spin()
