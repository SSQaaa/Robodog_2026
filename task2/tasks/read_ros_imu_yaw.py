# -*- coding: utf-8 -*-
from __future__ import print_function

"""
只读取 ROS 的 /imu/data，不包含任何运动学控制。

用途：
1. 从 /imu/data 读取 IMU 四元数
2. 换算成 roll / pitch / yaw 角度
3. 打印 yaw 和 yaw_vel，方便后续标定 task2_3 的目标角度
"""

import argparse
import math
import sys
import time

import rospy
from sensor_msgs.msg import Imu


last_print_time = 0.0
print_interval = 0.20
print_once = False


def quaternion_to_euler_deg(x, y, z, w):
    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def imu_callback(msg):
    global last_print_time

    now = time.time()
    if now - last_print_time < print_interval:
        return
    last_print_time = now

    q = msg.orientation
    roll, pitch, yaw = quaternion_to_euler_deg(q.x, q.y, q.z, q.w)

    roll_vel = float(msg.angular_velocity.x)
    pitch_vel = float(msg.angular_velocity.y)
    yaw_vel = float(msg.angular_velocity.z)

    print(
        "roll={:.3f} pitch={:.3f} yaw={:.3f} | "
        "roll_vel={:.4f} pitch_vel={:.4f} yaw_vel={:.4f}".format(
            roll,
            pitch,
            yaw,
            roll_vel,
            pitch_vel,
            yaw_vel,
        )
    )

    if print_once:
        rospy.signal_shutdown("print once finished")


def main():
    global print_interval
    global print_once

    parser = argparse.ArgumentParser(description="读取 ROS /imu/data 并打印 roll/pitch/yaw")
    parser.add_argument("--topic", default="/imu/data", help="IMU话题名，默认 /imu/data")
    parser.add_argument("--interval", type=float, default=0.20, help="打印间隔，单位秒")
    parser.add_argument("--once", action="store_true", help="只打印一次后退出")

    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    print_interval = args.interval
    print_once = args.once

    rospy.init_node("read_ros_imu_yaw", anonymous=True)
    rospy.Subscriber(args.topic, Imu, imu_callback, queue_size=1)

    print("开始读取 {}，按 Ctrl+C 退出".format(args.topic))
    rospy.spin()


if __name__ == "__main__":
    main()
