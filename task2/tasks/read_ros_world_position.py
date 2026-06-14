# -*- coding: utf-8 -*-
from __future__ import print_function

"""
只读取 ROS 的 /leg_odom2，打印机器人当前世界坐标系位置。

说明：
1. 这里读取 Odometry 的 position.x / position.y / position.z。
2. 根据当前机器狗文档，position.z 实际对应 pos_world[2]，也就是 yaw(rad)，不是高度。
3. 不包含任何运动学控制代码，只做只读调试。
"""

import argparse
import math
import sys
import time

import rospy
from nav_msgs.msg import Odometry


last_print_time = 0.0
print_interval = 0.20
print_once = False


def odom_callback(msg):
    global last_print_time

    now = time.time()
    if now - last_print_time < print_interval:
        return
    last_print_time = now

    pos = msg.pose.pose.position
    vel = msg.twist.twist.linear

    yaw_rad = float(pos.z)
    yaw_deg = math.degrees(yaw_rad)

    print(
        "pos_world: x={:.3f}m y={:.3f}m yaw_rad={:.3f} yaw_deg={:.2f} | "
        "vel_world: x={:.3f} y={:.3f} z={:.3f}".format(
            float(pos.x),
            float(pos.y),
            yaw_rad,
            yaw_deg,
            float(vel.x),
            float(vel.y),
            float(vel.z),
        )
    )

    if print_once:
        rospy.signal_shutdown("print once finished")


def main():
    global print_interval
    global print_once

    parser = argparse.ArgumentParser(description="读取 ROS 里程计并打印机器人世界坐标位置")
    parser.add_argument("--topic", default="/leg_odom2", help="里程计话题名，默认 /leg_odom2")
    parser.add_argument("--interval", type=float, default=0.20, help="打印间隔，单位秒")
    parser.add_argument("--once", action="store_true", help="只打印一次后退出")

    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    print_interval = args.interval
    print_once = args.once

    rospy.init_node("read_ros_world_position", anonymous=True)
    rospy.Subscriber(args.topic, Odometry, odom_callback, queue_size=1)

    print("开始读取 {}，按 Ctrl+C 退出".format(args.topic))
    rospy.spin()


if __name__ == "__main__":
    main()
