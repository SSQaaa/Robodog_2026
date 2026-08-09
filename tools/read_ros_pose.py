# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import sys
import threading

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


lock = threading.Lock()
world_pose = None
imu_yaw_deg = None


def quaternion_yaw_deg(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def odom_callback(msg):
    global world_pose
    pos = msg.pose.pose.position
    with lock:
        world_pose = (float(pos.x), float(pos.y), float(pos.z))


def imu_callback(msg):
    global imu_yaw_deg
    with lock:
        imu_yaw_deg = quaternion_yaw_deg(msg.orientation)


def publish_latest(_event):
    with lock:
        pose = world_pose
        yaw_deg = imu_yaw_deg
    if pose is None or yaw_deg is None:
        return
    sys.stdout.write(
        "POSE x={:.6f} y={:.6f} odom_yaw_rad={:.6f} imu_yaw_deg={:.6f}\n".format(
            pose[0], pose[1], pose[2], yaw_deg
        )
    )
    sys.stdout.flush()


def main():
    rospy.init_node("read_ros_pose", anonymous=True)
    rospy.Subscriber("/leg_odom2", Odometry, odom_callback, queue_size=1)
    rospy.Subscriber("/imu/data", Imu, imu_callback, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), publish_latest)
    rospy.spin()


if __name__ == "__main__":
    main()
