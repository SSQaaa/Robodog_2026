# -*- coding: utf-8 -*-
import atexit
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass

from project_config import PROJECT_DIR, TASK1_WORLD_POSE_PYTHON, TASK1_WORLD_POSE_TIMEOUT_S


_POSE_RE = re.compile(
    r"POSE x=(?P<x>[-+\d.eE]+) y=(?P<y>[-+\d.eE]+) "
    r"odom_yaw_rad=(?P<odom_yaw>[-+\d.eE]+) imu_yaw_deg=(?P<imu_yaw>[-+\d.eE]+)"
)


@dataclass
class WorldPose:
    x: float
    y: float
    yaw_rad: float
    yaw_deg: float


@dataclass
class ImuYaw:
    yaw_rad: float
    yaw_deg: float


@dataclass
class RelativePose:
    forward_m: float
    lateral_m: float


class RosPoseReader:
    def __init__(self, python_executable=TASK1_WORLD_POSE_PYTHON):
        script_path = os.path.join(PROJECT_DIR, "tools", "read_ros_pose.py")
        if not os.path.exists(script_path):
            raise RuntimeError("ROS pose reader not found: {}".format(script_path))
        self.condition = threading.Condition()
        self.snapshot = None
        self.sequence = 0
        self.process = subprocess.Popen(
            [python_executable, "-u", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read_output, daemon=True)
        self.thread.start()

    def _read_output(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                return
            if not isinstance(line, str):
                line = line.decode("utf-8", errors="replace")
            match = _POSE_RE.search(line)
            if match is None:
                continue
            imu_yaw_deg = float(match.group("imu_yaw"))
            snapshot = WorldPose(
                x=float(match.group("x")),
                y=float(match.group("y")),
                yaw_rad=math.radians(imu_yaw_deg),
                yaw_deg=imu_yaw_deg,
            )
            with self.condition:
                self.snapshot = snapshot
                self.sequence += 1
                self.condition.notify_all()

    def read(self, timeout_s=TASK1_WORLD_POSE_TIMEOUT_S):
        deadline = time.time() + float(timeout_s)
        with self.condition:
            previous_sequence = self.sequence
            while self.snapshot is None or self.sequence == previous_sequence:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RuntimeError("read ROS pose timeout after {:.1f}s".format(float(timeout_s)))
                self.condition.wait(remaining)
            return self.snapshot

    def close(self):
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


_reader = None
_reader_lock = threading.Lock()


def get_pose_reader():
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = RosPoseReader()
        return _reader


def close_pose_reader():
    global _reader
    with _reader_lock:
        if _reader is not None:
            _reader.close()
            _reader = None


def read_world_pose_once(timeout_s=TASK1_WORLD_POSE_TIMEOUT_S, python_executable=None):
    _ = python_executable
    return get_pose_reader().read(timeout_s)


def read_imu_yaw_once(timeout_s=TASK1_WORLD_POSE_TIMEOUT_S, python_executable=None):
    pose = read_world_pose_once(timeout_s, python_executable)
    return ImuYaw(yaw_rad=pose.yaw_rad, yaw_deg=pose.yaw_deg)


def project_relative_pose(start_pose, current_pose):
    dx = float(current_pose.x) - float(start_pose.x)
    dy = float(current_pose.y) - float(start_pose.y)
    yaw = float(start_pose.yaw_rad)
    return RelativePose(
        forward_m=dx * math.cos(yaw) + dy * math.sin(yaw),
        lateral_m=-dx * math.sin(yaw) + dy * math.cos(yaw),
    )


atexit.register(close_pose_reader)
