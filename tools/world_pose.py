# -*- coding: utf-8 -*-
import math
import os
import re
import subprocess
from dataclasses import dataclass

from project_config import PROJECT_DIR, TASK1_WORLD_POSE_PYTHON, TASK1_WORLD_POSE_TIMEOUT_S


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_POSE_RE = re.compile(
    r"pos_world:\s*x=(?P<x>{0})m\s*y=(?P<y>{0})m\s*"
    r"yaw_rad=(?P<yaw_rad>{0})\s*yaw_deg=(?P<yaw_deg>{0})".format(_NUMBER)
)
_IMU_YAW_RE = re.compile(r"\byaw=(?P<yaw>{0})\b".format(_NUMBER))


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


def read_world_pose_once(timeout_s=TASK1_WORLD_POSE_TIMEOUT_S, python_executable=TASK1_WORLD_POSE_PYTHON):
    script_path = os.path.join(PROJECT_DIR, "task2", "tasks", "read_ros_world_position.py")
    if not os.path.exists(script_path):
        raise RuntimeError("world pose reader not found: {}".format(script_path))

    try:
        output = subprocess.check_output(
            [python_executable, script_path, "--once"],
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
        )
    except subprocess.CalledProcessError as exc:
        text = exc.output.decode("utf-8", errors="replace") if exc.output else ""
        raise RuntimeError("read world pose failed: {}".format(text.strip()))
    except subprocess.TimeoutExpired:
        raise RuntimeError("read world pose timeout after {:.1f}s".format(float(timeout_s)))

    text = output.decode("utf-8", errors="replace")
    match = _POSE_RE.search(text)
    if match is None:
        raise RuntimeError("cannot parse world pose output: {}".format(text.strip()))

    return WorldPose(
        x=float(match.group("x")),
        y=float(match.group("y")),
        yaw_rad=float(match.group("yaw_rad")),
        yaw_deg=float(match.group("yaw_deg")),
    )


def read_imu_yaw_once(timeout_s=TASK1_WORLD_POSE_TIMEOUT_S, python_executable=TASK1_WORLD_POSE_PYTHON):
    script_path = os.path.join(PROJECT_DIR, "task2", "tasks", "read_ros_imu_yaw.py")
    if not os.path.exists(script_path):
        raise RuntimeError("IMU yaw reader not found: {}".format(script_path))

    try:
        output = subprocess.check_output(
            [python_executable, script_path, "--once"],
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
        )
    except subprocess.CalledProcessError as exc:
        text = exc.output.decode("utf-8", errors="replace") if exc.output else ""
        raise RuntimeError("read IMU yaw failed: {}".format(text.strip()))
    except subprocess.TimeoutExpired:
        raise RuntimeError("read IMU yaw timeout after {:.1f}s".format(float(timeout_s)))

    text = output.decode("utf-8", errors="replace")
    match = _IMU_YAW_RE.search(text)
    if match is None:
        raise RuntimeError("cannot parse IMU yaw output: {}".format(text.strip()))

    yaw_deg = float(match.group("yaw"))
    return ImuYaw(yaw_rad=math.radians(yaw_deg), yaw_deg=yaw_deg)


def project_relative_pose(start_pose, current_pose):
    dx = float(current_pose.x) - float(start_pose.x)
    dy = float(current_pose.y) - float(start_pose.y)
    yaw = float(start_pose.yaw_rad)
    forward_m = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral_m = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return RelativePose(forward_m=forward_m, lateral_m=lateral_m)
