# -*- coding: utf-8 -*-
import json
import math
import os
import time

from project_config import PROJECT_DIR
from tools.motion import DogControl
from tools.world_pose import read_imu_yaw_once, read_world_pose_once


PLAN_PATH = os.path.join(PROJECT_DIR, "tools", "task1_path_plan.json")

START_PLAN_X_M = -0.50
START_PLAN_Y_M = 0.75
FINISH_PLAN_X_M = 4.7
FINISH_PLAN_Y_M = 0.75
PLAN_FORWARD_SIGN = 1.0
PLAN_LATERAL_SIGN = -1.0
FORWARD_COMMAND_SIGN = 1.0
LATERAL_COMMAND_SIGN = 1.0
PLAN_YAW_OFFSET_RAD = 0.0

FORWARD_VX = 15000
FORWARD_SPEED_MPS = 0.4

LATERAL_VY = 18000
LATERAL_SPEED_MPS = 0.12

X_CORRECT_VX = 15000
X_CORRECT_SPEED_MPS = 0.5
X_CORRECT_MIN_STEP_S = 0.3
X_CORRECT_MAX_STEP_S = 1.2
Y_CORRECT_VY = 18000
Y_CORRECT_SPEED_MPS = 0.12
Y_CORRECT_MIN_STEP_S = 0.3
Y_CORRECT_MAX_STEP_S = 1.2
MOVE_SETTLE_S = 0.50

WAYPOINT_TOLERANCE_M = 0.12
MAX_X_CORRECT_STEPS = 10
MAX_Y_CORRECT_STEPS = 10
HEADING_CALIBRATION_MIN_DISTANCE_M = 0.50


def load_path_plan_data(path=PLAN_PATH):
    if not os.path.exists(path):
        raise RuntimeError(
            "Task1 path plan not found: {}. Run tools/task1_path_planner.py and press s to save first.".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    waypoints = data.get("waypoints_mm")
    if not waypoints or len(waypoints) < 2:
        raise RuntimeError("Task1 path plan must contain at least two waypoints")
    expected_start = [int(round(START_PLAN_X_M * 1000)), int(round(START_PLAN_Y_M * 1000))]
    expected_finish = [int(round(FINISH_PLAN_X_M * 1000)), int(round(FINISH_PLAN_Y_M * 1000))]
    if data.get("start_mm") != expected_start or data.get("finish_mm") != expected_finish:
        raise RuntimeError(
            "Task1 path plan start/finish mismatch. expected start_mm={} finish_mm={}, got start_mm={} finish_mm={}. Run tools/task1_path_planner.py and press s to save again.".format(
                expected_start,
                expected_finish,
                data.get("start_mm"),
                data.get("finish_mm"),
            )
        )
    return data


# 加载路径规划结果，mm->m
def load_path_plan(path=PLAN_PATH):
    data = load_path_plan_data(path)
    waypoints = data["waypoints_mm"]

    return [(float(x) / 1000.0, float(y) / 1000.0) for x, y in waypoints]


# 计算当前在路径规划坐标系下的位置，单位m
def current_plan_pose_with_yaw_m(start_world_pose):
    current_world_pose = read_world_pose_once()
    current_imu_yaw = read_imu_yaw_once()
    world_dx = float(current_world_pose.x) - float(start_world_pose.x)
    world_dy = float(current_world_pose.y) - float(start_world_pose.y)
    yaw = plan_world_yaw_rad(start_world_pose)
    forward_x, forward_y, right_x, right_y = plan_axis_vectors(yaw)
    plan_dx = world_dx * forward_x + world_dy * forward_y
    plan_dy = world_dx * right_x + world_dy * right_y
    return (
        START_PLAN_X_M + plan_dx,
        START_PLAN_Y_M + plan_dy,
        current_imu_yaw.yaw_rad,
        current_imu_yaw.yaw_deg,
    )


def current_plan_pose_m(start_world_pose):
    current_x, current_y, _, _ = current_plan_pose_with_yaw_m(start_world_pose)
    return current_x, current_y


def normalize_angle_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def plan_world_yaw_rad(start_world_pose):
    return normalize_angle_rad(float(start_world_pose.yaw_rad) + PLAN_YAW_OFFSET_RAD)


def plan_axis_vectors(yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward_x = PLAN_FORWARD_SIGN * cos_yaw
    forward_y = PLAN_FORWARD_SIGN * sin_yaw
    right_x = PLAN_LATERAL_SIGN * -sin_yaw
    right_y = PLAN_LATERAL_SIGN * cos_yaw
    return forward_x, forward_y, right_x, right_y


def lock_plan_frame_to_start_pose(world_pose):
    plan_yaw = plan_world_yaw_rad(world_pose)
    print(
        "[Task1][Frame] plan start locked to ROS pose: imu_yaw={:.3f}rad/{:.2f}deg plan_yaw={:.3f}rad/{:.2f}deg".format(
            world_pose.yaw_rad,
            world_pose.yaw_deg,
            plan_yaw,
            math.degrees(plan_yaw),
        )
    )


def correction_time_s(error_m, speed_mps, min_step_s, max_step_s):
    raw_time = abs(float(error_m)) / max(float(speed_mps), 1e-6)
    return max(float(min_step_s), min(float(max_step_s), raw_time))


# 根据规划结果在xy轴移动
def drive_axis_segment(dog: DogControl, axis, distance_m):
    distance_m = float(distance_m)
    if abs(distance_m) <= WAYPOINT_TOLERANCE_M:
        return

    if axis == "x":
        speed_mps = FORWARD_SPEED_MPS
        command_sign = 1 if distance_m > 0 else -1
        command = int(command_sign * FORWARD_COMMAND_SIGN * FORWARD_VX)
        last_time = abs(distance_m) / speed_mps
        print("[Task1][Move] x distance={:.3f}m vx={} time={:.2f}s".format(distance_m, command, last_time))
        dog.move(vx=command, last_time=last_time, duration=MOVE_SETTLE_S)
        return

    if axis == "y":
        speed_mps = LATERAL_SPEED_MPS
        command_sign = 1 if distance_m > 0 else -1
        command = int(command_sign * LATERAL_COMMAND_SIGN * LATERAL_VY)
        last_time = abs(distance_m) / speed_mps
        print("[Task1][Move] y distance={:.3f}m vy={} time={:.2f}s".format(distance_m, command, last_time))
        dog.move(vy=command, last_time=last_time, duration=MOVE_SETTLE_S)
        # time.sleep(0.5)
        dog.move(last_time=0.12*last_time, vx=10000)
        print("[Task1][Move] y correction 10000 0.12")
        return

    raise ValueError("unsupported axis: {}".format(axis))

# 修正因为左右平移带来的x轴误差，单位m
def correct_x_to_target(dog: DogControl, start_world_pose, target_x_m):
    for attempt in range(1, MAX_X_CORRECT_STEPS + 1):
        current_x, current_y, yaw_rad, yaw_deg = current_plan_pose_with_yaw_m(start_world_pose)
        error_x = float(target_x_m) - current_x
        print(
            "[Task1][Correct] attempt={} current=({:.3f}, {:.3f}) imu_yaw={:.3f}rad/{:.2f}deg target_x={:.3f} error_x={:.3f}".format(
                attempt, current_x, current_y, yaw_rad, yaw_deg, target_x_m, error_x
            )
        )
        if abs(error_x) <= WAYPOINT_TOLERANCE_M:
            return

        command_sign = 1 if error_x > 0 else -1
        vx = int(command_sign * FORWARD_COMMAND_SIGN * X_CORRECT_VX)
        step_s = correction_time_s(error_x, X_CORRECT_SPEED_MPS, X_CORRECT_MIN_STEP_S, X_CORRECT_MAX_STEP_S)
        dog.move(vx=vx, last_time=step_s, duration=MOVE_SETTLE_S)
        print("[Task1][CorrectX] x correction step: vx={} time={:.2f}s".format(vx, step_s))

    current_x, current_y = current_plan_pose_m(start_world_pose)
    raise RuntimeError(
        "Task1 x correction failed: current=({:.3f}, {:.3f}) target_x={:.3f} tolerance={:.3f}".format(
            current_x, current_y, target_x_m, WAYPOINT_TOLERANCE_M
        )
    )


def correct_y_to_target(dog: DogControl, start_world_pose, target_y_m):
    for attempt in range(1, MAX_Y_CORRECT_STEPS + 1):
        current_x, current_y, yaw_rad, yaw_deg = current_plan_pose_with_yaw_m(start_world_pose)
        error_y = float(target_y_m) - current_y
        print(
            "[Task1][CorrectY] attempt={} current=({:.3f}, {:.3f}) imu_yaw={:.3f}rad/{:.2f}deg target_y={:.3f} error_y={:.3f}".format(
                attempt, current_x, current_y, yaw_rad, yaw_deg, target_y_m, error_y
            )
        )
        if abs(error_y) <= WAYPOINT_TOLERANCE_M:
            return

        command_sign = 1 if error_y > 0 else -1
        vy = int(command_sign * LATERAL_COMMAND_SIGN * Y_CORRECT_VY)
        step_s = correction_time_s(error_y, Y_CORRECT_SPEED_MPS, Y_CORRECT_MIN_STEP_S, Y_CORRECT_MAX_STEP_S)
        dog.move(vy=vy, last_time=step_s, duration=MOVE_SETTLE_S)
        # time.sleep(0.5)
        dog.move(last_time=0.12*step_s, vx=10000)
        print("[Task1][CorrectY] y correction step: vy={} time={:.2f}s".format(vy, step_s))
        print("[Task1][Move] y correction 10000 0.1")

    current_x, current_y = current_plan_pose_m(start_world_pose)
    raise RuntimeError(
        "Task1 y correction failed: current=({:.3f}, {:.3f}) target_y={:.3f} tolerance={:.3f}".format(
            current_x, current_y, target_y_m, WAYPOINT_TOLERANCE_M
        )
    )


def execute_path(dog: DogControl, waypoints_m, start_world_pose):
    print("[Task1] loaded {} waypoints".format(len(waypoints_m)))
    for i, (x, y) in enumerate(waypoints_m):
        print("[Task1] waypoint {}: ({:.3f}, {:.3f})m".format(i, x, y))

    for index in range(1, len(waypoints_m)):
        target_x, target_y = waypoints_m[index]
        prev_x, prev_y = waypoints_m[index - 1]
        planned_dx = target_x - prev_x
        planned_dy = target_y - prev_y
        axis = "x" if abs(planned_dx) >= abs(planned_dy) else "y"

        current_x, current_y, yaw_rad, yaw_deg = current_plan_pose_with_yaw_m(start_world_pose)
        if axis == "x":
            distance_m = target_x - current_x
        else:
            distance_m = target_y - current_y

        print(
            "[Task1] segment {} axis={} current=({:.3f}, {:.3f}) imu_yaw={:.3f}rad/{:.2f}deg target=({:.3f}, {:.3f}) error=({:.3f}, {:.3f})".format(
                index,
                axis,
                current_x,
                current_y,
                yaw_rad,
                yaw_deg,
                target_x,
                target_y,
                target_x - current_x,
                target_y - current_y,
            )
        )
        drive_axis_segment(dog, axis, distance_m)
        if axis == "y":
            correct_y_to_target(dog, start_world_pose, target_y)
            correct_x_to_target(dog, start_world_pose, target_x)
        else:
            correct_x_to_target(dog, start_world_pose, target_x)
            correct_y_to_target(dog, start_world_pose, target_y)


def run(dog: DogControl, plan_path=PLAN_PATH):
    print("[Task1] DogControl class from {}.{}".format(dog.__class__.__module__, dog.__class__.__name__))
    plan_data = load_path_plan_data(plan_path)
    waypoints_m = [(float(x) / 1000.0, float(y) / 1000.0) for x, y in plan_data["waypoints_mm"]]
    start_world_pose = read_world_pose_once()
    start_imu_yaw = read_imu_yaw_once()
    start_world_pose.yaw_rad = start_imu_yaw.yaw_rad
    start_world_pose.yaw_deg = start_imu_yaw.yaw_deg
    lock_plan_frame_to_start_pose(start_world_pose)
    print(
        "[Task1] start world pose x={:.3f} y={:.3f} yaw={:.3f}rad/{:.2f}deg maps to plan=({:.3f}, {:.3f})m".format(
            start_world_pose.x,
            start_world_pose.y,
            start_world_pose.yaw_rad,
            start_world_pose.yaw_deg,
            START_PLAN_X_M,
            START_PLAN_Y_M,
        )
    )

    try:
        execute_path(dog, waypoints_m, start_world_pose)
        print("[Task1][Stop] finish reached")
        dog.stop()
        time.sleep(0.2)
        dog.revolve_90_r()
    finally:
        print("[Task1][Stop] task1 finally")
        dog.stop()
        time.sleep(0.2)
