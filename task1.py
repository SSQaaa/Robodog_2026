# -*- coding: utf-8 -*-
import json
import math
import os
import time

from project_config import PROJECT_DIR
from tools.motion import DogControl
from tools.world_pose import read_world_pose_once


PLAN_PATH = os.path.join(PROJECT_DIR, "tools", "task1_path_plan.json")

START_PLAN_X_M = -0.25
START_PLAN_Y_M = 0.75
FINISH_PLAN_X_M = 4.20
FINISH_PLAN_Y_M = 0.75
PLAN_LATERAL_SIGN = -1.0

FORWARD_VX = 20000
FORWARD_SPEED_MPS = 0.5

LATERAL_VY = 25000
LATERAL_SPEED_MPS = 0.25

DIAGONAL_VX = 20000
DIAGONAL_VY = 25000
DIAGONAL_X_SPEED_MPS = 0.40
DIAGONAL_Y_SPEED_MPS = 0.15
DIAGONAL_MIN_TIME_S = 1.0

X_CORRECT_VX = 10000
X_CORRECT_SPEED_MPS = 0.4
MOVE_SETTLE_S = 0.50

WAYPOINT_TOLERANCE_M = 0.10

# 用于计算脉冲和修正距离的速度映射
VX10000_SPEED_MPS = 0.4          # vx=10000 对应的实际速度，与 X_CORRECT_SPEED_MPS 一致
DIAGONAL_SPEED_MPS = math.hypot(DIAGONAL_X_SPEED_MPS, DIAGONAL_Y_SPEED_MPS)


def log_move(current, target, vx, vy, move_time, error, yaw_deg):
    """Print one compact, consistent line for every Task1 motion command."""
    print(
        "[Task1][Move] current=({:.3f},{:.3f}) target=({:.3f},{:.3f}) "
        "velocity=(vx={},vy={}) time={:.2f}s err=({:.3f},{:.3f}) yaw={:.2f}deg".format(
            current[0], current[1], target[0], target[1], vx, vy,
            move_time, error[0], error[1], yaw_deg,
        )
    )


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
    if waypoints[0] != expected_start or waypoints[-1] != expected_finish:
        raise RuntimeError(
            "Task1 path plan waypoint mismatch. expected first={} last={}, got first={} last={}. Run tools/task1_path_planner.py again.".format(
                expected_start,
                expected_finish,
                waypoints[0],
                waypoints[-1],
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
    world_dx = float(current_world_pose.x) - float(start_world_pose.x)
    world_dy = float(current_world_pose.y) - float(start_world_pose.y)
    yaw = plan_world_yaw_rad(start_world_pose)
    forward_x, forward_y, right_x, right_y = plan_axis_vectors(yaw)
    plan_dx = world_dx * forward_x + world_dy * forward_y
    plan_dy = world_dx * right_x + world_dy * right_y
    plan_x = START_PLAN_X_M + plan_dx
    plan_y = START_PLAN_Y_M + plan_dy
    return (
        plan_x,
        plan_y,
        current_world_pose.yaw_rad,
        current_world_pose.yaw_deg,
    )


def current_plan_pose_m(start_world_pose):
    current_x, current_y, _, _ = current_plan_pose_with_yaw_m(start_world_pose)
    return current_x, current_y


def read_start_world_pose(sample_count=3):
    poses = [read_world_pose_once() for _ in range(sample_count)]
    sin_sum = sum(math.sin(pose.yaw_rad) for pose in poses)
    cos_sum = sum(math.cos(pose.yaw_rad) for pose in poses)
    latest = poses[-1]
    latest.yaw_rad = math.atan2(sin_sum, cos_sum)
    latest.yaw_deg = math.degrees(latest.yaw_rad)
    return latest


def normalize_angle_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def plan_world_yaw_rad(start_world_pose):
    return normalize_angle_rad(float(start_world_pose.yaw_rad))


def plan_axis_vectors(yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward_x = cos_yaw
    forward_y = sin_yaw
    right_x = PLAN_LATERAL_SIGN * -sin_yaw
    right_y = PLAN_LATERAL_SIGN * cos_yaw
    return forward_x, forward_y, right_x, right_y


def lock_plan_frame_to_start_pose(world_pose):
    plan_world_yaw_rad(world_pose)


# 根据规划结果在xy轴移动
def drive_axis_segment(dog: DogControl, axis, distance_m, current, target, yaw_deg):
    """返回 (执行了移动?, 沿轴实际距离_米, 脉冲距离_米)"""
    distance_m = float(distance_m)
    if abs(distance_m) <= WAYPOINT_TOLERANCE_M + 1e-9:
        return False, 0.0, 0.0

    if axis == "x":
        speed_mps = FORWARD_SPEED_MPS
        command_sign = 1 if distance_m > 0 else -1
        command = command_sign * FORWARD_VX
        last_time = abs(distance_m) / speed_mps
        log_move(current, target, command, 0, last_time, (distance_m, target[1] - current[1]), yaw_deg)
        dog.move(vx=command, last_time=last_time, duration=MOVE_SETTLE_S)
        actual_dist = last_time * speed_mps
        return True, actual_dist, 0.0

    if axis == "y":
        speed_mps = LATERAL_SPEED_MPS
        command_sign = 1 if distance_m > 0 else -1
        command = command_sign * LATERAL_VY
        last_time = abs(distance_m) / speed_mps
        one_second_steps = max(int(last_time) - 1, 0)
        lateral_steps = [1.0] * one_second_steps
        lateral_steps.append(last_time - one_second_steps)

        total_y_dist = 0.0
        total_pulse_dist = 0.0
        for step_index, step_time in enumerate(lateral_steps, start=1):
            log_move(current, target, 0, command, step_time, (target[0] - current[0], distance_m), yaw_deg)
            dog.move(vy=command, last_time=step_time, duration=MOVE_SETTLE_S)
            total_y_dist += step_time * speed_mps

            pulse_time = 0.12
            log_move(current, target, 10000, 0, pulse_time, (target[0] - current[0], distance_m), yaw_deg)
            dog.move(last_time=pulse_time, vx=10000)
            total_pulse_dist += pulse_time * VX10000_SPEED_MPS

        return True, total_y_dist, total_pulse_dist

    raise ValueError("unsupported axis: {}".format(axis))


def drive_diagonal_segment(dog: DogControl, error_x_m, error_y_m, planned_dx, planned_dy,
                           current, target, yaw_deg):
    """返回 (执行了移动?, 对角线实际行走距离_米)"""
    if error_x_m * planned_dx <= 0 or error_y_m * planned_dy <= 0:
        return False, 0.0

    move_time = min(
        abs(error_x_m) / DIAGONAL_X_SPEED_MPS,
        abs(error_y_m) / DIAGONAL_Y_SPEED_MPS,
    )
    if move_time < DIAGONAL_MIN_TIME_S:
        return False, 0.0

    vx = DIAGONAL_VX if planned_dx > 0 else -DIAGONAL_VX
    vy = DIAGONAL_VY if planned_dy > 0 else -DIAGONAL_VY
    log_move(current, target, vx, vy, move_time, (error_x_m, error_y_m), yaw_deg)
    dog.move(vx=vx, vy=vy, last_time=move_time, duration=MOVE_SETTLE_S)
    dog.stop()
    actual_dist = move_time * DIAGONAL_SPEED_MPS
    return True, actual_dist


# 修正因为左右平移带来的x轴误差，单位m
def correct_x_to_target(dog: DogControl, start_world_pose, target):
    """返回 x 方向修正总距离（米）"""
    total_correct = 0.0
    target_x_m, target_y_m = target
    while True:
        current_x, current_y, _, yaw_deg = current_plan_pose_with_yaw_m(start_world_pose)
        error_x = float(target_x_m) - current_x
        if abs(error_x) <= WAYPOINT_TOLERANCE_M:
            break

        command_sign = 1 if error_x > 0 else -1
        vx = command_sign * X_CORRECT_VX
        step_s = abs(error_x) / X_CORRECT_SPEED_MPS
        log_move(
            (current_x, current_y), (target_x_m, target_y_m), vx, 0, step_s,
            (error_x, target_y_m - current_y), yaw_deg,
        )
        dog.move(vx=vx, last_time=step_s, duration=MOVE_SETTLE_S)
        step_dist = step_s * X_CORRECT_SPEED_MPS
        total_correct += step_dist
    return total_correct


def correct_y_to_target(dog: DogControl, start_world_pose, target):
    """返回 (y 方向修正距离_米, 前向脉冲总距离_米)"""
    total_correct = 0.0
    total_pulse = 0.0
    target_x_m, target_y_m = target
    while True:
        current_x, current_y, _, yaw_deg = current_plan_pose_with_yaw_m(start_world_pose)
        error_y = float(target_y_m) - current_y
        if abs(error_y) <= WAYPOINT_TOLERANCE_M:
            break

        command_sign = 1 if error_y > 0 else -1
        vy = command_sign * LATERAL_VY
        step_s = abs(error_y) / LATERAL_SPEED_MPS
        log_move(
            (current_x, current_y), (target_x_m, target_y_m), 0, vy, step_s,
            (target_x_m - current_x, error_y), yaw_deg,
        )
        dog.move(vy=vy, last_time=step_s, duration=MOVE_SETTLE_S)
        step_y_dist = step_s * LATERAL_SPEED_MPS
        total_correct += step_y_dist

        pulse_time = 0.12 * step_s
        log_move(
            (current_x, current_y), (target_x_m, target_y_m), 10000, 0, pulse_time,
            (target_x_m - current_x, error_y), yaw_deg,
        )
        dog.move(last_time=pulse_time, vx=10000)
        pulse_dist = pulse_time * VX10000_SPEED_MPS
        total_pulse += pulse_dist

    return total_correct, total_pulse


def execute_path(dog: DogControl, waypoints_m, start_world_pose, on_first_move=None):
    print("[Task1] loaded {} waypoints".format(len(waypoints_m)))

    total_plan_dist = 0.0
    total_actual_dist = 0.0
    total_pulse_dist = 0.0
    total_corr_x = 0.0
    total_corr_y = 0.0

    for index in range(1, len(waypoints_m)):
        target_x, target_y = waypoints_m[index]
        prev_x, prev_y = waypoints_m[index - 1]
        planned_dx = target_x - prev_x
        planned_dy = target_y - prev_y
        plan_seg_dist = math.hypot(planned_dx, planned_dy)
        total_plan_dist += plan_seg_dist

        is_diagonal = abs(planned_dx) > 1e-9 and abs(planned_dy) > 1e-9
        axis = "diagonal" if is_diagonal else ("x" if abs(planned_dx) >= abs(planned_dy) else "y")

        if index == 1:
            current_x, current_y = prev_x, prev_y
            yaw_deg = start_world_pose.yaw_deg
            error_x = planned_dx
            error_y = planned_dy
        else:
            current_x, current_y, _, yaw_deg = current_plan_pose_with_yaw_m(start_world_pose)
            error_x = target_x - current_x
            error_y = target_y - current_y

        if (
            abs(error_x) <= WAYPOINT_TOLERANCE_M + 1e-9
            and abs(error_y) <= WAYPOINT_TOLERANCE_M + 1e-9
        ):
            print("[Task1] segment {} actual position within tolerance, skip move".format(index))
            seg_actual = 0.0
            seg_pulse = 0.0
            seg_corr_x = 0.0
            seg_corr_y = 0.0
            moved = False
            main_dist = 0.0
        else:
            if on_first_move is not None:
                on_first_move()
                on_first_move = None

            moved = False
            main_dist = 0.0
            pulse_dist = 0.0

            if is_diagonal:
                moved, main_dist = drive_diagonal_segment(
                    dog, error_x, error_y, planned_dx, planned_dy,
                    (current_x, current_y), (target_x, target_y), yaw_deg,
                )
            elif axis == "y":
                moved, main_dist, pulse_dist = drive_axis_segment(
                    dog, axis, error_y, (current_x, current_y), (target_x, target_y), yaw_deg
                )
            else:
                moved, main_dist, _ = drive_axis_segment(
                    dog, axis, error_x, (current_x, current_y), (target_x, target_y), yaw_deg
                )

            target = (target_x, target_y)
            corr_x_dist = correct_x_to_target(dog, start_world_pose, target)
            corr_y_dist, corr_y_pulse = correct_y_to_target(dog, start_world_pose, target)

            seg_corr_x = corr_x_dist
            seg_corr_y = corr_y_dist
            seg_pulse = pulse_dist + corr_y_pulse
            seg_actual = main_dist + corr_x_dist + corr_y_dist + seg_pulse

        total_actual_dist += seg_actual
        total_pulse_dist += seg_pulse
        total_corr_x += seg_corr_x
        total_corr_y += seg_corr_y

        print("[Task1] waypoint {}/{} reached".format(index, len(waypoints_m) - 1))

    print("[Task1] completed: plan={:.3f}m actual={:.3f}m".format(total_plan_dist, total_actual_dist))


def run(dog: DogControl, plan_path=PLAN_PATH, on_navigation_ready=None):
    print("[Task1] loading path plan: {}".format(os.path.abspath(plan_path)))
    plan_data = load_path_plan_data(plan_path)
    waypoints_m = [(float(x) / 1000.0, float(y) / 1000.0) for x, y in plan_data["waypoints_mm"]]
    start_world_pose = read_start_world_pose(sample_count=3)
    lock_plan_frame_to_start_pose(start_world_pose)
    print("[Task1] start yaw={:.2f}deg".format(start_world_pose.yaw_deg))
    try:
        execute_path(dog, waypoints_m, start_world_pose, on_first_move=on_navigation_ready)
        dog.stop()
        time.sleep(0.2)
        dog.revolve_90_r()
    finally:
        dog.stop()
        time.sleep(0.2)
    return start_world_pose.yaw_deg
