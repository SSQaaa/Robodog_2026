# -*- coding: utf-8 -*-
import math
import os
import re
import subprocess
import time

from project_config import PROJECT_DIR, TASK1_WORLD_POSE_PYTHON
from tools.vision import DashboardInfer


def _pick_best_by_class(detections, class_id):
    best_det = None
    best_score = -1.0
    for det in detections:
        cid = int(det.get("class_id", -1))
        if cid != class_id:
            continue
        score = float(det.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_det = det
    return best_det


def _box_center_x(det):
    x1, y1, x2, y2 = det["xyxy"]
    _ = y1, y2
    return (float(x1) + float(x2)) / 2.0


def _normalize_yaw_error(target_yaw, current_yaw):
    error = float(target_yaw) - float(current_yaw)
    while error > 180.0:
        error -= 360.0
    while error < -180.0:
        error += 360.0
    return error


def _average_yaw_deg(yaw_list):
    sin_sum = 0.0
    cos_sum = 0.0
    for yaw in yaw_list:
        yaw_rad = math.radians(float(yaw))
        sin_sum += math.sin(yaw_rad)
        cos_sum += math.cos(yaw_rad)
    return math.degrees(math.atan2(sin_sum, cos_sum))


def read_current_yaw_deg(sample_count=3, python_executable=TASK1_WORLD_POSE_PYTHON):
    script_path = os.path.join(PROJECT_DIR, "task2", "tasks", "read_ros_imu_yaw.py")
    yaw_list = []

    for _ in range(sample_count):
        output = subprocess.check_output(
            [python_executable, script_path, "--once"],
            stderr=subprocess.STDOUT,
            timeout=8,
        )
        if not isinstance(output, str):
            output = output.decode("utf-8", errors="replace")

        match = re.search(r"yaw=([-+]?\d+\.?\d*)", output)
        if match is None:
            print(output)
            raise RuntimeError("cannot parse IMU yaw output")

        yaw_list.append(float(match.group(1)))
        time.sleep(0.05)

    return _average_yaw_deg(yaw_list)


def rotate_to_relative_yaw(dog, target_yaw_deg, tolerance_deg=3.0):
    stable_need_frames = 3
    max_adjust_steps = 40
    yaw_vz_small = 9555
    yaw_vz_large = 10000
    stable_count = 0

    for step in range(max_adjust_steps):
        current_yaw = read_current_yaw_deg()
        error = _normalize_yaw_error(target_yaw_deg, current_yaw)

        if abs(error) <= tolerance_deg:
            stable_count += 1
            print(
                "YAW: step={} stable={}/{} current={:.3f} target={:.3f} error={:.3f}".format(
                    step + 1,
                    stable_count,
                    stable_need_frames,
                    current_yaw,
                    target_yaw_deg,
                    error,
                )
            )
            if stable_count >= stable_need_frames:
                print("YAW: adjustment complete")
                return current_yaw
            time.sleep(0.2)
            continue

        stable_count = 0
        vz_abs = yaw_vz_large if abs(error) > 30.0 else yaw_vz_small
        vz_cmd = -vz_abs if error > 0 else vz_abs

        print(
            "YAW: step={} current={:.3f} target={:.3f} error={:.3f} vz={}".format(
                step + 1,
                current_yaw,
                target_yaw_deg,
                error,
                vz_cmd,
            )
        )
        dog.move(last_time=0.1, vz=vz_cmd)
        time.sleep(1)

    print("YAW: max adjustment steps reached, continue with current yaw")
    return read_current_yaw_deg()


def rotate_to_relative_yaw_once(dog, target_yaw_deg, tolerance_deg=3.0):
    """Check yaw once and issue at most one correction command."""
    yaw_vz_small = 9555
    yaw_vz_large = 10000

    current_yaw = read_current_yaw_deg()
    error = _normalize_yaw_error(target_yaw_deg, current_yaw)
    if abs(error) <= tolerance_deg:
        print(
            "YAW_ONCE: within tolerance current={:.3f} target={:.3f} error={:.3f}".format(
                current_yaw, target_yaw_deg, error
            )
        )
        return current_yaw

    vz_abs = yaw_vz_large if abs(error) > 30.0 else yaw_vz_small
    vz_cmd = -vz_abs if error > 0 else vz_abs
    print(
        "YAW_ONCE: correct once current={:.3f} target={:.3f} error={:.3f} vz={}".format(
            current_yaw, target_yaw_deg, error, vz_cmd
        )
    )
    dog.move(last_time=0.1, vz=vz_cmd)
    return current_yaw


def _adjust_c_distance(dog, detector, target_m, tolerance_m, stable_need_frames, max_adjust_steps):
    stable_count = 0

    for step in range(max_adjust_steps):
        infer_output = detector.infer_once()
        det_c = _pick_best_by_class(infer_output.get("detections", []), 2)

        if det_c is None:
            print("BRIDGE_DISTANCE: step={} C not detected".format(step + 1))
            stable_count = 0
            time.sleep(0.3)
            continue

        distance_m = det_c.get("distance_m", None)
        if distance_m is None:
            print("BRIDGE_DISTANCE: step={} C depth invalid".format(step + 1))
            stable_count = 0
            time.sleep(0.3)
            continue

        distance_m = float(distance_m)
        error_m = target_m - distance_m
        if abs(error_m) <= tolerance_m:
            stable_count += 1
            print(
                "BRIDGE_DISTANCE: step={} stable={}/{} current={:.3f}m target={:.3f}m error={:.3f}m".format(
                    step + 1,
                    stable_count,
                    stable_need_frames,
                    distance_m,
                    target_m,
                    error_m,
                )
            )
            if stable_count >= stable_need_frames:
                print("BRIDGE_DISTANCE: adjustment complete")
                return distance_m
            time.sleep(0.2)
            continue

        stable_count = 0
        vx_abs = 25000 if abs(error_m) > 0.50 else 20000
        vx_cmd = vx_abs if error_m < 0 else -vx_abs
        print(
            "BRIDGE_DISTANCE: step={} current={:.3f}m target={:.3f}m error={:.3f}m vx={}".format(
                step + 1,
                distance_m,
                target_m,
                error_m,
                vx_cmd,
            )
        )
        dog.move(last_time=0.2, vx=vx_cmd)
        time.sleep(0.4)

    print("BRIDGE_DISTANCE: max adjustment steps reached")
    return None


def task2_3(dog, detector, start_yaw_deg):
    c_x_center_min = 350
    c_x_center_max = 400
    c_distance_target_m = 1.50
    c_distance_tolerance_m = 0.20
    target_yaw_deg = start_yaw_deg - 90.0

    stable_need_frames = 3
    max_adjust_steps = 30
    max_distance_adjust_steps = 30
    stable_count = 0
    c_missing_count = 0

    # dog.move(last_time=7, vy=25000)
    # time.sleep(0.5)
    dog.move(last_time=3.2, vx=-20000)
    time.sleep(0.5)
    dog.revolve_90_r()
    time.sleep(0.5)

    print("[Task2_3] start IMU yaw adjustment")
    final_yaw = rotate_to_relative_yaw(dog, target_yaw_deg)
    print("[Task2_3] yaw adjusted current={:.3f}, start C centering".format(final_yaw))

    for step in range(max_adjust_steps):
        infer_output = detector.infer_once()
        det_c = _pick_best_by_class(infer_output.get("detections", []), 2)

        if det_c is None:
            c_missing_count += 1
            print("[Task2_3] step={} C not detected, wait {}/3".format(step + 1, c_missing_count))
            stable_count = 0
            time.sleep(0.1)

            if c_missing_count >= 3:
                print("[Task2_3] C not detected 3 times, move back a little")
                dog.move(last_time=0.10, vx=-8000)
                c_missing_count = 0
                time.sleep(0.5)
            continue

        c_x = _box_center_x(det_c)
        c_missing_count = 0
        if c_x < c_x_center_min:
            print("[Task2_3] step={} C left x={:.1f}, shift left".format(step + 1, c_x))
            dog.move(last_time=0.3, vy=-25000)
            stable_count = 0
            time.sleep(0.25)
            continue

        if c_x > c_x_center_max:
            print("[Task2_3] step={} C right x={:.1f}, shift right".format(step + 1, c_x))
            dog.move(last_time=0.3, vy=25000)
            stable_count = 0
            time.sleep(0.25)
            continue

        stable_count += 1
        print(
            "[Task2_3] step={} C centered stable={}/{} x={:.1f}".format(
                step + 1,
                stable_count,
                stable_need_frames,
                c_x,
            )
        )

        if stable_count >= stable_need_frames:
            print("[Task2_3] C centered, start distance adjustment")
            return _adjust_c_distance(
                dog,
                detector,
                c_distance_target_m,
                c_distance_tolerance_m,
                stable_need_frames,
                max_distance_adjust_steps,
            )

        time.sleep(0.3)

    print("[Task2_3] C centering max steps reached, start distance adjustment")
    return _adjust_c_distance(
        dog,
        detector,
        c_distance_target_m,
        c_distance_tolerance_m,
        stable_need_frames,
        max_distance_adjust_steps,
    )


def run(dog, show_stream=False, start_yaw_deg=0.0):
    detector = DashboardInfer(show_stream=show_stream)
    try:
        return task2_3(dog, detector, start_yaw_deg)
    finally:
        detector.close()
        print("[Task2_3] detector closed")
