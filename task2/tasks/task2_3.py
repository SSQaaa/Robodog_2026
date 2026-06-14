import math
import os
import re
import subprocess
import time


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


def _adjust_c_distance(robot, detector, target_m, tolerance_m, stable_need_frames, max_adjust_steps):
    """只根据C的深度距离前后移动，不再做左右和角度调整。"""
    stable_count = 0

    for step in range(max_adjust_steps):
        infer_output = detector.infer_once()
        detections = infer_output.get("detections", [])
        det_c = _pick_best_by_class(detections, 2)

        if det_c is None:
            print("BRIDGE_DISTANCE: step={} 未检测到C，继续等待".format(step + 1))
            stable_count = 0
            time.sleep(0.3)
            continue

        distance_m = det_c.get("distance_m", None)
        if distance_m is None:
            print("BRIDGE_DISTANCE: step={} C深度无效，继续等待".format(step + 1))
            stable_count = 0
            time.sleep(0.3)
            continue

        distance_m = float(distance_m)
        error_m = target_m - distance_m

        if abs(error_m) <= tolerance_m:
            stable_count += 1
            print(
                "BRIDGE_DISTANCE: step={} 通过帧 {}/{} 当前={:.3f}m 目标={:.3f}m 误差={:.3f}m".format(
                    step + 1,
                    stable_count,
                    stable_need_frames,
                    distance_m,
                    target_m,
                    error_m,
                )
            )
            if stable_count >= stable_need_frames:
                print("BRIDGE_DISTANCE: 距离调整完成")
                return distance_m
            time.sleep(0.2)
            continue

        stable_count = 0

        if abs(error_m) > 0.50:
            vx_abs = 15000
        else:
            vx_abs = 10000

        # error_m = target - current；远了前进，近了后退。
        if error_m < 0:
            vx_cmd = vx_abs
        else:
            vx_cmd = -vx_abs

        print(
            "BRIDGE_DISTANCE: step={} 当前={:.3f}m 目标={:.3f}m 误差={:.3f}m vx={}".format(
                step + 1,
                distance_m,
                target_m,
                error_m,
                vx_cmd,
            )
        )
        robot.move(last_time=0.12, vx=vx_cmd)
        time.sleep(0.4)

    print("BRIDGE_DISTANCE: 达到最大调整次数，按当前距离继续")
    return None


def _normalize_yaw_error(target_yaw, current_yaw):
    """把角度误差限制在 -180 到 180 之间，避免正负180跳变。"""
    error = float(target_yaw) - float(current_yaw)
    while error > 180.0:
        error -= 360.0
    while error < -180.0:
        error += 360.0
    return error


def _average_yaw_deg(yaw_list):
    """角度平均要用sin/cos，避免179度和-179度被平均成0度。"""
    sin_sum = 0.0
    cos_sum = 0.0
    for yaw in yaw_list:
        yaw_rad = math.radians(float(yaw))
        sin_sum += math.sin(yaw_rad)
        cos_sum += math.cos(yaw_rad)
    return math.degrees(math.atan2(sin_sum, cos_sum))


def read_current_yaw_deg(sample_count=3):
    """
    读取当前IMU yaw角度。
    注意：ROS melodic在这台机器上使用python2，所以这里通过/usr/bin/python2读取。
    """
    script_path = os.path.join(os.path.dirname(__file__), "read_ros_imu_yaw.py")
    yaw_list = []

    for _ in range(sample_count):
        output = subprocess.check_output(
            ["/usr/bin/python2", script_path, "--once"],
            stderr=subprocess.STDOUT,
            timeout=8,
        )
        if not isinstance(output, str):
            output = output.decode("utf-8")

        match = re.search(r"yaw=([-+]?\d+\.?\d*)", output)
        if match is None:
            print(output)
            raise RuntimeError("没有从IMU输出中读到yaw")

        yaw_list.append(float(match.group(1)))
        time.sleep(0.05)

    return _average_yaw_deg(yaw_list)


def rotate_to_relative_yaw(robot, target_yaw_deg, tolerance_deg=3.0):
    """
    转到相对于开机初始0度的目标角度。
    """
    stable_need_frames = 3
    max_adjust_steps = 40   #最大的调整次数，超过这个次数就认为调整失败
    yaw_vz_small = 9555
    yaw_vz_large = 10000
    stable_count = 0

    for step in range(max_adjust_steps):
        current_yaw = read_current_yaw_deg()
        error = _normalize_yaw_error(target_yaw_deg, current_yaw)

        if abs(error) <= tolerance_deg:
            stable_count += 1
            print(
                "YAW: step={} 通过帧 {}/{} 当前={:.3f} 目标={:.3f} 误差={:.3f}".format(
                    step + 1,
                    stable_count,
                    stable_need_frames,
                    current_yaw,
                    target_yaw_deg,
                    error,
                )
            )
            if stable_count >= stable_need_frames:
                print("YAW: 角度调整完成")
                return current_yaw
            time.sleep(0.2)
            continue

        stable_count = 0

        if abs(error) > 30.0:
            vz_abs = yaw_vz_large
        else:
            vz_abs = yaw_vz_small

        # 约定：正vz向右转，yaw会变小；负vz向左转，yaw会变大。
        if error > 0:
            vz_cmd = -vz_abs
        else:
            vz_cmd = vz_abs

        print(
            "YAW: step={} 当前={:.3f} 目标={:.3f} 误差={:.3f} vz={}".format(
                step + 1,
                current_yaw,
                target_yaw_deg,
                error,
                vz_cmd,
            )
        )
        robot.move(last_time=0.1, vz=vz_cmd)
        time.sleep(0.15)

    print("YAW: 达到最大调整次数，按当前角度继续")
    return read_current_yaw_deg()


def task2_3(robot, detector):
    """
    任务二和任务三衔接：
    1) 使用IMU yaw转到相对于开机0度的目标角度
    2) 让C尽量在画面中线附近
    """
    c_x_center_min = 350
    c_x_center_max = 400
    c_distance_target_m = 1.50
    c_distance_tolerance_m = 0.20
    target_yaw_deg = 93.0  #非常重要的一个参数，需要根据实际情况调整。这个角度是相对于开机初始0度的，也就是说如果开机时机器人朝向不正，那么这个目标角度也要相应调整。

    stable_need_frames = 3
    max_adjust_steps = 30
    max_distance_adjust_steps = 30

    stable_count = 0

    print("BRIDGE: 开始IMU角度调整")
    final_yaw = rotate_to_relative_yaw(robot, target_yaw_deg)
    print("BRIDGE: IMU角度调整完成，当前yaw={:.3f}，开始左右平移".format(final_yaw))

    for step in range(max_adjust_steps):
        infer_output = detector.infer_once()
        detections = infer_output.get("detections", [])

        det_c = _pick_best_by_class(detections, 2)

        if det_c is None:
            print("BRIDGE: step={} 未检测到C，直走搜索".format(step + 1))
            robot.move(last_time=0.2, vx=10000)
            stable_count = 0
            time.sleep(0.5)
            continue

        c_x = _box_center_x(det_c)
        if c_x < c_x_center_min:
            print("BRIDGE: step={} C偏左 x={:.1f}，左移微调".format(step + 1, c_x))
            robot.move(last_time=0.10, vy=-20000)
            stable_count = 0
            time.sleep(0.25)
            continue

        if c_x > c_x_center_max:
            print("BRIDGE: step={} C偏右 x={:.1f}，右移微调".format(step + 1, c_x))
            robot.move(last_time=0.10, vy=20000)
            stable_count = 0
            time.sleep(0.25)
            continue

        stable_count += 1
        print(
            "BRIDGE: step={} C居中通过帧 {}/{} | Cx={:.1f}".format(
                step + 1,
                stable_count,
                stable_need_frames,
                c_x,
            )
        )

        if stable_count >= stable_need_frames:
            print("BRIDGE: 左右调整完成，开始调整C距离")
            final_distance = _adjust_c_distance(
                robot,
                detector,
                c_distance_target_m,
                c_distance_tolerance_m,
                stable_need_frames,
                max_distance_adjust_steps,
            )
            if final_distance is not None:
                print("BRIDGE: C距离调整完成，当前距离={:.3f}m，进入任务三".format(final_distance))
            else:
                print("BRIDGE: C距离调整结束，进入任务三")
            break

        time.sleep(0.3)

    else:
        print("BRIDGE: C居中达到最大调整次数，按当前位置开始调整C距离")
        final_distance = _adjust_c_distance(
            robot,
            detector,
            c_distance_target_m,
            c_distance_tolerance_m,
            stable_need_frames,
            max_distance_adjust_steps,
        )
        if final_distance is not None:
            print("BRIDGE: C距离调整完成，当前距离={:.3f}m，进入任务三".format(final_distance))
        else:
            print("BRIDGE: C距离调整结束，进入任务三")
