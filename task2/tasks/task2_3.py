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


def task2_3(robot, detector):
    """
    任务二和任务三衔接：
    1) 让C尽量在画面中线附近
    2) 让B和D的距离尽量相等，用来修正朝向
    """
    c_x_center_min = 350
    c_x_center_max = 400
    bd_distance_diff_tolerance_m = 0.06

    stable_need_frames = 3
    max_adjust_steps = 30
    yaw_vz_small = 10000
    yaw_vz_large = 12000
    yaw_direction_sign = -1

    stable_count = 0
    stage = "LATERAL"

    for step in range(max_adjust_steps):
        infer_output = detector.infer_once()
        detections = infer_output.get("detections", [])

        det_b = _pick_best_by_class(detections, 1)
        det_c = _pick_best_by_class(detections, 2)
        det_d = _pick_best_by_class(detections, 3)

        if stage == "LATERAL":
            if det_c is None:
                print("BRIDGE: step={} 未检测到C，直走搜索".format(step + 1))
                robot.move(last_time=0.08, vx=10000)
                stable_count = 0
                time.sleep(1)
                continue

            c_x = _box_center_x(det_c)
            if c_x < c_x_center_min:
                print("BRIDGE: step={} C偏左 x={:.1f}，左移微调".format(step + 1, c_x))
                robot.move(last_time=0.10, vy=-15000)
                stable_count = 0
                time.sleep(0.5)
                continue

            if c_x > c_x_center_max:
                print("BRIDGE: step={} C偏右 x={:.1f}，右移微调".format(step + 1, c_x))
                robot.move(last_time=0.10, vy=15000)
                stable_count = 0
                time.sleep(0.5)
                continue

            stage = "YAW"
            stable_count = 0
            print("BRIDGE: step={} 左右调整完成，进入角度调整阶段".format(step + 1))

        # 进入YAW阶段后，不再做左右调整
        if det_b is None or det_d is None:
            print("BRIDGE: step={} B或D未检测到，直走搜索".format(step + 1))
            robot.move(last_time=0.08, vx=10000)
            stable_count = 0
            time.sleep(1)
            continue

        d_b = det_b.get("distance_m", None)
        d_d = det_d.get("distance_m", None)
        if d_b is None or d_d is None:
            print("BRIDGE: step={} B或D深度无效，后退搜索".format(step + 1))
            robot.move(last_time=0.08, vx=-10000)
            stable_count = 0
            time.sleep(1)
            continue

        depth_diff = float(d_b) - float(d_d)

        if abs(depth_diff) <= bd_distance_diff_tolerance_m:
            stable_count += 1
            if det_c is not None:
                c_x = _box_center_x(det_c)
            else:
                c_x = -1.0
            print(
                "BRIDGE: step={} 通过帧 {}/{} | Cx={:.1f} dB={:.3f} dD={:.3f} diff={:.3f}".format(
                    step + 1,
                    stable_count,
                    stable_need_frames,
                    c_x,
                    d_b,
                    d_d,
                    depth_diff,
                )
            )
            if stable_count >= stable_need_frames:
                print("BRIDGE: 调整完成，进入任务三")
                break
            time.sleep(0.5)
            continue

        stable_count = 0

        if abs(depth_diff) > 0.10:
            vz_abs = yaw_vz_large
        else:
            vz_abs = yaw_vz_small

        if depth_diff > 0:
            vz_cmd = yaw_direction_sign * vz_abs
        else:
            vz_cmd = -yaw_direction_sign * vz_abs

        print(
            "BRIDGE: step={} 角度修正 dB={:.3f} dD={:.3f} diff={:.3f} vz={}".format(
                step + 1,
                d_b,
                d_d,
                depth_diff,
                vz_cmd,
            )
        )
        robot.move(last_time=0.08, vz=vz_cmd)
        time.sleep(1.5)

    else:
        print("BRIDGE: 达到最大调整次数，按当前姿态进入任务三")
