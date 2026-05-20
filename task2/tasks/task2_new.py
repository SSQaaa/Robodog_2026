# -*- coding: utf-8 -*-
"""
任务2新版本（只做第一个仪表盘）

约束：
1. 不修改原 task2.py，只改本文件。
2. 只记录：第几个仪表盘、仪表盘状态、字母值。
3. 只有“居中+距离”使用状态机逻辑。
   - 状态机仅用于字母阶段：ALIGN -> DISTANCE
4. 其他部分不用状态机：
   - 找单仪表盘（SEARCH）用普通循环
   - 读取状态（READ_STATE）用普通循环
   - 读取字母（READ_LETTER）用普通循环
"""

import time

from tasks.dashboard_letter_detector import SimpleInfer, analyze_infer_output


def _calc_center_x_from_vertices(vertices):
    """根据四顶点计算中心x。"""
    x_sum = 0.0
    for p in vertices:
        x_sum += float(p[0])
    return x_sum / 4.0


def _pick_best_letter_detection(infer_output):
    """从当前帧里选置信度最高的字母框，并返回中心与距离。"""
    detections = infer_output.get("detections", [])
    best_det = None
    best_score = -1.0

    for det in detections:
        cid = int(det.get("class_id", -1))
        if cid < 0 or cid > 3:
            continue
        score = float(det.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_det = det

    if best_det is None:
        return None

    x1, y1, x2, y2 = best_det["xyxy"]
    _ = y1, y2
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    cid = int(best_det["class_id"])
    return {
        "letter": letter_map.get(cid, "unknown"),
        "x_center": (float(x1) + float(x2)) / 2.0,
        "distance_m": best_det.get("distance_m", None),
        "score": float(best_det.get("score", 0.0)),
    }


def _has_ssi_detection(infer_output):
    """当前帧是否检测到ssi。"""
    detections = infer_output.get("detections", [])
    for det in detections:
        if int(det.get("class_id", -1)) == 9:
            return True
    return False


def _wait_single_dashboard(detector):
    """普通循环：等待画面里只有一个仪表盘。"""
    while True:
        infer_output = detector.infer_once()
        result = analyze_infer_output(infer_output)

        if result["dashboard_count"] == 1:
            return result

        if result["dashboard_count"] <= 0:
            print("SEARCH: 没有仪表盘，继续等待")
        else:
            print("SEARCH: 多个仪表盘，继续等待")

        time.sleep(0.2)


def _read_state_normal_loop(detector, need_frames=3, max_frames=40):
    """普通循环：稳定读取仪表盘状态。"""
    last_state = None
    same_count = 0
    frame_count = 0

    while True:
        frame_count += 1
        infer_output = detector.infer_once()
        result = analyze_infer_output(infer_output)

        if result["dashboard_count"] != 1:
            print("READ_STATE: 非单仪表盘，继续读取")
            continue

        state_cn = result["dashboard_details"][0]["state_cn"]

        if state_cn == "未知":
            print("READ_STATE: 状态未知，继续读取")
            same_count = 0
            last_state = None
        else:
            if state_cn == last_state:
                same_count += 1
            else:
                last_state = state_cn
                same_count = 1

            print("READ_STATE: 当前={} 连续={}/{}".format(state_cn, same_count, need_frames))

            if same_count >= need_frames:
                return state_cn

        if frame_count >= max_frames:
            print("READ_STATE: 超时，返回未知")
            return "未知"


def _read_letter_normal_loop(detector, need_frames=3, max_frames=40):
    """普通循环：稳定读取字母。"""
    last_letter = None
    same_count = 0
    frame_count = 0

    while True:
        frame_count += 1
        infer_output = detector.infer_once()
        result = analyze_infer_output(infer_output)

        letter = result["letter"]

        if letter == "unknown":
            print("READ_LETTER: 字母未知，继续读取")
            same_count = 0
            last_letter = None
        else:
            if letter == last_letter:
                same_count += 1
            else:
                last_letter = letter
                same_count = 1

            print("READ_LETTER: 当前={} 连续={}/{}".format(letter, same_count, need_frames))

            if same_count >= need_frames:
                return letter

        if frame_count >= max_frames:
            print("READ_LETTER: 超时，返回unknown")
            return "unknown"


def task2_new(robot, show_stream=False):
    """
    第一个仪表盘识别。

    返回：
    [
      {
        "dashboard_index": 1,
        "dashboard_state": "正常/偏低/偏高/未知",
        "letter": "A/B/C/D/unknown"
      }
    ]
    """
    detector = SimpleInfer(show_stream=show_stream)
    records = []

    try:
        # ----------------------------
        # 一、先运动到第一个仪表盘附近（仿照原task2）
        # ----------------------------
        robot.move(last_time=2.5, vx=20000)
        time.sleep(0.5)

        robot.revolve_90_r()
        time.sleep(0.5)

        # ----------------------------
        # 第一个仪表盘阈值（你后续自己填）
        # ----------------------------
        # 字母先决阈值（你后续自己填）
        first_letter_x_center_min = 290
        first_letter_x_center_max = 320
        first_letter_distance_target_m = 0.35
        first_letter_distance_tolerance_m = 0.10

        # 每个状态最多微调次数
        max_letter_align_adjust_count = 5
        max_letter_distance_adjust_count = 8
        max_ssi_check_retry_count = 5

        # ----------------------------
        # 二、状态机：先让字母位置合适（居中+距离），再切换到匍匐
        # ----------------------------
        letter_state = "ALIGN"  # ALIGN / DISTANCE
        letter_align_adjust_count = 0
        letter_distance_adjust_count = 0

        while True:
            infer_output = detector.infer_once()
            letter_det = _pick_best_letter_detection(infer_output)

            if letter_det is None:
                print("LETTER_PRECHECK: 字母未识别，继续等待")
                letter_state = "ALIGN"
                letter_align_adjust_count = 0
                letter_distance_adjust_count = 0
                time.sleep(0.2)
                continue

            letter = letter_det["letter"]
            letter_x_center = float(letter_det["x_center"])
            letter_distance_m = letter_det["distance_m"]

            if letter_state == "ALIGN":
                if letter_x_center < first_letter_x_center_min:
                    robot.move(last_time=0.12, vy=-15000)
                    letter_align_adjust_count += 1
                    print("LETTER_ALIGN: {} 偏左，左移，x_center={:.1f} 次数={}".format(letter, letter_x_center, letter_align_adjust_count))
                    time.sleep(0.25)
                elif letter_x_center > first_letter_x_center_max:
                    robot.move(last_time=0.12, vy=15000)
                    letter_align_adjust_count += 1
                    print("LETTER_ALIGN: {} 偏右，右移，x_center={:.1f} 次数={}".format(letter, letter_x_center, letter_align_adjust_count))
                    time.sleep(0.25)
                else:
                    print("LETTER_ALIGN: 居中通过，进入LETTER_DISTANCE")
                    letter_state = "DISTANCE"
                    letter_distance_adjust_count = 0

                if letter_align_adjust_count > max_letter_align_adjust_count:
                    print("LETTER_ALIGN: 超限，重置到ALIGN")
                    letter_state = "ALIGN"
                    letter_align_adjust_count = 0
                    letter_distance_adjust_count = 0
                continue

            if letter_state == "DISTANCE":
                if letter_distance_m is None:
                    print("LETTER_DISTANCE: 深度无效，回到LETTER_ALIGN")
                    letter_state = "ALIGN"
                    letter_align_adjust_count = 0
                    letter_distance_adjust_count = 0
                    time.sleep(0.2)
                    continue

                letter_error_m = first_letter_distance_target_m - float(letter_distance_m)
                if abs(letter_error_m) <= first_letter_distance_tolerance_m:
                    print("LETTER_DISTANCE: 距离通过，字母阶段完成，letter={} x={:.1f} d={:.3f}m".format(letter, letter_x_center, letter_distance_m))
                    break

                if abs(letter_error_m) > 0.20:
                    letter_vx_abs = 15000
                elif abs(letter_error_m) > 0.10:
                    letter_vx_abs = 10000
                else:
                    letter_vx_abs = 7000

                letter_vx = letter_vx_abs if letter_error_m < 0 else -letter_vx_abs
                robot.move(last_time=0.10, vx=letter_vx)
                letter_distance_adjust_count += 1
                print(
                    "LETTER_DISTANCE: {} 当前={:.3f}m 目标={:.3f}m 误差={:.3f}m vx={} 次数={}".format(
                        letter,
                        letter_distance_m,
                        first_letter_distance_target_m,
                        letter_error_m,
                        letter_vx,
                        letter_distance_adjust_count,
                    )
                )
                time.sleep(0.25)

                if letter_distance_adjust_count > max_letter_distance_adjust_count:
                    print("LETTER_DISTANCE: 超限，回到LETTER_ALIGN")
                    letter_state = "ALIGN"
                    letter_align_adjust_count = 0
                    letter_distance_adjust_count = 0
                continue

        # ----------------------------
        # 三、普通循环：等待单仪表盘
        # ----------------------------
        _wait_single_dashboard(detector)

        # ----------------------------
        # 四、普通if：只做ssi可见性检查
        # ----------------------------
        ssi_check_retry_count = 0

        while True:
            infer_output = detector.infer_once()
            result = analyze_infer_output(infer_output)

            if result["dashboard_count"] != 1:
                print("SSI_CHECK: 非单仪表盘，重新等待")
                _wait_single_dashboard(detector)
                ssi_check_retry_count = 0
                continue

            has_ssi = _has_ssi_detection(infer_output)
            if has_ssi:
                print("SSI_CHECK: 已检测到ssi，进入读取仪表盘状态步骤")
                break

            robot.DOWNmove(last_time=0.10, vx=-7000)
            ssi_check_retry_count += 1
            print("SSI_CHECK: 未检测到ssi，后退微调，次数={}".format(ssi_check_retry_count))
            time.sleep(0.25)

            if ssi_check_retry_count > max_ssi_check_retry_count:
                print("SSI_CHECK: 超限，重新等待单仪表盘")
                _wait_single_dashboard(detector)
                ssi_check_retry_count = 0

        # ----------------------------
        # 五、普通循环：先读状态，再读字母
        # ----------------------------
        final_dashboard_state = _read_state_normal_loop(
            detector,
            need_frames=3,
            max_frames=40,
        )
        print("状态读取完成：{}".format(final_dashboard_state))

        final_letter = _read_letter_normal_loop(
            detector,
            need_frames=3,
            max_frames=40,
        )
        print("字母读取完成：{}".format(final_letter))

        # ----------------------------
        # 六、记录结果（只三项）
        # ----------------------------
        record = {
            "dashboard_index": 1,
            "dashboard_state": final_dashboard_state,
            "letter": final_letter,
        }
        records.append(record)
        print("第一个仪表盘记录完成：{}".format(record))

    finally:
        detector.close()
        print("detector.close() done, task2_new finished")

    return records
