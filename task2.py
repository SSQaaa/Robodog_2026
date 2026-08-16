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

from tools.audio_announce import announce_dashboard
from tools.vision import DashboardInfer, analyze_infer_output
from tools.world_pose import correct_yaw, read_yaw_deg


LETTER_ALIGN_MIN_DEPTH_M = 0.35
LETTER_ALIGN_BACKWARD_VX = -20000
LETTER_ALIGN_BACKWARD_TIME_S = 0.10


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


def _move_sideways_until_letter(dog, detector, direction, max_move_seconds=5.0):
    """向左或向右横移，直到检测到字母或达到最大移动时间。"""
    direction = str(direction).lower()
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right', got {!r}".format(direction))

    vy = -25000 if direction == "left" else 25000
    direction_cn = "左" if direction == "left" else "右"
    moved_seconds = 0.0
    while moved_seconds + 1e-9 < max_move_seconds:
        if _pick_best_letter_detection(detector.infer_once()) is not None:
            print("检测到字母，停止{}移".format(direction_cn))
            dog.stop()
            return True
        step_seconds = min(1, max_move_seconds - moved_seconds)
        dog.move(last_time=step_seconds, vy=vy)
        moved_seconds += step_seconds

    dog.stop()
    print("{}移达到最大时间{:.1f}s，未检测到字母".format(
        direction_cn, max_move_seconds
    ))
    return False


def _wait_single_dashboard(detector, dog):
    """普通循环：等待画面里只有一个仪表盘。"""
    while True:
        infer_output = detector.infer_once()
        result = analyze_infer_output(infer_output)

        if result["dashboard_count"] == 1:
            return result

        if result["dashboard_count"] <= 0:
            print("SEARCH: 没有仪表盘，继续等待")
            dog.move(last_time=0.10, vx=-10000)
            time.sleep(0.25)
        else:
            print("SEARCH: 多个仪表盘，继续等待")

        time.sleep(0.2)


def _read_state_normal_loop(detector, need_frames=3, max_frames=40, interval_s=0.5):
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

        time.sleep(interval_s)


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


def _run_single_dashboard_with_detector(
    detector,
    dog,
    dashboard_index,
    letter_x_center_min,
    letter_x_center_max,
    letter_distance_target_m,
    letter_distance_tolerance_m,
    max_letter_align_adjust_count,
    max_letter_distance_adjust_count,
    max_ssi_check_retry_count,
):
    # ----------------------------
    # 二、状态机：先让字母位置合适（居中+距离），再切换到匍匐
    # ----------------------------
    letter_state = "ALIGN"  # ALIGN / DISTANCE
    letter_align_adjust_count = 0
    letter_distance_adjust_count = 0
    letter_missing_count = 0

    while True:
        infer_output = detector.infer_once()
        letter_det = _pick_best_letter_detection(infer_output)

        if letter_det is None:
            letter_missing_count += 1
            print("D{} LETTER_PRECHECK: 字母未识别，等待第{}次".format(dashboard_index, letter_missing_count))
            letter_state = "ALIGN"
            letter_align_adjust_count = 0
            letter_distance_adjust_count = 0
            time.sleep(0.1)

            if letter_missing_count >= 3:
                print("D{} LETTER_PRECHECK: 连续3次未识别到字母，后退一点".format(dashboard_index))
                dog.move(last_time=0.10, vx=-20000)
                letter_missing_count = 0
                time.sleep(0.5)
            continue

        letter = letter_det["letter"]
        letter_x_center = float(letter_det["x_center"])
        letter_distance_m = letter_det["distance_m"]
        letter_missing_count = 0

        if letter_state == "ALIGN":
            if (
                letter_distance_m is not None
                and float(letter_distance_m) < LETTER_ALIGN_MIN_DEPTH_M
            ):
                dog.move(
                    last_time=LETTER_ALIGN_BACKWARD_TIME_S,
                    vx=LETTER_ALIGN_BACKWARD_VX,
                )
                print(
                    "D{} LETTER_ALIGN: {} 距离过近，当前={:.3f}m 阈值={:.3f}m，先后退".format(
                        dashboard_index,
                        letter,
                        float(letter_distance_m),
                        LETTER_ALIGN_MIN_DEPTH_M,
                    )
                )
                time.sleep(0.5)
                continue

            if letter_x_center < letter_x_center_min:
                dog.move(last_time=0.3, vy=-25000)
                time.sleep(0.5)
                dog.move(last_time=0.15, vx=7000)
                letter_align_adjust_count += 1
                print("D{} LETTER_ALIGN: {} 偏左，左移，x_center={:.1f} 次数={}".format(dashboard_index, letter, letter_x_center, letter_align_adjust_count))
                time.sleep(0.25)
            elif letter_x_center > letter_x_center_max:
                dog.move(last_time=0.3, vy=25000)
                letter_align_adjust_count += 1
                print("D{} LETTER_ALIGN: {} 偏右，右移，x_center={:.1f} 次数={}".format(dashboard_index, letter, letter_x_center, letter_align_adjust_count))
                time.sleep(0.25)
            else:
                print("D{} LETTER_ALIGN: 居中通过，进入LETTER_DISTANCE".format(dashboard_index))
                letter_state = "DISTANCE"
                letter_distance_adjust_count = 0

            if letter_align_adjust_count > max_letter_align_adjust_count:
                print("D{} LETTER_ALIGN: 超限，重置到ALIGN".format(dashboard_index))
                letter_state = "ALIGN"
                letter_align_adjust_count = 0
                letter_distance_adjust_count = 0
            continue

        if letter_state == "DISTANCE":
            if letter_distance_m is None:
                print("D{} LETTER_DISTANCE: 深度无效，先等待0.5秒再重新确认".format(dashboard_index))
                time.sleep(0.5)

                infer_output = detector.infer_once()
                letter_det = _pick_best_letter_detection(infer_output)

                if letter_det is None or letter_det["distance_m"] is None:
                    print("D{} LETTER_DISTANCE: 等待后仍然深度无效，后退一点".format(dashboard_index))
                    dog.move(last_time=0.10, vx=-20000)
                    time.sleep(0.5)
                    continue

                letter_distance_m = letter_det["distance_m"]

            letter_error_m = letter_distance_target_m - float(letter_distance_m)
            if abs(letter_error_m) <= letter_distance_tolerance_m:
                print("D{} LETTER_DISTANCE: 距离通过，字母阶段完成，letter={} x={:.1f} d={:.3f}m".format(dashboard_index, letter, letter_x_center, letter_distance_m))
                break

            if abs(letter_error_m) > 0.30:
                letter_vx_abs = 20000
                letter_move_time_s = 0.30
            else:
                letter_vx_abs = 10000
                letter_move_time_s = 0.10

            letter_vx = letter_vx_abs if letter_error_m < 0 else -letter_vx_abs
            dog.move(last_time=letter_move_time_s, vx=letter_vx)
            letter_distance_adjust_count += 1
            print(
                "D{} LETTER_DISTANCE: {} 当前={:.3f}m 目标={:.3f}m 误差={:.3f}m vx={} time={:.2f}s 次数={}".format(
                    dashboard_index,
                    letter,
                    letter_distance_m,
                    letter_distance_target_m,
                    letter_error_m,
                    letter_vx,
                    letter_move_time_s,
                    letter_distance_adjust_count,
                )
            )
            time.sleep(0.25)

            if letter_distance_adjust_count > max_letter_distance_adjust_count:
                print("D{} LETTER_DISTANCE: 超限，回到LETTER_ALIGN".format(dashboard_index))
                letter_state = "ALIGN"
                letter_align_adjust_count = 0
                letter_distance_adjust_count = 0
            continue

    # ----------------------------
    # 三、普通循环：等待单仪表盘
    # ----------------------------
    _wait_single_dashboard(detector, dog)

    # ----------------------------
    # 四、普通if：只做ssi可见性检查
    # ----------------------------
    time.sleep(1.8)
    dog.UPDOWN()
    # time.sleep(2)
    ssi_check_retry_count = 0
    while True:
        infer_output = detector.infer_once()
        result = analyze_infer_output(infer_output)

        if result["dashboard_count"] != 1:
            print("D{} SSI_CHECK: 非单仪表盘，重新等待".format(dashboard_index))
            _wait_single_dashboard(detector, dog)
            ssi_check_retry_count = 0
            continue

        has_ssi = _has_ssi_detection(infer_output)
        if has_ssi:
            print("D{} SSI_CHECK: 已检测到ssi，进入读取仪表盘状态步骤".format(dashboard_index))
            time.sleep(1.0)
            break

        dog.move(last_time=0.02, vx=-10000)
        ssi_check_retry_count += 1
        print("D{} SSI_CHECK: 未检测到ssi，后退微调，次数={}".format(dashboard_index, ssi_check_retry_count))
        time.sleep(0.25)

        if ssi_check_retry_count > max_ssi_check_retry_count:
            print("D{} SSI_CHECK: 超限，重新等待单仪表盘".format(dashboard_index))
            _wait_single_dashboard(detector, dog)
            ssi_check_retry_count = 0

    # ----------------------------
    # 五、普通循环：先读状态，再读字母
    # ----------------------------
    final_dashboard_state = _read_state_normal_loop(
        detector,
        need_frames=3,
        max_frames=40,
    )
    print("D{} 状态读取完成：{}".format(dashboard_index, final_dashboard_state))

    final_letter = _read_letter_normal_loop(
        detector,
        need_frames=3,
        max_frames=40,
    )
    print("D{} 字母读取完成：{}".format(dashboard_index, final_letter))
    announce_dashboard(final_letter, final_dashboard_state)

    record = {
        "dashboard_index": dashboard_index,
        "dashboard_state": final_dashboard_state,
        "letter": final_letter,
    }
    print("第{}个仪表盘记录完成：{}".format(dashboard_index, record))
    return record


def task2_new(dog, detector, show_stream=False):
    """
    识别四个仪表盘，返回四条记录。
    """
    records = []

    try:
        # ============================
        # 第1个仪表盘
        # ============================
        # ----------------------------
        # 一、先运动到第1个仪表盘附近
        # ----------------------------
        dog.move(last_time=6, vx=20000)
        time.sleep(0.5)
        dog.move(last_time=3.5, vy=25000)
        found_letter = _move_sideways_until_letter(
            dog,
            detector,
            direction="right",
        )
        if not found_letter:
            print("右移未找到字母，后退一点后开始左移")
            dog.move(last_time=0.5, vx=-20000)
            _move_sideways_until_letter(
                dog,
                detector,
                direction="left",
                max_move_seconds=2.0,
            )
        time.sleep(0.5)

        # ----------------------------
        # 第1个仪表盘阈值
        # ----------------------------
        first_letter_x_center_min = 335
        first_letter_x_center_max = 385
        first_letter_distance_target_m = 0.35
        first_letter_distance_tolerance_m = 0.10
        first_max_letter_align_adjust_count = 15
        first_max_letter_distance_adjust_count = 20
        first_max_ssi_check_retry_count = 15

        first_record = _run_single_dashboard_with_detector(
            detector=detector,
            dog=dog,
            dashboard_index=1,
            letter_x_center_min=first_letter_x_center_min,
            letter_x_center_max=first_letter_x_center_max,
            letter_distance_target_m=first_letter_distance_target_m,
            letter_distance_tolerance_m=first_letter_distance_tolerance_m,
            max_letter_align_adjust_count=first_max_letter_align_adjust_count,
            max_letter_distance_adjust_count=first_max_letter_distance_adjust_count,
            max_ssi_check_retry_count=first_max_ssi_check_retry_count,
        )
        records.append(first_record)

        # ============================
        # 第2个仪表盘
        # ============================
        # ----------------------------
        # 一、先运动到第2个仪表盘附近
        # ----------------------------
        dog.UPDOWN()
        time.sleep(0.5)
        dog.revolve_180()
        time.sleep(0.5)
        dog.move(last_time=1.5, vx=20000)
        time.sleep(0.5)
        # dog.move(last_time=0.18, vz=10000) #定
        # time.sleep(0.8)
        # dog.move(last_time =5.8, vx=20000)   #定
        # time.sleep(0.5)
        # dog.revolve_90_r()
        # time.sleep(0.5)

        # ----------------------------
        # 第2个仪表盘阈值
        # ----------------------------
        second_letter_x_center_min = 335
        second_letter_x_center_max = 385
        second_letter_distance_target_m = 0.35
        second_letter_distance_tolerance_m = 0.10
        second_max_letter_align_adjust_count = 15
        second_max_letter_distance_adjust_count = 20
        second_max_ssi_check_retry_count = 15

        second_record = _run_single_dashboard_with_detector(
            detector=detector,
            dog=dog,
            dashboard_index=2,
            letter_x_center_min=second_letter_x_center_min,
            letter_x_center_max=second_letter_x_center_max,
            letter_distance_target_m=second_letter_distance_target_m,
            letter_distance_tolerance_m=second_letter_distance_tolerance_m,
            max_letter_align_adjust_count=second_max_letter_align_adjust_count,
            max_letter_distance_adjust_count=second_max_letter_distance_adjust_count,
            max_ssi_check_retry_count=second_max_ssi_check_retry_count,
        )
        records.append(second_record)

        # ============================
        # 第3个仪表盘
        # ============================
        # ----------------------------
        # 一、先运动到第3个仪表盘附近
        # ----------------------------
        dog.UPDOWN()
        time.sleep(0.5)
        dog.move(last_time=5.0, vy=-25000)
        time.sleep(0.5)
        dog.move(last_time=2.3, vx=20000)
        time.sleep(0.5)
        dog.revolve_90_r()
        time.sleep(0.5)
        # ----------------------------
        # 第3个仪表盘阈值
        # ----------------------------
        third_letter_x_center_min = 335
        third_letter_x_center_max = 385
        third_letter_distance_target_m = 0.35
        third_letter_distance_tolerance_m = 0.10
        third_max_letter_align_adjust_count = 15
        third_max_letter_distance_adjust_count = 20
        third_max_ssi_check_retry_count = 15

        third_record = _run_single_dashboard_with_detector(
            detector=detector,
            dog=dog,
            dashboard_index=3,
            letter_x_center_min=third_letter_x_center_min,
            letter_x_center_max=third_letter_x_center_max,
            letter_distance_target_m=third_letter_distance_target_m,
            letter_distance_tolerance_m=third_letter_distance_tolerance_m,
            max_letter_align_adjust_count=third_max_letter_align_adjust_count,
            max_letter_distance_adjust_count=third_max_letter_distance_adjust_count,
            max_ssi_check_retry_count=third_max_ssi_check_retry_count,
        )
        records.append(third_record)

        # ============================
        # 第4个仪表盘
        # ============================
        # ----------------------------
        # 一、先运动到第4个仪表盘附近
        # ----------------------------
        dog.UPDOWN()
        time.sleep(0.5)
        yaw_before_turn = read_yaw_deg()
        dog.revolve_90_r()
        time.sleep(2.0)
        target_yaw_after_turn = yaw_before_turn - 90.0
        print(
            "D3_AFTER_TURN_YAW: start correction before={:.3f} target={:.3f}".format(
                yaw_before_turn, target_yaw_after_turn
            )
        )
        yaw_after_correction = correct_yaw(
            dog,
            target_yaw_after_turn,
            tolerance_deg = 2,
            max_adjust_steps=3,
            stable_need_frames=3,
            sample_count=3,
            settle_s=0.5,
        )
        print("D3_AFTER_TURN_YAW: correction finished current={:.3f}".format(yaw_after_correction))
        # 换了个足端感觉没有变斜情况了
        # dog.move(last_time=0.18, vz=10000)
        time.sleep(0.5)
        dog.move(last_time=5.8, vx=20000)
        time.sleep(0.5)
        dog.revolve_90_l()
        time.sleep(0.5)

        # ----------------------------
        # 第4个仪表盘阈值
        # ----------------------------
        fourth_letter_x_center_min = 335
        fourth_letter_x_center_max = 385
        fourth_letter_distance_target_m = 0.35
        fourth_letter_distance_tolerance_m = 0.10
        fourth_max_letter_align_adjust_count = 15
        fourth_max_letter_distance_adjust_count = 20
        fourth_max_ssi_check_retry_count = 15

        fourth_record = _run_single_dashboard_with_detector(
            detector=detector,
            dog=dog,
            dashboard_index=4,
            letter_x_center_min=fourth_letter_x_center_min,
            letter_x_center_max=fourth_letter_x_center_max,
            letter_distance_target_m=fourth_letter_distance_target_m,
            letter_distance_tolerance_m=fourth_letter_distance_tolerance_m,
            max_letter_align_adjust_count=fourth_max_letter_align_adjust_count,
            max_letter_distance_adjust_count=fourth_max_letter_distance_adjust_count,
            max_ssi_check_retry_count=fourth_max_ssi_check_retry_count,
        )
        records.append(fourth_record)

        summary_list = []
        for rec in records:
            summary_list.append([rec["dashboard_index"], rec["letter"], rec["dashboard_state"]])
        print("四个仪表盘汇总列表：{}".format(summary_list))
        print("task2_new finished")
        dog.UPDOWN()
        time.sleep(0.5)

    finally:
        pass

    return records


def run(dog, show_stream=False, detector=None):
    own_detector = detector is None
    if own_detector:
        detector = DashboardInfer(show_stream=show_stream)
    try:
        return task2_new(dog, detector, show_stream=show_stream)
    finally:
        if own_detector:
            detector.close()
            print("[Task2] detector closed")

