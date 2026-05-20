# -*- coding: utf-8 -*-
"""
任务2新版本（只做第一个仪表盘）

约束：
1. 不修改原 task2.py，只改本文件。
2. 只记录：第几个仪表盘、仪表盘状态、字母值。
3. 只有“居中+距离”使用状态机逻辑。
   - 状态机仅包含：ALIGN -> DISTANCE -> ALIGN_FINE
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


def task2_new(robot):
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
    detector = SimpleInfer()
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
        # 粗居中阈值
        first_x_center_min = 210
        first_x_center_max = 430

        # 精居中阈值
        first_x_center_fine_min = 300
        first_x_center_fine_max = 340

        # 距离闭环目标和容差
        first_distance_target_m = 0.35
        first_distance_tolerance_m = 0.40

        # 每个状态最多微调次数
        max_align_adjust_count = 3
        max_distance_adjust_count = 5
        max_align_fine_adjust_count = 3

        # ----------------------------
        # 二、普通循环：先等待单仪表盘
        # ----------------------------
        _wait_single_dashboard(detector)

        # ----------------------------
        # 三、状态机（仅居中+距离）
        # ----------------------------
        state = "ALIGN"  # 只会在 ALIGN / DISTANCE / ALIGN_FINE 三个状态里切换
        align_adjust_count = 0
        distance_adjust_count = 0
        align_fine_adjust_count = 0

        while True:
            infer_output = detector.infer_once()
            result = analyze_infer_output(infer_output)

            # 状态机内部也要求单仪表盘，否则回到外层重新等待
            if result["dashboard_count"] != 1:
                print("ADJUST: 非单仪表盘，跳出状态机并重新等待")
                _wait_single_dashboard(detector)
                state = "ALIGN"
                align_adjust_count = 0
                distance_adjust_count = 0
                align_fine_adjust_count = 0
                continue

            detail = result["dashboard_details"][0]
            x_center = _calc_center_x_from_vertices(detail["vertices"])
            distance_m = detail["distance_m"]

            if state == "ALIGN":
                if x_center < first_x_center_min:
                    robot.move(last_time=0.2, vy=-20000)
                    align_adjust_count += 1
                    print("ALIGN: 偏左，左移，x_center={:.1f} 次数={}".format(x_center, align_adjust_count))
                    time.sleep(0.3)
                elif x_center > first_x_center_max:
                    robot.move(last_time=0.2, vy=20000)
                    align_adjust_count += 1
                    print("ALIGN: 偏右，右移，x_center={:.1f} 次数={}".format(x_center, align_adjust_count))
                    time.sleep(0.3)
                else:
                    print("ALIGN: 粗居中通过，进入DISTANCE")
                    state = "DISTANCE"
                    distance_adjust_count = 0

                if align_adjust_count > max_align_adjust_count:
                    print("ALIGN: 超限，重新等待单仪表盘")
                    _wait_single_dashboard(detector)
                    state = "ALIGN"
                    align_adjust_count = 0
                    distance_adjust_count = 0
                    align_fine_adjust_count = 0
                continue

            if state == "DISTANCE":
                if distance_m is None:
                    print("DISTANCE: 深度无效，重新等待单仪表盘")
                    _wait_single_dashboard(detector)
                    state = "ALIGN"
                    align_adjust_count = 0
                    distance_adjust_count = 0
                    align_fine_adjust_count = 0
                    continue

                error_m = first_distance_target_m - float(distance_m)

                if abs(error_m) <= first_distance_tolerance_m:
                    print("DISTANCE: 距离通过，进入ALIGN_FINE")
                    state = "ALIGN_FINE"
                    align_fine_adjust_count = 0
                    continue

                if abs(error_m) > 0.20:
                    vx_abs = 15000
                elif abs(error_m) > 0.10:
                    vx_abs = 10000
                else:
                    vx_abs = 7000

                # 目标是：远了前进，近了后退
                # error_m = target - current
                # error_m < 0 -> current更远 -> 前进（正vx）
                # error_m > 0 -> current更近 -> 后退（负vx）
                vx = vx_abs if error_m < 0 else -vx_abs
                robot.move(last_time=0.12, vx=vx)
                distance_adjust_count += 1

                print(
                    "DISTANCE: 当前={:.3f}m 目标={:.3f}m 误差={:.3f}m vx={} 次数={}".format(
                        distance_m,
                        first_distance_target_m,
                        error_m,
                        vx,
                        distance_adjust_count,
                    )
                )
                time.sleep(0.25)

                if distance_adjust_count > max_distance_adjust_count:
                    print("DISTANCE: 超限，重新等待单仪表盘")
                    _wait_single_dashboard(detector)
                    state = "ALIGN"
                    align_adjust_count = 0
                    distance_adjust_count = 0
                    align_fine_adjust_count = 0
                continue

            if state == "ALIGN_FINE":
                if x_center < first_x_center_fine_min:
                    robot.move(last_time=0.1, vy=-15000)
                    align_fine_adjust_count += 1
                    print("ALIGN_FINE: 偏左，微调左移，x_center={:.1f} 次数={}".format(x_center, align_fine_adjust_count))
                    time.sleep(0.25)
                elif x_center > first_x_center_fine_max:
                    robot.move(last_time=0.1, vy=15000)
                    align_fine_adjust_count += 1
                    print("ALIGN_FINE: 偏右，微调右移，x_center={:.1f} 次数={}".format(x_center, align_fine_adjust_count))
                    time.sleep(0.25)
                else:
                    print("ALIGN_FINE: 精居中通过，结束调整状态机")
                    break

                if align_fine_adjust_count > max_align_fine_adjust_count:
                    print("ALIGN_FINE: 超限，重新等待单仪表盘")
                    _wait_single_dashboard(detector)
                    state = "ALIGN"
                    align_adjust_count = 0
                    distance_adjust_count = 0
                    align_fine_adjust_count = 0
                continue

            # 理论不会到这，保险处理
            print("ADJUST: 未知状态，重置到ALIGN")
            state = "ALIGN"

        # ----------------------------
        # 四、普通循环：先读状态，再读字母
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
        # 五、记录结果（只三项）
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
