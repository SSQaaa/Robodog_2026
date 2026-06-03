# -*- coding: utf-8 -*-
"""
任务二到任务三衔接阈值采集脚本

用途：
1. 让机器狗先走到你认为满意的位置与朝向
2. 运行本脚本采样
3. 直接得到 main.py 里可填写的阈值：
   - c_x_center_min
   - c_x_center_max
   - bd_distance_diff_tolerance_m

说明：
- 只做采样，不做运动控制
- 建议采样时机器狗和纸箱都保持稳定
"""

import argparse
import time

import numpy as np

from dashboard_letter_detector import SimpleInfer


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


def _percentile(values, p):
    if len(values) == 0:
        return None
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=80, help="有效采样帧数")
    parser.add_argument("--max-seconds", type=float, default=120.0, help="最长采样秒数")
    parser.add_argument("--stream", action="store_true", help="显示实时检测画面")
    args = parser.parse_args()

    detector = SimpleInfer(show_stream=args.stream)

    c_x_list = []
    bd_diff_list = []

    frame_id = 0
    start_ts = time.time()

    print("开始采样，请保持机器人和ABCD纸箱稳定...")

    try:
        while True:
            frame_id += 1

            if time.time() - start_ts > args.max_seconds:
                print("采样超时，提前结束")
                break

            infer_output = detector.infer_once()
            detections = infer_output.get("detections", [])

            det_b = _pick_best_by_class(detections, 1)  # B
            det_c = _pick_best_by_class(detections, 2)  # C
            det_d = _pick_best_by_class(detections, 3)  # D

            if det_b is None or det_c is None or det_d is None:
                if frame_id % 15 == 0:
                    print("frame={} B/C/D不全，跳过".format(frame_id))
                continue

            d_b = det_b.get("distance_m", None)
            d_d = det_d.get("distance_m", None)
            if d_b is None or d_d is None:
                if frame_id % 15 == 0:
                    print("frame={} B或D深度无效，跳过".format(frame_id))
                continue

            c_x = _box_center_x(det_c)
            bd_diff = float(d_b) - float(d_d)

            c_x_list.append(c_x)
            bd_diff_list.append(bd_diff)

            print(
                "sample={}/{} frame={} Cx={:.1f} dB={:.3f} dD={:.3f} diff={:.3f}".format(
                    len(c_x_list),
                    args.samples,
                    frame_id,
                    c_x,
                    d_b,
                    d_d,
                    bd_diff,
                )
            )

            if len(c_x_list) >= args.samples:
                print("采样数量达到目标，结束采样")
                break

    finally:
        detector.close()

    if len(c_x_list) < 10:
        print("有效样本太少（<10），请重新采样")
        return

    # C中线阈值建议
    c_p20 = _percentile(c_x_list, 20)
    c_p80 = _percentile(c_x_list, 80)
    c_pad = 5.0
    c_x_center_min = int(round(c_p20 - c_pad))
    c_x_center_max = int(round(c_p80 + c_pad))

    # B/D 等距差容差建议
    diff_abs = [abs(v) for v in bd_diff_list]
    diff_p80 = _percentile(diff_abs, 80)
    bd_distance_diff_tolerance_m = round(diff_p80 + 0.01, 3)

    print("\n================ 采样统计结果 ================")
    print("有效样本数: {}".format(len(c_x_list)))
    print("C x_center 中位数: {:.1f}".format(_percentile(c_x_list, 50)))
    print("|dB-dD| 中位数: {:.3f}m".format(_percentile(diff_abs, 50)))

    print("\n================ 建议填入 main.py 的阈值 ================")
    print("c_x_center_min = {}".format(c_x_center_min))
    print("c_x_center_max = {}".format(c_x_center_max))
    print("bd_distance_diff_tolerance_m = {:.3f}".format(bd_distance_diff_tolerance_m))

    print("\n你可以先直接用上面的建议值，再根据实测微调。")


if __name__ == "__main__":
    main()
