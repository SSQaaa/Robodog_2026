# -*- coding: utf-8 -*-
"""
第一个仪表盘阈值采集脚本

用途：
1. 让机器狗站在你认为“最合适读取”的位置。
2. 运行本脚本，自动采样多帧。
3. 输出建议阈值，直接填到 task2_new.py：
   - first_x_center_min
   - first_x_center_max
   - first_distance_min_m
   - first_distance_max_m

说明：
- 只做采样统计，不做运动控制。
- 建议采样时保持机器人静止、仪表盘稳定在画面中。
"""

import argparse
import time
from collections import Counter

import numpy as np

from dashboard_letter_detector import SimpleInfer, analyze_infer_output


def calc_center_x(vertices):
    """根据四顶点计算中心x坐标。"""
    x_sum = 0.0
    for p in vertices:
        x_sum += float(p[0])
    return x_sum / 4.0


def percentile(values, p):
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

    x_centers = []
    distances = []
    letters = []
    states = []

    frame_id = 0
    start_ts = time.time()

    print("开始采样，请保持机器狗和仪表盘稳定...")

    try:
        while True:
            frame_id += 1
            infer_output = detector.infer_once()
            result = analyze_infer_output(infer_output)

            # 超时保护
            if time.time() - start_ts > args.max_seconds:
                print("采样超时，提前结束")
                break

            # 只采样单仪表盘场景
            if result["dashboard_count"] != 1:
                if frame_id % 15 == 0:
                    print("frame={} 不是单仪表盘场景，跳过".format(frame_id))
                continue

            detail = result["dashboard_details"][0]
            distance_m = detail["distance_m"]
            if distance_m is None:
                if frame_id % 15 == 0:
                    print("frame={} 深度无效，跳过".format(frame_id))
                continue

            x_center = calc_center_x(detail["vertices"])
            state_cn = detail["state_cn"]
            letter = result["letter"]

            x_centers.append(x_center)
            distances.append(float(distance_m))
            states.append(state_cn)
            if letter != "unknown":
                letters.append(letter)

            print(
                "sample={}/{} frame={} x_center={:.1f} distance={:.3f}m state={} letter={}".format(
                    len(x_centers),
                    args.samples,
                    frame_id,
                    x_center,
                    distance_m,
                    state_cn,
                    letter,
                )
            )

            if len(x_centers) >= args.samples:
                print("采样数量达到目标，结束采样")
                break

    finally:
        detector.close()

    if len(x_centers) < 10:
        print("有效样本太少（<10），请重新采样")
        return

    # ----------------------------
    # 统计建议值
    # ----------------------------
    x_p20 = percentile(x_centers, 20)
    x_p80 = percentile(x_centers, 80)
    d_p20 = percentile(distances, 20)
    d_p80 = percentile(distances, 80)

    # 适度放宽一点，避免阈值过紧
    x_pad = 15.0
    d_pad = 0.03

    first_x_center_min = int(round(x_p20 - x_pad))
    first_x_center_max = int(round(x_p80 + x_pad))
    first_distance_min_m = round(max(0.05, d_p20 - d_pad), 3)
    first_distance_max_m = round(d_p80 + d_pad, 3)

    state_counter = Counter(states)
    letter_counter = Counter(letters)

    print("\n================ 采样统计结果 ================")
    print("有效样本数: {}".format(len(x_centers)))
    print("x_center 中位数: {:.1f}".format(percentile(x_centers, 50)))
    print("distance 中位数: {:.3f}m".format(percentile(distances, 50)))
    print("状态分布: {}".format(dict(state_counter)))
    print("字母分布: {}".format(dict(letter_counter)))

    print("\n================ 建议填入 task2_new 的阈值 ================")
    print("first_x_center_min = {}".format(first_x_center_min))
    print("first_x_center_max = {}".format(first_x_center_max))
    print("first_distance_min_m = {:.3f}".format(first_distance_min_m))
    print("first_distance_max_m = {:.3f}".format(first_distance_max_m))

    print("\n你可以先直接用上面的建议值，再根据实测微调。")


if __name__ == "__main__":
    main()
