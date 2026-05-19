# -*- coding: utf-8 -*-
"""
任务2：无运动学纯检测入口

当前只打印四项：
1. 仪表盘四顶点坐标
2. 仪表盘状态（正常/偏低/偏高）
3. 仪表盘距离
4. 字母
"""

import argparse
import time

from dashboard_letter_detector import SimpleInfer, analyze_infer_output


def print_result_cn(frame_id, infer_ms, result):
    print("\n==============================")
    print("帧号: {} | 推理耗时: {:.1f} ms".format(frame_id, infer_ms))
    print("仪表盘数量: {}".format(result["dashboard_count"]))
    print("字母: {}".format(result["letter"]))

    if result["dashboard_count"] == 0:
        print("当前帧没有仪表盘")
        return

    for detail in result["dashboard_details"]:
        distance_text = "{:.3f} m".format(detail["distance_m"]) if detail["distance_m"] is not None else "未知"
        print("仪表盘#{}".format(detail["index"]))
        print("  四顶点坐标: {}".format(detail["vertices"]))
        print("  状态: {}".format(detail["state_cn"]))
        print("  距离: {}".format(distance_text))
        print("  字母: {}".format(detail["letter"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="显示实时画面")
    parser.add_argument("--interval", type=float, default=0.02, help="打印间隔秒")
    args = parser.parse_args()

    detector = SimpleInfer(show_stream=args.stream)
    frame_id = 0

    try:
        while True:
            infer_output = detector.infer_once()
            result = analyze_infer_output(infer_output)

            frame_id += 1
            print_result_cn(frame_id, infer_output.get("infer_ms", 0.0), result)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n手动停止检测")
    finally:
        detector.close()
        print("相机与检测器已关闭")


if __name__ == "__main__":
    main()
