# -*- coding: utf-8 -*-
"""
仪表盘状态后处理只读调试脚本。

作用：
1. 打开新相机和新模型，只读取画面与检测结果。
2. 不包含任何机器狗运动学代码。
3. 显示 dashboard ROI、二值化图、mask 后二值化图、最大轮廓/射线扫描、指针点。
4. 打印 angle / cross_value / state，方便判断到底是阈值问题、ROI 问题，还是角度分类范围问题。
"""

import argparse
import math
import time

import cv2
import numpy as np

from dashboard_letter_detector import (
    DASHBOARD_ID,
    SSI_ID,
    POINTER_THRESHOLD,
    NORMAL_ANGLE_MIN,
    NORMAL_ANGLE_MAX,
    STATE_CN_MAP,
    SimpleInfer,
    center_of_box,
    nearest_ssi_box,
    refine_box,
)


def pick_best_dashboard(detections):
    """只取置信度最高的一个仪表盘，避免多个框干扰调试。"""
    best_det = None
    best_score = -1.0
    for det in detections:
        if int(det["class_id"]) != DASHBOARD_ID:
            continue
        score = float(det["score"])
        if score > best_score:
            best_score = score
            best_det = det
    return best_det


def collect_ssi_boxes(detections):
    """收集当前画面里所有 ssi 框，后面选择离仪表盘最近的一个。"""
    ssi_boxes = []
    for det in detections:
        if int(det["class_id"]) == SSI_ID:
            ssi_boxes.append(det["xyxy"])
    return ssi_boxes


def make_three_quarter_mask(binary_shape, cut_width_deg):
    """制作四分之三椭圆 mask：保留仪表盘上、左、右区域，挖掉下方 ssi 附近区域。"""
    h, w = binary_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    axes = (max(1, w // 2), max(1, h // 2))

    # 先画完整椭圆，再把下方一块扇形挖掉。
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    cut_width_deg = max(0, min(180, int(cut_width_deg)))
    start_angle = 90 - cut_width_deg // 2
    end_angle = 90 + cut_width_deg // 2
    cv2.ellipse(mask, center, axes, 0, start_angle, end_angle, 0, -1)

    return mask


def find_pointer_by_contour(binary, roi, x1, y1):
    """原来的方法：找最大轮廓，并用最小外接矩形中心作为指针点。"""
    contours_data = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[-2]

    contour_view = roi.copy()
    if len(contours) == 0:
        return None, contour_view, 0.0, "contour"

    max_contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(max_contour))
    rect = cv2.minAreaRect(max_contour)
    pointer_point = np.array([int(rect[0][0]) + x1, int(rect[0][1]) + y1], dtype=np.float32)

    cv2.drawContours(contour_view, [max_contour], -1, (0, 255, 0), 2)
    cv2.circle(contour_view, (int(rect[0][0]), int(rect[0][1])), 5, (0, 0, 255), -1)

    return pointer_point, contour_view, contour_area, "contour"


def is_black_near(binary, x, y):
    """看某个采样点附近 3x3 范围有没有黑色目标。"""
    h, w = binary.shape[:2]
    x1 = max(0, x - 1)
    y1 = max(0, y - 1)
    x2 = min(w, x + 2)
    y2 = min(h, y + 2)
    return np.max(binary[y1:y2, x1:x2]) > 0


def find_pointer_by_ray(binary, roi, x1, y1, dashboard_center, ray_min_r, ray_max_r):
    """射线扫描法：从仪表盘中心向外扫线，选择连续黑色像素最长的方向。"""
    h, w = binary.shape[:2]
    center_x = float(dashboard_center[0]) - float(x1)
    center_y = float(dashboard_center[1]) - float(y1)

    max_possible_r = int(min(w, h) * 0.50)
    ray_min_r = max(1, int(ray_min_r))
    ray_max_r = int(ray_max_r)
    if ray_max_r <= ray_min_r:
        ray_max_r = max_possible_r
    ray_max_r = min(ray_max_r, max_possible_r)

    best_score = -1
    best_angle = 0
    best_point_roi = None
    best_run_start = 0
    best_run_end = 0

    for angle_deg in range(0, 360):
        rad = math.radians(angle_deg)
        cos_v = math.cos(rad)
        sin_v = math.sin(rad)

        current_run = 0
        current_start = ray_min_r
        max_run = 0
        max_run_start = ray_min_r
        max_run_end = ray_min_r
        hit_count = 0
        farthest_hit = 0

        for r in range(ray_min_r, ray_max_r + 1):
            px = int(round(center_x + float(r) * cos_v))
            py = int(round(center_y + float(r) * sin_v))

            if px < 0 or px >= w or py < 0 or py >= h:
                if current_run > max_run:
                    max_run = current_run
                    max_run_start = current_start
                    max_run_end = r - 1
                current_run = 0
                continue

            if is_black_near(binary, px, py):
                hit_count += 1
                farthest_hit = r
                if current_run == 0:
                    current_start = r
                current_run += 1
            else:
                if current_run > max_run:
                    max_run = current_run
                    max_run_start = current_start
                    max_run_end = r - 1
                current_run = 0

        if current_run > max_run:
            max_run = current_run
            max_run_start = current_start
            max_run_end = ray_max_r

        # 连续长度最重要，其次是总命中数，最后偏向更远的点。
        score = max_run * 1000 + hit_count * 10 + farthest_hit
        if score > best_score:
            best_score = score
            best_angle = angle_deg
            best_run_start = max_run_start
            best_run_end = max_run_end
            use_r = max_run_end if max_run_end > 0 else farthest_hit
            best_point_roi = (
                int(round(center_x + float(use_r) * cos_v)),
                int(round(center_y + float(use_r) * sin_v)),
            )

    method_view = roi.copy()
    if best_point_roi is None or best_score <= 0:
        return None, method_view, 0.0, "ray"

    rad = math.radians(best_angle)
    line_start = (
        int(round(center_x + float(best_run_start) * math.cos(rad))),
        int(round(center_y + float(best_run_start) * math.sin(rad))),
    )
    line_end = (
        int(round(center_x + float(best_run_end) * math.cos(rad))),
        int(round(center_y + float(best_run_end) * math.sin(rad))),
    )

    cv2.line(method_view, line_start, line_end, (0, 255, 0), 4)
    cv2.circle(method_view, best_point_roi, 5, (0, 0, 255), -1)
    cv2.circle(method_view, (int(center_x), int(center_y)), 4, (255, 0, 0), -1)

    pointer_point = np.array([best_point_roi[0] + x1, best_point_roi[1] + y1], dtype=np.float32)
    return pointer_point, method_view, float(best_score), "ray"


def calc_state_debug(image_raw, dashboard_box, ssi_box, pointer_threshold, use_mask, mask_cut_width, pointer_method, ray_min_r, ray_max_r):
    """复刻仪表盘状态后处理，同时把中间结果全部返回出来。"""
    frame_h, frame_w = image_raw.shape[:2]
    x1, y1, x2, y2 = refine_box(dashboard_box, frame_w, frame_h, ratio=0.5)

    roi = image_raw[y1:y2, x1:x2].copy()
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_raw = cv2.threshold(gray, pointer_threshold, 255, cv2.THRESH_BINARY_INV)

    if use_mask:
        mask = make_three_quarter_mask(binary_raw.shape, mask_cut_width)
        binary = cv2.bitwise_and(binary_raw, binary_raw, mask=mask)
    else:
        mask = np.ones(binary_raw.shape, dtype=np.uint8) * 255
        binary = binary_raw.copy()

    dashboard_center = center_of_box(dashboard_box)
    ssi_center = center_of_box(ssi_box)

    if pointer_method == 1:
        pointer_point, method_view, contour_area, method_name = find_pointer_by_ray(binary, roi, x1, y1, dashboard_center, ray_min_r, ray_max_r)
        method_view[mask == 0] = method_view[mask == 0] // 3
    else:
        pointer_point, method_view, contour_area, method_name = find_pointer_by_contour(binary, roi, x1, y1)
        method_view[mask == 0] = method_view[mask == 0] // 3

    if pointer_point is None:
        return {
            "roi": roi,
            "binary_raw": binary_raw,
            "binary": binary,
            "mask": mask,
            "contour_view": method_view,
            "state": "unknown",
            "state_cn": STATE_CN_MAP.get("unknown", "未知"),
            "angle": None,
            "cross_value": None,
            "pointer_point": None,
            "roi_box": [x1, y1, x2, y2],
            "contour_area": contour_area,
            "method_name": method_name,
        }

    v1_x = ssi_center[0] - dashboard_center[0]
    v1_y = ssi_center[1] - dashboard_center[1]
    v2_x = pointer_point[0] - dashboard_center[0]
    v2_y = pointer_point[1] - dashboard_center[1]

    norm_v1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
    norm_v2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)
    if norm_v1 <= 1e-6 or norm_v2 <= 1e-6:
        state = "unknown"
        angle = None
        cross_value = None
    else:
        cos_value = (v1_x * v2_x + v1_y * v2_y) / (norm_v1 * norm_v2)
        cos_value = max(-1.0, min(1.0, cos_value))
        angle = math.degrees(math.acos(cos_value))
        cross_value = v1_x * v2_y - v2_x * v1_y

        if NORMAL_ANGLE_MIN <= angle <= NORMAL_ANGLE_MAX:
            state = "normal"
        elif cross_value > 0:
            state = "low"
        else:
            state = "high"

    return {
        "roi": roi,
        "binary_raw": binary_raw,
        "binary": binary,
        "mask": mask,
        "contour_view": method_view,
        "state": state,
        "state_cn": STATE_CN_MAP.get(state, "未知"),
        "angle": angle,
        "cross_value": cross_value,
        "pointer_point": pointer_point,
        "roi_box": [x1, y1, x2, y2],
        "contour_area": contour_area,
        "method_name": method_name,
    }


def draw_debug_raw(image_raw, dashboard_det, ssi_box, debug_info, pointer_threshold, use_mask, mask_cut_width, pointer_method, ray_min_r, ray_max_r, infer_ms):
    """在原图上画出仪表盘框、ssi 框、ROI 框、指针方向和状态数值。"""
    view = image_raw.copy()
    dashboard_box = dashboard_det["xyxy"]
    x1, y1, x2, y2 = dashboard_box

    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(view, "dashboard", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    sx1, sy1, sx2, sy2 = ssi_box
    cv2.rectangle(view, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
    cv2.putText(view, "ssi", (sx1, max(20, sy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    rx1, ry1, rx2, ry2 = debug_info["roi_box"]
    cv2.rectangle(view, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
    cv2.putText(view, "pointer ROI", (rx1, max(20, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    dashboard_center = center_of_box(dashboard_box)
    ssi_center = center_of_box(ssi_box)
    dc = (int(dashboard_center[0]), int(dashboard_center[1]))
    sc = (int(ssi_center[0]), int(ssi_center[1]))

    cv2.circle(view, dc, 5, (0, 255, 255), -1)
    cv2.circle(view, sc, 5, (255, 0, 0), -1)
    cv2.line(view, dc, sc, (255, 0, 0), 2)

    pointer_point = debug_info["pointer_point"]
    if pointer_point is not None:
        pc = (int(pointer_point[0]), int(pointer_point[1]))
        cv2.circle(view, pc, 6, (0, 0, 255), -1)
        cv2.line(view, dc, pc, (0, 0, 255), 2)

    distance_m = dashboard_det.get("distance_m", None)
    distance_text = "None" if distance_m is None else "{:.3f}m".format(distance_m)
    angle_text = "None" if debug_info["angle"] is None else "{:.2f}".format(debug_info["angle"])
    cross_text = "None" if debug_info["cross_value"] is None else "{:.2f}".format(debug_info["cross_value"])

    lines = [
        "infer_ms: {:.1f}".format(infer_ms),
        "threshold: {}".format(pointer_threshold),
        "mask: {} cut:{}".format(int(use_mask), int(mask_cut_width)),
        "method: {}".format(debug_info["method_name"]),
        "ray_r: {}~{}".format(int(ray_min_r), int(ray_max_r)),
        "distance: {}".format(distance_text),
        "angle: {}".format(angle_text),
        "cross: {}".format(cross_text),
        "score/area: {:.1f}".format(debug_info["contour_area"]),
        "state: {}".format(debug_info["state_cn"]),
    ]

    y = 28
    for text in lines:
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30

    return view


def show_waiting(image_raw, message):
    view = image_raw.copy()
    cv2.putText(view, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow("debug_raw", view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=POINTER_THRESHOLD, help="指针二值化阈值，默认使用当前后处理里的 POINTER_THRESHOLD")
    parser.add_argument("--interval", type=float, default=0.05, help="每次刷新后的等待时间，单位秒")
    parser.add_argument("--no-mask", action="store_true", help="启动时先不使用四分之三椭圆 mask")
    parser.add_argument("--mask-cut-width", type=int, default=90, help="底部挖掉的角度范围，默认 90 度")
    parser.add_argument("--method", type=int, default=1, help="指针方法：0=最大轮廓，1=射线扫描")
    parser.add_argument("--ray-min-r", type=int, default=18, help="射线从离中心多少像素开始统计，用来避开中心和短指针")
    parser.add_argument("--ray-max-r", type=int, default=70, help="射线统计到离中心多少像素，太大可能扫到刻度文字")
    args = parser.parse_args()

    pointer_threshold = int(args.threshold)
    use_mask_init = 0 if args.no_mask else 1
    mask_cut_width = int(args.mask_cut_width)
    pointer_method = 1 if int(args.method) == 1 else 0
    ray_min_r = int(args.ray_min_r)
    ray_max_r = int(args.ray_max_r)

    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("controls", 620, 220)
    cv2.createTrackbar("PointerThreshold", "controls", pointer_threshold, 255, lambda value: None)
    cv2.createTrackbar("UseMask", "controls", use_mask_init, 1, lambda value: None)
    cv2.createTrackbar("MaskCutWidth", "controls", mask_cut_width, 180, lambda value: None)
    cv2.createTrackbar("Method", "controls", pointer_method, 1, lambda value: None)
    cv2.createTrackbar("RayMinR", "controls", ray_min_r, 120, lambda value: None)
    cv2.createTrackbar("RayMaxR", "controls", ray_max_r, 160, lambda value: None)

    detector = SimpleInfer(show_stream=False)

    print("开始仪表盘状态只读调试，按 q 退出")
    print("窗口说明：debug_raw=原图叠加信息，dashboard_roi=仪表盘ROI")
    print("pointer_binary_raw=原始二值化图，pointer_binary=mask后参与找指针的二值化图，pointer_contour=当前方法结果")
    print("Method=0 表示最大轮廓方法，Method=1 表示射线扫描方法")
    print("RayMinR 越大，越能避开中心和短指针；RayMaxR 太大可能扫到刻度文字")
    print("当前正常角度范围：{:.1f} ~ {:.1f}".format(NORMAL_ANGLE_MIN, NORMAL_ANGLE_MAX))

    last_print_time = 0.0

    try:
        while True:
            pointer_threshold = cv2.getTrackbarPos("PointerThreshold", "controls")
            use_mask = cv2.getTrackbarPos("UseMask", "controls") == 1
            mask_cut_width = cv2.getTrackbarPos("MaskCutWidth", "controls")
            pointer_method = cv2.getTrackbarPos("Method", "controls")
            ray_min_r = cv2.getTrackbarPos("RayMinR", "controls")
            ray_max_r = cv2.getTrackbarPos("RayMaxR", "controls")

            infer_output = detector.infer_once()
            image_raw = infer_output["image_raw"]
            detections = infer_output["detections"]

            if image_raw is None:
                time.sleep(args.interval)
                continue

            dashboard_det = pick_best_dashboard(detections)
            if dashboard_det is None:
                show_waiting(image_raw, "no dashboard")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                time.sleep(args.interval)
                continue

            ssi_box = nearest_ssi_box(dashboard_det["xyxy"], collect_ssi_boxes(detections))
            if ssi_box is None:
                show_waiting(image_raw, "no ssi")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                time.sleep(args.interval)
                continue

            debug_info = calc_state_debug(
                image_raw,
                dashboard_det["xyxy"],
                ssi_box,
                pointer_threshold,
                use_mask,
                mask_cut_width,
                pointer_method,
                ray_min_r,
                ray_max_r,
            )
            if debug_info is None:
                show_waiting(image_raw, "roi invalid")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                time.sleep(args.interval)
                continue

            debug_raw = draw_debug_raw(
                image_raw=image_raw,
                dashboard_det=dashboard_det,
                ssi_box=ssi_box,
                debug_info=debug_info,
                pointer_threshold=pointer_threshold,
                use_mask=use_mask,
                mask_cut_width=mask_cut_width,
                pointer_method=pointer_method,
                ray_min_r=ray_min_r,
                ray_max_r=ray_max_r,
                infer_ms=float(infer_output.get("infer_ms", 0.0)),
            )

            cv2.imshow("debug_raw", debug_raw)
            cv2.imshow("dashboard_roi", debug_info["roi"])
            cv2.imshow("pointer_binary_raw", debug_info["binary_raw"])
            cv2.imshow("pointer_binary", debug_info["binary"])
            cv2.imshow("pointer_mask", debug_info["mask"])
            cv2.imshow("pointer_contour", debug_info["contour_view"])

            now = time.time()
            if now - last_print_time >= 0.5:
                angle_text = "None" if debug_info["angle"] is None else "{:.2f}".format(debug_info["angle"])
                cross_text = "None" if debug_info["cross_value"] is None else "{:.2f}".format(debug_info["cross_value"])
                distance_m = dashboard_det.get("distance_m", None)
                distance_text = "None" if distance_m is None else "{:.3f}m".format(distance_m)
                print(
                    "threshold={} mask={} cut={} method={} ray={}~{} state={} angle={} cross={} score_area={:.1f} distance={}".format(
                        pointer_threshold,
                        int(use_mask),
                        mask_cut_width,
                        debug_info["method_name"],
                        ray_min_r,
                        ray_max_r,
                        debug_info["state_cn"],
                        angle_text,
                        cross_text,
                        debug_info["contour_area"],
                        distance_text,
                    )
                )
                last_print_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            time.sleep(args.interval)

    finally:
        detector.close()
        cv2.destroyAllWindows()
        print("debug_dashboard_state_view finished")


if __name__ == "__main__":
    main()
