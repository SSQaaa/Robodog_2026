#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_detect_cones.py
利用深度相机 + YOLO 自动检测锥桶，输出规划坐标系坐标 (mm)
直接复用现有避障代码的所有模块，仅添加多帧稳定和坐标转换。
"""

import os
import sys
import time
import json
import ctypes
from collections import defaultdict, deque

import cv2
import numpy as np
import orbbec_native

# ========================= 路径配置（与原辅助代码完全一致） =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LIB_DIR = os.path.join(BASE_DIR, "/home/ysc/Desktop/2026Project/libs/")
ENGINE_PATH = os.path.join(TRT_LIB_DIR, "bigdog_0427.engine")
sys.path.append(TRT_LIB_DIR)
ctypes.CDLL(os.path.join(TRT_LIB_DIR, "libmyplugins.so"))
import yolov5_trt_cpp

# ========================= 参数 =========================
CONF_THRESH = 0.5               # 置信度阈值（可调）
MIN_VALID_DEPTH_COUNT = 10
DEPTH_HISTORY_LEN = 5
DETECTION_FRAMES = 15           # 采集多少帧用于稳定检测（实际会取中位数）
TARGET_CLASS = 7                # Traffic_cone

# 相机内参
DEPTH_FX = 478.547
DEPTH_FY = 478.547
DEPTH_CX = 321.087
DEPTH_CY = 201.625
REAL_CONE_WIDTH = 0.32
COLOR_FX = 453.72

# ---------- 重要：相机在机器人上的安装偏移 ----------
# 机器人坐标系：x 向前，y 向右，原点在机器人中心地面投影
CAM_ON_ROBOT_X = 0.2            # 相机在机器人中心前方多少米
CAM_ON_ROBOT_Y = 0.0            # 相机在机器人中心右方多少米
# ----------------------------------------------------

WINDOW_NAME = "Auto Cone Detection - press SPACE to confirm"

# ========================= 工具函数（直接从你的辅助代码复制） =========================
def scale_box(box, src_size, dst_size):
    x1, y1, x2, y2 = box
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    dx1 = int(round(x1 * dst_w / float(src_w)))
    dy1 = int(round(y1 * dst_h / float(src_h)))
    dx2 = int(round(x2 * dst_w / float(src_w)))
    dy2 = int(round(y2 * dst_h / float(src_h)))
    dx1 = max(0, min(dst_w - 1, dx1))
    dx2 = max(0, min(dst_w - 1, dx2))
    dy1 = max(0, min(dst_h - 1, dy1))
    dy2 = max(0, min(dst_h - 1, dy2))
    return dx1, dy1, dx2, dy2

def yolo_to_original(box, img_w, img_h, input_size=640):
    cx, cy, w, h = box
    scale = min(input_size / img_w, input_size / img_h)
    new_w = img_w * scale
    new_h = img_h * scale
    pad_x = (input_size - new_w) / 2
    pad_y = (input_size - new_h) / 2
    cx = (cx - pad_x) / scale
    cy = (cy - pad_y) / scale
    w = w / scale
    h = h / scale
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w - 1, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2

def pixel_to_camera_3d(u, v, depth_mm, fx, fy, cx, cy):
    z = depth_mm / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def get_cone_depth_roi(color_box, color_size, depth_size):
    depth_box = scale_box(color_box, color_size, depth_size)
    dx1, dy1, dx2, dy2 = depth_box
    h = dy2 - dy1
    if h <= 0:
        return None
    target_h = max(5, h // 4)
    roi_y2 = dy2
    roi_y1 = dy2 - target_h
    if roi_y1 < 0:
        roi_y1 = 0
        roi_y2 = min(depth_size[1], roi_y1 + target_h)
    if roi_y2 - roi_y1 < 5:
        roi_y1 = dy1
        roi_y2 = dy1 + min(target_h, depth_size[1] - dy1)
    w = dx2 - dx1
    target_w = max(5, w // 2)
    center = (dx1 + dx2) // 2
    roi_x1 = max(0, center - target_w // 2)
    roi_x2 = min(depth_size[0], roi_x1 + target_w)
    if roi_x2 - roi_x1 < 5:
        roi_x1 = dx1
        roi_x2 = dx2
    return (roi_x1, roi_y1, roi_x2, roi_y2)

def draw_detection(frame, box, depth_mm, pos_3d=None, is_estimated=False):
    x1, y1, x2, y2 = box
    cls_name = "Cone"
    if depth_mm is not None and depth_mm > 0:
        text = f"{cls_name} {depth_mm/1000.0:.2f}m"
        if is_estimated:
            text += " [est]"
        color = (0, 255, 0)
    else:
        text = f"{cls_name} Depth Invalid"
        color = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(25, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if pos_3d is not None:
        x_3d, y_3d, z_3d = pos_3d
        coord_text = f"X:{x_3d:.3f} Y:{y_3d:.3f} Z:{z_3d:.3f}m"
        cv2.putText(frame, coord_text, (x1, min(frame.shape[0]-5, y2+40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

# ========================= 主检测逻辑 =========================
def auto_detect():
    print("[TRT] loading engine...")
    detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
    print("[TRT] engine loaded")

    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(1.0)
    depth_w, depth_h = cam.get_depth_size()
    color_w, color_h = cam.get_color_size()
    print(f"[Orbbec] color {color_w}x{color_h}, depth {depth_w}x{depth_h}")

    # 深度平滑器（按 track_id）
    depth_history = defaultdict(lambda: deque(maxlen=DEPTH_HISTORY_LEN))
    position_estimates = {}       # tid -> (x_cam, z_cam) 平均值
    last_centers = {}             # tid -> (cx, cy) 用于简单跟踪

    print(f"[采集] 将采集 {DETECTION_FRAMES} 帧，请保持机器人静止...")
    for frame_idx in range(DETECTION_FRAMES):
        color_frame = cam.get_color_frame()
        if color_frame is None:
            time.sleep(0.01)
            continue
        frame = np.asarray(color_frame, dtype=np.uint8).copy()
        detections = detector.detect(frame)

        for det in detections:
            cx_y, cy_y, w_y, h_y, conf, cls_id = det
            if conf < CONF_THRESH or int(cls_id) != TARGET_CLASS:
                continue
            color_box = yolo_to_original((cx_y, cy_y, w_y, h_y), color_w, color_h)
            box_center = ((color_box[0]+color_box[2])//2, (color_box[1]+color_box[3])//2)

            # 简单关联：找最近 center 的已有 track（阈值50像素）
            tid = None
            for existing_tid, last_center in last_centers.items():
                if abs(box_center[0] - last_center[0]) < 50 and abs(box_center[1] - last_center[1]) < 50:
                    tid = existing_tid
                    break
            if tid is None:
                tid = len(last_centers)
            last_centers[tid] = box_center

            depth_roi = get_cone_depth_roi(color_box, (color_w, color_h), (depth_w, depth_h))
            if depth_roi is None:
                continue
            raw_depth, valid_cnt = cam.get_depth_in_box(*depth_roi)

            # 视觉估计距离（基于宽度）
            x1, y1, x2, y2 = color_box
            box_w = x2 - x1
            visual_z = (REAL_CONE_WIDTH * COLOR_FX) / max(box_w, 1) if box_w > 0 else 5.0

            # 决定使用何种深度
            if raw_depth > 0 and valid_cnt >= MIN_VALID_DEPTH_COUNT:
                depth_history[tid].append(raw_depth)
                # 使用中位数深度
                median_depth = int(np.median(depth_history[tid]))
                u = (depth_roi[0] + depth_roi[2]) // 2
                v = depth_roi[3] - 1
                x_cam, y_cam, z_cam = pixel_to_camera_3d(u, v, median_depth,
                                                         DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)
                is_estimated = False
            else:
                if visual_z <= 0:
                    continue
                z_cam = visual_z
                u = (depth_roi[0] + depth_roi[2]) // 2
                v = depth_roi[3] - 1
                x_cam = (u - DEPTH_CX) * visual_z / DEPTH_FX
                y_cam = (v - DEPTH_CY) * visual_z / DEPTH_FY
                is_estimated = True

            # 平滑位置
            alpha = 0.5
            if tid in position_estimates:
                prev = position_estimates[tid]
                position_estimates[tid] = (
                    alpha*x_cam + (1-alpha)*prev[0],
                    alpha*z_cam + (1-alpha)*prev[1]
                )
            else:
                position_estimates[tid] = (x_cam, z_cam)

        sys.stdout.write(f"\r[采集] 帧 {frame_idx+1}/{DETECTION_FRAMES}  当前跟踪ID: {list(position_estimates.keys())}   ")
        sys.stdout.flush()
        time.sleep(0.05)
    print("\n[采集] 完成")

    cam.stop()

    if not position_estimates:
        print("[错误] 未检测到任何锥桶")
        sys.exit(1)

    # ---------- 坐标转换到规划坐标系 (mm) ----------
    # 规划起点: START_PLAN_X_M = -0.5, START_PLAN_Y_M = 0.75  (米)
    # 转换为 mm: (-500, 750)
    START_PLAN_X_MM = -500
    START_PLAN_Y_MM = 750

    cones_mm = []
    for tid, (x_cam, z_cam) in position_estimates.items():
        # 相机坐标系: x_cam 指向右, z_cam 指向前
        # 规划坐标系: x 向前, y 向右
        # 所以 plan_x = 起点x + (相机在机器人的前向偏移 + 检测的前向距离)
        #     plan_y = 起点y + (相机在机器人的横向偏移 + 检测的横向距离)
        plan_x = START_PLAN_X_MM + int(round((CAM_ON_ROBOT_X + z_cam) * 1000))
        plan_y = START_PLAN_Y_MM + int(round((CAM_ON_ROBOT_Y + x_cam) * 1000))
        cones_mm.append((plan_x, plan_y))
        print(f"  Cone {tid}: 相机坐标 ({x_cam:.3f}, {z_cam:.3f}) m  -> 规划坐标 ({plan_x}, {plan_y}) mm")

    # 保存 JSON
    json_path = "detected_cones.json"
    with open(json_path, "w") as f:
        json.dump({"cones_plan_mm": [list(c) for c in cones_mm]}, f, indent=2)
    print(f"\n结果已保存到 {json_path}")

    # 输出可直接使用的命令行参数
    cmd_args = " ".join([f"--cone{i} {x} {y}" for i, (x, y) in enumerate(cones_mm, 1)])
    print("\n" + "="*50)
    print("复制以下参数运行 task1_path_planner.py：")
    print(cmd_args)
    print("="*50)
    return cones_mm, cmd_args

# ========================= 交互式确认模式（可选） =========================
def interactive_mode():
    """
    实时显示检测画面，按空格键输出结果，按 ESC 退出。
    """
    print("[交互模式] 将实时显示检测结果，按 SPACE 确认输出，按 Q 退出")
    detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
    cam = orbbec_native.OrbbecCamera()
    cam.start()
    time.sleep(1.0)
    depth_w, depth_h = cam.get_depth_size()
    color_w, color_h = cam.get_color_size()

    # 用于稳定化的变量
    depth_history = defaultdict(lambda: deque(maxlen=DEPTH_HISTORY_LEN))
    position_estimates = {}
    last_centers = {}

    while True:
        color_frame = cam.get_color_frame()
        if color_frame is None:
            time.sleep(0.01)
            continue
        frame = np.asarray(color_frame, dtype=np.uint8).copy()
        detections = detector.detect(frame)

        # 更新位置估计
        for det in detections:
            cx_y, cy_y, w_y, h_y, conf, cls_id = det
            if conf < CONF_THRESH or int(cls_id) != TARGET_CLASS:
                continue
            color_box = yolo_to_original((cx_y, cy_y, w_y, h_y), color_w, color_h)
            box_center = ((color_box[0]+color_box[2])//2, (color_box[1]+color_box[3])//2)

            tid = None
            for t, c in last_centers.items():
                if abs(box_center[0]-c[0])<50 and abs(box_center[1]-c[1])<50:
                    tid = t; break
            if tid is None:
                tid = len(last_centers)
            last_centers[tid] = box_center

            depth_roi = get_cone_depth_roi(color_box, (color_w,color_h), (depth_w,depth_h))
            if depth_roi is None: continue
            raw_depth, valid_cnt = cam.get_depth_in_box(*depth_roi)
            x1, y1, x2, y2 = color_box
            box_w = x2 - x1
            visual_z = (REAL_CONE_WIDTH * COLOR_FX) / max(box_w,1) if box_w>0 else 5.0
            if raw_depth>0 and valid_cnt>=MIN_VALID_DEPTH_COUNT:
                depth_history[tid].append(raw_depth)
                median_depth = int(np.median(depth_history[tid]))
                u = (depth_roi[0]+depth_roi[2])//2
                v = depth_roi[3]-1
                x_cam, y_cam, z_cam = pixel_to_camera_3d(u,v,median_depth, DEPTH_FX,DEPTH_FY,DEPTH_CX,DEPTH_CY)
                is_est=False
            else:
                if visual_z<=0: continue
                z_cam = visual_z
                u = (depth_roi[0]+depth_roi[2])//2
                v = depth_roi[3]-1
                x_cam = (u-DEPTH_CX)*visual_z/DEPTH_FX
                is_est=True
            alpha=0.5
            if tid in position_estimates:
                p = position_estimates[tid]
                position_estimates[tid] = (alpha*x_cam+(1-alpha)*p[0], alpha*z_cam+(1-alpha)*p[1])
            else:
                position_estimates[tid] = (x_cam,z_cam)

            draw_detection(frame, color_box, raw_depth, (x_cam,0,z_cam), is_est)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        elif key == 32:   # 空格键
            if not position_estimates:
                print("无锥桶数据，按空格无效")
                continue
            # 转换坐标并输出
            START_PLAN_X_MM = -500
            START_PLAN_Y_MM = 750
            cones_mm = []
            for tid, (x_cam,z_cam) in position_estimates.items():
                plan_x = START_PLAN_X_MM + int(round((CAM_ON_ROBOT_X + z_cam)*1000))
                plan_y = START_PLAN_Y_MM + int(round((CAM_ON_ROBOT_Y + x_cam)*1000))
                cones_mm.append((plan_x, plan_y))
                print(f"Cone {tid}: ({plan_x}, {plan_y}) mm")
            cmd_args = " ".join([f"--cone{i} {x} {y}" for i,(x,y) in enumerate(cones_mm,1)])
            print("\n复制参数：", cmd_args)
            with open("detected_cones.json","w") as f:
                json.dump({"cones_plan_mm":[list(c) for c in cones_mm]}, f, indent=2)
            # 确认后不清空位置，可继续修改

    cam.stop()
    cv2.destroyAllWindows()

# ========================= 入口 =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="进入实时交互模式")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        auto_detect()