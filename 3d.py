# -*- coding: utf-8 -*-
import os
import sys
import time
import ctypes
from collections import defaultdict, deque

import cv2
import numpy as np
import orbbec_native

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LIB_DIR = os.path.join(BASE_DIR, "/home/ysc/Desktop/2026Project/libs/")
ENGINE_PATH = os.path.join(TRT_LIB_DIR, "bigdog_0427.engine")
sys.path.append(TRT_LIB_DIR)
ctypes.CDLL(os.path.join(TRT_LIB_DIR, "libmyplugins.so"))
import yolov5_trt_cpp

# =========================
# 参数区
# =========================
CONF_THRESH = 0.6
MIN_VALID_DEPTH_COUNT = 10          # 最小有效深度点数
DEPTH_HISTORY_LEN = 5

WINDOW_NAME = "GeminiPro YOLO Depth"

CLASS_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "Green",
    5: "MPa", 6: "Red", 7: "Traffic_cone",
    8: "dashboard", 9: "ssi"
}

# 相机内参（深度）
DEPTH_FX = 478.547
DEPTH_FY = 478.547
DEPTH_CX = 321.087
DEPTH_CY = 201.625

# 锥桶实际宽度（米）用于视觉估计
REAL_CONE_WIDTH = 0.32
COLOR_FX = 453.72   # 彩色相机焦距（用于宽度估距）

# =========================
# 深度平滑器
# =========================
class DepthSmoother:
    def __init__(self, max_len=5):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, obj_id, depth_mm):
        if depth_mm <= 0:
            return None
        self.history[obj_id].append(depth_mm)
        return int(np.median(np.array(self.history[obj_id])))

# =========================
# 工具函数
# =========================
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
    """
    自适应锥桶深度ROI：取底部1/4高度，中间1/2宽度，
    并确保ROI不越界且至少有5x5像素。
    返回 (x1,y1,x2,y2) 或 None
    """
    depth_box = scale_box(color_box, color_size, depth_size)
    dx1, dy1, dx2, dy2 = depth_box
    h = dy2 - dy1
    if h <= 0:
        return None
    # 目标高度：原始高度的1/4，但至少5像素
    target_h = max(5, h // 4)
    # 从底部向上取 target_h 像素
    roi_y2 = dy2
    roi_y1 = dy2 - target_h
    # 如果超出图像上边界，则上移整个ROI
    if roi_y1 < 0:
        roi_y1 = 0
        roi_y2 = min(depth_size[1], roi_y1 + target_h)
    # 如果仍然高度不足（图像底部太靠上），则使用原始框的上半部分
    if roi_y2 - roi_y1 < 5:
        roi_y1 = dy1
        roi_y2 = dy1 + min(target_h, depth_size[1] - dy1)
    # 宽度方向：取中间1/2，至少5像素
    w = dx2 - dx1
    target_w = max(5, w // 2)
    center = (dx1 + dx2) // 2
    roi_x1 = max(0, center - target_w // 2)
    roi_x2 = min(depth_size[0], roi_x1 + target_w)
    if roi_x2 - roi_x1 < 5:
        roi_x1 = dx1
        roi_x2 = dx2
    return (roi_x1, roi_y1, roi_x2, roi_y2)

def draw_detection(frame, box, cls_id, conf, depth_mm, valid_count, pos_3d=None, depth_roi_box=None, is_estimated=False):
    x1, y1, x2, y2 = box
    cls_name = CLASS_NAMES.get(int(cls_id), f"id{int(cls_id)}")
    if depth_mm is not None and depth_mm > 0:
        text = f"{cls_name} {conf:.2f} {depth_mm/1000.0:.2f}m"
        if is_estimated:
            text += " [est]"
        color = (0, 255, 0)
    else:
        text = f"{cls_name} {conf:.2f} Depth Invalid"
        color = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if depth_roi_box is not None:
        rx1, ry1, rx2, ry2 = depth_roi_box
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
    cv2.putText(frame, text, (x1, max(25, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"ValidPts: {valid_count}", (x1, min(frame.shape[0]-5, y2+20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    if pos_3d is not None:
        x_3d, y_3d, z_3d = pos_3d
        coord_text = f"X:{x_3d:.3f} Y:{y_3d:.3f} Z:{z_3d:.3f}m"
        cv2.putText(frame, coord_text, (x1, min(frame.shape[0]-5, y2+40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

# =========================
# 主程序
# =========================
def main():
    print("[TRT] loading engine...")
    detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
    print("[TRT] engine loaded")

    cam = orbbec_native.OrbbecCamera()
    print("[Orbbec] start...")
    cam.start()
    time.sleep(1.0)

    depth_w, depth_h = cam.get_depth_size()
    color_w, color_h = cam.get_color_size()
    print(f"[Orbbec] color size: {color_w}x{color_h}")
    print(f"[Orbbec] depth size : {depth_w}x{depth_h}")

    smoother = DepthSmoother(max_len=DEPTH_HISTORY_LEN)
    frame_id = 0

    try:
        while True:
            # 获取彩色图
            color_frame = cam.get_color_frame()
            if color_frame is None:
                time.sleep(0.01)
                continue
            frame = np.asarray(color_frame, dtype=np.uint8).copy()
            color_h, color_w = frame.shape[:2]
            depth_w, depth_h = cam.get_depth_size()

            # YOLO检测
            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()
            print(f"[YOLO] detect num={len(detections)}, infer={(t1 - t0) * 1000:.2f} ms")

            for det in detections:
                cx_yolo, cy_yolo, w_yolo, h_yolo, conf, cls_id = det
                if conf < CONF_THRESH:
                    continue

                # 原始检测框（彩色图）
                color_box = yolo_to_original(
                    (cx_yolo, cy_yolo, w_yolo, h_yolo),
                    img_w=color_w, img_h=color_h
                )

                # 获取自适应深度ROI
                depth_roi = get_cone_depth_roi(color_box, (color_w, color_h), (depth_w, depth_h))
                if depth_roi is None:
                    continue

                # 从深度相机获取ROI内的平均深度和有效点数
                raw_depth, valid_cnt = cam.get_depth_in_box(*depth_roi)

                # 视觉估计距离（基于宽度）
                x1, y1, x2, y2 = color_box
                box_w = x2 - x1
                visual_z = (REAL_CONE_WIDTH * COLOR_FX) / max(box_w, 1) if box_w > 0 else 5.0

                stable_depth = None
                pos_3d = None
                is_estimated = False

                if raw_depth > 0 and valid_cnt >= MIN_VALID_DEPTH_COUNT:
                    # 使用深度传感器测量值，通过平滑器得到中位数
                    stable_depth = smoother.update(int(cls_id), raw_depth)
                    if cls_id == 7 and stable_depth is not None:
                        # 取ROI底部中心点计算3D坐标
                        u = (depth_roi[0] + depth_roi[2]) // 2
                        v = depth_roi[3] - 1   # 底部像素
                        x_m, y_m, z_m = pixel_to_camera_3d(u, v, stable_depth,
                                                           DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)
                        pos_3d = (x_m, y_m, z_m)
                        print(f"[3D] TrafficCone: x={x_m:.3f}m, y={y_m:.3f}m, z={z_m:.3f}m, valid={valid_cnt}")
                else:
                    # 深度无效：使用视觉估计
                    if visual_z > 0:
                        stable_depth = int(visual_z * 1000)
                        is_estimated = True
                        # 计算3D坐标（仍然使用深度ROI的中心点，但深度值用视觉估计）
                        u = (depth_roi[0] + depth_roi[2]) // 2
                        v = depth_roi[3] - 1
                        x_m, y_m, z_m = pixel_to_camera_3d(u, v, stable_depth,
                                                           DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)
                        pos_3d = (x_m, y_m, z_m)
                        print(f"[视觉估计] 锥桶: 视觉深度={visual_z:.2f}m, 估算3D: ({x_m:.2f},{y_m:.2f},{z_m:.2f})")
                    else:
                        # 完全无法估计，跳过
                        print(f"[警告] 无法获取锥桶深度和视觉估计，跳过")
                        continue

                # 可视化（depth_roi 需要映射回彩色图用于显示）
                color_roi_box = scale_box(depth_roi, (depth_w, depth_h), (color_w, color_h))
                draw_detection(frame, color_box, cls_id, conf, stable_depth, valid_cnt,
                               pos_3d, color_roi_box, is_estimated)

                # 如果需要，可以在此处添加避障逻辑（例如记录锥桶3D坐标到列表）
                # ...

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break
            frame_id += 1

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[Exit] stopped")

if __name__ == "__main__":
    main()
