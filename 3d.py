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

CONF_THRESH = 0.4
MIN_VALID_DEPTH_COUNT = 20
DEPTH_HISTORY_LEN = 5

WINDOW_NAME = "GeminiPro YOLO Depth"

CLASS_NAMES = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "Green",
    5: "MPa",
    6: "Red",
    7: "Traffic_cone",
    8: "dashboard",
    9: "ssi"
}

# =========================
# 相机内参（临时硬编码，请根据实际相机型号修改）
# 若后续在 orbbec_native 中暴露 get_depth_intrinsics()，可替换为动态获取
# =========================
DEPTH_FX = 570.34
DEPTH_FY = 570.34
DEPTH_CX = 320.0
DEPTH_CY = 240.0


# =========================
# 工具类 / 函数
# =========================

class DepthSmoother:
    def __init__(self, max_len=5):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, obj_id, depth_mm):
        if depth_mm <= 0:
            return None

        self.history[obj_id].append(depth_mm)
        return int(np.median(np.array(self.history[obj_id])))


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

    # step1: scale
    scale = min(input_size / img_w, input_size / img_h)

    new_w = img_w * scale
    new_h = img_h * scale

    pad_x = (input_size - new_w) / 2
    pad_y = (input_size - new_h) / 2

    # step2: 去padding
    cx = (cx - pad_x) / scale
    cy = (cy - pad_y) / scale
    w = w / scale
    h = h / scale

    # step3: 转xyxy
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    # clip
    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w - 1, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h - 1, y2))

    return x1, y1, x2, y2


def pixel_to_camera_3d(u, v, depth_mm, fx, fy, cx, cy):
    """
    将深度图上的像素坐标(u, v)和深度值(mm)转换为相机坐标系下的3D坐标(米)
    相机坐标系：X轴向右，Y轴向下，Z轴向前
    """
    z = depth_mm / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def draw_detection(frame, box, cls_id, conf, depth_mm, valid_count,pos_3d=None):
    x1, y1, x2, y2 = box

    cls_name = CLASS_NAMES.get(int(cls_id), f"id{int(cls_id)}")

    if depth_mm is not None and depth_mm > 0:
        text = f"{cls_name} {conf:.2f} {depth_mm / 1000.0:.2f}m"
    else:
        text = f"{cls_name} {conf:.2f} Depth Invalid"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        frame,
        f"ValidPts: {valid_count}",
        (x1, min(frame.shape[0] - 5, y2 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )
    
    if pos_3d is not None:
        x_3d,y_3d,z_3d=pos_3d
        coord_text=f"X:{x_3d:.3f} Y:{y_3d:.3f} Z:{z_3d:.3f}"
        y_offset=min(frame.shape[0]-5,y2+40)
        cv2.putText(
            frame,
            coord_text,
            (x1,y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,0),
            2,
        )



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
            frame = cam.get_color_frame()

            if frame is None:
                print("[Orbbec] no color frame")
                time.sleep(0.01)
                continue

            frame = np.asarray(frame, dtype=np.uint8).copy()

            color_h, color_w = frame.shape[:2]
            depth_w, depth_h = cam.get_depth_size()

            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()

            print(f"[YOLO] detect num={len(detections)}, infer={(t1 - t0) * 1000:.2f} ms")

            for i, det in enumerate(detections):
                cx_yolo, cy_yolo, w_yolo, h_yolo, conf, cls_id = det

                if conf < CONF_THRESH:
                    continue

                color_box = yolo_to_original(
                    (cx_yolo, cy_yolo, w_yolo, h_yolo),
                    img_w=color_w,
                    img_h=color_h,
                )

                depth_box = scale_box(
                    color_box,
                    src_size=(color_w, color_h),
                    dst_size=(depth_w, depth_h),
                )

                raw_depth, valid_count = cam.get_depth_in_box(*depth_box)

                stable_depth = None

                if raw_depth > 0 and valid_count >= MIN_VALID_DEPTH_COUNT:
                    obj_key = int(cls_id)
                    stable_depth = smoother.update(obj_key, raw_depth)

                    print(
                        f"[Target] cls={int(cls_id)}, conf={conf:.2f}, "
                        f"raw={raw_depth} mm, stable={stable_depth} mm, "
                        f"valid={valid_count}, "
                        f"color_box={color_box}, depth_box={depth_box}"
                    )

                    # ---------- 新增：如果是 Traffic_cone，计算 3D 位置 ----------
                    if cls_id == 7 and stable_depth is not None:
                        # 使用深度框底部中心作为测量点（更贴近地面）
                        u_cam = (depth_box[0] + depth_box[2]) // 2
                        v_cam = depth_box[3]  # 底部 y2
                        # 若想用框的中心，可改为 v_cam = (depth_box[1] + depth_box[3]) // 2

                        x_m, y_m, z_m = pixel_to_camera_3d(
                            u_cam, v_cam, stable_depth,
                            DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY
                        )
                        print(f"[3D] TrafficCone: x={x_m:.3f}m, y={y_m:.3f}m, z={z_m:.3f}m")
                    # -----------------------------------------------------------
                else:
                    print(
                        f"[Target] cls={int(cls_id)}, conf={conf:.2f}, "
                        f"depth invalid, code={raw_depth}, valid={valid_count}, "
                        f"color_box={color_box}, depth_box={depth_box}"
                    )

                
                pos_3d = None
                if cls_id ==7 and stable_depth is not None:
                    pos_3d = (x_m,y_m,z_m)
                draw_detection(
                    frame,
                    color_box,
                    cls_id,
                    conf,
                    stable_depth,
                    valid_count,
                    pos_3d,
                )

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