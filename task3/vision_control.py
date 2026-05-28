# -*- coding: utf-8 -*-
"""视觉识别模块：封装 TensorRT YOLO 和 Orbbec 深度相机。"""

import ctypes
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRT_LIB_DIR = "/home/ysc/Desktop/2026Project/libs/"
ENGINE_PATH = os.path.join(TRT_LIB_DIR, "bigdog_0427.engine")

CONF_THRESH = 0.4
MIN_VALID_DEPTH_COUNT = 20
DEPTH_HISTORY_LEN = 5

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
    9: "ssi",
}


@dataclass
class Detection:
    class_name: str
    conf: float
    box: tuple
    center: tuple
    depth_mm: Optional[int]
    valid_count: int

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return abs(x2 - x1) * abs(y2 - y1)


class DepthSmoother:
    def __init__(self, max_len=DEPTH_HISTORY_LEN):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    # 对同一类别的深度做短时间中值滤波，减少单帧跳变。
    def update(self, key, depth_mm):
        if depth_mm <= 0:
            return None
        self.history[key].append(int(depth_mm))
        return int(np.median(np.asarray(self.history[key], dtype=np.int32)))


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


class YoloDepthDetector:
    def __init__(self, engine_path=ENGINE_PATH, conf_thresh=CONF_THRESH):
        self.engine_path = engine_path
        self.conf_thresh = float(conf_thresh)
        self.detector = None
        self.camera = None
        self.color_intrinsics = None
        self.smoother = DepthSmoother()

    # 加载 TensorRT 引擎并启动 Orbbec 相机。
    def start(self):
        sys.path.append(TRT_LIB_DIR)
        ctypes.CDLL(os.path.join(TRT_LIB_DIR, "libmyplugins.so"))
        import orbbec_native
        import yolov5_trt_cpp

        print("[TRT] loading engine...")
        self.detector = yolov5_trt_cpp.Yolov5TRT(self.engine_path)
        print("[TRT] engine loaded")

        self.camera = orbbec_native.OrbbecCamera()
        self.camera.start()
        time.sleep(1.0)
        self.color_intrinsics = self.camera.get_color_intrinsics()
        print(f"[Orbbec] color size: {self.get_color_size()}")
        print(f"[Orbbec] depth size : {self.get_depth_size()}")
        return self

    def stop(self):
        if self.camera is not None:
            self.camera.stop()

    def get_color_size(self):
        return self.camera.get_color_size()

    def get_depth_size(self):
        return self.camera.get_depth_size()

    def get_frame(self):
        frame = self.camera.get_color_frame()
        if frame is None:
            return None
        return np.asarray(frame, dtype=np.uint8).copy()

    # 检测当前帧中的所有 YOLO 目标，并为每个框附加深度。
    def detect(self):
        frame = self.get_frame()
        if frame is None:
            return None, []

        color_h, color_w = frame.shape[:2]
        depth_w, depth_h = self.get_depth_size()
        detections = []

        for raw in self.detector.detect(frame):
            cx, cy, w, h, conf, cls_id = raw
            if float(conf) < self.conf_thresh:
                continue

            class_name = CLASS_NAMES.get(int(cls_id), f"id{int(cls_id)}")
            color_box = yolo_to_original((cx, cy, w, h), img_w=color_w, img_h=color_h)
            depth_box = scale_box(color_box, src_size=(color_w, color_h), dst_size=(depth_w, depth_h))
            depth_mm, valid_count = self.camera.get_depth_in_box(*depth_box)
            stable_depth = None
            if depth_mm > 0 and valid_count >= MIN_VALID_DEPTH_COUNT:
                stable_depth = self.smoother.update(class_name, depth_mm)

            x1, y1, x2, y2 = color_box
            detections.append(
                Detection(
                    class_name=class_name,
                    conf=float(conf),
                    box=color_box,
                    center=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                    depth_mm=stable_depth,
                    valid_count=int(valid_count),
                )
            )

        return frame, detections

