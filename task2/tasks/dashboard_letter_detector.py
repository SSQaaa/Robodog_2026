# -*- coding: utf-8 -*-
"""
任务2：仪表盘纯识别与后处理模块（新模型 + 新深度相机）

当前版本只围绕四项核心输出：
1. 仪表盘四顶点坐标
2. 仪表盘状态（正常/偏低/偏高）
3. 仪表盘距离（米）
4. 字母

说明：
- 不包含运动学控制。
- 保留 SimpleInfer / analyze_infer_output / print_dashboard_result 三个接口。
"""

import ctypes
import math
import os
import sys
import time

import cv2
import numpy as np
import orbbec_native


# =========================
# 路径与类别配置
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK2_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(TASK2_DIR)
LIBS_DIR = os.path.join(PROJECT_DIR, "libs")

ENGINE_PATH = os.path.join(LIBS_DIR, "bigdog_0427.engine")
PLUGIN_PATH = os.path.join(LIBS_DIR, "libmyplugins.so")

if LIBS_DIR not in sys.path:
    sys.path.append(LIBS_DIR)

ctypes.CDLL(PLUGIN_PATH)
import yolov5_trt_cpp

# 新模型类别
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

LETTER_ID_TO_NAME = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

DASHBOARD_ID = 8
SSI_ID = 9

CONF_THRESH = 0.25
MIN_VALID_DEPTH_COUNT = 20
POINTER_THRESHOLD = 118
NORMAL_ANGLE_MIN = 120.0
NORMAL_ANGLE_MAX = 180.0

STATE_CN_MAP = {
    "normal": "正常",
    "low": "偏低",
    "high": "偏高",
    "unknown": "未知",
}

WINDOW_NAME = "dashboard_detector_v2"


# =========================
# 工具函数
# =========================

def yolo_to_xyxy(box, img_w, img_h, input_size=640):
    """把检测框从(cx, cy, w, h)映射回原图xyxy。"""
    cx, cy, w, h = box

    scale = min(float(input_size) / float(img_w), float(input_size) / float(img_h))
    new_w = img_w * scale
    new_h = img_h * scale
    pad_x = (input_size - new_w) / 2.0
    pad_y = (input_size - new_h) / 2.0

    cx = (cx - pad_x) / scale
    cy = (cy - pad_y) / scale
    w = w / scale
    h = h / scale

    x1 = int(cx - w / 2.0)
    y1 = int(cy - h / 2.0)
    x2 = int(cx + w / 2.0)
    y2 = int(cy + h / 2.0)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return [x1, y1, x2, y2]


def scale_box(box, src_size, dst_size):
    """把彩色图检测框映射到深度图坐标系。"""
    x1, y1, x2, y2 = box
    src_w, src_h = src_size
    dst_w, dst_h = dst_size

    dx1 = int(round(float(x1) * float(dst_w) / float(src_w)))
    dy1 = int(round(float(y1) * float(dst_h) / float(src_h)))
    dx2 = int(round(float(x2) * float(dst_w) / float(src_w)))
    dy2 = int(round(float(y2) * float(dst_h) / float(src_h)))

    dx1 = max(0, min(dst_w - 1, dx1))
    dy1 = max(0, min(dst_h - 1, dy1))
    dx2 = max(0, min(dst_w - 1, dx2))
    dy2 = max(0, min(dst_h - 1, dy2))
    return [dx1, dy1, dx2, dy2]


def center_of_box(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def vertices_from_box(box):
    """按顺时针输出四顶点：左上、右上、右下、左下。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def refine_box(box, frame_w, frame_h, ratio=0.5):
    """为了提取指针，按中心缩小dashboard框。"""
    x1, y1, x2, y2 = [float(v) for v in box]

    x1 = max(0.0, min(x1, float(frame_w - 1)))
    y1 = max(0.0, min(y1, float(frame_h - 1)))
    x2 = max(0.0, min(x2, float(frame_w - 1)))
    y2 = max(0.0, min(y2, float(frame_h - 1)))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    x1 = cx + (x1 - cx) * ratio
    y1 = cy + (y1 - cy) * ratio
    x2 = cx + (x2 - cx) * ratio
    y2 = cy + (y2 - cy) * ratio
    return [int(x1), int(y1), int(x2), int(y2)]


def find_pointer_point(image_raw, dashboard_box):
    """用二值化+最大轮廓找指针中心点。"""
    frame_h, frame_w = image_raw.shape[:2]
    x1, y1, x2, y2 = refine_box(dashboard_box, frame_w, frame_h, ratio=0.5)

    roi = image_raw[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, POINTER_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    contours_data = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[-2]
    if len(contours) == 0:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    px = int(rect[0][0]) + x1
    py = int(rect[0][1]) + y1
    return np.array([px, py], dtype=np.float32)


def nearest_ssi_box(dashboard_box, ssi_boxes):
    if len(ssi_boxes) == 0:
        return None

    db_center = center_of_box(dashboard_box)
    best_ssi = ssi_boxes[0]
    best_dist = float(np.linalg.norm(center_of_box(best_ssi) - db_center))

    for ssi in ssi_boxes[1:]:
        dist = float(np.linalg.norm(center_of_box(ssi) - db_center))
        if dist < best_dist:
            best_dist = dist
            best_ssi = ssi

    return best_ssi


def state_from_dashboard(image_raw, dashboard_box, ssi_box):
    """根据ssi方向与指针方向关系判断状态。"""
    dashboard_center = center_of_box(dashboard_box)
    ssi_center = center_of_box(ssi_box)
    pointer_point = find_pointer_point(image_raw, dashboard_box)
    if pointer_point is None:
        return "unknown"

    v1_x = ssi_center[0] - dashboard_center[0]
    v1_y = ssi_center[1] - dashboard_center[1]
    v2_x = pointer_point[0] - dashboard_center[0]
    v2_y = pointer_point[1] - dashboard_center[1]

    norm_v1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
    norm_v2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)
    if norm_v1 <= 1e-6 or norm_v2 <= 1e-6:
        return "unknown"

    cos_value = (v1_x * v2_x + v1_y * v2_y) / (norm_v1 * norm_v2)
    cos_value = max(-1.0, min(1.0, cos_value))
    angle = math.degrees(math.acos(cos_value))

    cross_value = v1_x * v2_y - v2_x * v1_y

    if NORMAL_ANGLE_MIN <= angle <= NORMAL_ANGLE_MAX:
        return "normal"
    if cross_value > 0:
        return "low"
    return "high"


def best_letter_from_detections(detections):
    """从A/B/C/D中取置信度最高字母。"""
    best_letter = "unknown"
    best_score = -1.0

    for det in detections:
        cid = int(det["class_id"])
        if cid in LETTER_ID_TO_NAME:
            score = float(det["score"])
            if score > best_score:
                best_score = score
                best_letter = LETTER_ID_TO_NAME[cid]

    return best_letter


# =========================
# 推理器
# =========================

class SimpleInfer:
    """单帧推理器：输出检测框、类别、深度距离。"""

    def __init__(self, show_stream=False, conf_thresh=CONF_THRESH, min_valid_depth_count=MIN_VALID_DEPTH_COUNT):
        self.show_stream = bool(show_stream)
        self.conf_thresh = float(conf_thresh)
        self.min_valid_depth_count = int(min_valid_depth_count)

        self.detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
        self.cam = orbbec_native.OrbbecCamera()
        self.cam.start()

        time.sleep(0.8)
        self.depth_w, self.depth_h = self.cam.get_depth_size()
        self.color_w, self.color_h = self.cam.get_color_size()

    def infer_once(self):
        frame = self.cam.get_color_frame()
        if frame is None:
            return {
                "image_raw": None,
                "detections": [],
                "infer_ms": 0.0,
            }

        image_raw = np.asarray(frame, dtype=np.uint8).copy()
        color_h, color_w = image_raw.shape[:2]

        t0 = time.time()
        raw_detections = self.detector.detect(image_raw)
        t1 = time.time()

        detections = []
        for det in raw_detections:
            cx, cy, w, h, conf, cls_id = det
            if conf < self.conf_thresh:
                continue

            xyxy = yolo_to_xyxy((cx, cy, w, h), img_w=color_w, img_h=color_h, input_size=640)
            depth_box = scale_box(xyxy, src_size=(color_w, color_h), dst_size=(self.depth_w, self.depth_h))
            raw_depth_mm, valid_count = self.cam.get_depth_in_box(*depth_box)

            if raw_depth_mm > 0 and valid_count >= self.min_valid_depth_count:
                distance_m = float(raw_depth_mm) / 1000.0
            else:
                distance_m = None

            detections.append(
                {
                    "class_id": int(cls_id),
                    "score": float(conf),
                    "xyxy": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "distance_m": distance_m,
                }
            )

        output = {
            "image_raw": image_raw,
            "detections": detections,
            "infer_ms": (t1 - t0) * 1000.0,
        }

        if self.show_stream:
            self._show_infer_frame(output)

        return output

    def _show_infer_frame(self, infer_output):
        image_raw = infer_output["image_raw"]
        if image_raw is None:
            return

        for det in infer_output["detections"]:
            x1, y1, x2, y2 = det["xyxy"]
            cid = det["class_id"]
            score = det["score"]
            name = CLASS_NAMES.get(cid, "id_{}".format(cid))

            cv2.rectangle(image_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_raw, "{} {:.2f}".format(name, score), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            dis_text = "{:.2f}m".format(det["distance_m"]) if det["distance_m"] is not None else "depth invalid"
            cv2.putText(image_raw, dis_text, (x1, min(image_raw.shape[0] - 8, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.putText(image_raw, "infer: {:.1f} ms".format(infer_output["infer_ms"]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, image_raw)
        cv2.waitKey(1)

    def close(self):
        self.cam.stop()
        if self.show_stream:
            cv2.destroyAllWindows()


# =========================
# 后处理（仅四项）
# =========================

def analyze_infer_values(image_raw, detections, infer_ms=0.0):
    """后处理主函数：只产出当前需要的四项信息。"""
    _ = infer_ms

    dashboard_dets = []
    ssi_dets = []

    for det in detections:
        cid = int(det["class_id"])
        if cid == DASHBOARD_ID:
            dashboard_dets.append(det)
        if cid == SSI_ID:
            ssi_dets.append(det)

    dashboard_dets.sort(key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) / 2.0)

    letter = best_letter_from_detections(detections)
    ssi_boxes = [det["xyxy"] for det in ssi_dets]

    dashboard_details = []
    for idx, db_det in enumerate(dashboard_dets, start=1):
        db_box = db_det["xyxy"]

        best_ssi = nearest_ssi_box(db_box, ssi_boxes)
        if best_ssi is None or image_raw is None:
            state_key = "unknown"
        else:
            state_key = state_from_dashboard(image_raw, db_box, best_ssi)

        dashboard_details.append(
            {
                "index": idx,
                "vertices": vertices_from_box(db_box),
                "state": state_key,
                "state_cn": STATE_CN_MAP.get(state_key, "未知"),
                "distance_m": db_det["distance_m"],
                "letter": letter,
            }
        )

    return {
        "dashboard_count": len(dashboard_dets),
        "letter": letter,
        "dashboard_details": dashboard_details,
    }


def analyze_infer_output(infer_output):
    image_raw = infer_output["image_raw"]
    detections = infer_output["detections"]
    infer_ms = infer_output.get("infer_ms", 0.0)
    return analyze_infer_values(image_raw, detections, infer_ms=infer_ms)


def print_dashboard_result(infer_output, tag="frame"):
    result = analyze_infer_output(infer_output)

    print("[{}] 仪表盘数量: {}".format(tag, result["dashboard_count"]))
    print("[{}] 字母: {}".format(tag, result["letter"]))

    for detail in result["dashboard_details"]:
        distance_text = "{:.3f}".format(detail["distance_m"]) if detail["distance_m"] is not None else "未知"
        print("[{}] 仪表盘#{} 顶点={} 状态={} 距离={}m".format(tag, detail["index"], detail["vertices"], detail["state_cn"], distance_text))
