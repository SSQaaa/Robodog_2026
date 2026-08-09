# -*- coding: utf-8 -*-
import ctypes
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from project_config import (
    CLASS_NAMES,
    DASHBOARD_ID,
    DEFAULT_DASHBOARD_STATUS,
    ENGINE_PATH,
    LETTER_ID_TO_NAME,
    LIBS_DIR,
    PLUGIN_PATH,
    SSI_ID,
    STATE_CN_MAP,
    STATUS_CN_TO_TASK3,
    UNKNOWN_STATE_CN,
)


CONF_THRESH = 0.25
MIN_VALID_DEPTH_COUNT = 20
POINTER_THRESHOLD = 118
POINTER_MASK_CUT_WIDTH_DEG = 60
POINTER_RAY_MIN_R = 11
POINTER_RAY_MAX_R = 85
NORMAL_ANGLE_MIN = 120.0
NORMAL_ANGLE_MAX = 180.0


def load_yolo_runtime():
    if LIBS_DIR not in sys.path:
        sys.path.append(LIBS_DIR)
    ctypes.CDLL(PLUGIN_PATH)
    import yolov5_trt_cpp

    return yolov5_trt_cpp


class DepthSmoother:
    def __init__(self, max_len=5):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, obj_id, depth_mm):
        if depth_mm <= 0:
            return None
        self.history[obj_id].append(depth_mm)
        return int(np.median(np.array(self.history[obj_id])))


def yolo_to_xyxy(box, img_w, img_h, input_size=640):
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


def pixel_to_camera_3d(u, v, depth_mm, fx, fy, cx, cy):
    z = depth_mm / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def center_of_box(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def vertices_from_box(box):
    x1, y1, x2, y2 = [int(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def refine_box(box, frame_w, frame_h, ratio=0.5):
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


def make_pointer_mask(binary_shape):
    h, w = binary_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(1, w // 2), max(1, h // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    cut_width_deg = max(0, min(180, int(POINTER_MASK_CUT_WIDTH_DEG)))
    start_angle = 90 - cut_width_deg // 2
    end_angle = 90 + cut_width_deg // 2
    cv2.ellipse(mask, center, axes, 0, start_angle, end_angle, 0, -1)
    return mask


def is_black_near(binary, x, y):
    h, w = binary.shape[:2]
    x1 = max(0, x - 1)
    y1 = max(0, y - 1)
    x2 = min(w, x + 2)
    y2 = min(h, y + 2)
    return np.max(binary[y1:y2, x1:x2]) > 0


def find_pointer_point(image_raw, dashboard_box):
    frame_h, frame_w = image_raw.shape[:2]
    x1, y1, x2, y2 = refine_box(dashboard_box, frame_w, frame_h, ratio=0.5)
    roi = image_raw[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_raw = cv2.threshold(gray, POINTER_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    mask = make_pointer_mask(binary_raw.shape)
    binary = cv2.bitwise_and(binary_raw, binary_raw, mask=mask)

    dashboard_center = center_of_box(dashboard_box)
    center_x = float(dashboard_center[0]) - float(x1)
    center_y = float(dashboard_center[1]) - float(y1)
    h, w = binary.shape[:2]
    max_possible_r = int(min(w, h) * 0.50)
    ray_min_r = max(1, int(POINTER_RAY_MIN_R))
    ray_max_r = min(int(POINTER_RAY_MAX_R), max_possible_r)
    if ray_max_r <= ray_min_r:
        ray_max_r = max_possible_r

    best_score = -1
    best_point_roi = None
    for angle_deg in range(0, 360):
        rad = math.radians(angle_deg)
        cos_v = math.cos(rad)
        sin_v = math.sin(rad)
        current_run = 0
        max_run = 0
        max_run_end = ray_min_r
        hit_count = 0
        farthest_hit = 0
        for r in range(ray_min_r, ray_max_r + 1):
            px = int(round(center_x + float(r) * cos_v))
            py = int(round(center_y + float(r) * sin_v))
            if px < 0 or px >= w or py < 0 or py >= h:
                if current_run > max_run:
                    max_run = current_run
                    max_run_end = r - 1
                current_run = 0
                continue
            if is_black_near(binary, px, py):
                hit_count += 1
                farthest_hit = r
                current_run += 1
            else:
                if current_run > max_run:
                    max_run = current_run
                    max_run_end = r - 1
                current_run = 0
        if current_run > max_run:
            max_run = current_run
            max_run_end = ray_max_r
        score = max_run * 1000 + hit_count * 10 + farthest_hit
        if score > best_score:
            best_score = score
            use_r = max_run_end if max_run_end > 0 else farthest_hit
            best_point_roi = (
                int(round(center_x + float(use_r) * cos_v)),
                int(round(center_y + float(use_r) * sin_v)),
            )

    if best_point_roi is None or best_score <= 0:
        return None
    return np.array([best_point_roi[0] + x1, best_point_roi[1] + y1], dtype=np.float32)


def nearest_ssi_box(dashboard_box, ssi_boxes):
    if len(ssi_boxes) == 0:
        return None
    db_center = center_of_box(dashboard_box)
    return min(ssi_boxes, key=lambda box: float(np.linalg.norm(center_of_box(box) - db_center)))


def state_from_dashboard(image_raw, dashboard_box, ssi_box):
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


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    box: tuple
    depth_box: tuple
    center: tuple
    depth_mm: Optional[int]
    distance_m: Optional[float]
    valid_count: int
    position_3d: Optional[tuple] = None

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return abs(x2 - x1) * abs(y2 - y1)


class YoloDepthDetector:
    def __init__(self, engine_path=ENGINE_PATH, conf_thresh=0.4, min_valid_depth_count=20):
        self.engine_path = engine_path
        self.conf_thresh = float(conf_thresh)
        self.min_valid_depth_count = int(min_valid_depth_count)
        self.detector = None
        self.camera = None
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_size = None
        self.color_size = None

    def start(self):
        import orbbec_native

        yolov5_trt_cpp = load_yolo_runtime()
        print("[TRT] loading engine...")
        self.detector = yolov5_trt_cpp.Yolov5TRT(self.engine_path)
        print("[TRT] engine loaded")
        self.camera = orbbec_native.OrbbecCamera()
        self.camera.start()
        time.sleep(1.0)
        self.color_intrinsics = self.camera.get_color_intrinsics()
        self.depth_intrinsics = self.camera.get_depth_intrinsics()
        self.depth_size = self.camera.get_depth_size()
        self.color_size = self.camera.get_color_size()
        print(f"[Orbbec] color size: {self.color_size}")
        print(f"[Orbbec] depth size : {self.depth_size}")
        return self

    def stop(self):
        if self.camera is not None:
            self.camera.stop()

    def get_frame(self):
        frame = self.camera.get_color_frame()
        if frame is None:
            return None
        return np.asarray(frame, dtype=np.uint8).copy()

    def detect(self):
        frame = self.get_frame()
        if frame is None:
            return None, []
        color_h, color_w = frame.shape[:2]
        depth_w, depth_h = self.depth_size or self.camera.get_depth_size()
        detections = []
        for raw in self.detector.detect(frame):
            cx, cy, w, h, conf, cls_id = raw
            if float(conf) < self.conf_thresh:
                continue
            class_id = int(cls_id)
            class_name = CLASS_NAMES.get(int(cls_id), f"id{int(cls_id)}")
            color_box = yolo_to_xyxy((cx, cy, w, h), img_w=color_w, img_h=color_h)
            depth_box = scale_box(color_box, src_size=(color_w, color_h), dst_size=(depth_w, depth_h))
            depth_mm, valid_count = self.camera.get_depth_in_box(*depth_box)
            current_depth = int(depth_mm) if depth_mm > 0 and valid_count >= self.min_valid_depth_count else None
            distance_m = float(current_depth) / 1000.0 if current_depth is not None else None
            position_3d = None
            if current_depth is not None and self.depth_intrinsics is not None:
                u_depth = (depth_box[0] + depth_box[2]) // 2
                v_depth = depth_box[3]
                position_3d = pixel_to_camera_3d(
                    u_depth,
                    v_depth,
                    current_depth,
                    float(self.depth_intrinsics["fx"]),
                    float(self.depth_intrinsics["fy"]),
                    float(self.depth_intrinsics["cx"]),
                    float(self.depth_intrinsics["cy"]),
                )
            x1, y1, x2, y2 = color_box
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    conf=float(conf),
                    box=tuple(color_box),
                    depth_box=tuple(depth_box),
                    center=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                    depth_mm=current_depth,
                    distance_m=distance_m,
                    valid_count=int(valid_count),
                    position_3d=position_3d,
                )
            )
        return frame, detections


class DashboardInfer:
    def __init__(
        self,
        show_stream=False,
        conf_thresh=CONF_THRESH,
        min_valid_depth_count=MIN_VALID_DEPTH_COUNT,
        task3_conf_thresh=0.4,
    ):
        import orbbec_native

        yolov5_trt_cpp = load_yolo_runtime()
        self.show_stream = bool(show_stream)
        self.conf_thresh = float(conf_thresh)
        self.task3_conf_thresh = float(task3_conf_thresh)
        self.min_valid_depth_count = int(min_valid_depth_count)
        self.detector = yolov5_trt_cpp.Yolov5TRT(ENGINE_PATH)
        self.cam = orbbec_native.OrbbecCamera()
        self.cam.start()
        time.sleep(0.8)
        self.depth_w, self.depth_h = self.cam.get_depth_size()
        self.color_w, self.color_h = self.cam.get_color_size()
        self.color_intrinsics = self.cam.get_color_intrinsics()
        self.depth_intrinsics = self.cam.get_depth_intrinsics()
        self.infer_frame_index = 0

    def infer_once(self):
        frame = self.cam.get_color_frame()
        if frame is None:
            return {"image_raw": None, "detections": [], "infer_ms": 0.0}
        image_raw = np.asarray(frame, dtype=np.uint8).copy()
        self.infer_frame_index += 1
        color_h, color_w = image_raw.shape[:2]
        t0 = time.time()
        raw_detections = self.detector.detect(image_raw)
        t1 = time.time()
        detections = []
        for det in raw_detections:
            cx, cy, w, h, conf, cls_id = det
            if conf < self.conf_thresh:
                continue
            xyxy = yolo_to_xyxy((cx, cy, w, h), img_w=color_w, img_h=color_h)
            depth_box = scale_box(xyxy, src_size=(color_w, color_h), dst_size=(self.depth_w, self.depth_h))
            raw_depth_mm, valid_count = self.cam.get_depth_in_box(*depth_box)
            distance_m = float(raw_depth_mm) / 1000.0 if raw_depth_mm > 0 and valid_count >= self.min_valid_depth_count else None
            detections.append(
                {
                    "class_id": int(cls_id),
                    "score": float(conf),
                    "xyxy": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "distance_m": distance_m,
                }
            )
        output = {"image_raw": image_raw, "detections": detections, "infer_ms": (t1 - t0) * 1000.0}
        if self.show_stream:
            self._show_infer_frame(output)
        return output

    def detect(self):
        """使用同一摄像头和模型返回 Task3 所需的 Detection 对象。"""
        frame = self.cam.get_color_frame()
        if frame is None:
            return None, []
        frame = np.asarray(frame, dtype=np.uint8).copy()
        color_h, color_w = frame.shape[:2]
        detections = []
        for raw in self.detector.detect(frame):
            cx, cy, w, h, conf, cls_id = raw
            if float(conf) < self.task3_conf_thresh:
                continue
            class_id = int(cls_id)
            color_box = yolo_to_xyxy((cx, cy, w, h), img_w=color_w, img_h=color_h)
            depth_box = scale_box(
                color_box,
                src_size=(color_w, color_h),
                dst_size=(self.depth_w, self.depth_h),
            )
            depth_mm, valid_count = self.cam.get_depth_in_box(*depth_box)
            current_depth = (
                int(depth_mm)
                if depth_mm > 0 and valid_count >= self.min_valid_depth_count
                else None
            )
            position_3d = None
            if current_depth is not None:
                u_depth = (depth_box[0] + depth_box[2]) // 2
                v_depth = depth_box[3]
                position_3d = pixel_to_camera_3d(
                    u_depth,
                    v_depth,
                    current_depth,
                    float(self.depth_intrinsics["fx"]),
                    float(self.depth_intrinsics["fy"]),
                    float(self.depth_intrinsics["cx"]),
                    float(self.depth_intrinsics["cy"]),
                )
            x1, y1, x2, y2 = color_box
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, "id{}".format(class_id)),
                    conf=float(conf),
                    box=tuple(color_box),
                    depth_box=tuple(depth_box),
                    center=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                    depth_mm=current_depth,
                    distance_m=None if current_depth is None else current_depth / 1000.0,
                    valid_count=int(valid_count),
                    position_3d=position_3d,
                )
            )
        return frame, detections

    def _show_infer_frame(self, infer_output):
        image_raw = infer_output["image_raw"]
        if image_raw is None:
            return
        display = image_raw.copy()
        for det in infer_output["detections"]:
            x1, y1, x2, y2 = det["xyxy"]
            name = CLASS_NAMES.get(det["class_id"], f"id_{det['class_id']}")
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{name} {det['score']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        status = "frame: {}  infer: {:.1f} ms".format(self.infer_frame_index, infer_output["infer_ms"])
        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("dashboard_detector", display)
        cv2.waitKey(1)

    def stop(self):
        self.cam.stop()
        if self.show_stream:
            cv2.destroyAllWindows()

    def close(self):
        self.stop()


def analyze_infer_values(image_raw, detections, infer_ms=0.0):
    _ = infer_ms
    dashboard_dets = [det for det in detections if int(det["class_id"]) == DASHBOARD_ID]
    ssi_dets = [det for det in detections if int(det["class_id"]) == SSI_ID]
    dashboard_dets.sort(key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) / 2.0)
    letter = best_letter_from_detections(detections)
    ssi_boxes = [det["xyxy"] for det in ssi_dets]
    dashboard_details = []
    for idx, db_det in enumerate(dashboard_dets, start=1):
        db_box = db_det["xyxy"]
        best_ssi = nearest_ssi_box(db_box, ssi_boxes)
        state_key = "unknown" if best_ssi is None or image_raw is None else state_from_dashboard(image_raw, db_box, best_ssi)
        dashboard_details.append(
            {
                "index": idx,
                "vertices": vertices_from_box(db_box),
                "state": state_key,
                "state_cn": STATE_CN_MAP.get(state_key, UNKNOWN_STATE_CN),
                "distance_m": db_det["distance_m"],
                "letter": letter,
            }
        )
    return {"dashboard_count": len(dashboard_dets), "letter": letter, "dashboard_details": dashboard_details}


def analyze_infer_output(infer_output):
    return analyze_infer_values(
        infer_output["image_raw"],
        infer_output["detections"],
        infer_ms=infer_output.get("infer_ms", 0.0),
    )


def resolve_dashboard_status(records, default_status=None):
    default_status = dict(default_status or DEFAULT_DASHBOARD_STATUS)
    recognized = {}
    fallback_all_default = False
    for record in records or []:
        letter = str(record.get("letter", "unknown")).strip().upper()
        if letter not in DEFAULT_DASHBOARD_STATUS:
            continue
        state_cn = str(record.get("dashboard_state", UNKNOWN_STATE_CN)).strip()
        if state_cn == UNKNOWN_STATE_CN or state_cn not in STATUS_CN_TO_TASK3:
            fallback_all_default = True
            continue
        recognized[letter] = STATUS_CN_TO_TASK3[state_cn]
    letters = set(DEFAULT_DASHBOARD_STATUS)
    missing = sorted(letters - set(recognized))
    normal_count = list(recognized.values()).count("NORMAL")
    abnormal_count = list(recognized.values()).count("ABNORMAL")
    if fallback_all_default:
        return dict(default_status)
    if len(missing) == 0:
        if normal_count > 2 or abnormal_count > 2:
            return dict(default_status)
        return recognized
    if len(missing) == 1:
        if normal_count > 2 or abnormal_count > 2:
            return dict(default_status)
        status = dict(recognized)
        status[missing[0]] = "NORMAL" if normal_count < 2 else "ABNORMAL"
        if list(status.values()).count("NORMAL") != 2 or list(status.values()).count("ABNORMAL") != 2:
            return dict(default_status)
        return status
    status = dict(default_status)
    status.update(recognized)
    return status
