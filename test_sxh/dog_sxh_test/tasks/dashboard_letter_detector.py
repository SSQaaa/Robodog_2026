import ctypes
import math
import sys

import cv2
import numpy as np



# Config (hard-coded)
ENGINE_PATH = r"/home/ysc/Desktop/Robodog_2026/test_sjh/best0330.engine"
PLUGIN_PATH = r"/home/ysc/Desktop/Robodog_2026/test_sjh/TRTX/yolov5/build/libmyplugins.so"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
SHOW_STREAM_DEFAULT = False
WINDOW_NAME = "robotdog_stream"

ROOT_DIR = r"/home/ysc/Desktop/Robodog_2026/test_sxh/dog_sxh_test"
sys.path.insert(0, ROOT_DIR)
from yolov5trt import YoLov5TRT
DASHBOARD_ID = 6
SSI_ID = 7
CLASS_NAMES = ["A", "B", "C", "D", "GC", "RC", "dashboard", "ssi"]
LETTER_ID_TO_NAME = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

POINTER_THRESHOLD = 118
NORMAL_ANGLE_MIN = 120.0
NORMAL_ANGLE_MAX = 180.0


class SimpleInfer:
    def __init__(self, show_stream=None):
        ctypes.CDLL(PLUGIN_PATH)
        self.model = YoLov5TRT(ENGINE_PATH)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if show_stream is None:
            self.show_stream = ("--stream" in sys.argv) or ("--show-stream" in sys.argv)
        else:
            self.show_stream = bool(show_stream)

    def infer_once(self):
        _, frame = self.cap.read()
        infer_output = self.model.infer(frame)
        if self.show_stream:
            self._show_infer_frame(infer_output)
        return infer_output

    def close(self):
        self.cap.release()
        self.model.destroy()
        if self.show_stream:
            cv2.destroyAllWindows()

    def _show_infer_frame(self, infer_output):
        image_raw, result_boxes, result_scores, result_classid, use_time = infer_output
        for i in range(len(result_classid)):
            x1, y1, x2, y2 = [int(v) for v in result_boxes[i]]
            cid = int(result_classid[i])
            score = float(result_scores[i])
            if 0 <= cid < len(CLASS_NAMES):
                label = f"{CLASS_NAMES[cid]} {score:.2f}"
            else:
                label = f"id_{cid} {score:.2f}"
            cv2.rectangle(image_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_raw, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image_raw, f"infer: {use_time*1000:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, image_raw)
        cv2.waitKey(1)


def _center_of_box(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def _length_width_from_box(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    length = max(w, h)
    width = min(w, h)
    return float(length), float(width)


def _area_from_box(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return float(abs((x2 - x1) * (y2 - y1)))


def _refine_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w - 1))
    y2 = max(0, min(y2, frame_h - 1))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    x1 = cx + (x1 - cx) * 0.5
    y1 = cy + (y1 - cy) * 0.5
    x2 = cx + (x2 - cx) * 0.5
    y2 = cy + (y2 - cy) * 0.5

    return [int(x1), int(y1), int(x2), int(y2)]


def _find_pointer_point_old(image_raw, dashboard_box):
    frame_h, frame_w = image_raw.shape[:2]
    x1, y1, x2, y2 = _refine_box(dashboard_box, frame_w, frame_h)

    roi = image_raw[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # old logic: threshold + contours + minAreaRect
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


def _state_from_dashboard_old(image_raw, dashboard_box, ssi_box):
    dashboard_center = _center_of_box(dashboard_box)
    ssi_center = _center_of_box(ssi_box)
    pointer_point = _find_pointer_point_old(image_raw, dashboard_box)
    if pointer_point is None:
        return "unknown"

    v1_x = ssi_center[0] - dashboard_center[0]
    v1_y = ssi_center[1] - dashboard_center[1]
    v2_x = pointer_point[0] - dashboard_center[0]
    v2_y = pointer_point[1] - dashboard_center[1]

    try:
        angle = math.degrees(
            math.acos(
                (v1_x * v2_x + v1_y * v2_y)
                / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))
            )
        )
    except Exception:
        angle = 65545.0

    if v1_x * v2_y - v2_x * v1_y > 0:
        direction = -1
    else:
        direction = 1

    if NORMAL_ANGLE_MIN <= angle <= NORMAL_ANGLE_MAX:
        return "normal"
    if direction == -1:
        return "low"
    return "high"


def _nearest_ssi_box(dashboard_box, ssi_boxes):
    if len(ssi_boxes) == 0:
        return None
    db_center = _center_of_box(dashboard_box)
    best_ssi = ssi_boxes[0]
    best_dist = float(np.linalg.norm(_center_of_box(ssi_boxes[0]) - db_center))

    for ssi in ssi_boxes[1:]:
        dist = float(np.linalg.norm(_center_of_box(ssi) - db_center))
        if dist < best_dist:
            best_dist = dist
            best_ssi = ssi

    return best_ssi


def _best_letter_from_detections(result_scores, result_classid):
    best_letter = "unknown"
    best_score = -1.0

    for i in range(len(result_classid)):
        cid = int(result_classid[i])
        if cid in LETTER_ID_TO_NAME:
            score = float(result_scores[i])
            if score > best_score:
                best_score = score
                best_letter = LETTER_ID_TO_NAME[cid]

    return best_letter


def analyze_infer_values(image_raw, result_boxes, result_scores, result_classid, use_time):
    _ = use_time

    dashboard_boxes = []
    ssi_boxes = []
    best_letter = _best_letter_from_detections(result_scores, result_classid)

    for i in range(len(result_classid)):
        cid = int(result_classid[i])
        if cid == DASHBOARD_ID:
            dashboard_boxes.append(result_boxes[i])
        if cid == SSI_ID:
            ssi_boxes.append(result_boxes[i])

    dashboard_boxes.sort(key=lambda b: (b[0] + b[2]) / 2.0)

    dashboard_count = len(dashboard_boxes)
    xyxy_list = []
    size_list = []
    area_list = []
    state_list = []

    for n, db in enumerate(dashboard_boxes, start=1):
        x1, y1, x2, y2 = [float(v) for v in db]
        length, width = _length_width_from_box(db)
        area = _area_from_box(db)

        best_ssi = _nearest_ssi_box(db, ssi_boxes)
        if best_ssi is None:
            state = "unknown"
        else:
            state = _state_from_dashboard_old(image_raw, db, best_ssi)

        xyxy_list.append([n, x1, y1, x2, y2])
        size_list.append([n, length, width])
        area_list.append([n, area])
        state_list.append([n, state])

    return {
        "dashboard_count": dashboard_count,
        "best_letter": best_letter,
        "xyxy_list": xyxy_list,
        "size_list": size_list,
        "area_list": area_list,
        "state_list": state_list,
    }


def analyze_infer_output(infer_output):
    image_raw, result_boxes, result_scores, result_classid, use_time = infer_output
    return analyze_infer_values(image_raw, result_boxes, result_scores, result_classid, use_time)


def print_dashboard_result(infer_output, tag):
    result = analyze_infer_output(infer_output)
    print(f"[{tag}] dashboard_count={result['dashboard_count']}")
    print(f"[{tag}] xyxy_list={result['xyxy_list']}")
    print(f"[{tag}] size_list={result['size_list']}")
    print(f"[{tag}] area_list={result['area_list']}")
    print(f"[{tag}] state_list={result['state_list']}")
