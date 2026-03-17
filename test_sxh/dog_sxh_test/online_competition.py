import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict, deque

import cv2
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
ONLINE_COMP_DIR = os.path.join(CURRENT_DIR, "online_competition")

if ONLINE_COMP_DIR not in sys.path:
    sys.path.insert(0, ONLINE_COMP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DASHBOARD_CLASS_ID = 14
SSI_CLASS_ID = 15
STATE_TEXT = {
    "low": ("偏低", "异常"),
    "normal": ("正常", "正常"),
    "high": ("偏高", "异常"),
}
COLOR_BY_STATE = {
    "low": (0, 255, 255),      # yellow
    "normal": (0, 255, 0),     # green
    "high": (0, 0, 255),       # red
    "unknown": (200, 200, 200)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Online competition dashboard state detector")
    parser.add_argument("--camera", type=int, default=2, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=640, help="Camera height")
    parser.add_argument("--threshold", type=int, default=118, help="Pointer binarization threshold")
    parser.add_argument("--normal-min", type=float, default=120.0, help="Normal angle lower bound")
    parser.add_argument("--normal-max", type=float, default=180.0, help="Normal angle upper bound")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Terminal report interval seconds")
    parser.add_argument("--max-dashboards", type=int, default=6, help="How many dashboards to report")
    return parser.parse_args()


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def refine_box(bbox, frame_w, frame_h, scale=0.5):
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    bbox[0] = clamp(bbox[0], 0, frame_w - 1)
    bbox[1] = clamp(bbox[1], 0, frame_h - 1)
    bbox[2] = clamp(bbox[2], 0, frame_w - 1)
    bbox[3] = clamp(bbox[3], 0, frame_h - 1)
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bbox[0] = cx + (bbox[0] - cx) * scale
    bbox[1] = cy + (bbox[1] - cy) * scale
    bbox[2] = cx + (bbox[2] - cx) * scale
    bbox[3] = cy + (bbox[3] - cy) * scale
    return bbox.astype(np.int32)


def center_of_box(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def match_ssi(dashboard_box, ssi_centers):
    x1, y1, x2, y2 = dashboard_box
    inside = []
    board_center = center_of_box(dashboard_box)
    for c in ssi_centers:
        if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
            inside.append(c)
    if inside:
        return min(inside, key=lambda c: np.linalg.norm(c - board_center))
    if not ssi_centers:
        return None
    nearest = min(ssi_centers, key=lambda c: np.linalg.norm(c - board_center))
    diag = np.linalg.norm(np.array([x2 - x1, y2 - y1], dtype=np.float32))
    if np.linalg.norm(nearest - board_center) <= 0.65 * diag:
        return nearest
    return None


def detect_pointer_tip(frame, dashboard_box, board_center, threshold):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = refine_box(dashboard_box, frame_w, frame_h, scale=0.5)
    if x2 <= x1 or y2 <= y1:
        return None, None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return None, thresh

    best_contour = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        points = contour.reshape(-1, 2).astype(np.float32)
        points[:, 0] += x1
        points[:, 1] += y1
        dists = np.linalg.norm(points - board_center, axis=1)
        max_dist = float(np.max(dists))
        score = max_dist + area * 0.03
        if score > best_score:
            best_score = score
            best_contour = points

    if best_contour is None:
        return None, thresh

    dists = np.linalg.norm(best_contour - board_center, axis=1)
    tip = best_contour[int(np.argmax(dists))]
    return tip, thresh


def classify_state(board_center, ssi_center, pointer_tip, normal_min, normal_max):
    v1 = ssi_center - board_center
    v2 = pointer_tip - board_center
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None, None
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))

    cross = v1[0] * v2[1] - v2[0] * v1[1]
    direction = -1 if cross > 0 else 1

    if normal_min <= angle <= normal_max:
        return "normal", angle
    if direction == -1:
        return "low", angle
    return "high", angle


def state_vote(history):
    if not history:
        return None
    counter = Counter(history)
    return counter.most_common(1)[0][0]


def render_state_text(state):
    if state not in STATE_TEXT:
        return "未识别"
    zone, abnormal = STATE_TEXT[state]
    return f"{zone}（{abnormal}）"


def main():
    args = parse_args()

    try:
        from detect_trt import Detect_image  # noqa: E402
    except Exception as exc:
        raise RuntimeError(
            "导入 detect_trt 失败，请检查 TensorRT/PyCUDA/OpenCV 环境是否正确安装。"
        ) from exc

    engine_path = os.path.join(PROJECT_ROOT, "build", "yolov5s.engine")
    plugin_path = os.path.join(PROJECT_ROOT, "build", "libmyplugins.so")
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"未找到TensorRT引擎文件: {engine_path}")
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"未找到TensorRT插件文件: {plugin_path}")

    os.chdir(PROJECT_ROOT)
    detector = Detect_image()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头打开失败，索引: {args.camera}")

    state_histories = defaultdict(lambda: deque(maxlen=8))
    last_report_time = 0.0
    last_report_text = ""

    print(f"启动成功: camera={args.camera}, 分辨率={args.width}x{args.height}")
    print("按 q 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头帧失败，正在重试...")
                time.sleep(0.05)
                continue

            _, boxes, scores, class_ids, _ = detector.yolov5_wrapper.infer(frame)
            if boxes is None:
                boxes = np.empty((0, 4), dtype=np.float32)
            if class_ids is None:
                class_ids = np.empty((0,), dtype=np.float32)

            board_boxes = [boxes[i] for i in range(len(class_ids)) if int(class_ids[i]) == DASHBOARD_CLASS_ID]
            ssi_centers = [center_of_box(boxes[i]) for i in range(len(class_ids)) if int(class_ids[i]) == SSI_CLASS_ID]

            board_boxes.sort(key=lambda b: ((b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0))
            board_boxes = board_boxes[: args.max_dashboards]

            current_states = {}
            for idx, box in enumerate(board_boxes, 1):
                board_center = center_of_box(box)
                ssi_center = match_ssi(box, ssi_centers)
                state = None
                angle = None

                if ssi_center is not None:
                    tip, _ = detect_pointer_tip(frame, box, board_center, args.threshold)
                    if tip is not None:
                        state, angle = classify_state(
                            board_center, ssi_center, tip, args.normal_min, args.normal_max
                        )

                        cv2.circle(frame, (int(board_center[0]), int(board_center[1])), 4, (255, 0, 0), -1)
                        cv2.circle(frame, (int(ssi_center[0]), int(ssi_center[1])), 4, (0, 255, 255), -1)
                        cv2.circle(frame, (int(tip[0]), int(tip[1])), 5, (255, 255, 0), -1)
                        cv2.line(
                            frame,
                            (int(board_center[0]), int(board_center[1])),
                            (int(tip[0]), int(tip[1])),
                            (255, 255, 0),
                            2
                        )

                if state is not None:
                    state_histories[idx].append(state)
                voted_state = state_vote(state_histories[idx])
                current_states[idx] = voted_state

                draw_color = COLOR_BY_STATE.get(voted_state or "unknown", COLOR_BY_STATE["unknown"])
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                status_text = render_state_text(voted_state)
                if angle is not None:
                    status_text += f" angle={angle:.1f}"
                cv2.putText(
                    frame,
                    f"Dashboard-{idx}: {status_text}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    draw_color,
                    2,
                    cv2.LINE_AA
                )

            now = time.time()
            if now - last_report_time >= args.report_interval:
                lines = []
                for idx in range(1, args.max_dashboards + 1):
                    lines.append(f"仪表盘{idx}: {render_state_text(current_states.get(idx))}")
                report = "\n".join(lines)
                if report != last_report_text:
                    print("-" * 40)
                    print(report)
                    last_report_text = report
                last_report_time = now

            cv2.imshow("online_competition_dashboard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.destroy()


if __name__ == "__main__":
    main()
