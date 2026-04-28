import ctypes
import math
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== 硬编码配置 ====================
ENGINE_PATH = r"/home/ysc/Desktop/Robodog_2026/test_sjh/best0330.engine"
PLUGIN_PATH = r"/home/ysc/Desktop/Robodog_2026/test_sjh/TRTX/yolov5/build/libmyplugins.so"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
SHOW_STREAM_DEFAULT = False
WINDOW_NAME = "dashboard_meter_recognition"

# 模型配置：【专门为SSI+仪表盘降低阈值，减少漏检】
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.15               # 通用置信度调低（防漏检）
CONF_THRESH_DASHBOARD = 0.15     # 仪表盘更低
CONF_THRESH_SSI = 0.10           # SSI最低（最容易漏检）
IOU_THRESHOLD = 0.40             # NMS适中
DETECTION_SIZE = 38
DASHBOARD_ID = 6
SSI_ID = 7
CLASS_NAMES = ["A", "B", "C", "D", "GC", "RC", "dashboard", "ssi"]

# 表计识别参数（完全不变）
POINTER_THRESHOLD = 118
NORMAL_ANGLE_MIN = 120.0
NORMAL_ANGLE_MAX = 180.0

# ==================== 中文乱码修复（不变） ====================
FONT_CANDIDATES = [
    "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
font = None
for font_path in FONT_CANDIDATES:
    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, 20)
            break
        except:
            continue

# ==================== TensorRT推理类（只改post_process） ====================
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
except ImportError:
    print("❌ 请安装依赖：pip install pycuda tensorrt opencv-python pillow numpy")
    sys.exit(1)

class YoLov5TRT:
    def __init__(self, engine_file_path):
        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        self.stream = cuda.Stream()

    def preprocess_image(self, image_raw):
        h, w, _ = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0

        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h

        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def nms(self, boxes, scores, iou_threshold=IOU_THRESHOLD):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        keep = []
        index = scores.argsort()[::-1]

        while index.size > 0:
            i = index[0]
            keep.append(i)
            if index.size == 1:
                break
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx + 1]
        return keep

    def nms_classwise(self, boxes, scores, classid, iou_threshold=IOU_THRESHOLD):
        keep_all = []
        classid_int = classid.astype(np.int32)
        unique_cls = np.unique(classid_int)

        for cid in unique_cls:
            idx = np.where(classid_int == cid)[0]
            if idx.size == 0:
                continue
            sub_keep = self.nms(boxes[idx, :], scores[idx], iou_threshold)
            keep_all.extend(idx[sub_keep].tolist())

        if len(keep_all) == 0:
            return np.empty((0,), dtype=np.int32)
        keep_all = np.array(keep_all, dtype=np.int32)
        order = scores[keep_all].argsort()[::-1]
        return keep_all[order]

    # ==================== 【关键修改：只保留dashboard+ssi，强制过滤其他】 ====================
    def post_process(self, output, origin_h, origin_w):
        num = int(output[0])
        if num <= 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0)

        raw = output[1:]
        valid_len = (len(raw) // DETECTION_SIZE) * DETECTION_SIZE
        pred = np.reshape(raw[:valid_len], (-1, DETECTION_SIZE))
        if num > pred.shape[0]:
            num = pred.shape[0]
        pred = pred[:num, :]

        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]

        # 1. 只保留 dashboard(6) 和 ssi(7)
        keep_idx = (classid == DASHBOARD_ID) | (classid == SSI_ID)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        classid = classid[keep_idx]
        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0)

        # 2. 分类别置信度（ssi最低，防漏检）
        classid_int = classid.astype(np.int32)
        score_thresh = np.where(
            classid_int == DASHBOARD_ID, CONF_THRESH_DASHBOARD,
            np.where(classid_int == SSI_ID, CONF_THRESH_SSI, CONF_THRESH)
        )
        keep_score = scores > score_thresh
        boxes = boxes[keep_score, :]
        scores = scores[keep_score]
        classid = classid[keep_score]
        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0), np.empty(0)

        # 3. 坐标还原 + NMS
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        indices = self.nms_classwise(boxes, scores, classid, IOU_THRESHOLD)

        return boxes[indices, :], scores[indices], classid[indices], indices

    def infer(self, image_raw):
        start_time = cv2.getTickCount()
        input_image, origin_h, origin_w = self.preprocess_image(image_raw)
        np.copyto(self.host_inputs[0], input_image.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]
        boxes, scores, classid, _ = self.post_process(output, origin_h, origin_w)
        use_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        return image_raw, boxes, scores, classid, use_time

    def destroy(self):
        if hasattr(self, 'context'): del self.context
        if hasattr(self, 'engine'): del self.engine
        if hasattr(self, 'stream'): del self.stream

# ==================== 工具函数（完全不变） ====================
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
        return "未识别"

    v1_x = ssi_center[0] - dashboard_center[0]
    v1_y = ssi_center[1] - dashboard_center[1]
    v2_x = pointer_point[0] - dashboard_center[0]
    v2_y = pointer_point[1] - dashboard_center[1]

    try:
        angle = math.degrees(
            math.acos(
                (v1_x * v2_x + v1_y * v2_y)
                / (((v1_x **2 + v1_y** 2) ** 0.5) * ((v2_x **2 + v2_y** 2) ** 0.5))
            )
        )
    except Exception:
        angle = 65545.0

    if v1_x * v2_y - v2_x * v1_y > 0:
        direction = -1
    else:
        direction = 1

    if NORMAL_ANGLE_MIN <= angle <= NORMAL_ANGLE_MAX:
        return "正常"
    if direction == -1:
        return "偏低"
    return "偏高"

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

def draw_chinese_text(img, text, pos, color=(0, 255, 0)):
    global font
    if font is not None:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

# ==================== 分析与打印（不变） ====================
def analyze_infer_values(image_raw, result_boxes, result_scores, result_classid, use_time):
    dashboard_boxes = []
    ssi_boxes = []
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
            state = "未识别"
        else:
            state = _state_from_dashboard_old(image_raw, db, best_ssi)

        xyxy_list.append([n, x1, y1, x2, y2])
        size_list.append([n, length, width])
        area_list.append([n, area])
        state_list.append([n, state])

    return {
        "dashboard_count": dashboard_count,
        "xyxy_list": xyxy_list,
        "size_list": size_list,
        "area_list": area_list,
        "state_list": state_list,
        "raw_boxes": result_boxes,
        "raw_scores": result_scores,
        "raw_classid": result_classid,
    }

def analyze_infer_output(infer_output):
    image_raw, result_boxes, result_scores, result_classid, use_time = infer_output
    return analyze_infer_values(image_raw, result_boxes, result_scores, result_classid, use_time), use_time

last_result_str = ""
def print_dashboard_result(analyze_result, tag):
    global last_result_str
    result_str = f"[{tag}] 表计数量：{analyze_result['dashboard_count']}\n"
    for item in analyze_result['state_list']:
        n, s = item
        result_str += f"  表计{n}：{s}\n"
    if result_str != last_result_str:
        last_result_str = result_str
        print("===== 识别结果 =====")
        print(result_str, end="")
        print("==================\n")

# ==================== 推理主类（不变） ====================
class SimpleInfer:
    def __init__(self, show_stream=None):
        ctypes.CDLL(PLUGIN_PATH)
        self.model = YoLov5TRT(ENGINE_PATH)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if show_stream is None:
            self.show_stream = ("--stream" in sys.argv) or ("--show-stream" in sys.argv) or SHOW_STREAM_DEFAULT
        else:
            self.show_stream = bool(show_stream)

    def infer_once(self):
        _, frame = self.cap.read()
        if frame is None:
            return None
        infer_output = self.model.infer(frame)
        analyze_result, use_time = analyze_infer_output(infer_output)
        if self.show_stream:
            self._show_infer_frame(infer_output, analyze_result, use_time)
        return infer_output, analyze_result

    def close(self):
        self.cap.release()
        self.model.destroy()
        if self.show_stream:
            cv2.destroyAllWindows()

    def _show_infer_frame(self, infer_output, analyze_result, use_time):
        image_raw, result_boxes, result_scores, result_classid, _ = infer_output
        for i in range(len(result_classid)):
            x1, y1, x2, y2 = [int(v) for v in result_boxes[i]]
            cid = int(result_classid[i])
            score = float(result_scores[i])
            if 0 <= cid < len(CLASS_NAMES):
                label = f"{CLASS_NAMES[cid]} {score:.2f}"
                color = (0, 0, 255) if cid == SSI_ID else (0, 255, 0)
            else:
                label = f"id{cid} {score:.2f}"
                color = (0, 255, 0)
            cv2.rectangle(image_raw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_raw, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        for item in analyze_result['state_list']:
            n, state = item
            _, x1, y1, _, _ = analyze_result['xyxy_list'][n-1]
            image_raw = draw_chinese_text(image_raw, f"表计{n}:{state}", (int(x1), int(y1)-35), (0, 255, 0))

        cv2.putText(image_raw, f"infer: {use_time*1000:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, image_raw)
        cv2.waitKey(1)

# ==================== 主程序（不变） ====================
if __name__ == "__main__":
    infer = SimpleInfer()
    print("✅ 表计识别（SSI+仪表盘增强版）启动")
    print("📌 已：降低ssi/仪表盘阈值 + 只保留这两类目标")
    print("📌 按 Ctrl+C 退出\n")

    try:
        while True:
            result = infer.infer_once()
            if result is None:
                continue
            _, analyze_result = result
            print_dashboard_result(analyze_result, "实时")
    except KeyboardInterrupt:
        print("\n🛑 程序终止")
    finally:
        infer.close()
        print("✅ 资源已释放")