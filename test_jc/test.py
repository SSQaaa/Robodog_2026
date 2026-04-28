import os
import sys
import math
import ctypes
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

# ==================== 固定配置（你的引擎 38 维输出）====================
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.2
CONF_THRESH_DASHBOARD = 0.1
IOU_THRESHOLD = 0.4

categories = ["A", "B", "C", "D", "GC", "RC", "dashboard", "ssi"]
DASHBOARD_ID = 6
SSI_ID = 7
DETECTION_SIZE = 38

# 角度判断阈值（比赛可微调）
NORMAL_ANGLE_MIN = 120.0
NORMAL_ANGLE_MAX = 180.0
POINTER_THRESHOLD = 118

# ==================== 引擎路径（改成你自己的）====================
ENGINE_PATH = "/home/ysc/Desktop/Robodog_2026/test_sjh/best0330.engine"
PLUGIN_PATH = "/home/ysc/Desktop/Robodog_2026/test_sjh/TRTX/yolov5/build/libmyplugins.so"
CAMERA_INDEX = 0

# ==================== YoLov5TRT 类（修复版 38 维，不崩溃）====================
class YoLov5TRT:
    def __init__(self, engine_file_path, plugin_path):
        cuda.init()
        self.device = cuda.Device(0)
        self.cfx = self.device.make_context()
        self.cfx.pop()
        self.stream = cuda.Stream()

        if os.path.exists(plugin_path):
            ctypes.CDLL(plugin_path)

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

    def infer(self, image_raw):
        with self.cfx:
            input_image, _, origin_h, origin_w = self.preprocess_image(image_raw)
            np.copyto(self.host_inputs[0], input_image.ravel())

            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            self.stream.synchronize()

            output = self.host_outputs[0]
            boxes, scores, classid = self.post_process(output, origin_h, origin_w)
            return image_raw, boxes, scores, classid, 0.0

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
        return image, image_raw, h, w

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

    def post_process(self, output, origin_h, origin_w):
        num = int(output[0])
        if num <= 0:
            return np.empty((0,4)), np.empty(0), np.empty(0)
        raw = output[1:]
        valid_len = (len(raw) // DETECTION_SIZE) * DETECTION_SIZE
        pred = np.reshape(raw[:valid_len], (-1, DETECTION_SIZE))[:num]

        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]

        valid = (classid >=0) & (classid < len(categories))
        boxes = boxes[valid]
        scores = scores[valid]
        classid = classid[valid]

        cid_int = classid.astype(int)
        thresh = np.where(cid_int == 6, CONF_THRESH_DASHBOARD, CONF_THRESH)
        keep = scores > thresh
        boxes = boxes[keep]
        scores = scores[keep]
        classid = classid[keep]

        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        return boxes, scores, classid

    def destroy(self):
        del self.context
        del self.engine

# ==================== 表计指针识别核心算法 ====================
def _center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=np.float32)

def _find_pointer(image, db_box):
    x1,y1,x2,y2 = map(int, db_box)
    roi = image[y1:y2, x1:x2]
    if roi.size ==0: return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, POINTER_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cnts: return None
    m = max(cnts, key=cv2.contourArea)
    ((cx,cy),_) = cv2.minEnclosingCircle(m)
    return np.array([x1+cx, y1+cy], dtype=np.float32)

def get_meter_state(img, db_box, ssi_box):
    c_db = _center(db_box)
    c_ssi = _center(ssi_box)
    ptr = _find_pointer(img, db_box)
    if ptr is None: return "未识别"

    v1 = c_ssi - c_db
    v2 = ptr - c_db
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1<1e-3 or n2<1e-3: return "未识别"

    cos = np.dot(v1,v2)/(n1*n2)
    cos = np.clip(cos, -1,1)
    angle = math.degrees(math.acos(cos))
    cross = v1[0]*v2[1] - v1[1]*v2[0]

    if NORMAL_ANGLE_MIN <= angle <= NORMAL_ANGLE_MAX:
        return "正常"
    return "偏高" if cross>0 else "偏低"

# ==================== 主程序：摄像头实时识别 ====================
def main():
    model = YoLov5TRT(ENGINE_PATH, PLUGIN_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print("? 摄像头启动，表计识别运行中...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        img, boxes, scores, cids, _ = model.infer(frame)
        dbs = []
        ssis = []

        for b,s,c in zip(boxes, scores, cids):
            if int(c)==DASHBOARD_ID:
                dbs.append(b)
            if int(c)==SSI_ID:
                ssis.append(b)

        # 匹配每个表盘最近的SSI
        for i,db in enumerate(dbs):
            best = None
            min_d = 9999
            for ssi in ssis:
                d = np.linalg.norm(_center(db)-_center(ssi))
                if d<min_d:
                    min_d=d
                    best=ssi
            if best is not None:
                state = get_meter_state(img, db, best)
                x1,y1,x2,y2 = map(int, db)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img,f"表计{i+1}:{state}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                # 终端中文输出
                print(f"?? 表计 {i+1} 状态：{state}")

        cv2.imshow("Dashboard Meter (Table Competition)", img)
        if cv2.waitKey(1)&0xFF==27:
            break

    cap.release()
    cv2.destroyAllWindows()
    model.destroy()

if __name__ == "__main__":
    main()