import time
import cv2
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# ===================== 配置和你原来完全一致 =====================
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.45

categories = ['A', 'B', 'C', 'D', 'GC', 'RC', 'dashboard', 'ssi']
NUM_CLASSES = len(categories)

ENGINE_PATH = "/home/ysc/Desktop/Robodog_2026/test_jc/best0330_onnx_2.engine"

# 颜色映射
def get_color_map(num_classes):
    np.random.seed(42)
    return {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}
color_map = get_color_map(NUM_CLASSES)

# ===================== YOLOv5 TRT 类（完全沿用你的逻辑）=====================
class Yolov5TRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(map(int, self.engine.get_binding_shape(i)))
            num_elements = np.prod(shape)

            host_mem = cuda.pagelocked_empty(num_elements, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(INPUT_W / w, INPUT_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pad_w = (INPUT_W - new_w) // 2
        pad_h = (INPUT_H - new_h) // 2

        img = cv2.resize(img, (new_w, new_h))
        img = cv2.copyMakeBorder(img, pad_h, INPUT_H - new_h - pad_h, pad_w, INPUT_W - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img, scale, pad_w, pad_h, h, w

    def infer(self, img):
        input_img, scale, pad_w, pad_h, orig_h, orig_w = self.preprocess(img)
        np.copyto(self.host_inputs[0], input_img.ravel())

        cuda.memcpy_htod(self.device_inputs[0], self.host_inputs[0])
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.host_outputs[0], self.device_outputs[0])

        output = self.host_outputs[0].reshape(tuple(map(int, self.engine.get_binding_shape(1))))
        boxes, scores, class_ids = self.postprocess(output, orig_h, orig_w, scale, pad_w, pad_h)
        return boxes, scores, class_ids

    def postprocess(self, output, orig_h, orig_w, scale, pad_w, pad_h):
        prediction = output[0]
        xc = prediction[..., 4] > CONF_THRESH
        x = prediction[xc]
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])

        # xywh -> xyxy
        xywh = x[:, :4]
        boxes = np.zeros_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        # 置信度
        x[:, 5:] *= x[:, 4:5]
        conf = np.max(x[:, 5:], axis=1)
        j = np.argmax(x[:, 5:], axis=1)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), CONF_THRESH, IOU_THRESHOLD)
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        indices = indices.flatten()
        boxes = boxes[indices]
        scores = conf[indices]
        class_ids = j[indices].astype(int)

        # 映射回原图
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, scores, class_ids

    def draw(self, img, boxes, scores, cls_ids):
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cls_id = int(cls_ids[i])
            score = scores[i]
            color = color_map.get(cls_id, (0, 255, 0))
            label = f"{categories[cls_id]}: {score:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def destroy(self):
        for d in self.device_inputs:
            d.free()
        for d in self.device_outputs:
            d.free()
        del self.context
        del self.engine
        del self.runtime

# ===================== 摄像头主逻辑（修复版）=====================
def main():
    print("Loading TensorRT engine...")
    model = Yolov5TRT(ENGINE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Start detection, press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        t0 = time.time()
        boxes, scores, cls_ids = model.infer(frame)
        frame = model.draw(frame, boxes, scores, cls_ids)
        t1 = time.time()

        # FPS
        fps = 1 / (t1 - t0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("TRT YOLOv5 Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    model.destroy()

if __name__ == "__main__":
    main()