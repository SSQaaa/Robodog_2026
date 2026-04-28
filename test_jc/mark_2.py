import time
import os
import cv2
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from tqdm import tqdm

# ===================== 配置（严格对齐训练/转换参数）=====================
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.45

# 必须和训练data.yaml的names顺序完全一致！
categories = ['A', 'B', 'C', 'D', 'GC', 'RC', 'dashboard', 'ssi']
NUM_CLASSES = len(categories)

# 路径（修改为你的TensorRTx生成的engine路径）
ENGINE_PATH = "/home/ysc/Desktop/Robodog_2026/test_jc/best_final_v6.engine"
VAL_IMG_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/train/images/"
VAL_LABEL_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/train/labels/"
RESULT_SAVE_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/val_infer_results_tensorrtx"

os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# 类别颜色映射
def get_color_map(num_classes):
    np.random.seed(42)
    return {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}
color_map = get_color_map(NUM_CLASSES)

# 自动判断精度模式
IS_FP16 = "fp16" in ENGINE_PATH.lower()
INPUT_DTYPE = np.float16 if IS_FP16 else np.float32

# -------------------------- TensorRTx专用推理类 --------------------------
class YoLov5TRT(object):
    def __init__(self, engine_file_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if not self.engine:
            raise RuntimeError("TensorRTx Engine加载失败，请检查文件路径和版本")
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 绑定输入输出
        self.bindings = []
        self.input_name = None
        self.output_name = None
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.input_name = name
                self.input_shape = shape
                self.input_host = host_mem
                self.input_device = device_mem
                print(f"输入绑定 {name}：形状 {shape}，数据类型 {dtype}")
            else:
                self.output_name = name
                self.output_shape = shape
                self.output_host = host_mem
                self.output_device = device_mem
                print(f"输出绑定 {name}：形状 {shape}，数据类型 {dtype}")
        
        print("YOLOv5 TRT 模型初始化成功")

    def infer(self, image):
        origin_h, origin_w = image.shape[:2]
        
        # 预处理（和TensorRTx编译参数完全对齐）
        input_blob = self.preprocess(image)
        np.copyto(self.input_host, input_blob.ravel())
        
        # 显存拷贝+推理
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()
        
        # 后处理（适配TensorRTx的输出格式）
        output = self.output_host.reshape(self.output_shape)
        boxes, scores, class_ids = self.postprocess(output, origin_h, origin_w)
        
        return boxes, scores, class_ids, origin_h, origin_w

    def preprocess(self, img):
        # 官方letterbox预处理，和TensorRTx完全一致
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        scale = min(INPUT_W / w, INPUT_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        dx, dy = (INPUT_W - new_w) // 2, (INPUT_H - new_h) // 2
        
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized, dy, INPUT_H - new_h - dy, 
            dx, INPUT_W - new_w - dx, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        img = padded.astype(INPUT_DTYPE) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        
        return img

    def postprocess(self, output, origin_h, origin_w):
        # 适配TensorRTx的输出格式 [1, 25200, 4+1+num_classes]
        if output.shape[0] == 1:
            output = output[0]
        
        xywh = output[:, :4]
        obj_conf = output[:, 4]
        cls_prob = output[:, 5:]
        
        # 低置信度过滤
        mask = obj_conf > CONF_THRESH
        if not np.any(mask):
            return np.array([]), np.array([]), np.array([])
        
        xywh, obj_conf, cls_prob = xywh[mask], obj_conf[mask], cls_prob[mask]
        cls_ids = np.argmax(cls_prob, axis=1)
        cls_conf = cls_prob[np.arange(len(cls_ids)), cls_ids]
        scores = obj_conf * cls_conf
        
        # xywh -> xyxy
        boxes = np.zeros_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        
        # 坐标映射回原始图像
        scale = min(INPUT_W / origin_w, INPUT_H / origin_h)
        pad_x = (INPUT_W - origin_w * scale) / 2
        pad_y = (INPUT_H - origin_h * scale) / 2
        
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, origin_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, origin_h)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            CONF_THRESH, IOU_THRESHOLD
        )
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        indices = indices.flatten() if isinstance(indices, np.ndarray) else list(indices)
        return boxes[indices], scores[indices], cls_ids[indices]

    def draw(self, img, boxes, scores, cls_ids):
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id)
            color = color_map.get(cls_id, (0, 255, 0))
            name = categories[cls_id] if cls_id < len(categories) else "unknown"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name}:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    def destroy(self):
        self.input_device.free()
        self.output_device.free()
        self.context = None
        self.engine = None
        self.stream = None

# -------------------------- 精度指标计算（和ONNX版本完全一致） --------------------------
def parse_yolo_label(label_path, img_w, img_h):
    boxes, cls_ids = [], []
    if not os.path.exists(label_path):
        return np.array(boxes), np.array(cls_ids)
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            cls_ids.append(cls_id)
    
    return np.array(boxes), np.array(cls_ids)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def calculate_ap(recalls, precisions):
    ap = 0.0
    for t in np.linspace(0, 1.0, 11):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precisions[mask]) / 11.0
    return ap

def evaluate(model, img_dir, label_dir):
    stats = {cls: {"tp": [], "conf": [], "gt_count": 0} for cls in range(NUM_CLASSES)}
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_img = len(img_files)
    
    print(f"\n开始计算精度指标（共{total_img}张图片）...")
    for img_name in tqdm(img_files, desc="计算指标"):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        pred_boxes, pred_scores, pred_cls, _, _ = model.infer(img)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        gt_boxes, gt_cls = parse_yolo_label(label_path, w, h)
        
        for cls in gt_cls:
            if 0 <= cls < NUM_CLASSES:
                stats[cls]["gt_count"] += 1
        
        for cls in range(NUM_CLASSES):
            cls_pred_mask = pred_cls == cls
            cls_pred_boxes = pred_boxes[cls_pred_mask]
            cls_pred_scores = pred_scores[cls_pred_mask]
            
            cls_gt_mask = gt_cls == cls
            cls_gt_boxes = gt_boxes[cls_gt_mask]
            gt_matched = np.zeros(len(cls_gt_boxes), dtype=bool)
            
            sort_idx = np.argsort(-cls_pred_scores)
            cls_pred_boxes = cls_pred_boxes[sort_idx]
            cls_pred_scores = cls_pred_scores[sort_idx]
            
            tp = np.zeros(len(cls_pred_boxes), dtype=bool)
            for i, pred_box in enumerate(cls_pred_boxes):
                if len(cls_gt_boxes) == 0:
                    continue
                ious = np.array([compute_iou(pred_box, gt_box) for gt_box in cls_gt_boxes])
                max_iou = np.max(ious)
                max_iou_idx = np.argmax(ious)
                
                if max_iou >= 0.5 and not gt_matched[max_iou_idx]:
                    tp[i] = True
                    gt_matched[max_iou_idx] = True
            
            stats[cls]["tp"].extend(tp.tolist())
            stats[cls]["conf"].extend(cls_pred_scores.tolist())
    
    aps = {}
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for cls in range(NUM_CLASSES):
        cls_name = categories[cls]
        cls_stats = stats[cls]
        tp = np.array(cls_stats["tp"])
        conf = np.array(cls_stats["conf"])
        gt_count = cls_stats["gt_count"]
        
        total_gt += gt_count
        total_tp += np.sum(tp)
        total_fp += np.sum(1 - tp)
        
        if gt_count == 0 or len(tp) == 0:
            aps[cls_name] = 0.0
            continue
        
        sort_idx = np.argsort(-conf)
        tp_sorted = tp[sort_idx]
        tp_cumsum = np.cumsum(tp_sorted)
        fp_cumsum = np.cumsum(1 - tp_sorted)
        
        recalls = tp_cumsum / gt_count
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        ap = calculate_ap(recalls, precisions)
        aps[cls_name] = ap
    
    global_precision = total_tp / (total_tp + total_fp + 1e-6)
    global_recall = total_tp / (total_gt + 1e-6)
    map50 = np.mean(list(aps.values()))
    
    return global_precision, global_recall, map50, aps

# -------------------------- 批量推理（和ONNX版本格式完全一致） --------------------------
def batch_infer(model, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_img = len(img_files)
    
    success_count = 0
    total_infer_time = 0
    
    print(f"\n开始批量推理验证集（共{total_img}张图片）...")
    for img_name in tqdm(img_files, desc="批量推理"):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        t_start = time.time()
        boxes, scores, cls_ids, h, w = model.infer(img)
        infer_time = (time.time() - t_start) * 1000
        
        total_infer_time += infer_time
        success_count += 1
        
        result_img = model.draw(img.copy(), boxes, scores, cls_ids)
        cv2.putText(result_img, f"{infer_time:.1f}ms", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_dir, img_name), result_img)
    
    avg_infer_time = total_infer_time / success_count if success_count > 0 else 0
    return success_count, total_img, avg_infer_time

# -------------------------- 主程序 --------------------------
def main():
    print("加载TensorRTx Engine模型...")
    model = YoLov5TRT(ENGINE_PATH)
    
    try:
        success_count, total_img, avg_infer_time = batch_infer(model, VAL_IMG_DIR, RESULT_SAVE_DIR)
        success_rate = (success_count / total_img) * 100 if total_img > 0 else 0
        
        print(f"\n=== 批量推理统计 ===")
        print(f"验证集总数：{total_img}")
        print(f"成功数：{success_count} | 成功率：{success_rate:.2f}%")
        print(f"失败数：{total_img - success_count}")
        
        precision, recall, map50, aps = evaluate(model, VAL_IMG_DIR, VAL_LABEL_DIR)
        
        print(f"\n=== 验证集精度指标 ===")
        print(f"Precision（精确率）: {precision:.4f}")
        print(f"Recall（召回率）: {recall:.4f}")
        print(f"mAP@0.5: {map50:.4f}")
        
        print(f"\n=== 按类别AP@0.5 ===")
        for name, ap in aps.items():
            print(f"{name}: {ap:.4f}")
        
        print(f"\n=== 验证集推理验证总结 ===")
        print(f"推理成功率：{success_rate:.2f}%")
        print(f"平均单帧推理耗时：{avg_infer_time:.1f} ms")
        print(f"Precision：{precision:.4f}")
        print(f"Recall：{recall:.4f}")
        print(f"mAP@0.5：{map50:.4f}")
        print(f"推理结果保存路径：{RESULT_SAVE_DIR}")
        
    finally:
        model.destroy()
        print("资源已释放。")

if __name__ == "__main__":
    main()