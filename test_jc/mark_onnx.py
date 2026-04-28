import time
import os
import cv2
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from tqdm import tqdm

# ===================== 配置（必须严格对齐训练参数）=====================
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.45

# ===================== 【必须修改】和训练时data.yaml的names完全一致！！！=====================
categories = ['A', 'B', 'C', 'D', 'GC', 'RC', 'dashboard', 'ssi']
NUM_CLASSES = len(categories)

# 路径配置
ENGINE_PATH = "/home/ysc/Desktop/Robodog_2026/test_jc/best0330_onnx_2.engine"
VAL_IMG_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/train/images/"
VAL_LABEL_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/train/labels/"
RESULT_SAVE_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/val_infer_results"

os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# 类别颜色映射
def get_color_map(num_classes):
    np.random.seed(42)
    return {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}
color_map = get_color_map(NUM_CLASSES)

# -------------------------- 最终正确版ONNX TRT推理类 --------------------------
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
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            
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
                                cv2.BORDER_CONSTANT, value=(114,114,114))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img, scale, pad_w, pad_h, h, w

    def infer(self, img):
        input_img, scale, pad_w, pad_h, orig_h, orig_w = self.preprocess(img)
        self.host_inputs[0] = input_img.ravel()
        cuda.memcpy_htod(self.device_inputs[0], self.host_inputs[0])
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.host_outputs[0], self.device_outputs[0])
        output = self.host_outputs[0].reshape(tuple(map(int, self.engine.get_binding_shape(1))))
        boxes, scores, class_ids = self.postprocess(output, orig_h, orig_w, scale, pad_w, pad_h)
        return boxes, scores, class_ids, orig_h, orig_w

    def postprocess(self, output, orig_h, orig_w, scale, pad_w, pad_h):
        prediction = output[0]  # shape: (25200, 13)
        
        # 1. 置信度过滤
        xc = prediction[..., 4] > CONF_THRESH
        x = prediction[xc]
        
        if not x.shape[0]:
            return np.array([]), np.array([]), np.array([])
        
        # 2. 【必须转换】xywh (中心+宽高) → xyxy (左上角+右下角)
        xywh = x[:, :4]
        boxes = np.zeros_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2 = cx + w/2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2 = cy + h/2
        
        # 3. 置信度 × 类别概率
        x[:, 5:] *= x[:, 4:5]
        conf = np.max(x[:, 5:], axis=1)
        j = np.argmax(x[:, 5:], axis=1)
        
        # 4. NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(),
                                   CONF_THRESH, IOU_THRESHOLD)
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        indices = indices.flatten()
        boxes = boxes[indices]
        scores = conf[indices]
        class_ids = j[indices].astype(int)
        
        # 5. 坐标映射回原图
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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def destroy(self):
        pass

# -------------------------- 精度指标计算（保持不变） --------------------------
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

# -------------------------- 批量推理函数 --------------------------
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
    print("加载TensorRT Engine模型...")
    model = Yolov5TRT(ENGINE_PATH)
    
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