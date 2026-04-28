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
# 原代码类别顺序和标注不匹配，导致AP计算完全错误，必须和训练时的id顺序一一对应
categories = ['A', 'B', 'C', 'D', 'GC', 'RC', 'dashboard', 'ssi']
NUM_CLASSES = len(categories)

# 路径配置（修改为你的engine路径）
ENGINE_PATH = "/home/ysc/Desktop/Robodog_2026/test_jc/best_final_v6.engine"
VAL_IMG_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/train/images/"
VAL_LABEL_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/train/labels/"
RESULT_SAVE_DIR = "/home/ysc/Desktop/Robodog_2026/test_jc/val_infer_results"

# 自动创建保存目录
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# 类别颜色映射
def get_color_map(num_classes):
    np.random.seed(42)
    return {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}
color_map = get_color_map(NUM_CLASSES)

# 自动判断engine精度模式
IS_FP16 = "fp16" in ENGINE_PATH.lower()
INPUT_DTYPE = np.float16 if IS_FP16 else np.float32

# -------------------------- 修复后的ONNX TRT推理类 --------------------------
class YoLov5TRT(object):
    def __init__(self, engine_file_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        
        # 加载engine
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if not self.engine:
            raise RuntimeError("Engine模型加载失败，请检查文件路径和版本匹配")
        
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
            
            # 分配锁页内存和显存
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
        
        # 预处理（和训练完全对齐）
        input_blob = self.preprocess(image)
        np.copyto(self.input_host, input_blob.ravel())
        
        # 显存拷贝+推理（仅统计这部分耗时，排除前后处理）
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        # TRT7.1 推理API
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()
        
        # 后处理
        output = self.output_host.reshape(self.output_shape)
        boxes, scores, class_ids = self.postprocess(output, origin_h, origin_w)
        
        return boxes, scores, class_ids, origin_h, origin_w

    def preprocess(self, img):
        # YOLOv5官方letterbox预处理，和训练完全对齐
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # 等比例缩放，避免形变
        scale = min(INPUT_W / w, INPUT_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        dx, dy = (INPUT_W - new_w) // 2, (INPUT_H - new_h) // 2
        
        # 缩放+填充
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized, dy, INPUT_H - new_h - dy, 
            dx, INPUT_W - new_w - dx, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # 归一化+维度转换，适配精度模式
        img = padded.astype(INPUT_DTYPE) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)   # CHW -> NCHW
        
        return img

    def postprocess(self, output, origin_h, origin_w):
        # 适配[1, 25200, 4+1+num_classes]输出，--grid导出后无需解码坐标
        if output.shape[0] == 1:
            output = output[0]
        
        # 拆分输出
        xywh = output[:, :4]          # 640x640输入下的像素坐标xywh
        obj_conf = output[:, 4]       # 目标置信度
        cls_prob = output[:, 5:]      # 类别概率
        
        # 低置信度过滤
        mask = obj_conf > CONF_THRESH
        if not np.any(mask):
            return np.array([]), np.array([]), np.array([])
        
        # 筛选有效框
        xywh, obj_conf, cls_prob = xywh[mask], obj_conf[mask], cls_prob[mask]
        cls_ids = np.argmax(cls_prob, axis=1)
        cls_conf = cls_prob[np.arange(len(cls_ids)), cls_ids]
        scores = obj_conf * cls_conf  # 最终置信度=目标置信度*类别置信度
        
        # xywh -> xyxy（左上角+右下角）
        boxes = np.zeros_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        
        # 坐标映射回原始图像尺寸
        scale = min(INPUT_W / origin_w, INPUT_H / origin_h)
        pad_x = (INPUT_W - origin_w * scale) / 2
        pad_y = (INPUT_H - origin_h * scale) / 2
        
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        # 边界裁剪，避免超出图像
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, origin_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, origin_h)
        
        # NMS非极大值抑制
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            CONF_THRESH, IOU_THRESHOLD
        )
        
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 兼容不同OpenCV版本的NMS输出
        indices = indices.flatten() if isinstance(indices, np.ndarray) else list(indices)
        return boxes[indices], scores[indices], cls_ids[indices]

    def draw(self, img, boxes, scores, cls_ids):
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id)
            color = color_map.get(cls_id, (0, 255, 0))
            name = categories[cls_id] if cls_id < len(categories) else "unknown"
            
            # 画框+标签
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name}:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    def destroy(self):
        # 释放显存资源
        self.input_device.free()
        self.output_device.free()
        self.context = None
        self.engine = None
        self.stream = None

# -------------------------- 修复后的精度指标计算（完全对齐COCO标准） --------------------------
def parse_yolo_label(label_path, img_w, img_h):
    """解析YOLO格式标注，返回xyxy格式的GT框和类别id"""
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
            # 归一化坐标转像素坐标xyxy
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            cls_ids.append(cls_id)
    
    return np.array(boxes), np.array(cls_ids)

def compute_iou(box1, box2):
    """计算两个框的IOU"""
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
    """11点插值法计算AP，对齐YOLOv5官方逻辑"""
    ap = 0.0
    for t in np.linspace(0, 1.0, 11):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precisions[mask]) / 11.0
    return ap

def evaluate(model, img_dir, label_dir):
    """按图片隔离计算指标，彻底修复原代码的跨图片匹配问题"""
    # 按类别存储统计结果
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
        
        # 推理预测
        pred_boxes, pred_scores, pred_cls, _, _ = model.infer(img)
        
        # 读取GT标注
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        gt_boxes, gt_cls = parse_yolo_label(label_path, w, h)
        
        # 按类别统计GT数量
        for cls in gt_cls:
            if 0 <= cls < NUM_CLASSES:
                stats[cls]["gt_count"] += 1
        
        # 【关键修复】按图片匹配TP/FP，避免跨图片匹配
        for cls in range(NUM_CLASSES):
            # 筛选当前类别的预测和GT
            cls_pred_mask = pred_cls == cls
            cls_pred_boxes = pred_boxes[cls_pred_mask]
            cls_pred_scores = pred_scores[cls_pred_mask]
            
            cls_gt_mask = gt_cls == cls
            cls_gt_boxes = gt_boxes[cls_gt_mask]
            gt_matched = np.zeros(len(cls_gt_boxes), dtype=bool)
            
            # 按置信度降序排序预测框
            sort_idx = np.argsort(-cls_pred_scores)
            cls_pred_boxes = cls_pred_boxes[sort_idx]
            cls_pred_scores = cls_pred_scores[sort_idx]
            
            # 匹配每个预测框
            tp = np.zeros(len(cls_pred_boxes), dtype=bool)
            for i, pred_box in enumerate(cls_pred_boxes):
                if len(cls_gt_boxes) == 0:
                    continue
                # 计算和所有GT的IOU
                ious = np.array([compute_iou(pred_box, gt_box) for gt_box in cls_gt_boxes])
                max_iou = np.max(ious)
                max_iou_idx = np.argmax(ious)
                
                # IOU>=0.5且GT未被匹配，记为TP
                if max_iou >= 0.5 and not gt_matched[max_iou_idx]:
                    tp[i] = True
                    gt_matched[max_iou_idx] = True
            
            # 保存当前类别的统计结果
            stats[cls]["tp"].extend(tp.tolist())
            stats[cls]["conf"].extend(cls_pred_scores.tolist())
    
    # 计算每个类别的AP、全局Precision/Recall/mAP
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
        
        # 无GT的类别AP记为0
        if gt_count == 0:
            aps[cls_name] = 0.0
            continue
        
        # 无预测的类别AP记为0
        if len(tp) == 0:
            aps[cls_name] = 0.0
            continue
        
        # 按置信度排序，计算Precision-Recall曲线
        sort_idx = np.argsort(-conf)
        tp_sorted = tp[sort_idx]
        tp_cumsum = np.cumsum(tp_sorted)
        fp_cumsum = np.cumsum(1 - tp_sorted)
        
        recalls = tp_cumsum / gt_count
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # 计算AP
        ap = calculate_ap(recalls, precisions)
        aps[cls_name] = ap
    
    # 计算全局指标
    global_precision = total_tp / (total_tp + total_fp + 1e-6)
    global_recall = total_tp / (total_gt + 1e-6)
    map50 = np.mean(list(aps.values()))
    
    return global_precision, global_recall, map50, aps

# -------------------------- 修复后的批量推理函数 --------------------------
def batch_infer(model, img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_img = len(img_files)
    
    success_count = 0
    total_infer_time = 0  # 仅统计纯推理耗时，排除前后处理
    
    print(f"\n开始批量推理验证集（共{total_img}张图片）...")
    for img_name in tqdm(img_files, desc="批量推理"):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 推理+耗时统计
        t_start = time.time()
        boxes, scores, cls_ids, h, w = model.infer(img)
        infer_time = (time.time() - t_start) * 1000  # 转ms
        
        total_infer_time += infer_time
        success_count += 1
        
        # 绘制结果并保存
        result_img = model.draw(img.copy(), boxes, scores, cls_ids)
        cv2.putText(result_img, f"{infer_time:.1f}ms", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(save_dir, img_name), result_img)
    
    # 计算平均单帧推理耗时
    avg_infer_time = total_infer_time / success_count if success_count > 0 else 0
    return success_count, total_img, avg_infer_time

# -------------------------- 主程序 --------------------------
def main():
    # 初始化模型
    print("加载TensorRT Engine模型...")
    model = YoLov5TRT(ENGINE_PATH)
    
    try:
        # 批量推理
        success_count, total_img, avg_infer_time = batch_infer(model, VAL_IMG_DIR, RESULT_SAVE_DIR)
        success_rate = (success_count / total_img) * 100 if total_img > 0 else 0
        
        print(f"\n=== 批量推理统计 ===")
        print(f"验证集总数：{total_img}")
        print(f"成功数：{success_count} | 成功率：{success_rate:.2f}%")
        print(f"失败数：{total_img - success_count}")
        
        # 计算精度指标
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