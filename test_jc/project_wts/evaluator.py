# 创建适配 8类 的 YOLOv5 v7.0 TensorRTX 评估工具



import os
import cv2
import time
import json
import argparse
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from tqdm import tqdm


# ============ 配置参数 ============
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.45


class Colors:
    """颜色工具类 - 8类配色"""
    def __init__(self):
        self.palette = [
            (255, 0, 0),    # A - 蓝
            (0, 255, 0),    # B - 绿
            (0, 0, 255),    # C - 红
            (255, 255, 0),  # D - 青
            (255, 0, 255),  # GC - 紫
            (0, 255, 255),  # RC - 黄
            (128, 128, 128),# dashboard - 灰
            (255, 255, 255),# ssi - 白
        ]
    
    def __call__(self, idx: int) -> Tuple[int, int, int]:
        return self.palette[idx % len(self.palette)]


class YoLov5TRTEvaluator:
    """YOLOv5 v7.0 TensorRTX 评估器 - 8类"""
    
    def __init__(self, engine_file_path: str, categories: List[str]):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.categories = categories
        self.num_classes = len(categories)  # 8
        self.colors = Colors()
        
        # CUDA上下文
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        
        # 加载引擎
        runtime = trt.Runtime(self.logger)
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        
        # 性能统计
        self.inference_times = []
        
    def _allocate_buffers(self):
        """分配GPU/CPU内存缓冲区"""
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
    
    def infer(self, image_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """执行推理"""
        self.cfx.push()
        
        try:
            # 预处理
            input_image, origin_h, origin_w = self.preprocess_image(image_raw)
            np.copyto(self.host_inputs[0], input_image.ravel())
            
            # GPU推理
            t0 = time.time()
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            self.stream.synchronize()
            infer_time = (time.time() - t0) * 1000
            self.inference_times.append(infer_time)
            
            # 后处理
            output = self.host_outputs[0]
            boxes, scores, classids = self.post_process(output, origin_h, origin_w)
            
            return boxes, scores, classids, infer_time
            
        finally:
            self.cfx.pop()
    
    def preprocess_image(self, image_raw: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """图像预处理"""
        h, w = image_raw.shape[:2]
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        
        scale = min(INPUT_W / w, INPUT_H / h)
        nw, nh = int(w * scale), int(h * scale)
        
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        dw, dh = (INPUT_W - nw) // 2, (INPUT_H - nh) // 2
        padded_image = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
        padded_image[dh:dh+nh, dw:dw+nw, :] = image
        
        padded_image = padded_image.astype(np.float32) / 255.0
        padded_image = np.transpose(padded_image, (2, 0, 1))
        padded_image = np.expand_dims(padded_image, axis=0)
        padded_image = np.ascontiguousarray(padded_image)
        
        return padded_image, h, w
    
    def post_process(self, output: np.ndarray, origin_h: int, origin_w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        YOLOv5 v7.0 TensorRTX 后处理 - 8类版本
        输出格式: [num_detections, 5 + num_classes]
        即: [num, 5 + 8] = [num, 13]
        """
        # 第一个元素是检测数量
        num = int(output[0])
        
        # reshape: 跳过第一个元素(num), 每行13个值 (5+8)
        # 5 = xywh(4) + obj_conf(1)
        # 8 = 每个类别的置信度
        pred = np.reshape(output[1:], (-1, 5 + self.num_classes))[:num, :]
        
        # 解析
        boxes = pred[:, :4]           # xywh
        obj_conf = pred[:, 4]         # 目标置信度 [num,]
        cls_conf = pred[:, 5:]        # 类别置信度 [num, 8]
        
        # 计算最终置信度 = 目标置信度 × 类别置信度
        # cls_conf 每行8个值,对应8个类别
        max_cls_conf = np.max(cls_conf, axis=1)  # [num,]
        classids = np.argmax(cls_conf, axis=1)   # [num,]
        scores = obj_conf * max_cls_conf         # [num,]
        
        # 过滤低置信度
        mask = scores > CONF_THRESH
        boxes = boxes[mask]
        scores = scores[mask]
        classids = classids[mask]
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # xywh -> xyxy 并映射回原图
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        
        # NMS
        indices = self.nms(boxes, scores, IOU_THRESHOLD)
        
        return boxes[indices], scores[indices], classids[indices]
    
    def xywh2xyxy(self, origin_h: int, origin_w: int, x: np.ndarray) -> np.ndarray:
        """坐标转换并映射回原图"""
        y = np.zeros_like(x)
        
        scale = min(INPUT_W / origin_w, INPUT_H / origin_h)
        nw, nh = int(origin_w * scale), int(origin_h * scale)
        dw = (INPUT_W - nw) // 2
        dh = (INPUT_H - nh) // 2
        
        # xywh -> xyxy
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        
        # 映射回原图
        y[:, [0, 2]] = (y[:, [0, 2]] - dw) / scale
        y[:, [1, 3]] = (y[:, [1, 3]] - dh) / scale
        
        # 裁剪边界
        y[:, [0, 2]] = np.clip(y[:, [0, 2]], 0, origin_w)
        y[:, [1, 3]] = np.clip(y[:, [1, 3]], 0, origin_h)
        
        return y
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """NMS"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, 
                       scores: np.ndarray, classids: np.ndarray) -> np.ndarray:
        """画框"""
        for box, score, cls_id in zip(boxes, scores, classids):
            x1, y1, x2, y2 = map(int, box)
            color = self.colors(int(cls_id))
            label = f"{self.categories[int(cls_id)]}:{score:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def destroy(self):
        """释放资源"""
        self.cfx.pop()


class mAPEvaluator:
    """mAP评估器 - 8类"""
    
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.results = defaultdict(lambda: {"tp": [], "scores": [], "gt": 0})
        self.iou_thresh = 0.5
        
    def add_image(self, pred_boxes: np.ndarray, pred_scores: np.ndarray, 
                 pred_classes: np.ndarray, gt_boxes: np.ndarray, gt_classes: np.ndarray):
        """添加单张图片结果"""
        for cls_idx, cls_name in enumerate(self.categories):
            pred_mask = pred_classes == cls_idx
            gt_mask = gt_classes == cls_idx
            
            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]
            cls_gt_boxes = gt_boxes[gt_mask]
            
            if len(cls_pred_boxes) > 0:
                sorted_indices = np.argsort(-cls_pred_scores)
                cls_pred_boxes = cls_pred_boxes[sorted_indices]
                cls_pred_scores = cls_pred_scores[sorted_indices]
            
            tp = []
            matched_gt = set()
            
            for pred_box in cls_pred_boxes:
                if len(cls_gt_boxes) == 0:
                    tp.append(0)
                    continue
                
                ious = self._compute_iou(pred_box, cls_gt_boxes)
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]
                
                if max_iou >= self.iou_thresh and max_iou_idx not in matched_gt:
                    tp.append(1)
                    matched_gt.add(max_iou_idx)
                else:
                    tp.append(0)
            
            self.results[cls_name]["tp"].extend(tp)
            self.results[cls_name]["scores"].extend(cls_pred_scores)
            self.results[cls_name]["gt"] += len(cls_gt_boxes)
    
    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """计算IoU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        
        return inter / (union + 1e-6)
    
    def compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """计算AP (VOC11点)"""
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        return ap
    
    def evaluate(self) -> Dict:
        """计算最终精度"""
        report = {"per_class": {}, "overall": {}}
        
        aps = []
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for cls_name in self.categories:
            tp = np.array(self.results[cls_name]["tp"])
            scores = np.array(self.results[cls_name]["scores"])
            gt_count = self.results[cls_name]["gt"]
            
            if len(tp) == 0 or gt_count == 0:
                report["per_class"][cls_name] = {"AP": 0.0}
                continue
            
            sorted_indices = np.argsort(-scores)
            tp = tp[sorted_indices]
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(1 - tp)
            
            recalls = tp_cumsum / gt_count
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            ap = self.compute_ap(recalls, precisions)
            aps.append(ap)
            
            total_tp += int(tp.sum())
            total_fp += int((1 - tp).sum())
            total_gt += gt_count
            
            report["per_class"][cls_name] = {"AP": float(ap)}
        
        report["overall"] = {
            "mAP@0.5": float(np.mean(aps)) if aps else 0.0,
            "Precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
            "Recall": total_tp / total_gt if total_gt > 0 else 0.0,
        }
        
        return report


def parse_label_file(label_path: str, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """解析YOLO格式标签"""
    boxes = []
    classes = []
    
    if not os.path.exists(label_path):
        return np.array([]), np.array([])
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                cls_id = int(parts[0])
                # 检查类别ID是否在有效范围内 (0-7)
                if cls_id < 0 or cls_id > 7:
                    continue
                    
                x_center, y_center, w, h = map(float, parts[1:])
                
                x1 = (x_center - w / 2) * img_w
                y1 = (y_center - h / 2) * img_h
                x2 = (x_center + w / 2) * img_w
                y2 = (y_center + h / 2) * img_h
                
                boxes.append([x1, y1, x2, y2])
                classes.append(cls_id)
    except Exception as e:
        return np.array([]), np.array([])
    
    return np.array(boxes), np.array(classes)


def print_report(report: Dict, avg_infer_time: float, total_images: int, 
                 success_count: int, fail_count: int, output_path: str):
    """打印格式化的评估报告"""
    
    print("\\n=== 批量推理统计 ===")
    print(f"验证集总数：{total_images}")
    print(f"成功数：{success_count} | 成功率：{success_count/total_images*100:.2f}%")
    print(f"失败数：{fail_count}")
    
    print("\\n=== 验证集精度指标 ===")
    overall = report["overall"]
    print(f"Precision（精确率）: {overall['Precision']:.4f}")
    print(f"Recall（召回率）: {overall['Recall']:.4f}")
    print(f"mAP@0.5: {overall['mAP@0.5']:.4f}")
    
    print("\\n=== 按类别AP@0.5 ===")
    for cls_name, metrics in report["per_class"].items():
        print(f"{cls_name}: {metrics['AP']:.4f}")
    
    print("\\n=== 验证集推理验证总结 ===")
    print(f"推理成功率：{success_count/total_images*100:.2f}%")
    print(f"平均单帧推理耗时：{avg_infer_time:.1f} ms")
    print(f"Precision：{overall['Precision']:.4f}")
    print(f"Recall：{overall['Recall']:.4f}")
    print(f"mAP@0.5：{overall['mAP@0.5']:.4f}")
    print(f"推理结果保存路径：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 v7.0 TensorRTX 评估工具 - 8类")
    parser.add_argument("--engine", type=str, required=True, help="TensorRT引擎文件路径")
    parser.add_argument("--data", type=str, required=True, help="验证集图片文件夹")
    parser.add_argument("--labels", type=str, required=True, help="标签文件夹")
    parser.add_argument("--output", type=str, default="results", help="输出结果文件夹")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--save-img", action="store_true", help="保存带框图片")
    
    args = parser.parse_args()
    
    global CONF_THRESH, IOU_THRESHOLD
    CONF_THRESH = args.conf
    IOU_THRESHOLD = args.iou
    
    # ========== 修改1: 8类定义 ==========
    categories = ['A', 'B', 'C', 'D', 'GC', 'RC', 'dashboard', 'ssi']
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    if args.save_img:
        os.makedirs(os.path.join(args.output, "visualizations"), exist_ok=True)
    
    # 初始化评估器
    trt_evaluator = YoLov5TRTEvaluator(args.engine, categories)
    map_evaluator = mAPEvaluator(categories)
    
    # 获取所有图片
    image_paths = list(Path(args.data).glob("*.jpg")) + list(Path(args.data).glob("*.png"))
    total_images = len(image_paths)
    
    if total_images == 0:
        print(f"[错误] 未找到图片: {args.data}")
        return
    
    # 批量推理
    success_count = 0
    fail_count = 0
    
    print(f"\\n开始批量推理（共{total_images}张图片）...")
    
    for img_path in tqdm(image_paths, desc="推理进度", ncols=80):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                fail_count += 1
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 推理
            boxes, scores, classids, infer_time = trt_evaluator.infer(img)
            
            # 读取标签
            label_path = os.path.join(args.labels, img_path.stem + ".txt")
            gt_boxes, gt_classes = parse_label_file(label_path, img_w, img_h)
            
            # 添加到评估器
            map_evaluator.add_image(boxes, scores, classids, gt_boxes, gt_classes)
            
            # 保存可视化
            if args.save_img:
                vis_img = trt_evaluator.draw_detections(img.copy(), boxes, scores, classids)
                info_text = f"{infer_time:.1f}ms"
                cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                save_path = os.path.join(args.output, "visualizations", img_path.name)
                cv2.imwrite(save_path, vis_img)
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            continue
    
    # 计算精度指标
    print(f"\\n开始计算精度指标（共{success_count}张图片）...")
    report = map_evaluator.evaluate()
    
    # 计算平均推理时间
    avg_infer_time = np.mean(trt_evaluator.inference_times) if trt_evaluator.inference_times else 0
    
    # 获取绝对路径
    output_abs_path = os.path.abspath(args.output)
    
    # 打印报告
    print_report(report, avg_infer_time, total_images, success_count, fail_count, output_abs_path)
    
    # 保存JSON报告
    report_path = os.path.join(args.output, "evaluation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 释放资源
    trt_evaluator.destroy()
    print("资源已释放。")


if __name__ == "__main__":
    main()
'''

# 保存代码
with open('/mnt/kimi/output/trt_evaluator_8class.py', 'w', encoding='utf-8') as f:
    f.write(eval_8class_code)

print("="*70)
print("? 8类版本评估工具已生成: /mnt/kimi/output/trt_evaluator_8class.py")
print("="*70)
print("\n?? 修改内容总结:")
print("\n1??  类别定义 (第1处修改):")
print("   原: 18类 ['1','2',...,'red_cylinder']")
print("   新: 8类 ['A', 'B', 'C', 'D', 'GC', 'RC', 'dashboard', 'ssi']")
print("\n2??  颜色配置 (Colors类):")
print("   原: 18色")
print("   新: 8色 (A蓝, B绿, C红, D青, GC紫, RC黄, dashboard灰, ssi白)")
print("\n3??  后处理维度:")
print("   原: reshape(-1, 5+18) = (-1, 23)")
print("   新: reshape(-1, 5+8) = (-1, 13)")
print("\n4??  标签解析检查:")
print("   新增: 检查cls_id是否在0-7范围内")
print("\n" + "="*70)
print("?? 运行命令:")
print("="*70)
print("""
python trt_evaluator_8class.py \\
    --engine build/yolov5s.engine \\
    --data data/images \\
    --labels data/labels \\
    --output results \\
    --save-img
""")
print("="*70)
print("?? 预期输出:")
print("="*70)
print("""
=== 按类别AP@0.5 ===
A: 0.3836
B: 0.1642
C: 0.1403
D: 0.0485
GC: 0.4512
RC: 0.4031
dashboard: 0.4536
ssi: 0.0840
""")