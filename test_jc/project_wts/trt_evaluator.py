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

import ctypes
import sys


# ============ 配置参数（完全匹配你的引擎） ============
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.2
CONF_THRESH_DASHBOARD = 0.1  # dashboard单独阈值
IOU_THRESHOLD = 0.4

categories = ["A", "B", "C", "D", "GC", "RC", "dashboard", "ssi"]

# 【关键】你的引擎输出维度是38！
DETECTION_SIZE = 38
DETECTION_SIZE_CANDIDATES = [6, 13, 38]
DEBUG_DETECTION_SIZE = ("--debug-size" in sys.argv)
DEBUG_PRINT_EVERY = 15


class Colors:
    """颜色工具类"""
    def __init__(self):
        self.palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 255, 255),
        ]
    
    def __call__(self, idx: int) -> Tuple[int, int, int]:
        return self.palette[idx % len(self.palette)]


class YoLov5TRT:
    """【完全匹配你引擎的推理器】38维输出，带num头"""
    
    def __init__(self, engine_file_path: str, plugin_path: str):
        # 加载插件
        if os.path.exists(plugin_path):
            ctypes.CDLL(plugin_path)
            print(f"[INFO] 已加载插件: {plugin_path}")
        else:
            print(f"[WARN] 插件不存在: {plugin_path}")
        
        # CUDA初始化（完全沿用你的代码）
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        self.debug_frame_count = 0

        # 加载引擎
        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # 分配缓冲区
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
        """执行推理（完全沿用你的逻辑）"""
        self.cfx.push()
        try:
            input_image, _, origin_h, origin_w = self.preprocess_image(image_raw)
            np.copyto(self.host_inputs[0], input_image.ravel())

            start = time.time()
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            self.stream.synchronize()
            end = time.time()
            infer_time = (end - start) * 1000

            output = self.host_outputs[0]
            result_boxes, result_scores, result_classid = self.post_process(output, origin_h, origin_w)
            return result_boxes, result_scores, result_classid, infer_time
        finally:
            self.cfx.pop()

    def destroy(self):
        """完整释放资源"""
        self.cfx.pop()
        self.cfx.detach()
        # 释放CUDA显存
        if hasattr(self, 'cuda_inputs'):
            for mem in self.cuda_inputs:
                try:
                    mem.free()
                except:
                    pass
        if hasattr(self, 'cuda_outputs'):
            for mem in self.cuda_outputs:
                try:
                    mem.free()
                except:
                    pass
        print("[INFO] 资源已完整释放")

    def preprocess_image(self, image_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """图像预处理（完全沿用你的Letterbox逻辑）"""
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

    def xywh2xyxy(self, origin_h: int, origin_w: int, x: np.ndarray) -> np.ndarray:
        """坐标转换（完全沿用你的Letterbox还原逻辑）"""
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

    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = IOU_THRESHOLD) -> List[int]:
        """NMS"""
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

    def nms_classwise(self, boxes: np.ndarray, scores: np.ndarray, classid: np.ndarray, iou_threshold: float = IOU_THRESHOLD) -> np.ndarray:
        """类别级别NMS（完全沿用你的逻辑）"""
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

    def post_process(self, output: np.ndarray, origin_h: int, origin_w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """【核心】完全沿用你的38维输出解析逻辑"""
        if DEBUG_DETECTION_SIZE:
            self.debug_frame_count += 1
            if self.debug_frame_count % DEBUG_PRINT_EVERY == 1:
                self._debug_detection_size(output)

        num = int(output[0])
        if num <= 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        raw = output[1:]
        valid_len = (len(raw) // DETECTION_SIZE) * DETECTION_SIZE
        pred = np.reshape(raw[:valid_len], (-1, DETECTION_SIZE))
        if num > pred.shape[0]:
            num = pred.shape[0]
        pred = pred[:num, :]

        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]

        # 过滤无效类别
        valid_class = (classid >= 0) & (classid < len(categories))
        boxes = boxes[valid_class, :]
        scores = scores[valid_class]
        classid = classid[valid_class]

        # 【关键】dashboard单独阈值
        classid_int = classid.astype(np.int32)
        score_thresh = np.where(classid_int == 6, CONF_THRESH_DASHBOARD, CONF_THRESH)
        keep_score = scores > score_thresh
        boxes = boxes[keep_score, :]
        scores = scores[keep_score]
        classid = classid[keep_score]

        # 坐标转换 + 类别NMS
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        indices = self.nms_classwise(boxes, scores, classid, IOU_THRESHOLD)

        result_boxes = boxes[indices, :]
        result_scores = scores[indices]
        result_classid = classid[indices]
        return result_boxes, result_scores, result_classid

    def _debug_detection_size(self, output: np.ndarray):
        """调试输出维度（完全沿用你的逻辑）"""
        raw = output[1:]
        num_raw = int(output[0])
        print(
            "[debug_size] num_raw={}, raw_len={}, now_DETECTION_SIZE={}".format(
                num_raw, len(raw), DETECTION_SIZE
            )
        )

        for ds in DETECTION_SIZE_CANDIDATES:
            rows = len(raw) // ds
            if rows <= 0:
                print("[debug_size] ds={} rows=0 skip".format(ds))
                continue

            pred = np.reshape(raw[: rows * ds], (-1, ds))
            valid_num = min(max(num_raw, 0), pred.shape[0])
            pred = pred[:valid_num, :]

            if pred.shape[0] == 0:
                print("[debug_size] ds={} rows={} valid_num=0 skip".format(ds, rows))
                continue

            if pred.shape[1] < 6:
                print("[debug_size] ds={} columns<6 skip".format(ds))
                continue

            score = pred[:, 4]
            cid = pred[:, 5]
            score_ok = float(np.mean((score >= 0.0) & (score <= 1.0)))
            cid_int = float(np.mean(np.abs(cid - np.round(cid)) < 1e-3))
            cid_range = float(np.mean((cid >= 0.0) & (cid < float(len(categories)))))

            print(
                "[debug_size] ds={} rows={} used={} score01={:.3f} cid_int={:.3f} cid_in_range={:.3f}".format(
                    ds, rows, valid_num, score_ok, cid_int, cid_range
                )
            )
    
    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, 
                       scores: np.ndarray, classids: np.ndarray) -> np.ndarray:
        """画框（新增，用于可视化）"""
        colors = Colors()
        for box, score, cls_id in zip(boxes, scores, classids):
            x1, y1, x2, y2 = map(int, box)
            color = colors(int(cls_id))
            label = f"{categories[int(cls_id)]}:{score:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image


class mAPEvaluator:
    """mAP评估器 - 8类（无需修改）"""
    
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
                if cls_id < 0 or cls_id > 7:
                    print(f"[WARN] 标签 {label_path} 中存在无效类别 ID: {cls_id}，已跳过")
                    continue
                    
                x_center, y_center, w, h = map(float, parts[1:])
                
                x1 = (x_center - w / 2) * img_w
                y1 = (y_center - h / 2) * img_h
                x2 = (x_center + w / 2) * img_w
                y2 = (y_center + h / 2) * img_h
                
                boxes.append([x1, y1, x2, y2])
                classes.append(cls_id)
    except Exception as e:
        print(f"[WARN] 解析标签失败 {label_path}: {e}")
        return np.array([]), np.array([])
    
    return np.array(boxes), np.array(classes)


def print_report(report: Dict, avg_infer_time: float, total_images: int, 
                 success_count: int, fail_count: int, output_path: str):
    """打印格式化的评估报告"""
    
    print("\n=== 批量推理统计 ===")
    print(f"验证集总数：{total_images}")
    print(f"成功数：{success_count} | 成功率：{success_count/total_images*100:.2f}%")
    print(f"失败数：{fail_count}")
    
    print("\n=== 验证集精度指标 ===")
    overall = report["overall"]
    print(f"Precision（精确率）: {overall['Precision']:.4f}")
    print(f"Recall（召回率）: {overall['Recall']:.4f}")
    print(f"mAP@0.5: {overall['mAP@0.5']:.4f}")
    
    print("\n=== 按类别AP@0.5 ===")
    for cls_name, metrics in report["per_class"].items():
        print(f"{cls_name}: {metrics['AP']:.4f}")
    
    print("\n=== 验证集推理验证总结 ===")
    print(f"推理成功率：{success_count/total_images*100:.2f}%")
    print(f"平均单帧推理耗时：{avg_infer_time:.1f} ms")
    print(f"Precision：{overall['Precision']:.4f}")
    print(f"Recall：{overall['Recall']:.4f}")
    print(f"mAP@0.5：{overall['mAP@0.5']:.4f}")
    print(f"推理结果保存路径：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 38维引擎评估工具（完全匹配你的引擎）")
    parser.add_argument("--engine", type=str, required=True, help="TensorRT引擎文件路径")
    parser.add_argument("--data", type=str, required=True, help="验证集图片文件夹")
    parser.add_argument("--labels", type=str, required=True, help="标签文件夹")
    parser.add_argument("--output", type=str, default="results", help="输出结果文件夹")
    parser.add_argument("--conf", type=float, default=0.2, help="置信度阈值（dashboard除外）")
    parser.add_argument("--conf-dashboard", type=float, default=0.1, help="dashboard置信度阈值")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU阈值")
    parser.add_argument("--save-img", action="store_true", help="保存带框图片")
    parser.add_argument("--plugin", type=str, required=True, help="插件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式(只处理前5张图)")
    parser.add_argument("--debug-size", action="store_true", help="启用输出维度调试")
    
    args = parser.parse_args()
    
    # 更新全局阈值
    global CONF_THRESH, CONF_THRESH_DASHBOARD, IOU_THRESHOLD
    CONF_THRESH = args.conf
    CONF_THRESH_DASHBOARD = args.conf_dashboard
    IOU_THRESHOLD = args.iou
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    if args.save_img:
        os.makedirs(os.path.join(args.output, "visualizations"), exist_ok=True)
    
    # 初始化评估器
    trt_evaluator = YoLov5TRT(args.engine, args.plugin)
    map_evaluator = mAPEvaluator(categories)
    
    # 获取所有图片
    image_paths = list(Path(args.data).glob("*.jpg")) + list(Path(args.data).glob("*.png"))
    total_images = len(image_paths)
    
    if total_images == 0:
        print(f"[错误] 未找到图片: {args.data}")
        return
    
    # 调试模式: 只处理前5张
    if args.debug:
        image_paths = image_paths[:5]
        print(f"[调试模式] 只处理前5张图片")
    
    # 批量推理
    success_count = 0
    fail_count = 0
    total_pred_boxes = 0
    total_gt_boxes = 0
    inference_times = []
    
    print(f"\n开始批量推理（共{len(image_paths)}张图片）...")
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="推理进度", ncols=80)):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                fail_count += 1
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 推理
            boxes, scores, classids, infer_time = trt_evaluator.infer(img)
            inference_times.append(infer_time)
            
            # 读取标签
            label_path = os.path.join(args.labels, img_path.stem + ".txt")
            gt_boxes, gt_classes = parse_label_file(label_path, img_w, img_h)
            
            # 统计
            total_pred_boxes += len(boxes)
            total_gt_boxes += len(gt_boxes)
            
            # 调试信息
            if args.debug:
                print(f"\n[调试] 图片 {img_path.name}:")
                print(f"  预测框: {len(boxes)} 个")
                print(f"  真实框: {len(gt_boxes)} 个 (来自 {label_path})")
                if len(boxes) > 0:
                    print(f"  预测类别: {[categories[int(c)] for c in classids[:3]]}...")
                    print(f"  预测分数: {scores[:3]}")
                if len(gt_boxes) > 0:
                    print(f"  真实类别: {[categories[int(c)] for c in gt_classes[:3]]}...")
            
            # 添加到评估器
            map_evaluator.add_image(boxes, scores, classids, gt_boxes, gt_classes)
            
            # 保存可视化
            if args.save_img:
                vis_img = trt_evaluator.draw_detections(img.copy(), boxes, scores, classids)
                # 也画真实框 (绿色)
                for gt_box in gt_boxes:
                    x1, y1, x2, y2 = map(int, gt_box)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info_text = f"{infer_time:.1f}ms"
                cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                save_path = os.path.join(args.output, "visualizations", img_path.name)
                cv2.imwrite(save_path, vis_img)
            
            success_count += 1
            
        except Exception as e:
            print(f"[错误] 处理 {img_path.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue
    
    # 打印统计信息
    print(f"\n[统计] 总预测框数: {total_pred_boxes}, 总真实框数: {total_gt_boxes}")
    
    # 计算精度指标
    print(f"\n开始计算精度指标（共{success_count}张图片）...")
    report = map_evaluator.evaluate()
    
    # 计算平均推理时间
    avg_infer_time = np.mean(inference_times) if inference_times else 0
    
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