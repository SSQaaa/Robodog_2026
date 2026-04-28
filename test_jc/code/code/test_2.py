# -*- coding: utf-8 -*-
import time
import cv2
import pycuda.autoinit
import numpy as np
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import os
import glob
from collections import defaultdict
from yolov5trt import YoLov5TRT
from tqdm import tqdm

# -------------------------- 原代码参数复用 + 新增路径参数 --------------------------
INPUT_W = 1024
INPUT_H = 576
CONF_THRESH = 0.2
IOU_THRESHOLD = 0.4
# 类别映射（与原detect_trt.py完全一致）
categories = ['1', '2', '3', '4', '5', '6', 'red_barrel','yellow_barrel', 
              'blue_barrel', 'orange_barrel', 'red_ball', 'yellow_ball',
              'blue_ball' ,'orange_ball','dashboard','ssi']
# 类别ID转名称（反向映射）
id2cat = {i: name for i, name in enumerate(categories)}
# 数据集路径（需替换为你的文件夹根路径）
DATA_ROOT = "/home/ysc/Desktop/Robodog_2026/test_jc/new_doglabels.v6i.yolov5pytorch/test/"  # 该文件夹下有images和labels子文件夹
IMG_PATH = os.path.join(DATA_ROOT, "images")
LABEL_PATH = os.path.join(DATA_ROOT, "labels")

# -------------------------- 原代码Detect_image类改造 + 评估功能 --------------------------
class DetectImageTRT:
    def __init__(self):
        # 加载TensorRT插件和引擎（原代码核心逻辑）
        PLUGIN_LIBRARY = "build/libmyplugins.so"
        ctypes.CDLL(PLUGIN_LIBRARY)
        engine_file_path = "build/yolov5s.engine"
        self.yolov5_wrapper = YoLov5TRT(engine_file_path)
        # 评估用变量
        self.infer_times = []  # 存储单张图片推理时长
        self.all_preds = []    # 所有预测框 (img_name, cls_id, x1, y1, x2, y2, conf)
        self.all_gts = []      # 所有标注框 (img_name, cls_id, x1, y1, x2, y2)

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # 原代码的画框函数，完全复用
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
        color = color or [np.random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, 
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def draw_boxes(self, image_raw, result_boxes, result_scores, result_classid):
        # 原代码的后处理筛选函数，完全复用（选面积最大目标）
        index_dict = {'number': [], 'barrel': [], 'board': [],'ssi': [],'ball':[]}
        areas = []
        result_new_boxes = []
        result_new_classid = []
        for idx, val in enumerate(result_classid):
            class_name = categories[int(val)]
            box = result_boxes[idx]
            area = abs(box[2] - box[0]) * abs(box[3] - box[1])
            areas.append(area)
            if class_name in ['1', '2', '3', '4', '5', '6']:
                index_dict['number'].append(idx)
            elif class_name in ['blue_barrel', 'orange_barrel', 'red_barrel', 'yellow_barrel']:
                index_dict['barrel'].append(idx)
            elif class_name == 'dashboard':
                index_dict['board'].append(idx)
            elif class_name == 'ssi':
                index_dict['ssi'].append(idx)
            elif class_name in ['red_ball', 'yellow_ball','blue_ball' ,'orange_ball']:
                index_dict['ball'].append(idx)
        # 筛选面积最大的目标
        for group in ['number', 'barrel','board','ball']:
            if index_dict[group]:
                max_idx = max(index_dict[group], key=lambda x: areas[x])
                box = result_boxes[max_idx]
                result_new_classid.append(result_classid[max_idx])
                result_new_boxes.append(result_boxes[max_idx])
        # 匹配dashboard和ssi
        for j in range(len(result_new_classid)):
            if int(result_new_classid[j]) == 14:
                for i in range(len(index_dict['ssi'])):
                    s = index_dict['ssi'][i]
                    middle_x = (result_boxes[s][0]+result_boxes[s][2])//2
                    if middle_x > min(result_new_boxes[j][0],result_new_boxes[j][2]) and \
                       middle_x < max(result_new_boxes[j][0],result_new_boxes[j][2]):
                        result_new_classid.append(result_classid[s])
                        result_new_boxes.append(result_boxes[s])
        return image_raw, result_new_boxes, result_new_classid

    def infer_single(self, frame, img_name):
        """单张图片推理：含时长统计、预测框存储"""
        # 记录推理开始时间（仅统计TensorRT模型推理耗时，与原代码use_time一致）
        start_time = time.time()
        # TensorRT核心推理（原代码逻辑）
        _, result_boxes, result_scores, result_classid, trt_use_time = self.yolov5_wrapper.infer(frame)
        # 统计推理时长（两种方式可选：trt_use_time/自定义计算）
        infer_time = (time.time() - start_time) * 1000  # 转毫秒
        self.infer_times.append(infer_time)
        
        # 后处理筛选检测框
        _, pred_boxes, pred_cls_ids = self.draw_boxes(frame.copy(), result_boxes, result_scores, result_classid)
        
        # 存储预测框（格式：img_name, cls_id, x1, y1, x2, y2, conf）
        for box, cls_id in zip(pred_boxes, pred_cls_ids):
            x1, y1, x2, y2 = box
            # 取原推理的置信度（匹配筛选后的框）
            conf = result_scores[np.where((result_boxes == box).all(axis=1))[0][0]]
            self.all_preds.append([img_name, int(cls_id), x1, y1, x2, y2, conf])

    def load_gt(self):
        """加载标注框：适配labels文件夹的txt标注（YOLO格式转像素坐标）"""
        label_files = glob.glob(os.path.join(LABEL_PATH, "*.txt"))
        for lbl_file in label_files:
            img_name = os.path.basename(lbl_file).replace(".txt", ".jpg")  # 需保证图片/标注后缀一致
            img_path = os.path.join(IMG_PATH, img_name)
            if not os.path.exists(img_path):
                continue
            # 获取图片尺寸（用于YOLO格式转像素）
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            # 读取标注txt
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) < 5:
                    continue
                cls_id = int(line[0])
                # YOLO格式：x_center, y_center, w, h（归一化）→ 转像素x1,y1,x2,y2
                xc, yc, bw, bh = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                self.all_gts.append([img_name, cls_id, x1, y1, x2, y2])

    def iou(self, box1, box2):
        """计算两个框的IOU：box=(x1,y1,x2,y2)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def calculate_metrics(self, iou_thresh=0.5):
        """计算模型评估指标：精确率(P)、召回率(R)、mAP@0.5"""
        metrics = defaultdict(lambda: {'tp':0, 'fp':0, 'fn':0})
        # 按图片分组标注框和预测框
        img2gts = defaultdict(list)
        for gt in self.all_gts:
            img2gts[gt[0]].append(gt[1:])  # [cls_id, x1,y1,x2,y2]
        img2preds = defaultdict(list)
        for pred in self.all_preds:
            img2preds[pred[0]].append(pred[1:])  # [cls_id, x1,y1,x2,y2, conf]

        # 逐图计算TP/FP/FN
        for img_name in img2gts:
            gts = img2gts[img_name]
            preds = img2preds.get(img_name, [])
            gt_matched = [False] * len(gts)  # 标注框是否被匹配

            # 预测框按置信度降序排列
            preds_sorted = sorted(preds, key=lambda x: x[-1], reverse=True)
            for pred in preds_sorted:
                p_cls, p_x1, p_y1, p_x2, p_y2, p_conf = pred
                best_iou = 0
                best_gt_idx = -1
                # 匹配同类别标注框
                for gt_idx, gt in enumerate(gts):
                    g_cls, g_x1, g_y1, g_x2, g_y2 = gt
                    if g_cls == p_cls and not gt_matched[gt_idx]:
                        current_iou = self.iou([p_x1,p_y1,p_x2,p_y2], [g_x1,g_y1,g_x2,g_y2])
                        if current_iou > best_iou and current_iou >= iou_thresh:
                            best_iou = current_iou
                            best_gt_idx = gt_idx
                # 判定TP/FP
                if best_gt_idx != -1:
                    metrics[p_cls]['tp'] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    metrics[p_cls]['fp'] += 1
            # 计算FN（未被匹配的标注框）
            for gt_idx, matched in enumerate(gt_matched):
                if not matched:
                    g_cls = gts[gt_idx][0]
                    metrics[g_cls]['fn'] += 1

        # 计算各类别P/R，以及平均P/R/mAP
        avg_p, avg_r, avg_ap = 0.0, 0.0, 0.0
        class_metrics = {}
        for cls_id in metrics:
            tp = metrics[cls_id]['tp']
            fp = metrics[cls_id]['fp']
            fn = metrics[cls_id]['fn']
            # 精确率：TP/(TP+FP)，避免除0
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # 召回率：TP/(TP+FN)，避免除0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # AP@0.5 简化计算（单IOU阈值下AP=P*R）
            ap = p * r
            class_metrics[id2cat[cls_id]] = {'P': round(p, 4), 'R': round(r, 4), 'AP': round(ap, 4)}
            avg_p += p
            avg_r += r
            avg_ap += ap

        # 计算mAP（各类别AP的平均值）
        n_classes = len(metrics)
        if n_classes > 0:
            avg_p /= n_classes
            avg_r /= n_classes
            avg_ap /= n_classes
        overall = {'Avg_P': round(avg_p, 4), 'Avg_R': round(avg_r, 4), 'mAP@0.5': round(avg_ap, 4)}
        return class_metrics, overall

    def run_evaluation(self):
        """主执行函数：加载标注→批量推理→计算时长+指标"""
        # 1. 加载所有标注框
        print("开始加载标注文件...")
        self.load_gt()
        if len(self.all_gts) == 0:
            print("未加载到标注框，请检查labels文件夹路径和格式！")
            return
        print(f"成功加载 {len(self.all_gts)} 个标注框")

        # 2. 批量推理images文件夹下的图片
        print("\n开始TensorRT加速批量推理...")
        img_files = glob.glob(os.path.join(IMG_PATH, "*.jpg")) + glob.glob(os.path.join(IMG_PATH, "*.png"))
        for img_path in tqdm(img_files, total=len(img_files)):
            img_name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            # 单张推理
            self.infer_single(frame, img_name)

        # 3. 计算平均推理时长
        if len(self.infer_times) > 0:
            avg_time = np.mean(self.infer_times)
            max_time = np.max(self.infer_times)
            min_time = np.min(self.infer_times)
            print(f"\n===== 推理时长统计（单位：毫秒） =====")
            print(f"平均推理时长：{avg_time:.2f} ms")
            print(f"最大推理时长：{max_time:.2f} ms")
            print(f"最小推理时长：{min_time:.2f} ms")
            print(f"推理图片总数：{len(self.infer_times)}")

        # 4. 计算模型评估指标
        print(f"\n===== 模型精度评估（IOU阈值：0.5） =====")
        class_metrics, overall_metrics = self.calculate_metrics()
        # 打印各类别指标
        for cls_name, res in class_metrics.items():
            print(f"{cls_name}: P={res['P']}, R={res['R']}, AP={res['AP']}")
        # 打印整体指标
        print(f"整体：{overall_metrics}")

        return overall_metrics, avg_time

    def destroy(self):
        # 释放TensorRT引擎资源（原代码逻辑）
        self.yolov5_wrapper.destroy()

# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    # 实例化TensorRT检测评估类
    detector = DetectImageTRT()
    try:
        # 运行批量推理和评估
        detector.run_evaluation()
    finally:
        # 释放资源
        detector.destroy()