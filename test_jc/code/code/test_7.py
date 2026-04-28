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
import random

# -------------------------- 16类配置（匹配标注文件） --------------------------
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.2
IOU_THRESHOLD = 0.4

# 16类（匹配你的标注文件）
categories = ['1', '2', '3', '4', '5', '6', 
              'red_barrel', 'yellow_barrel', 'blue_barrel', 'orange_barrel',
              'red_ball', 'yellow_ball', 'blue_ball', 'orange_ball',
              'dashboard', 'ssi']

id2cat = {i: name for i, name in enumerate(categories)}
VALID_CLS_IDS = set(range(16))  # 0-15

# 类别分组（匹配16类定义）
NUMBER_CLASSES = {'1', '2', '3', '4', '5', '6'}
BARREL_CLASSES = {'red_barrel', 'yellow_barrel', 'blue_barrel', 'orange_barrel'}  # 单r
BOARD_CLASSES = {'dashboard'}  # 小写
BALL_CLASSES = {'red_ball', 'yellow_ball', 'blue_ball', 'orange_ball'}
SSI_CLASSES = {'ssi'}

DATA_ROOT = "/home/ysc/Desktop/Robodog_2026/test_jc/new_doglabels.v6i.yolov5pytorch/test/"
IMG_PATH = os.path.join(DATA_ROOT, "images")
LABEL_PATH = os.path.join(DATA_ROOT, "labels")

# -------------------------- 检测评估类 --------------------------
class DetectImageTRT:
    def __init__(self):
        PLUGIN_LIBRARY = "build/libmyplugins.so"
        ctypes.CDLL(PLUGIN_LIBRARY)
        engine_file_path = "/home/ysc/Desktop/Robodog_2026/test_jc/code/code/build/yolov5s.engine"
        self.yolov5_wrapper = YoLov5TRT(engine_file_path)
        
        self.infer_times = []
        self.all_preds = []
        self.all_gts = []
        self.warmup_engine()

    def warmup_engine(self):
        print("TensorRT引擎预热中...")
        dummy_img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        self.yolov5_wrapper.infer(dummy_img)
        print("引擎预热完成！")

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
        color = color or [random.randint(0, 255) for _ in range(3)]
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
        """16类后处理：包含ssi"""
        index_dict = {'number': [], 'barrel': [], 'board': [], 'ball': [], 'ssi': []}
        areas = []
        result_new_boxes = []
        result_new_classid = []
        
        for idx, val in enumerate(result_classid):
            class_name = categories[int(val)]
            box = result_boxes[idx]
            area = abs(box[2] - box[0]) * abs(box[3] - box[1])
            areas.append(area)
            
            if class_name in NUMBER_CLASSES:
                index_dict['number'].append(idx)
            elif class_name in BARREL_CLASSES:
                index_dict['barrel'].append(idx)
            elif class_name in BOARD_CLASSES:
                index_dict['board'].append(idx)
            elif class_name in BALL_CLASSES:
                index_dict['ball'].append(idx)
            elif class_name in SSI_CLASSES:
                index_dict['ssi'].append(idx)
        
        # 每类保留最大面积
        for group in ['number', 'barrel', 'board', 'ball', 'ssi']:
            if index_dict[group]:
                max_idx = max(index_dict[group], key=lambda x: areas[x])
                box = result_boxes[max_idx]
                self.plot_one_box(
                    box, image_raw,
                    label="{}:{:.2f}".format(categories[int(result_classid[max_idx])], result_scores[max_idx]),
                )
                result_new_classid.append(result_classid[max_idx])
                result_new_boxes.append(result_boxes[max_idx])
        
        # SSI与Dashboard配对逻辑（如果检测到dashboard且ssi在其范围内）
        dashboard_idx = None
        ssi_boxes = []
        for i, cls_id in enumerate(result_new_classid):
            if categories[int(cls_id)] == 'dashboard':
                dashboard_idx = i
                dashboard_box = result_new_boxes[i]
        
        # 检查是否有ssi在dashboard范围内
        if dashboard_idx is not None:
            for idx in index_dict['ssi']:
                ssi_box = result_boxes[idx]
                middle_x = (ssi_box[0] + ssi_box[2]) // 2
                if (middle_x > min(dashboard_box[0], dashboard_box[2]) and 
                    middle_x < max(dashboard_box[0], dashboard_box[2])):
                    # 添加ssi到结果
                    self.plot_one_box(
                        ssi_box, image_raw,
                        label="{}:{:.2f}".format(categories[int(result_classid[idx])], result_scores[idx]),
                    )
                    result_new_classid.append(result_classid[idx])
                    result_new_boxes.append(ssi_box)
        
        return image_raw, result_new_boxes, result_new_classid

    def infer_single(self, frame, img_name):
        img, result_boxes, result_scores, result_classid, trt_use_time = self.yolov5_wrapper.infer(frame)
        
        if trt_use_time > 0:
            self.infer_times.append(trt_use_time * 1000)
        
        _, pred_boxes, pred_cls_ids = self.draw_boxes(img, result_boxes, result_scores, result_classid)
        
        for box, cls_id in zip(pred_boxes, pred_cls_ids):
            cls_id = int(cls_id)
            if cls_id not in VALID_CLS_IDS:
                print(f"警告：检测到无效类别ID {cls_id}，跳过")
                continue
            x1, y1, x2, y2 = box
            conf = 0.0
            match_idx = np.where((result_boxes == box).all(axis=1))[0]
            if len(match_idx) > 0:
                conf = result_scores[match_idx[0]]
            self.all_preds.append([img_name, cls_id, x1, y1, x2, y2, conf])

    def load_gt(self):
        label_files = glob.glob(os.path.join(LABEL_PATH, "*.txt"))
        if not label_files:
            print(f"未找到标注文件！请检查路径：{LABEL_PATH}")
            return
        
        for lbl_file in label_files:
            img_name = os.path.basename(lbl_file).replace(".txt", ".jpg")
            if not os.path.exists(os.path.join(IMG_PATH, img_name)):
                img_name = img_name.replace(".jpg", ".png")
            
            img_path = os.path.join(IMG_PATH, img_name)
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    continue
                
                if cls_id not in VALID_CLS_IDS:
                    print(f"无效类别ID：{cls_id}（仅支持0-15），跳过")
                    continue
                
                try:
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                
                x1 = max(0, min(w, (xc - bw/2) * w))
                y1 = max(0, min(h, (yc - bh/2) * h))
                x2 = max(0, min(w, (xc + bw/2) * w))
                y2 = max(0, min(h, (yc + bh/2) * h))
                
                self.all_gts.append([img_name, cls_id, x1, y1, x2, y2])
        
        print(f"成功加载 {len(self.all_gts)} 个有效标注框")

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
        area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0

    def calculate_metrics(self, iou_thresh=0.5):
        metrics = defaultdict(lambda: {'tp':0, 'fp':0, 'fn':0})
        
        img2gts = defaultdict(list)
        for gt in self.all_gts:
            img2gts[gt[0]].append(gt[1:])
        
        img2preds = defaultdict(list)
        for pred in self.all_preds:
            img2preds[pred[0]].append(pred[1:])

        for img_name in img2gts:
            gts = img2gts[img_name]
            preds = img2preds.get(img_name, [])
            gt_matched = [False] * len(gts)

            preds_sorted = sorted(preds, key=lambda x: x[-1], reverse=True)
            
            for pred in preds_sorted:
                p_cls, p_x1, p_y1, p_x2, p_y2, p_conf = pred
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    g_cls, g_x1, g_y1, g_x2, g_y2 = gt
                    if g_cls == p_cls and not gt_matched[gt_idx]:
                        current_iou = self.iou([p_x1,p_y1,p_x2,p_y2], [g_x1,g_y1,g_x2,g_y2])
                        if current_iou > best_iou and current_iou >= iou_thresh:
                            best_iou = current_iou
                            best_gt_idx = gt_idx
                
                if best_gt_idx != -1:
                    metrics[p_cls]['tp'] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    metrics[p_cls]['fp'] += 1
            
            for gt_idx, matched in enumerate(gt_matched):
                if not matched:
                    g_cls = gts[gt_idx][0]
                    metrics[g_cls]['fn'] += 1

        avg_p, avg_r, avg_ap = 0.0, 0.0, 0.0
        class_metrics = {}
        valid_cls = [cls for cls in metrics if cls in VALID_CLS_IDS]
        n_classes = len(valid_cls) if len(valid_cls) > 0 else 1

        for cls_id in valid_cls:
            tp = metrics[cls_id]['tp']
            fp = metrics[cls_id]['fp']
            fn = metrics[cls_id]['fn']
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            ap = p * r
            
            class_metrics[id2cat[cls_id]] = {'P': round(p, 4), 'R': round(r, 4), 'AP': round(ap, 4)}
            avg_p += p
            avg_r += r
            avg_ap += ap

        overall = {
            'Avg_P': round(avg_p / n_classes, 4),
            'Avg_R': round(avg_r / n_classes, 4),
            'mAP@0.5': round(avg_ap / n_classes, 4)
        }
        
        return class_metrics, overall

    def run_evaluation(self):
        print("开始加载标注文件...")
        self.load_gt()
        if len(self.all_gts) == 0:
            print("无有效标注框，终止评估！")
            return

        print("\n开始TensorRT加速批量推理...")
        img_files = glob.glob(os.path.join(IMG_PATH, "*.jpg")) + \
                    glob.glob(os.path.join(IMG_PATH, "*.png"))
        
        if not img_files:
            print(f"未找到图片！请检查路径：{IMG_PATH}")
            return
        
        for img_path in tqdm(img_files, total=len(img_files)):
            img_name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            self.infer_single(frame, img_name)

        if len(self.infer_times) > 0:
            avg_time = np.mean(self.infer_times)
            print(f"\n===== 推理时长统计（单位：毫秒） =====")
            print(f"平均推理时长：{avg_time:.2f} ms")
            print(f"有效推理图片数：{len(self.infer_times)}")
        else:
            return

        print(f"\n===== 模型精度评估（IOU阈值：0.5） =====")
        class_metrics, overall_metrics = self.calculate_metrics()
        
        for cls_name, res in sorted(class_metrics.items()):
            print(f"{cls_name:<15}: P={res['P']}, R={res['R']}, AP={res['AP']}")
        
        print(f"整体指标：{overall_metrics}")

        return overall_metrics, avg_time

    def destroy(self):
        self.yolov5_wrapper.destroy()

if __name__ == "__main__":
    detector = None
    try:
        detector = DetectImageTRT()
        detector.run_evaluation()
    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector:
            detector.destroy()
        print("释放内存完成")