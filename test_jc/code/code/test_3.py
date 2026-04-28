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

# -------------------------- 完全复用detect_trt.py的参数 --------------------------
INPUT_W = 1024
INPUT_H = 576
CONF_THRESH = 0.2
IOU_THRESHOLD = 0.4
# 严格复用原文件的categories，0-15共16个类别，无16！
categories = ['1', '2', '3', '4', '5', '6', 'red_barrel','yellow_barrel', 
              'blue_barrel', 'orange_barrel', 'red_ball', 'yellow_ball',
              'blue_ball' ,'orange_ball','dashboard','ssi']
# 类别ID转名称（增加越界判断）
id2cat = {i: name for i, name in enumerate(categories)}
# 数据集路径（替换为你的根路径）
DATA_ROOT = "/home/ysc/Desktop/Robodog_2026/test_jc/new_doglabels.v6i.yolov5pytorch/test/"  # 下含images/labels子文件夹
IMG_PATH = os.path.join(DATA_ROOT, "images")
LABEL_PATH = os.path.join(DATA_ROOT, "labels")
# 有效类别ID范围（0-15）
VALID_CLS_IDS = set(range(len(categories)))

# -------------------------- 修复后的检测评估类 --------------------------
class DetectImageTRT:
    def __init__(self):
        # 加载TensorRT插件和引擎（完全复用原代码）
        PLUGIN_LIBRARY = "build/libmyplugins.so"
        ctypes.CDLL(PLUGIN_LIBRARY)
        engine_file_path = "build/yolov5s.engine"
        self.yolov5_wrapper = YoLov5TRT(engine_file_path)
        # 评估用变量
        self.infer_times = []  # 存储单张图片推理时长
        self.all_preds = []    # 所有预测框 (img_name, cls_id, x1, y1, x2, y2, conf)
        self.all_gts = []      # 所有标注框 (img_name, cls_id, x1, y1, x2, y2)
        # 预热引擎（解决首次推理耗时偏高问题）
        self.warmup_engine()

    def warmup_engine(self):
        """引擎预热：执行一次空推理，跳过初始化耗时"""
        print("TensorRT引擎预热中...")
        dummy_img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        self.yolov5_wrapper.infer(dummy_img)
        print("引擎预热完成！")

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # 完全复用原detect_trt.py的画框函数
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
        # 完全复用原detect_trt.py的后处理函数
        index_dict = {'number': [], 'barrel': [], 'board': [],'ssi': [],'ball':[]}
        areas = []
        result_new_boxes = []
        result_new_classid = []
        for idx, val in enumerate(result_classid):
            class_name = categories[int(val)]
            box = result_boxes[idx]
            area = abs(box[2] - box[0]) * abs(box[3] - box[1])
            areas.append(area)
            if class_name in set(['1', '2', '3', '4', '5', '6']):
                index_dict['number'].append(idx)
            elif class_name in set(['blue_barrel', 'orange_barrel', 'red_barrel', 'yellow_barrel']):
                index_dict['barrel'].append(idx)
            elif class_name in set(['dashboard']):
                index_dict['board'].append(idx)
            elif class_name in set(['ssi']):
                index_dict['ssi'].append(idx)
            elif class_name in set(['red_ball', 'yellow_ball','blue_ball' ,'orange_ball']):
                index_dict['ball'].append(idx)
        for group in ['number', 'barrel','board','ball']:
            if index_dict[group]:
                max_idx = max(index_dict[group], key=lambda x: areas[x])
                box = result_boxes[max_idx]
                self.plot_one_box(
                    box,
                    image_raw,
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[max_idx])], result_scores[max_idx]
                    ),
                )
                result_new_classid.append(result_classid[max_idx])
                result_new_boxes.append(result_boxes[max_idx])
        for j in range(len(result_new_classid)):
            if int(result_new_classid[j]) == 14:
                for i in range(len(index_dict['ssi'])):
                    s = index_dict['ssi'][i]
                    middle_x = (result_boxes[s][0]+result_boxes[s][2])//2
                    if middle_x > min(result_new_boxes[j][0],result_new_boxes[j][2]) and \
                       middle_x < max(result_new_boxes[j][0],result_new_boxes[j][2]):
                        self.plot_one_box(
                            result_boxes[s],
                            image_raw,
                            label="{}:{:.2f}".format(
                                categories[int(result_classid[s])],result_scores[s]
                            ),
                        )
                        result_new_classid.append(result_classid[s])
                        result_new_boxes.append(result_boxes[s])
        return image_raw, result_new_boxes, result_new_classid

    def infer_single(self, frame, img_name):
        """单张图片推理：仅统计纯推理耗时，跳过预处理/后处理"""
        # 仅统计TensorRT模型infer的耗时（复用原代码返回的use_time，更精准）
        _, result_boxes, result_scores, result_classid, trt_use_time = self.yolov5_wrapper.infer(frame)
        # 转毫秒，仅统计有效推理（过滤0耗时）
        if trt_use_time > 0:
            self.infer_times.append(trt_use_time * 1000)
        
        # 后处理筛选检测框
        _, pred_boxes, pred_cls_ids = self.draw_boxes(frame.copy(), result_boxes, result_scores, result_classid)
        
        # 存储预测框：过滤无效类别ID，匹配原categories
        for box, cls_id in zip(pred_boxes, pred_cls_ids):
            cls_id = int(cls_id)
            if cls_id not in VALID_CLS_IDS:
                continue
            x1, y1, x2, y2 = box
            # 匹配置信度（容错：避免框匹配失败）
            conf = 0.0
            match_idx = np.where((result_boxes == box).all(axis=1))[0]
            if len(match_idx) > 0:
                conf = result_scores[match_idx[0]]
            self.all_preds.append([img_name, cls_id, x1, y1, x2, y2, conf])

    def load_gt(self):
        """加载标注框：过滤无效类别ID+容错YOLO格式"""
        label_files = glob.glob(os.path.join(LABEL_PATH, "*.txt"))
        if not label_files:
            print(f"未找到标注文件！请检查路径：{LABEL_PATH}")
            return
        for lbl_file in label_files:
            img_name = os.path.basename(lbl_file).replace(".txt", ".jpg")
            # 兼容png格式
            if not os.path.exists(os.path.join(IMG_PATH, img_name)):
                img_name = img_name.replace(".jpg", ".png")
            img_path = os.path.join(IMG_PATH, img_name)
            if not os.path.exists(img_path):
                print(f"图片缺失：{img_path}，跳过该标注")
                continue
            # 获取图片尺寸（YOLO归一化转像素）
            img = cv2.imread(img_path)
            if img is None:
                print(f"图片读取失败：{img_path}，跳过该标注")
                continue
            h, w = img.shape[:2]
            # 读取标注txt
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    print(f"标注格式错误：{lbl_file} → {line}，跳过")
                    continue
                # 解析类别ID（强制转int，过滤非数字）
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    print(f"类别ID非数字：{lbl_file} → {line}，跳过")
                    continue
                # 核心修复：过滤无效类别ID（如16）
                if cls_id not in VALID_CLS_IDS:
                    print(f"无效类别ID：{cls_id}（仅支持0-15），{lbl_file} → {line}，跳过")
                    continue
                # 解析YOLO坐标（容错：非浮点数）
                try:
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    print(f"坐标格式错误：{lbl_file} → {line}，跳过")
                    continue
                # YOLO→像素坐标（限制0~w/h，避免越界）
                x1 = max(0, min(w, (xc - bw/2) * w))
                y1 = max(0, min(h, (yc - bh/2) * h))
                x2 = max(0, min(w, (xc + bw/2) * w))
                y2 = max(0, min(h, (yc + bh/2) * h))
                self.all_gts.append([img_name, cls_id, x1, y1, x2, y2])
        print(f"成功加载 {len(self.all_gts)} 个有效标注框（已过滤无效ID/格式）")

    def iou(self, box1, box2):
        """计算IOU：容错空框/越界框"""
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
        """计算P/R/mAP@0.5：完全容错有效类别ID"""
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
                # 匹配同类别有效标注框
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

        # 计算各类别P/R/AP，容错无检测结果的类别
        avg_p, avg_r, avg_ap = 0.0, 0.0, 0.0
        class_metrics = {}
        valid_cls = [cls for cls in metrics if cls in VALID_CLS_IDS]
        n_classes = len(valid_cls) if len(valid_cls) > 0 else 1

        for cls_id in valid_cls:
            tp = metrics[cls_id]['tp']
            fp = metrics[cls_id]['fp']
            fn = metrics[cls_id]['fn']
            # 避免除0错误
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            ap = p * r  # 单IOU阈值下AP简化计算
            class_metrics[id2cat[cls_id]] = {'P': round(p, 4), 'R': round(r, 4), 'AP': round(ap, 4)}
            avg_p += p
            avg_r += r
            avg_ap += ap

        # 计算整体平均指标
        overall = {
            'Avg_P': round(avg_p / n_classes, 4),
            'Avg_R': round(avg_r / n_classes, 4),
            'mAP@0.5': round(avg_ap / n_classes, 4)
        }
        return class_metrics, overall

    def run_evaluation(self):
        """主执行函数：加载标注→批量推理→计算指标"""
        # 1. 加载所有有效标注框
        print("开始加载标注文件...")
        self.load_gt()
        if len(self.all_gts) == 0:
            print("无有效标注框，终止评估！")
            return

        # 2. 批量推理images文件夹下的图片
        print("\n开始TensorRT加速批量推理...")
        img_files = glob.glob(os.path.join(IMG_PATH, "*.jpg")) + glob.glob(os.path.join(IMG_PATH, "*.png"))
        if not img_files:
            print(f"未找到图片！请检查路径：{IMG_PATH}")
            return
        for img_path in tqdm(img_files, total=len(img_files)):
            img_name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"图片读取失败：{img_path}，跳过")
                continue
            # 单张推理
            self.infer_single(frame, img_name)

        # 3. 计算推理时长统计（更精准）
        if len(self.infer_times) > 0:
            avg_time = np.mean(self.infer_times)
            max_time = np.max(self.infer_times)
            min_time = np.min(self.infer_times)
            print(f"\n===== 推理时长统计（单位：毫秒） =====")
            print(f"平均推理时长：{avg_time:.2f} ms")
            print(f"最大推理时长：{max_time:.2f} ms")
            print(f"最小推理时长：{min_time:.2f} ms")
            print(f"有效推理图片数：{len(self.infer_times)}")
        else:
            print("\n无有效推理时长数据！")
            return

        # 4. 计算模型精度评估指标
        print(f"\n===== 模型精度评估（IOU阈值：0.5） =====")
        class_metrics, overall_metrics = self.calculate_metrics()
        # 打印各类别指标
        for cls_name, res in sorted(class_metrics.items()):
            print(f"{cls_name:<15}: P={res['P']}, R={res['R']}, AP={res['AP']}")
        # 打印整体指标
        print(f"整体指标：{overall_metrics}")

        return overall_metrics, avg_time

    def destroy(self):
        # 释放TensorRT引擎资源
        self.yolov5_wrapper.destroy()

# -------------------------- 执行入口 + 资源释放容错 --------------------------
if __name__ == "__main__":
    detector = None
    try:
        # 实例化并运行评估
        detector = DetectImageTRT()
        detector.run_evaluation()
    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源释放
        if detector:
            detector.destroy()
        print("释放内存完成")