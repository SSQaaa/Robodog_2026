import time
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
from collections import defaultdict


import threading
import random
from yolov5trt import YoLov5TRT

INPUT_W = 1024
INPUT_H = 576
CONF_THRESH = 0.2
IOU_THRESHOLD = 0.4
categories = ['1', '2', '3', '4', '5', '6', 'red_barrel','yellow_barrel', 'blue_barrel', 'orange_barrel',
 'red_ball', 'yellow_ball','blue_ball' ,'orange_ball','dashboard','ssi','yellow_cylinder' ,'red_cylinder'   ]
number_classes = set(['1', '2', '3', '4', '5', '6'])
barrel_classes = set(['blue_barrel', 'orange_barrel', 'red_barrel', 'yellow_barrel'])
board_classes = set(['dashboard'])
ssi_classes = set(['ssi'])
ball_classes = set(['red_ball', 'yellow_ball','blue_ball' ,'orange_ball'])
cylinder_classes = set(['red_cylinder'])
class Detect_image(object):
    def __init__(self):
        PLUGIN_LIBRARY = "build/libmyplugins.so"
        ctypes.CDLL(PLUGIN_LIBRARY)
        engine_file_path = "build/yolov5s.engine"
        self.yolov5_wrapper = YoLov5TRT(engine_file_path)
        

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(
                img,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )


    def draw_boxes(self, image_raw, result_boxes, result_scores, result_classid):
        index_dict = {'number': [], 'barrel': [], 'board': [],'ssi': [],'ball':[],'cylinder':[]}
        areas = []
        result_new_boxes = []
        result_new_classid = []
        areas = []
        for idx, val in enumerate(result_classid):
            class_name = categories[int(val)]
            box = result_boxes[idx]
            area = abs(box[2] - box[0]) * abs(box[3] - box[1])
            areas.append(area)
            if class_name in number_classes:
                index_dict['number'].append(idx)
            elif class_name in barrel_classes:
                index_dict['barrel'].append(idx)
            elif class_name in board_classes:
                index_dict['board'].append(idx)
            elif class_name in ssi_classes:
                index_dict['ssi'].append(idx)
            elif class_name in ball_classes:
                index_dict['ball'].append(idx)
            elif class_name in cylinder_classes:
                self.plot_one_box(
                    box,
                    image_raw,
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[idx])], result_scores[idx]
                    ),
                )
                result_new_classid.append(result_classid[idx])
                result_new_boxes.append(result_boxes[idx])
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
                    if middle_x > min(result_new_boxes[j][0],result_new_boxes[j][2]) and middle_x < max(result_new_boxes[j][0],result_new_boxes[j][2]):
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
    def detect_image(self, frame):
        img, result_boxes, result_scores, result_classid ,use_time = self.yolov5_wrapper.infer(frame)
        img, result_new_boxes, result_new_classid = self.draw_boxes(frame, result_boxes, result_scores, result_classid)
        return  img, result_new_boxes, result_new_classid
    def destroy(self):
        self.yolov5_wrapper.destroy()

if __name__ == '__main__':
    detector = Detect_image()
    cap = cv2.VideoCapture(2)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame, _, _ = detector.detect_image(frame)
        cv2.imshow('YOLOv5 TRT Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.destroy()

    

