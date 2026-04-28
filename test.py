import threading
import time
import cv2 as cv
import numpy as np
import multiprocessing
from pyzbar.pyzbar import decode

from udp import UDPClient
from detect_trt import Detect_image
from detect import Element_Detect
from sound_output import sound_output

categories = ['1', '2', '3', '4', '5', '6', 'red_barrel', 'yellow_barrel', 'blue_barrel', 'orange_barrel',
              'red_ball', 'yellow_ball', 'blue_ball', 'orange_ball', 'dashboard', 'ssi']
dict_abnormal = {'yellow_barrel': 'yellow_ball', 'red_barrel': 'red_ball', 'orange_barrel': 'orange_ball',
                 'blue_barrel': 'blue_ball'}



class DogControl:
    def __init__(self):
        # 心脏包
        self.__udp_client = UDPClient('192.168.1.120', 43893)
        self.__heart_beat_thread = threading.Thread(target=self.__heart_beat)
        self.__heart_beat_thread.start()
        # 摄像头
        self.cap_index = 6
        self.frame = None
        self.cap_thread = threading.Thread(target=self._cap)
        self.cap_thread.start()

        self.Detect = Detect_image()
        self.Element_detect = Element_Detect()

        #指针阈值变量
        self.niddle_threshold_value = 110
        # 巡线阈值变量
        self.line_threshold_value = 160

        # 标记控制窗口是否已创建
        self.controls_created = False
        self.niddle_controls_created = False

    def __heart_beat(self):
        while True:
            self.__udp_client.send(0x21040001, duration=0.2)

    def _cap(self):
        cap = cv.VideoCapture(self.cap_index)
        while True:
            ret, self.frame = cap.read()

    def stand_up(self):
        self.__udp_client.send(0x21010202, duration=3)

    """
    move：正常低步态
    EXmove：转换步态且转回低步态
    NEXmove：转换步态且不转回低步态
    控制前进后退和旋转
    Args:
        last_time (int): 发送持续时间
        vx (int, optional): 前后平移速度，死区 [-6553,6553]，正值向前
        vy (int, optional): 左右平移速度，死区 [-12553,12553]，正值向右
        vz (int, optional): 旋转角速度，死区 [-9553,9553] ，正值顺时针
        duration (int, optional): 发送后的间隔时间
    """
    def move(self, last_time: float = 0, vx=0, vy=0, vz=0, duration=0.0):
        self.__udp_client.send(0x21010D06)
        start_time = time.time()
        self.__udp_client.send(0x21010300)
        while True:
            self.__udp_client.send(0x21010130, vx)
            self.__udp_client.send(0x21010131, vy)
            self.__udp_client.send(0x21010135, vz)

            if time.time() - start_time > last_time:
                break

        time.sleep(duration)

    def EXmove(self, last_time: float = 0, vx=0, vy=0, vz=0, duration=0.0,case = 1):
        self.__udp_client.send(0x21010D06)
        start_time = time.time()
        if case == 1 :
            self.__udp_client.send(0x21010307) #中步态
        elif case == 2:
            self.__udp_client.send(0x21010303) #高步态
        time.sleep(0.3)
        while True:
            self.__udp_client.send(0x21010130, vx)
            self.__udp_client.send(0x21010131, vy)
            self.__udp_client.send(0x21010135, vz)

            if time.time() - start_time > last_time + 0.3:
                time.sleep(1)
                self.__udp_client.send(0x21010300)
                break
        time.sleep(duration)

    def NEXmove(self, last_time: float = 0, vx=0, vy=0, vz=0, duration=0.0, case = 1):
        self.__udp_client.send(0x21010D06)
        start_time = time.time()
        if case == 1:
            self.__udp_client.send(0x21010307) # 中步态
        elif case ==2:
            self.__udp_client.send(0x21010303) #高步态
        while True:
            self.__udp_client.send(0x21010130, vx)
            self.__udp_client.send(0x21010131, vy)
            self.__udp_client.send(0x21010135, vz)

            if time.time() - start_time > last_time:
                break

        time.sleep(duration)

    def shake_head(self):
        self.__udp_client.send(0x21010D05, duration=1)  # 开启原地模式

        self.__udp_client.send(0x21010135, 22767, last_time=0.5)
        self.__udp_client.send(0x21010135, -22767, last_time=0.5)
        self.__udp_client.send(0x21010135, 22767, last_time=0.5)
        self.__udp_client.send(0x21010135, -22767, last_time=0.5)
        self.__udp_client.send(0x21010135, 0, last_time=0.5)
        time.sleep(0.5)
        self.__udp_client.send(0x21010D06)  # 开启移动模式

    def nod_head(self):
        self.__udp_client.send(0x21010D05, duration=1)

        self.__udp_client.send(0x21010130, 12000, last_time=0.6)
        self.__udp_client.send(0x21010130, 0, last_time=0.1)
        self.__udp_client.send(0x21010130, -12000, last_time=0.5)
        self.__udp_client.send(0x21010130, 0, last_time=0.1)
        self.__udp_client.send(0x21010130, 12000, last_time=0.6)
        self.__udp_client.send(0x21010130, 0, last_time=0.1)
        self.__udp_client.send(0x21010130, -12000, last_time=0.5)
        self.__udp_client.send(0x21010130, 0, last_time=0.1)
        time.sleep(0.5)
        self.__udp_client.send(0x21010D06)

    def move_state(self):
        self.__udp_client.send(0x21010D06)
        time.sleep(1)

    def revolve_180(self):
        self.__udp_client.send(0x21010C0A, value=15)  # lite3
        time.sleep(4)
        self.__udp_client.send(0x21010C0A, value=7)

    def revolve_90(self):
        self.__udp_client.send(0x21010C0A, value=14)
        time.sleep(2)
        self.__udp_client.send(0x21010C0A, value=7)

    def stop(self):
        self.__udp_client.send(0x21010C0A, value=7)

    # 持续运动模式开启后，机械狗就算不收到命令也会持续踏步
    def close_continue(self):
        self.__udp_client.send(0x21010C06, value=2)

    def start_continue(self):
        self.__udp_client.send(0x21010C06, value=-1)

    #测试摄像头
    def video_test(self):
        while True:
            if self.frame is not None:
                cv.imshow('frame', self.frame)
                cv.waitKey(1)

    #显示yolov5检测结果图像
    def yolov5_test(self):
        while True:
            if self.frame is not None:
                frame, box, classid = self.Detect.detect_image(self.frame)
                cv.imshow("result", frame)
                cv.waitKey(1)

    def line_test(self):
        # 创建调节窗口（如果尚未创建）
        if not self.controls_created:
            cv.namedWindow('controls')
            cv.createTrackbar('Threshold', 'controls', self.line_threshold_value, 255, self.on_trackbar)
            self.controls_created = True

        while True:
            if self.frame is not None:
                h, w, channels = self.frame.shape
                frame = self.frame[h // 4:, :]
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # 使用类变量作为阈值
                _, gray = cv.threshold(gray, self.line_threshold_value, 255, cv.THRESH_BINARY)

                kernel = np.ones((5, 5), np.uint8)
                gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
                edges = cv.Canny(gray, 50, 150, apertureSize=3)

                cv.imshow('edge', edges)
                cv.imshow('binary',gray)
                cv.waitKey(1)

    # 新增的滑条回调函数
    def on_trackbar(self, val):
        self.line_threshold_value = val

    def clamp(self ,value , min_val, max_val):
        return max(min_val,min(value,max_val))

    def refine_box(self, bbox, w, h):
        bbox[0] = self.clamp(bbox[0], 0, w)
        bbox[1] = self.clamp(bbox[1], 0, h)
        bbox[2] = self.clamp(bbox[2], 0, w)
        bbox[3] = self.clamp(bbox[3], 0, h)
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        bbox[0] = center_x + (bbox[0] - center_x) * 0.5
        bbox[1] = center_y + (bbox[1] - center_y) * 0.5
        bbox[2] = center_x + (bbox[2] - center_x) * 0.5
        bbox[3] = center_y + (bbox[3] - center_y) * 0.5

        return bbox

    def find_niddle(self,frame,bbox):
        h,w,rgb = frame.shape

        bbox = self.refine_box(bbox,w,h)
        bbox = bbox.astype(np.int32)
        # print(bbox)
        # print(frame.shape)
        image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if image is not None:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # 二值化，将黑色区域设为255，其他区域设为0
            _, thresh = cv.threshold(gray, self.niddle_threshold_value, 255, cv.THRESH_BINARY_INV)
            # 寻找轮廓
            cont_image,contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # 找到面积最大的轮廓
            max_area = 0
            max_contour = None
            for c in contours:
                area = cv.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_contour = c

            # 如果找到了最大轮廓，绘制它并计算它的角度
            if max_contour is not None:
                cv.drawContours(image, [max_contour], -1, (0, 255, 0), 2)

            cv.imshow('max_contours', image)
            cv.imshow('binary', thresh)
            cv.waitKey(1)

    def niddle_test(self):
        # 创建指针检测控制窗口（如果尚未创建）
        if not self.niddle_controls_created:
            cv.namedWindow('niddle_controls')
            cv.createTrackbar('Niddle Threshold', 'niddle_controls',
                              self.niddle_threshold_value, 255, self.on_niddle_trackbar)
            self.niddle_controls_created = True

        while True:
            if self.frame is not None:
                frame_1 = self.frame.copy()
                frame, box, classid = self.Detect.detect_image(self.frame)

                for index, val in enumerate(classid):
                    if val == 14:  # 假设14是指针类别的ID
                        self.find_niddle(frame_1, box[index])

    # 新增的指针阈值滑条回调函数
    def on_niddle_trackbar(self, val):
        self.niddle_threshold_value = val

if __name__ == "__main__":
    control = DogControl()
    time.sleep(1)
    control.close_continue()
    control.stop()
    time.sleep(0.5)
    control.stand_up()
    time.sleep(0.5)
    control.nod_head()
    #测试摄像头
    # control.video_test()
    #测试模型效果
    # control.yolov5_test()
    #巡线阈值调节函数
    # control.line_test()
    #指针阈值测试代码
    # control.niddle_test()
    