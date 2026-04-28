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
        self._cap_running = True
        self.cap_index = 5
        self.cap_index_catch = 2
        self.frame = None
        self.cap_thread = threading.Thread(target=self._cap)
        self.cap_thread.start()
        # 元素检测
        self.Detect = Detect_image()  # 实例化yolov5检测
        self.Element_detect = Element_Detect()  # 元素检测结果处理
        self.abnormal = ['orange_barrel', 'red_barrel'] # 存放异常区域
        self.detect_list = []  # 记录当前纸箱上三个元素状态
        self.num = 0
        # 巡线矫正
        self.line_cnt = 0
        self.line_x = [[0, 0], [0, 0]]
        self.line_y = [[0, 0], [0, 0]]
        self.point = [0, 0]
        self.line_flag = True
        self.line_flag_1 = False
        #存放需要踢的球的颜色
        self.ball_classes = []
        #语音模块
        self.msg_output = sound_output()

    def __del__(self):  # 析构函数 对象销毁时让狗趴下
        self.stand_up()

    def __heart_beat(self):
        while True:
            self.__udp_client.send(0x21040001, duration=0.2)

    def _cap(self):
        cap = cv.VideoCapture(self.cap_index)
        while self._cap_running:
            ret, self.frame = cap.read()
        cap.release()

    #用于抓取任务切换摄像头
    def switch_camera(self):
        # 关闭旧线程
        self._cap_running = False
        self.cap_thread.join()
        # 切换摄像头
        self.cap_index = self.cap_index_catch
        self._cap_running = True
        self.cap_thread = threading.Thread(target=self._cap)
        self.cap_thread.start()

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

        self.__udp_client.send(0x21010130, 22767, last_time=0.6)
        self.__udp_client.send(0x21010130, 0, last_time=0.1)
        self.__udp_client.send(0x21010130, -22767, last_time=0.5)
        self.__udp_client.send(0x21010130, 0, last_time=0.1)
        self.__udp_client.send(0x21010130, 22767, last_time=0.6)
        self.__udp_client.send(0x21010130, 0, last_time=0.1)
        self.__udp_client.send(0x21010130, -22767, last_time=0.5)
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

if __name__ == "__main__":
    control = DogControl()
    time.sleep(1)
    control.close_continue()
    control.stand_up()
    control.stop()
    # control.stand_up()
    # control.close_continue()
    time.sleep(1)
    Detect = Detect_image()
    # Element = Element_Detect()
    # 创建VideoCapture对象

    capture = cv.VideoCapture(6) # 0表示内置摄像头 
    name = []
    while True:
    # 读取摄像头的一帧图像
        ret, frame = capture.read()
        if not ret:
            break
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #     # 二值化，将黑色区域设为255，其他区域设为0
        # _, thresh = cv.threshold(gray, 170, 255, cv.THRESH_BINARY_INV)
        # frame = cv2.resize(frame, (640, 640))
        # frame_1 = frame.copy()
        frame, box, classid = Detect.detect_image(frame)
        # Element.judge(frame_1,box,classid)
        # # print(classid)
        # # print(box)
        # cv.namedWindow("result", 0)
        # # cv2.resizeWindow("result", 640, 480)
        cv.imshow("result", frame)
        # cv.imshow('result_1',thresh)
        # cv.imshow('result_1',frame_1)
    # 显示图像
        if cv.waitKey(1) == ord('q'):
            Detect.destroy()
            break
