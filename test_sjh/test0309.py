# -*- coding: utf-8 -*-
import threading
import time
import cv2
import numpy as np
import multiprocessing
from pyzbar.pyzbar import decode

from udp import UDPClient

class DogControl:
    def __init__(self):
        # 心脏包
        self.__udp_client = UDPClient('192.168.1.120', 43893)
        self.__heart_beat_thread = threading.Thread(target=self.__heart_beat)
        self.__heart_beat_thread.start()
        # 摄像头
        self._cap_running = True        
        self.cap_index = 3
        self.cap_index_catch = 2
        self.frame = None
        self.cap_thread = threading.Thread(target=self._cap)
        self.cap_thread.start()

    def __del__(self):  # 析构函数 对象销毁时让狗趴下
        self.stand_up()

    def __heart_beat(self):
        while True:
            self.__udp_client.send(0x21040001, duration=0.2)

    def _cap(self):
        cap = cv2.VideoCapture(self.cap_index)
        if not cap.isOpened():
            print("camera error")
            return
        while self._cap_running:
            ret, frame = cap.read()
            if ret:
              self.frame = frame
            # if ret:
            #     cv2.imshow('x1',self.frame)
            #     cv2.waitKey(1)
            time.sleep(0.001)
        cap.release()


    def stand_up(self):
        self.__udp_client.send(0x21010202, duration=3)

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
            self.__udp_client.send(0x21010307) # 中步态Process_One()
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

        self.__udp_client.send(0x21010130, 12000 , last_time=0.6)
        self.__udp_client.send(0x21010130, 0, last_time=1)
        self.__udp_client.send(0x21010130, -12000, last_time=0.5)
        self.__udp_client.send(0x21010130, 0, last_time=1)
        self.__udp_client.send(0x21010130, 12000, last_time=0.6)
        self.__udp_client.send(0x21010130, 0, last_time=1)
        self.__udp_client.send(0x21010130, -12000, last_time=0.5)
        self.__udp_client.send(0x21010130, 0, last_time=1)
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


    # 仪表盘校正
    def dashboard_correction(self):
        while True:
            if self.frame is not None:
                frame = self.frame.copy()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([35, 255, 255])
                
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = contours[-2] 
                 
                frame_h, frame_w, channels = frame.shape
                
                
                if cnts:
                  max_cnt = max(cnts, key=cv2.contourArea)
                  if cv2.contourArea(max_cnt) > 500:
                      x, y, w, h = cv2.boundingRect(max_cnt)
                      print(x+w//2, y+h//2, "Area:", w*h)
                      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                      
                      self.start_continue()
                      center_x = x + w / 2
                      
                      if center_x - frame_w // 2 >= 80:  # 球在右边，狗往右走
                          self.move(vy=40000, last_time=0.2)  # 向右
                          time.sleep(0.5)
                          print("right")
                      elif center_x - frame_w // 2 <= -80:  # 球在左边，狗往左走
                          self.move(vy=-40000, last_time=0.2)  # 向右
                          time.sleep(0.5)
                          print("left")
                      else:
                          area = w * h
                          print("area:", area)
                          frame_area = frame_w * frame_h
                          print("frame_area:", frame_area)
                          
                          if(area < frame_area * 0.25):  # too small, approaching
                              self.start_continue()
                              self.move(vx=9000, last_time=0.15, duration=0.5)
                              print("too small, approaching")
                          elif(area > frame_area * 0.5):
                              self.start_continue()
                              self.move(vx=-9000, last_time=0.15, duration=0.5)
                              print("too big, back back")
                          else:
                              self.close_continue()
                              break





if __name__ == '__main__':
    control = DogControl()
    time.sleep(1)
    control.close_continue()
    control.stop()
    time.sleep(0.5)
    control.stand_up()
    time.sleep(2.5)
    
    control.dashboard_correction()
    
    control.close_continue()
    control.stop()
    time.sleep(2)
