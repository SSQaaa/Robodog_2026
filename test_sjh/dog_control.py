# -*- coding: utf-8 -*-
import threading
import time
from udp import UDPClient


class DogControl:
    def __init__(self):
        # 心脏包
        self.__udp_client = UDPClient('192.168.1.120', 43893)
        self.__heart_beat_thread = threading.Thread(target=self.__heart_beat)
        self.__heart_beat_thread.start()

    def __del__(self):  # 析构函数 对象销毁时让狗趴下
        self.stand_up()

    def __heart_beat(self):
        while True:
            self.__udp_client.send(0x21040001, duration=0.2)

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
    
    def get_info(self):
        self.data = self.__udp_client.send(0x0901)
        print(self.data)


    def stop(self):
        self.__udp_client.send(0x21010C0A, value=7)

    # 持续运动模式开启后，机械狗就算不收到命令也会持续踏步
    def close_continue(self):
        self.__udp_client.send(0x21010C06, value=2)

    def start_continue(self):
        self.__udp_client.send(0x21010C06, value=-1)