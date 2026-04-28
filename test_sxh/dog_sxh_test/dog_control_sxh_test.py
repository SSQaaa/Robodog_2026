import threading
from udp import UDPClient
import time

class DogControl:
    def __init__(self):
        # 心脏包 这里的ip地址为上下位机通讯的ip地址 不要随便修改
        self.__udp_client = UDPClient('192.168.1.120', 43893)
        #开始双线程，在后台源源不断执行这个心脏发送
        self.__heart_beat_thread = threading.Thread(target=self.__heart_beat)
        self.__heart_beat_thread.start()

    def __del__(self):  # 析构函数 对象销毁时让狗趴下
        self.stand_up()

    #发送心脏，不低于2HZ，不低于0.5s
    def __heart_beat(self):
        while True:
            self.__udp_client.send(0x21040001, duration=0.2)

    #发送这个指令，站起或者坐下（是一个指令）
    def stand_up(self):
        self.__udp_client.send(0x21010202, duration=3)

    def move(self, last_time: float = 0, vx=0, vy=0, vz=0, duration=0.0):
        #开启移动模式
        self.__udp_client.send(0x21010D06)
        start_time = time.time()
        #平地低速步态
        self.__udp_client.send(0x21010300)
        while True:
            self.__udp_client.send(0x21010130, vx) #前后平移
            self.__udp_client.send(0x21010131, vy) #左右平移
            self.__udp_client.send(0x21010135, vz) #左右转弯
            #这个move在调用一次只会启动一次，会立马break掉
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
        time.sleep(0.3)#转换步态之后需要停0.3s
        while True:
            self.__udp_client.send(0x21010130, vx)
            self.__udp_client.send(0x21010131, vy)
            self.__udp_client.send(0x21010135, vz)
            #为了和上面的0.3s打平回来 因此这里需要加一个0.3
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

    def revolve_90_r(self):#右转
        self.__udp_client.send(0x21010C0A, value=14)
        time.sleep(2)
        self.__udp_client.send(0x21010C0A, value=7)

    def revolve_90_l(self):#左转
        self.__udp_client.send(0x21010C0A, value=13)
        time.sleep(2)
        self.__udp_client.send(0x21010C0A, value=7)

    def stop(self):
        self.__udp_client.send(0x21010C0A, value=7)

    # 持续运动模式开启后，机械狗就算不收到命令也会持续踏步
    def close_continue(self):
        self.__udp_client.send(0x21010C06, value=2)

    def start_continue(self):
        self.__udp_client.send(0x21010C06, value=-1)





