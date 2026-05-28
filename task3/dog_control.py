# -*- coding: utf-8 -*-
import threading
import time

from udp import UDPClient


DOG_IP = "192.168.1.120"
DOG_PORT = 43893
COMMAND_SLEEP_S = 0.5

FORWARD_SPEED = 20000
BACKWARD_SPEED = -9000
PRE_GRASP_MOVE_SPEED_X = 7000
PRE_GRASP_MOVE_SPEED_Y = 25000
SIDE_SPEED = 25000

class DogCommand:
    HEARTBEAT = 0x21040001
    STAND_UP = 0x21010202
    MOVE_MODE = 0x21010D06
    NORMAL_GAIT = 0x21010300
    VX = 0x21010130
    VY = 0x21010131
    VZ = 0x21010135
    ACTION = 0x21010C0A


class DogAction:
    STOP = 7
    TURN_180 = 15


class DogControl:
    def __init__(self, ip=DOG_IP, port=DOG_PORT):
        self._running = True
        self._udp_client = UDPClient(ip, port)
        self._heartbeat_thread = None
        self._send_lock = threading.Lock()
        self._move_mode_ready = False

        self._heartbeat_thread = threading.Thread(target=self._heart_beat, daemon=True)
        self._heartbeat_thread.start()

    # 停止机器狗，并结束心跳线程循环。
    def close(self):
        self._running = False
        self.stop()

    def _heart_beat(self):
        while self._running:
            self._udp_client.send(DogCommand.HEARTBEAT)

    # 设置速度前，先进入正常移动模式。
    def enter_move_mode(self):
        if self._move_mode_ready:
            return
        self._udp_client.send(DogCommand.MOVE_MODE)
        self._udp_client.send(DogCommand.NORMAL_GAIT)
        self._move_mode_ready = True

    # 任务开始前先让机器狗站立。
    def stand_up(self):
        self._udp_client.send(DogCommand.STAND_UP)
        self._move_mode_ready = False
        time.sleep(2.5)


    def move(self, last_time: float = 0, vx=0, vy=0, vz=0, duration=0.0):
        self._udp_client.send(DogCommand.MOVE_MODE)
        start_time = time.time()
        self._udp_client.send(DogCommand.NORMAL_GAIT)
        while True:
            self._udp_client.send(DogCommand.VX, vx)
            self._udp_client.send(DogCommand.VY, vy)
            self._udp_client.send(DogCommand.VZ, vz)

            if time.time() - start_time > last_time:
                break

        time.sleep(duration)

    # 调用机器狗内置动作原地转 180 度。
    def revolve_180(self):
        print("[Dog] revolve 180")
        self._move_mode_ready = False
        self._udp_client.send(DogCommand.ACTION, DogAction.TURN_180)
        time.sleep(4.0)
        self._udp_client.send(DogCommand.ACTION, DogAction.STOP)

    def stop(self):
        self.move(0, 0, 0)
        self._udp_client.send(DogCommand.ACTION, DogAction.STOP)
        self._move_mode_ready = False
