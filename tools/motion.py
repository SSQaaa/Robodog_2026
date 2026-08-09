# -*- coding: utf-8 -*-
import socket
import struct
import threading
import time

from project_config import DOG_IP, DOG_PORT


ZERO_SPEED_HOLD_S = 0.15


class UDPClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.send_addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)

    def close(self):
        self.sock.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def send(self, code, value=0, type=0, last_time=0, duration=0):
        data = struct.pack("<3i", code, value, type)
        start_time = time.time()
        if last_time == 0:
            self.sock.sendto(data, self.send_addr)
            time.sleep(0.05)
        else:
            while time.time() - start_time < last_time:
                self.sock.sendto(data, self.send_addr)
                time.sleep(0.05)
        if duration != 0:
            time.sleep(duration)


class DogCommand:
    HEARTBEAT = 0x21040001
    STAND_UP = 0x21010202
    MOVE_MODE = 0x21010D06
    NORMAL_GAIT = 0x21010300
    MEDIUM_GAIT = 0x21010307
    HIGH_GAIT = 0x21010303
    DOWN_GAIT = 0x21010406
    SPOT_MODE = 0x21010D05
    VX = 0x21010130
    VY = 0x21010131
    VZ = 0x21010135
    ACTION = 0x21010C0A
    CONTINUE = 0x21010C06


class DogAction:
    STOP = 7
    TURN_90_L = 13
    TURN_90_R = 14
    TURN_180 = 15


class DogControl:
    def __init__(self, ip=DOG_IP, port=DOG_PORT):
        self._udp_client = UDPClient(ip, port)
        self._running = True
        self._send_lock = threading.Lock()
        self._heartbeat_thread = threading.Thread(target=self._heart_beat, daemon=True)
        self._heartbeat_thread.start()

    def _send(self, code, value=0, type=0, last_time=0, duration=0):
        with self._send_lock:
            self._udp_client.send(code, value=value, type=type, last_time=last_time, duration=duration)

    def _heart_beat(self):
        while self._running:
            self._send(DogCommand.HEARTBEAT)
            time.sleep(0.2)

    def close(self):
        self._running = False
        self.stop()
        try:
            self._udp_client.close()
        except Exception:
            pass

    def stand_up(self):
        self._send(DogCommand.STAND_UP, duration=0.1)

    def move(self, last_time: float = 0, vx: int = 0, vy: int = 0, vz: int = 0, duration: float = 0.0) -> None:
        self._send(DogCommand.MOVE_MODE)
        start_time = time.time()
        self._send(DogCommand.NORMAL_GAIT)
        while True:
            self._send(DogCommand.VX, vx)
            self._send(DogCommand.VY, vy)
            self._send(DogCommand.VZ, vz)
            if time.time() - start_time > last_time:
                break
        zero_start = time.time()
        while time.time() - zero_start < ZERO_SPEED_HOLD_S:
            self._send(DogCommand.VX, 0)
            self._send(DogCommand.VY, 0)
            self._send(DogCommand.VZ, 0)
        time.sleep(duration)

    def EXmove(self, last_time: float = 0, vx=0, vy=0, vz=0, duration=0.0, case=1):
        self._send(DogCommand.MOVE_MODE)
        gait = DogCommand.MEDIUM_GAIT if case == 1 else DogCommand.HIGH_GAIT
        self._send(gait)
        time.sleep(0.3)
        self.move(last_time=last_time, vx=vx, vy=vy, vz=vz, duration=duration)
        self._send(DogCommand.NORMAL_GAIT)

    def DOWNmove(self, last_time: float = 0, vx=0, vy=0, vz=0, duration=0.0):
        self._send(DogCommand.MOVE_MODE)
        self._send(DogCommand.DOWN_GAIT)
        time.sleep(0.5)
        self.move(last_time=last_time, vx=vx, vy=vy, vz=vz, duration=duration)

    def UPDOWN(self):
        self._send(DogCommand.MOVE_MODE)
        self._send(DogCommand.DOWN_GAIT)
        time.sleep(1)

    def revolve_180(self):
        self._send(DogCommand.ACTION, value=DogAction.TURN_180)
        time.sleep(4)
        self._send(DogCommand.ACTION, value=DogAction.STOP)

    def revolve_90_r(self):
        self._send(DogCommand.ACTION, value=DogAction.TURN_90_R)
        time.sleep(2)
        self._send(DogCommand.ACTION, value=DogAction.STOP)

    def revolve_90_l(self):
        self._send(DogCommand.ACTION, value=DogAction.TURN_90_L)
        time.sleep(2)
        self._send(DogCommand.ACTION, value=DogAction.STOP)

    def stop(self):
        try:
            self._send(DogCommand.ACTION, value=DogAction.STOP)
        except Exception:
            pass

    def close_continue(self):
        self._send(DogCommand.CONTINUE, value=2)

    def start_continue(self):
        self._send(DogCommand.CONTINUE, value=-1)
