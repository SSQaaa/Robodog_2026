# -*- coding: utf-8 -*-

import threading
import time
import socket
import struct
import os

class UDPClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.send_addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)

    def __del__(self):
        self.sock.close()

    """
    发送 UDP 指令
    Args:
        code (int): 指令码
        value (int, optional): 指令值.
        type (int, optional): 指令类型..
        last_time (int, optional): 发送持续时间.
        duration (int, optional): 发送后的间隔时间.
    """
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




