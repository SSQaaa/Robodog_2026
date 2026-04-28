# -*- coding: utf-8 -*-
import sys
import time
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 明确导入所需的类
from scservo_sdk.port_handler import PortHandler
from scservo_sdk.sms_sts import sms_sts

# 配置参数
BAUDRATE = 500000
DEVICENAME = '/dev/ttyUSB0'  # 请根据实际情况修改

# 初始化
portHandler = PortHandler(DEVICENAME)
packetHandler = sms_sts(portHandler)

# 打开串口
if not portHandler.openPort():
    print("serial failed")
    exit()

# 设置波特率
if not portHandler.setBaudRate(BAUDRATE):
    print("setBaudRate")
    exit()

print("Succeed init, start to control")

# 控制舵机 - 例如让3号舵机运动
# 参数：舵机ID, 目标位置(0-4095), 速度, 加速度
packetHandler.WritePosEx(3, 2047, 1500, 50)  # 3号舵机到中位

time.sleep(2)

# 关闭串口
portHandler.closePort()
print("control finish")