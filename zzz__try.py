import time
from scservo_sdk import *

SCS_ID_1 = 1
BAUDRATE = 500000
DEVICENAME = '/dev/ttyUSB0'
SCS_MOVING_SPEED = 1500
SCS_MOVING_ACC = 50

portHandler = PortHandler(DEVICENAME)
packetHandler = sms_sts(portHandler)

# 打开串口
if not portHandler.openPort():
    print("❌ 串口打开失败")
    exit()
if not portHandler.setBaudRate(BAUDRATE):
    print("❌ 波特率设置失败")
    exit()

# 测试1号舵机：张开 → 闭合
print("✅ 测试1号舵机张开")
packetHandler.WritePosEx(SCS_ID_1, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
time.sleep(2)

print("✅ 测试1号舵机闭合")
packetHandler.WritePosEx(SCS_ID_1, 2400, SCS_MOVING_SPEED, SCS_MOVING_ACC)
time.sleep(2)

portHandler.closePort()