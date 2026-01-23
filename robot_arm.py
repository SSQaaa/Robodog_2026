# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scservo_sdk import *                 # Uses SCServo SDK library
# from three_Inverse_kinematics import Arm

# 舵机编号，抓手为1，底盘为6
SCS_ID_1                      = 1                 # SCServo ID : 1  抓手开合
SCS_ID_2                      = 2                 # SCServo ID : 2  抓收旋转
SCS_ID_3                      = 3                 # SCServo ID : 3  第三连杆
SCS_ID_4                      = 4                 # SCServo ID : 4  第二连杆
SCS_ID_5                      = 5                 # SCServo ID : 5  第一连杆
SCS_ID_6                      = 6                 # SCServo ID : 6  控制整个机械臂旋转

# 舵机旋转角度数值，以2047为中间值
# 由于4、5号舵机连接结构与 3号的不同,
#    3号:角度为正数 逆时针旋转 数值减小 -- 角度为负数 顺时针旋转 数值增大。
# 4、5号:角度为正数 逆时针旋转 数值增大 -- 角度为负数 顺时针旋转 数值减小。
SCS_1_INIT_VALUE  = 2400             # 抓手初始状态 闭合。闭合最大数值：2450
SCS_1_STATUS_VALUE  = 2047           # 抓手张开。最大张开角度数值：1600；张开-->小    闭合-->大

SCS_2_INIT_VALUE  = 2047             # 抓手初始旋转状态
SCS_2_STATUS_VALUE  = 2047           # 抓手旋转值，2047中间值水平，大-->逆时针   小-->顺时针 取值范围：0~4095

SCS_3_INIT_VALUE  = 3080             # 3关节 初始状态
SCS_3_STATUS_VALUE  = 2800           # 3关节 2047中间值与前臂水平，顺-->大   逆-->小  取值范围尽可能在：3060~1060
SCS_3_MOVE_VALUE = 1070              # 运动姿态，使用机械臂上得摄像头
SCS_3_TRANSPORT1_VALUE = 3060        # 运输姿态1：药瓶水平
SCS_3_TRANSPORT2_VALUE = 2940        # 运输姿态2：药瓶垂直


SCS_4_INIT_VALUE  = 800              # 4关节 初始状态
SCS_4_STATUS_VALUE  = 1100           # 4关节 2047中间值与前臂水平，顺-->小   逆-->大  取值范围尽可能在：1060~3060
SCS_4_MOVE_VALUE = 540               # 运动姿态，使用机械臂上得摄像头
SCS_4_TRANSPORT1_VALUE = 1024        # 运输姿态1：药瓶水平
SCS_4_TRANSPORT2_VALUE = 1430        # 运输姿态2：药瓶垂直

SCS_5_INIT_VALUE  = 2400             # 5关节 初始状态
SCS_5_STATUS_VALUE  = 3030           # 5关节 2047中间值与前臂水平，顺-->小   逆-->大  取值范围尽可能在：1060~3060
SCS_5_MOVE_VALUE = 1540              # 运动姿态，使用机械臂上得摄像头
SCS_5_TRANSPORT1_VALUE = 2200        # 运输姿态1：药瓶水平
SCS_5_TRANSPORT2_VALUE = 2540        # 运输姿态2：药瓶垂直

SCS_6_INIT_VALUE  = 2047             # 机械臂整体旋转初始状态：正前方
SCS_6_STATUS_VALUE  = 2047           # 2047中间值，顺时针(右)-->大     逆时针(左)-->小     取值范围：0~4095

SCS_MOVING_SPEED       = 1500        # SCServo moving speed 旋转速度
SCS_MOVING_ACC         = 50          # SCServo moving acc   旋转加速度
class Robot_Arm:
    def __init__(self):
        self.M = None

        self.BAUDRATE = 500000
        self.DEVICENAME = '/dev/ttyUSB0' #Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = sms_sts(self.portHandler)

    def detect_red_cylinder(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([156, 43, 46])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        processed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)


        contours= cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]


        area = 0
        center = (0, 0)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
                return cx,cy,area
            else:
                return None,None,0
        
    def catch(self, SCS_1_INIT_VALUE, SCS_2_INIT_VALUE, SCS_3_INIT_VALUE,
                SCS_4_INIT_VALUE, SCS_5_INIT_VALUE, SCS_6_INIT_VALUE):
        scs_comm_result, scs_error =  self.packetHandler.WritePosEx(SCS_ID_1, SCS_1_INIT_VALUE, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error =  self.packetHandler.WritePosEx(SCS_ID_2, SCS_2_INIT_VALUE, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error =  self.packetHandler.WritePosEx(SCS_ID_3, SCS_3_INIT_VALUE, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error =  self.packetHandler.WritePosEx(SCS_ID_4, SCS_4_INIT_VALUE, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error =  self.packetHandler.WritePosEx(SCS_ID_5, SCS_5_INIT_VALUE, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error =  self.packetHandler.WritePosEx(SCS_ID_6, SCS_6_INIT_VALUE, SCS_MOVING_SPEED, SCS_MOVING_ACC)

    def catch_init(self):
        flag_port = 0
        flag_baudrate = 0

        if  self.portHandler.openPort():
            print("Succeeded to open the port")
            flag_port = 1
        else:
            print("Failed to open the port")

        if  self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
            flag_baudrate = 1
        else:
            print("Failed to change the baudrate")

        if flag_port and flag_baudrate:
            self.catch(1413,2047,2940,1430,2540,2047)
            time.sleep(1)
            # self.catch(2300,2050,3180,707,3037,2047)
    def put_cy(self):
        flag_port = 0
        flag_baudrate = 0

        if  self.portHandler.openPort():
            print("Succeeded to open the port")
            flag_port = 1
        else:
            print("Failed to open the port")

        if  self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
            flag_baudrate = 1
        else:
            print("Failed to change the baudrate")

        if flag_port and flag_baudrate:

            self.catch(2300,2044,1985,1759,1269,2047)
            time.sleep(2)
            self.catch(1500,2044,1985,1759,1269,2047)

    def catch_cy(self):
        flag_port = 0
        flag_baudrate = 0

        if  self.portHandler.openPort():
            print("Succeeded to open the port")
            flag_port = 1
        else:
            print("Failed to open the port")

        if  self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
            flag_baudrate = 1
        else:
            print("Failed to change the baudrate")

        if flag_port and flag_baudrate:
            self.catch(1500,2044,1450,1420,1140,1970)
            time.sleep(2)
            self.catch(2300,2044,1450,1420,1140,1970)
            time.sleep(2)
            self.catch(2300,2086,2012,1072,2626,2047)
            time.sleep(1)
    def catch_start(self):
        flag_port = 0
        flag_baudrate = 0

        if  self.portHandler.openPort():
            print("Succeeded to open the port")
            flag_port = 1
        else:
            print("Failed to open the port")

        if  self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
            flag_baudrate = 1
        else:
            print("Failed to change the baudrate")

        if flag_port and flag_baudrate:
            self.catch(1615,2053,3000,524,3050,2047)
            time.sleep(1)

if __name__ == '__main__' :
    arm = Robot_Arm()
    arm.catch_init()
    arm.catch_start()
    # arm.catch_cy()
    time.sleep(1)
    # arm.catch_init()
    
 
