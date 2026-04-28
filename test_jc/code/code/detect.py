# -*- coding: utf-8 -*-

import math
import cv2 as cv
import numpy as np
class Element_Detect:
    def __init__(self):
        self.direction = 0
        self.center = [0, 0]
        self.middle = [0, 0] #ssi
        self.rect = [0, 0]

        self.temp_boardState = None
        self.temp_number = None
        self.temp_barrel = None

        self.flag_number = 0
        self.flag_barrel = 0
        self.flag_board = 0
        self.flag_process = 0

        self.board_state = {"low":0 , "high":0 , "normal":0}
        self.barrel_color = {"red_barrel":0, "yellow_barrel":0, "blue_barrel":0, "orange_barrel":0}
        self.number = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0 }
        self.category = ['1', '2', '3', '4', '5', '6',
                         'red_barrel','yellow_barrel','blue_barrel','orange_barrel',
                         'red_ball', 'yellow_ball', 'blue_ball', 'orange_ball',
                         'dashboard', 'ssi']
        self.list_all = [[],[],[],
                        [],[],[]]
        self.list_state = []    # 存放异常纸箱的index
        self.abnormal_color = []

    def judge(self,frame,box, classid):
        if frame is not None and len(classid) != 0:
            for i in range(len(classid)):
                # 对应数字1-6
                if 0 <= classid[i] <= 5 and self.flag_number == 0:
                    self.number[self.category[int(classid[i])]] += 1
                    for index in self.number.keys():
                        if self.number[index] >= 10:
                            self.temp_number = index
                            self.flag_number = 1
                    print(f'number={self.flag_number}')
                # 对应锥桶
                elif 6 <= classid[i] <= 9 and self.flag_barrel == 0:
                    self.barrel_color[self.category[int(classid[i])]] += 1
                    for index in self.barrel_color.keys():
                        if self.barrel_color[index] >= 10:
                            self.temp_barrel = index
                            self.flag_barrel = 1
                # 对应仪表盘状态
                elif classid[i] == 14 and (15 in classid) and self.flag_board == 0:
                    self.center[0] = (box[i][0] + box[i][2]) // 2
                    self.center[1] = (box[i][1] + box[i][3]) // 2
                    for k in range(len(classid)):   #ssi
                        if classid[k] == 15:
                            self.middle[0] = (box[k][0] + box[k][2]) // 2
                            self.middle[1] = (box[k][1] + box[k][3]) // 2
                    state = None
                    if self.find_niddle(frame,box[i]):
                        state = self.Get_dashboardstate()
                        print(f'state={state}')
                    if state != None:
                        self.board_state[state] += 1
                        print(self.board_state)
                    for index in self.board_state.keys():
                        if self.board_state[index] >= 10:
                            self.temp_boardState = index
                            self.flag_board = 1

            if (self.flag_number and self.flag_barrel and self.flag_board):
                self.list_all[self.flag_process].append(self.temp_boardState)
                self.list_all[self.flag_process].append(self.temp_number)
                self.list_all[self.flag_process].append(self.temp_barrel)
                self.clear()

            if len(self.list_all[self.flag_process]) == 3:
                if self.list_all[self.flag_process][0] != 'normal':
                    self.list_state.append(self.flag_process)
                self.flag_process += 1
                return self.list_all[self.flag_process - 1]
            else:
                return []
        
        else:
            return []
    def return_abnormal(self):
        for num in self.list_state:
            if self.list_all[num][2] not in self.abnormal_color:
                self.abnormal_color.append(self.list_all[num][2])
        return self.abnormal_color
    def clear(self):
        self.flag_number = 0
        self.flag_barrel = 0
        self.flag_board = 0
        self.temp_boardState = None
        self.temp_number = None
        self.temp_barrel = None
        for index in self.number.keys():
            self.number[index] = 0
        for index in self.barrel_color.keys():
            self.barrel_color[index] = 0
        for index in self.board_state.keys():
            self.board_state[index] = 0

    def clamp(self ,value , min_val, max_val):
        return max(min_val,min(value,max_val))

    def refine_box(self,bbox,w,h):
        bbox[0] = self.clamp(bbox[0],0,w)
        bbox[1] = self.clamp(bbox[1],0,h)
        bbox[2] = self.clamp(bbox[2],0,w)
        bbox[3] = self.clamp(bbox[3],0,h)
        center_x = (bbox[0]+bbox[2])//2
        center_y = (bbox[1]+bbox[3])//2
        bbox[0] = center_x + (bbox[0]-center_x)*0.5
        bbox[1] = center_y + (bbox[1]-center_y)*0.5
        bbox[2] = center_x + (bbox[2]-center_x)*0.5
        bbox[3] = center_y + (bbox[3]-center_y)*0.5
            
        return bbox

    def find_niddle(self,frame,bbox):
        h,w,rgb = frame.shape

        bbox = self.refine_box(bbox,w,h)
        bbox = bbox.astype(np.int32)
        image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if image is not None:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 110, 255, cv.THRESH_BINARY_INV)
            cont_image,contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            max_area = 0
            max_contour = None
            for c in contours:
                area = cv.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_contour = c

            # 如果找到了最大轮廓，绘制它并计算它的角度
            if max_contour is not None:
                cv.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
                rect = cv.minAreaRect(max_contour)
                self.rect = [int(rect[0][0]) + bbox[0], int(rect[0][1]) + bbox[1]]
                return 1
            else:
                return 0
        else:
            return 0

    def Get_dashboardstate(self):
        v1_x = self.middle[0] - self.center[0]
        v1_y = self.middle[1] - self.center[1]
        v2_x = self.rect[0] - self.center[0]
        v2_y = self.rect[1] - self.center[1]
        try:
            angle_ = math.degrees(math.acos(
                (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))
            ))
        except:
            angle_ = 65545.

        if v1_x * v2_y - v2_x * v1_y > 0:
            # 指针在基准线顺时针方向
            self.direction = -1
        else:
            self.direction = 1

        if 120 <= angle_ <= 180:
            return "normal"
        else:
            if self.direction == -1:
                return "low"
            elif self.direction == 1:
                return "high"

