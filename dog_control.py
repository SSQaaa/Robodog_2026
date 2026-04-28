import threading
import time
import cv2 as cv
import numpy as np
import multiprocessing
from pyzbar.pyzbar import decode

from udp import UDPClient
from detect_trt import Detect_image
from detect import Element_Detect
from sound_output import sound_output
from robot_arm import Robot_Arm

categories = ['1', '2', '3', '4', '5', '6', 'red_barrel', 'yellow_barrel', 'blue_barrel', 'orange_barrel',
              'red_ball', 'yellow_ball', 'blue_ball', 'orange_ball', 'dashboard', 'ssi' ,'yellow_cylinder' , 'red_cylinder']
dict_abnormal = {'yellow_barrel': 'yellow_ball', 'red_barrel': 'red_ball', 'orange_barrel': 'orange_ball',
                 'blue_barrel': 'blue_ball'}


class DogControl:
    def __init__(self):
        # 心脏包
        self.__udp_client = UDPClient('192.168.1.120', 43893)
        self.__heart_beat_thread = threading.Thread(target=self.__heart_beat)
        self.__heart_beat_thread.start()
        # 摄像头
        self._cap_running = True
        self.cap_index = 6
        self.cap_index_catch = 2
        self.frame = None
        self.cap_thread = threading.Thread(target=self._cap)
        self.cap_thread.start()
        # 元素检测
        self.Detect = Detect_image()  # 实例化yolov5检测
        self.Element_detect = Element_Detect()  # 元素检测结果处理
        self.abnormal = [] # 存放异常区域
        self.detect_list = []  # 记录当前纸箱上三个元素状态
        self.num = 0
        # 巡线矫正
        self.line_cnt = 0
        self.line_x = [[0, 0], [0, 0]]
        self.line_y = [[0, 0], [0, 0]]
        self.point = [0, 0]
        self.line_flag = True
        self.line_flag_1 = False
        #存放需要踢的球的颜色
        self.ball_classes = []
        #语音模块
        self.msg_output = sound_output()

        self.roboarm = Robot_Arm()

    def __del__(self):  # 析构函数 对象销毁时让狗趴下
        self.stand_up()

    def __heart_beat(self):
        while True:
            self.__udp_client.send(0x21040001, duration=0.2)

    def _cap(self):
        cap = cv.VideoCapture(self.cap_index)
        while self._cap_running:
            ret, self.frame = cap.read()
            # if ret:
            #     cv.imshow('x1',self.frame)
            #     cv.waitKey(1)
        cap.release()

    #用于抓取任务切换摄像头
    def switch_camera(self):
        # 关闭旧线程
        self._cap_running = False
        self.cap_thread.join()
        # 切换摄像头
        self.cap_index = self.cap_index_catch
        self._cap_running = True
        self.cap_thread = threading.Thread(target=self._cap)
        self.cap_thread.start()

    def stand_up(self):
        self.__udp_client.send(0x21010202, duration=3)

    """
    move：正常低步态
    EXmove：转换步态且转回低步态
    NEXmove：转换步态且不转回低步态
    控制前进后退和旋转
    Args:
        last_time (int): 发送持续时间
        vx (int, optional): 前后平移速度，死区 [-6553,6553]，正值向前
        vy (int, optional): 左右平移速度，死区 [-12553,12553]，正值向右
        vz (int, optional): 旋转角速度，死区 [-9553,9553] ，正值顺时针
        duration (int, optional): 发送后的间隔时间
    """
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

# ---------------------------------巡检任务--------------------------------------

    def detect_process(self):
        self.close_continue()
        while True:
            if self.frame is not None:
                """
                frame:返回的图像
                box：[[x1,y1,x2,y2],[]]此类型的数组，记录每一个目标的左上、右下坐标
                classid：检测出来的物体的对应下标，这个下表是Detect_TRT_new.py中的列表的顺序
                """
                frame_1 = self.frame.copy()
                frame, box, classid = self.Detect.detect_image(self.frame)
                self.detect_list = self.Element_detect.judge(frame_1, box, classid)
                self.abnormal = self.Element_detect.return_abnormal()
                # print(self.abnormal)

                cv.imshow('result_1', frame)
                cv.imshow('result_2', frame_1)
                cv.waitKey(1)

                if len(self.detect_list) == 3:
                    p = multiprocessing.Process(target=self.msg_output.output,
                                                args=(self.detect_list[2], self.detect_list[1], self.detect_list[0]))
                    p.start()
                    self.num+=1
                    if self.detect_list[0] == 'normal':
                        # self.msg_output.output(self.detect_list[2],self.detect_list[1],self.detect_list[0])
                        self.shake_head()
                    else:
                        # self.msg_output.output(self.detect_list[2],self.detect_list[1],self.detect_list[0])
                        self.nod_head()

                    self.start_continue()
                    break

    # 仪表盘校正
    def dashboard_correction(self):
        while True:
            if self.frame is not None:
                h, w, channels = self.frame.shape
                index_dashboard = None
                flag_dashboard_correction = 0
                frame, box, classid = self.Detect.detect_image(self.frame)

                for i in range(len(classid)):
                    if classid[i] == 14:
                        index_dashboard = i
                        break

                if index_dashboard != None:
                    self.start_continue()
                    middle_x = int((box[index_dashboard][0] + box[index_dashboard][2]) / 2)
                    if middle_x - w // 2 < -80:
                        self.move(vy=-40000, last_time=0.2)  # 向左
                        time.sleep(0.5)
                        print("left")
                    elif middle_x - w // 2 > 80:
                        self.move(vy=40000, last_time=0.2)  # 向右
                        time.sleep(0.5)
                        print("right")
                    else:
                        flag_dashboard_correction = 1
                        self.close_continue()

                    if flag_dashboard_correction:
                        dashboard_area = int(abs(int(box[index_dashboard][0]) - int(box[index_dashboard][2])) * abs(
                            int(box[index_dashboard][1]) - int(box[index_dashboard][3])))
                        print(dashboard_area)
                        if self.num == 2:
                            if dashboard_area <= 12000:  # 仪表盘面积过小
                                self.start_continue()
                                self.move(vx=15000, last_time=0.15, duration=0.5)
                            elif dashboard_area >= 18000:  # 仪表盘面积过大
                                self.start_continue()
                                self.move(vx=-9000, last_time=0.1)
                                time.sleep(0.5)
                            else:
                                self.close_continue()
                                break  
                        else:  
                            if dashboard_area <= 11000:  # 仪表盘面积过小
                                self.start_continue()
                                self.move(vx=15000, last_time=0.15, duration=0.5)
                            elif dashboard_area >= 18000:  # 仪表盘面积过大
                                self.start_continue()
                                self.move(vx=-9000, last_time=0.1)
                                time.sleep(0.5)
                            else:
                                self.close_continue()
                                break

    # 做延长线
    def extend_line(self, x1, y1, x2, y2, w, h):
        if x1 == x2:  # 垂直线
            return x1, 0, x2, h
        elif y1 == y2:  # 水平线
            return 0, y1, w, y2
        else:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            # 与左边界交点
            x_left = 0
            y_left = int(b)
            # 与右边界交点
            x_right = w
            y_right = int(k * w + b)
            # 与上边界交点
            y_top = 0
            x_top = int(-b / k)
            # 与下边界交点
            y_bottom = h
            x_bottom = int((h - b) / k)
            # 选取在图像范围内的两个点
            points = []
            if 0 <= y_left <= h:
                points.append((x_left, y_left))
            if 0 <= y_right <= h:
                points.append((x_right, y_right))
            if 0 <= x_top <= w:
                points.append((x_top, y_top))
            if 0 <= x_bottom <= w:
                points.append((x_bottom, y_bottom))
            if len(points) >= 2:
                return points[0][0], points[0][1], points[1][0], points[1][1]
            else:
                return x1, y1, x2, y2  # 原线段

    #巡线
    def line_process(self):
        while self.line_flag:
            if self.frame is not None:
                h, w, channels = self.frame.shape
                frame = self.frame[h // 4:, :]
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                _, gray = cv.threshold(gray, 160, 255, cv.THRESH_BINARY) #170
                kernel = np.ones((5, 5), np.uint8)
                # gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
                gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
                # cv.imshow('gray',gray)
                edges = cv.Canny(gray, 50, 150, apertureSize=3)
                # cv.imshow('edge',edges)
                lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
                line_image_1 = np.copy(frame)
                flag_vertical = False

                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        x1, y1, x2, y2 = self.extend_line(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
                        if y1 >= y2:
                            bottom_x = x1
                        else:
                            bottom_x = x2

                        if abs(y2 - y1) >= abs(x2 - x1) and not flag_vertical:
                            vertical_line_x = bottom_x
                            cv.line(line_image_1, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            flag_vertical = True

                        if flag_vertical:
                            if vertical_line_x < int(0.45 * w):
                                self.move(vy=-20000, last_time=0.2)
                                time.sleep(0.5)
                                print('left')
                            elif vertical_line_x > int(0.55 * w):
                                self.move(vy=20000, last_time=0.2)
                                time.sleep(0.5)
                                print('right')
                            else:
                                print('ok')
                                self.line_flag = False
                cv.imshow('edge', edges)
                cv.imshow('1', line_image_1)
                cv.waitKey(1)


# ---------------------------------救援任务--------------------------------------

    # 踢球
    def track_ball(self):
        while True:
            if self.frame is not None:
                classid_ball = [10, 11, 12, 13]
                h, w, channels = self.frame.shape
                flag_ball = False
                idx_ball = -1
                correction_index = -1
                frame, box, classid = self.Detect.detect_image(self.frame)
                cv.imshow('1', frame)
                cv.waitKey(1)

                for color in self.abnormal:
                    self.ball_classes.append(dict_abnormal[color])

                for index, val in enumerate(classid):
                    if int(val) in classid_ball:
                        correction_index = index
                        if categories[int(val)] in self.ball_classes:
                            flag_ball = True

                if correction_index != -1:
                    ball_center_x = (box[correction_index][0] + box[correction_index][2]) // 2
                    ball_area = (abs(int(box[correction_index][0]) - int(box[correction_index][2])) *
                                 abs(int(box[correction_index][1]) - int(box[correction_index][3])))
                    print(f'ball_area={ball_area}')
                    print(f'center={ball_center_x}')
                    # 左右校正
                    if flag_ball and ball_center_x < int(0.3 * w):
                        print('abnormal_left')
                        self.move(vy=-25000, last_time=0.2, duration=1)
                        continue
                    elif flag_ball and ball_center_x > int(0.36 * w):
                        print('abnormal_right')
                        self.move(vy=25000, last_time=0.2, duration=1)
                        continue
                    # 左右校正
                    elif not flag_ball and ball_center_x > int(0.7 * w):
                        print('normal_right')
                        self.move(vy=40000, last_time=0.3, duration=0.5)
                        continue
                    elif not flag_ball and ball_center_x < int(0.3 * w):
                        print('normal_left')
                        self.move(vy=-40000, last_time=0.3, duration=0.5)
                        continue
                        # 前后
                    elif ball_area < 9000:
                        print('forward')
                        self.move(vx=20000, last_time=0.2, duration=1)
                        continue
                    elif ball_area > 18000:
                        print('back')
                        self.move(vx=-20000, last_time=0.2, duration=1)
                        continue
                    elif flag_ball:
                        self.start_continue()
                        print('success_forword')
                        self.move(vx=40000, last_time=2)
                        time.sleep(1)
                        self.move(vx=-20000, last_time=1.1)
                        time.sleep(1)
                        self.move(vy=-40000,last_time=0.4,duration=0.5)
                        self.close_continue()
                        break
                    else:
                       print('normal ball')
                       break 

                else:
                    self.EXmove(vx=-10000, last_time=0.15, duration=1.5, case=1)
                    print('no goal')

                # for idx, val_ in enumerate(classid):
                #     if categories[int(val_)] in self.ball_classes:
                #         idx_ball = idx
                #         break

                # if idx_ball != -1:
                #     self.start_continue()
                #     print('success_forword')
                #     self.move(vx=40000, last_time=2)
                #     time.sleep(1)
                #     self.EXmove(vx=-20000, last_time=0.7, case=1)
                #     time.sleep(1)
                #     self.close_continue()
                #     break
                # elif idx_ball == -1:
                #     print('normal ball')
                    

#---------------------------------抓取任务--------------------------------------

    def decode_image(self, image):
        center = None
        barcode_area = 0
        barcodes = decode(image)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            center = (x + w // 2, y + h // 2)
            barcode_area = w * h
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type
            print(f" Center: {center}, Area: {barcode_area}")
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv.imshow('x',image)
        # cv.waitKey(1)

        return image, center, barcode_area

    def qr_code_correction(self):
        # self.switch_camera()
        while True:
            if self.frame is not None:
                flag_qr_code_correction = 0
                h, w, channels = self.frame.shape
                frame, qr_code_center, qr_code_area = self.decode_image(self.frame)
                print(qr_code_area)
                if qr_code_area != 0:
                    self.start_continue()
                    if qr_code_center[0] - w // 2 < -80:
                        self.move(vy=-40000, last_time=0.2)  # 向左
                        time.sleep(1)
                        print("qr_left")
                    elif qr_code_center[0] - w // 2 > 80:
                        self.move(vy=40000, last_time=0.2)  # 向右
                        time.sleep(1)
                        print("qr_right")
                    else:
                        flag_qr_code_correction = 1
                        self.close_continue()

                if flag_qr_code_correction:
                    # print(qr_code_area)
                    if qr_code_area <= 4000:  # 仪表盘面积过小
                        self.start_continue()
                        self.move(vx=15000, last_time=0.15, duration=0.5)
                    elif qr_code_area >= 7500:  # 仪表盘面积过大
                        self.start_continue()
                        self.move(vx=-9000, last_time=0.1)
                    else:
                        self.close_continue()
                        break

    def cylinder_correction(self):
        while True:
            if self.frame is not None:
                flag_cylinder_correction = 1
                h, w, channels = self.frame.shape
                # x ,y , area = self.roboarm.detect_red_cylinder(self.frame)
                frame,box,classid = self.Detect.detect_image(self.frame)
                for idx,val in enumerate(classid):
                    if int(val) == 17:
                        area = (abs(int(box[idx][0]) - int(box[idx][2])) *
                                 abs(int(box[idx][1]) - int(box[idx][3])))
                        x = (box[idx][0]+box[idx][2]//2)
                if area != 0:
                    self.start_continue()
                    if x - w // 2 < -60:
                        self.move(vy=-20000, last_time=0.15)  # 向左
                        time.sleep(0.6)
                        print("cy_left")
                        continue
                    elif x - w // 2 > 60:
                        self.move(vy=18000, last_time=0.15)  # 向右
                        time.sleep(0.6)
                        print("cy_right")
                        continue
                    else:
                        flag_cylinder_correction = 1
                        self.close_continue()

                if flag_cylinder_correction:
                    print(area)
                    if area <= 27500:  # 圆柱面积过小
                        self.start_continue()
                        self.move(vx=8000, last_time=0.12, duration=0.5)
                        continue
                    # elif area >= 30000 :
                    #     self.move(vx=-8000, last_time=0.1, duration=0.5)
                    #     continue
                    else:
                        self.close_continue()
                        time.sleep(1)
                        break
                




    def Process_One(self):
        self.roboarm.catch_start()
        self.detect_process()
        time.sleep(0.5)
        # 第二个
        self.NEXmove(vy=-150000, last_time=1, case = 1)
        time.sleep(0.5)
        self.NEXmove(vx=11000, last_time=1.4, case = 1)
        self.__udp_client.send(0x21010300)
        time.sleep(0.5)
        self.dashboard_correction()
        time.sleep(0.5)
        self.detect_process()
        time.sleep(0.5)
        # 第三个
        self.NEXmove(vy=-150000, last_time=1, case = 1)
        time.sleep(0.5)
        self.NEXmove(vx=11000, last_time=1, case = 1)
        self.__udp_client.send(0x21010300)
        time.sleep(0.5)
        self.dashboard_correction()
        time.sleep(0.5)
        self.detect_process()
        time.sleep(0.5)
        # 巡线
        self.EXmove(vy=-150000, last_time=0.9, case=1)
        time.sleep(0.5)
        self.line_process()
        time.sleep(0.5)
        self.EXmove(vx=12000, last_time=1.8, case=2)
        time.sleep(0.5)
        self.revolve_180()
        time.sleep(0.5)
        #第四个
        self.EXmove(vy=-150000, last_time=0.7, case=1)
        self.dashboard_correction()  # 使用仪表盘校正
        self.detect_process()
        #第五个
        self.NEXmove(vy=-150000, last_time=1, case = 1)
        time.sleep(0.5)
        self.NEXmove(vx=11000, last_time=1, case = 1)
        self.__udp_client.send(0x21010300)
        time.sleep(0.5)
        self.dashboard_correction()
        self.detect_process()
        #第六个
        self.NEXmove(vy=-150000, last_time=0.85, case = 1)
        time.sleep(0.5)
        self.NEXmove(vx=11000, last_time=0.9, case = 1)
        self.__udp_client.send(0x21010300)
        time.sleep(0.5)
        self.dashboard_correction()
        self.detect_process()
        #转身
        self.revolve_90()
        self.start_continue()
        self.EXmove(vy=180000, last_time=2.55, case=2)
        time.sleep(0.5)
        #走到第二部分起点
        print("turn_2")
        self.EXmove(vx=11000, last_time=0.85,case=1)
        time.sleep(0.5)
        self.revolve_90()

    def Process_Two(self):
        self.line_flag = True
        self.line_process()
        time.sleep(1)
        self.EXmove(vx=20000,last_time=1.2, case = 1)
        # 踢第一个球
        self.EXmove(vy=150000, last_time=2.3, case = 2)
        time.sleep(1)
        self.track_ball()
        # 第二个
        self.revolve_180()
        self.track_ball()
        self.EXmove(vy=150000, last_time=4, case = 2)
        time.sleep(1)
        # 第三个
        self.track_ball()
        # 第四个
        self.revolve_180()
        self.track_ball()
        # 回到原点
        self.EXmove(vy=150000, last_time=2.4, case = 2)

    def Process_Three(self):
        self.EXmove(vx=20000, last_time=1.1, duration=1, case = 1)
        self.move(vy=50000, last_time=1, duration=1)
        self.roboarm.catch_init()
        self.qr_code_correction()
        self.cylinder_correction()
        self.roboarm.catch_cy()
        self.move(vy=-150000, last_time=3.35,duration=0.5)
        self.start_continue()
        self.move(vx=8000, last_time=0.15, duration=1)
        self.close_continue()
        self.roboarm.put_cy()
        # self.roboarm.catch_init()


if __name__ == '__main__':
    control = DogControl()
    time.sleep(1)
    control.close_continue()
    control.stop()
    time.sleep(0.5)
    control.stand_up()
    time.sleep(0.5)
    control.Process_One()
    control.Process_Two()
    control.Process_Three()

    control.close_continue()
    control.stop()
    time.sleep(2)
