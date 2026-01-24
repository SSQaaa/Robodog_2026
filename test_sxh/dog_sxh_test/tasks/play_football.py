from dog_control_sxh_test import DogControl
import time

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
                    self.move(vy=-40000 ,last_time=0.4 ,duration=0.5)
                    self.close_continue()
                    break
                else:
                    print('normal ball')
                    break

            else:
                self.EXmove(vx=-10000, last_time=0.15, duration=1.5, case=1)
                print('no goal')