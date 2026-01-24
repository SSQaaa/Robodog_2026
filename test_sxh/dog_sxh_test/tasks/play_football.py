from dog_control_sxh_test import DogControl
import time
import cv2
import numpy as np
import time

def color_detect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_cnt = 0
    fps = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera Error")
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([5, 100, 100])
        upper_blue = np.array([15, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours[-2]

        if cnts:
            max_cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 500:
                x, y, w, h = cv2.boundingRect(max_cnt)
                print(x + w // 2, y + h // 2, "Area:", w * h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_cnt += 1

        if frame_cnt == 5:
            end_time = time.time()
            fps = 1 / 5 / (end_time - start_time)
            frame_cnt = 0

        cv2.putText(frame, f"FPS: {int(fps)}", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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