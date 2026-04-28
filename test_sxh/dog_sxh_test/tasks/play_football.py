from dog_control_sxh_test import DogControl
import time
import cv2
import numpy as np

# HSV色彩空间的颜色范围配置
COLOR_RANGES = {
    'orange': {'lower': np.array([5, 100, 100]), 'upper': np.array([25, 255, 255])},
    # 'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
    # 'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
    # 'green': {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
    # 'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([40, 255, 255])}
}

# 检测参数配置（可根据实际情况调整）
CONFIG = {
    'min_area': 500,           # 最小检测面积，过滤噪点
    'target_area_min': 60000,   # 目标踢球距离的最小面积
    'target_area_max': 80000,  # 目标踢球距离的最大面积
    'center_threshold': 0.10,  # 中心对齐阈值（屏幕宽度的比例）
    # 'left_threshold': 0.35,    # 左侧边界阈值
    # 'right_threshold': 0.65,   # 右侧边界阈值
}


def play_football(robot, color='orange', show_video=False):
    """
    机器狗踢球任务函数

    参数:
        robot: DogControl实例
        color: 球的颜色，可选 'orange', 'red', 'blue', 'green', 'yellow'
        show_video: 是否显示视频窗口（默认False，避免无显示器系统的Qt/xcb错误）

    功能流程:
        1. 打开摄像头检测指定颜色的球
        2. 根据球的位置调整机器狗的位置（左右、前后）
        3. 当球在合适位置时执行踢球动作
        4. 完成后返回
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # if color not in COLOR_RANGES:
    #     print(f"警告: 未知颜色 '{color}'，使用默认橙色")
    #     color = 'orange'

    lower_color = COLOR_RANGES[color]['lower']
    upper_color = COLOR_RANGES[color]['upper']

    print(f"开始踢球任务，检测颜色: {color}")
    print("等待检测到球...")

    frame_cnt = 0
    fps = 0
    fps_start_time = time.time()
    kicked = False  # 是否已经踢球

    try:
        while not kicked:
            ret, frame = cap.read()
            if not ret:
                print("摄像头错误")
                break

            h, w, _ = frame.shape
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 颜色检测
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # 形态学，减少噪声
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours_tuple = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_tuple[-2]

            ball_detected = False
            ball_center_x = 0
            ball_center_y = 0
            ball_area = 0

            if contours:
                # 找到最大的轮廓（假设是球）
                max_cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_cnt)

                if area > CONFIG['min_area']:
                    ball_detected = True
                    x, y, w_box, h_box = cv2.boundingRect(max_cnt)
                    ball_center_x = x + w_box // 2
                    ball_center_y = y + h_box // 2
                    ball_area = w_box * h_box

                    # 绘制检测框和中心点
                    # if show_video:
                    #     cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    #     cv2.circle(frame, (ball_center_x, ball_center_y), 5, (0, 0, 255), -1)
                    #     cv2.putText(frame, f"Area: {ball_area}", (x, y - 10),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制屏幕中心参考线
            # if show_video:
            #     cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)
            #     cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)

            # FPS计算
            frame_cnt += 1
            if frame_cnt >= 10:
                fps = 10 / (time.time() - fps_start_time)
                frame_cnt = 0
                fps_start_time = time.time()

            # if show_video:
            #     cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
            #               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     cv2.putText(frame, f"Color: {color}", (10, 60),
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 球的位置判断和机器狗调整
            if ball_detected:
                # 计算球相对于屏幕中心的位置
                ball_x_ratio = ball_center_x / w

                print(f"球检测: 中心=({ball_center_x}, {ball_center_y}), 面积={ball_area}, X比例={ball_x_ratio:.2f}")

                # 显示当前状态
                # if show_video:
                #     cv2.putText(frame, "Ball Found!", (10, h - 10),
                #               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 左右对齐调整
                if ball_x_ratio < (0.9 - CONFIG['center_threshold']):
                    # 球在左侧，向左
                    print("-> 动作: 向左调整")
                    robot.move(vy=-25000, last_time=0.2, duration=1)
                    continue

                elif ball_x_ratio > (0.9 + CONFIG['center_threshold']):
                    # 球在右侧，向右
                    print("-> 动作: 向右调整")
                    robot.move(vy=25000, last_time=0.2, duration=1)
                    continue

                # 前后距离调整
                elif ball_area < CONFIG['target_area_min']:
                    # 球太远，向前
                    print("-> 动作: 向前移动（球太远）")
                    robot.move(vx=20000, last_time=0.2, duration=1)
                    continue

                elif ball_area > CONFIG['target_area_max']:
                    # 球太近，向后
                    print("-> 动作: 向后移动（球太近）")
                    robot.move(vx=-20000, last_time=0.2, duration=1)
                    continue

                # 球在合适位置，执行踢球动作
                else:
                    print("\n=== 球在最佳位置，准备踢球 ===")
                    print(f"最终参数: 中心X={ball_x_ratio}, 面积={ball_area}")

                    # if show_video:
                    #     cv2.putText(frame, "KICKING!", (w//2 - 80, h//2),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    #     cv2.imshow('Football Detection', frame)
                    #     cv2.waitKey(500)  # 显示半秒

                    # 执行踢球序列（参考历史代码track_ball的安全参数）
                    robot.start_continue()  # 开启持续运动模式
                    print("1. 向前冲刺踢球")
                    robot.move(vx=40000, last_time=2)  # 使用历史代码参数：vx=40000, last_time=2
                    time.sleep(1)  # 重要：让机器狗稳定！

                    print("2. 后退复位")
                    robot.move(vx=-20000, last_time=1.1)  # 使用历史代码参数：vx=-20000, last_time=1.1
                    time.sleep(1)  # 重要：让机器狗稳定！

                    # print("3. 侧移调整")
                    # robot.move(vy=-40000, last_time=0.4, duration=0.5)  # 使用历史代码参数

                    robot.close_continue()  # 关闭持续运动模式

                    print("\n踢球完成！")
                    kicked = True
                    break
            else:
                # 没有检测到球
                # if show_video:
                #     cv2.putText(frame, "Searching for ball...", (10, h - 10),
                #               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                # 每隔一段时间打印一次
                if frame_cnt % 30 == 0:
                    print("等待球")

            # 显示画面
            # if show_video:
            #     cv2.imshow('Football Detection', frame)
                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("中断")
                    break

    finally:
        # 清理资源
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        print("踢球任务结束")


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