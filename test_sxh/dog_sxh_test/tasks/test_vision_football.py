import time
import cv2
import numpy as np

# HSV色彩空间的颜色范围配置
COLOR_RANGES = {
    'orange': {'lower': np.array([5, 100, 100]), 'upper': np.array([25, 255, 255])},
}

# 检测参数配置（与play_football.py完全一致）
CONFIG = {
    'min_area': 500,           # 最小检测面积，过滤噪点
    'target_area_min': 60000,   # 目标踢球距离的最小面积
    'target_area_max': 80000,  # 目标踢球距离的最大面积
    'center_threshold': 0.10,  # 中心对齐阈值（屏幕宽度的比例）
    # 'left_threshold': 0.35,    # 左侧边界阈值
    # 'right_threshold': 0.65,   # 右侧边界阈值
}


def test_vision_football(color='orange', show_video=False):
    """
    纯视觉测试函数 - 不包含机器狗运动，只测试视觉识别

    功能:
        - 检测指定颜色的球
        - 计算球的位置和面积
        - 打印详细的决策信息（应该如何移动机器狗）
        - 可选显示实时视频流

    参数:
        color: 球的颜色 ('orange', 'red', 'blue', 'green', 'yellow')
        show_video: 是否显示视频窗口（默认False，避免无显示器系统的Qt/xcb错误）
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # if color not in COLOR_RANGES:
    #     print(f"警告未知颜色 '{color}'，使用默认橙色")
    #     color = 'orange'

    lower_color = COLOR_RANGES[color]['lower']
    upper_color = COLOR_RANGES[color]['upper']

    print("\n" + "="*70)
    print(f"视觉测试开始 - 检测颜色: {color}")
    # print(f"HSV范围: Lower={lower_color}, Upper={upper_color}")
    # print(f"检测配置:")
    # print(f"  - 最小面积过滤: {CONFIG['min_area']}")
    # print(f"  - 目标面积范围: {CONFIG['target_area_min']} ~ {CONFIG['target_area_max']}")
    # print(f"  - 中心对齐阈值: ±{CONFIG['center_threshold']*100}%")
    # print("按 'q' 键退出测试")
    print("="*70 + "\n")

    frame_cnt = 0
    fps = 0
    fps_start_time = time.time()
    kicked = False  # 模拟踢球完成标志
    action_count = 0  # 动作计数

    try:
        while not kicked:
            ret, frame = cap.read()
            if not ret:
                print("[错误] 摄像头读取失败")
                break

            h, w, _ = frame.shape
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 颜色检测
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # 形态学操作，减少噪声
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
                    if show_video:
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        cv2.circle(frame, (ball_center_x, ball_center_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"Area: {ball_area}", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制屏幕中心参考线和区域标记
            # if show_video:
            #     # 中心十字线
            #     cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)
            #     cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 1)
            #
            #     # 对齐区域标记（中心区域）
            #     center_left = int(w * (0.5 - CONFIG['center_threshold']))
            #     center_right = int(w * (0.5 + CONFIG['center_threshold']))
            #     cv2.line(frame, (center_left, 0), (center_left, h), (0, 255, 255), 1)
            #     cv2.line(frame, (center_right, 0), (center_right, h), (0, 255, 255), 1)
            #     cv2.putText(frame, "Center Zone", (center_left + 5, 20),
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

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

            # 球的位置判断和决策打印
            if ball_detected:
                # 计算球相对于屏幕中心的位置
                ball_x_ratio = ball_center_x / w
                ball_y_ratio = ball_center_y / h
                # center_offset_x = ball_center_x - w//2
                # center_offset_y = ball_center_y - h//2

                print("\n" + "-"*70)
                print(f"[检测信息] 帧#{action_count}")
                print(f"  球中心坐标: ({ball_center_x}, {ball_center_y})")
                print(f"  球面积: {ball_area} ")
                print(f"  屏幕尺寸: {w}x{h}")
                print(f"  水平位置: {ball_x_ratio:.3f} ")
                print(f"  垂直位置: {ball_y_ratio:.3f} ")
                # print(f"  中心偏移: X={center_offset_x:+d}px, Y={center_offset_y:+d}px")

                # 显示当前状态
                # if show_video:
                #     cv2.putText(frame, "Ball Found!", (10, h - 10),
                #               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #     cv2.putText(frame, f"Pos: {ball_x_ratio:.2f}", (10, h - 40),
                #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 决策逻辑
                action_count += 1

                # 左右对齐判断
                if ball_x_ratio < (0.9 - CONFIG['center_threshold']):
                    # 球在左侧，需要向左转
                    print(f"\n[决策] 球在左侧 (X比例={ball_x_ratio:.3f} < {0.9-CONFIG['center_threshold']:.3f})")
                    # print(f"[动作] 应执行: robot.move(vy=-25000, last_time=0.2, duration=1)")
                    # print(f"  -> 含义: 向左平移（vy负值=左移），速度25000，持续0.2秒，之后等待1秒")
                    # print(f"  -> 目的: 让球进入中心对齐区域")
                    # if show_video:
                    #     cv2.putText(frame, "Action: TURN LEFT", (w//2 - 120, h//2 + 50),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    #     cv2.arrowedLine(frame, (w//2, h//2), (w//2 - 80, h//2), (0, 255, 255), 3)

                elif ball_x_ratio > (0.9 + CONFIG['center_threshold']):
                    # 球在右侧，需要向右转
                    print(f"\n[决策] 球在右侧 (X比例={ball_x_ratio:.3f} > {0.9+CONFIG['center_threshold']:.3f})")
                    # print(f"[动作] 应执行: robot.move(vy=25000, last_time=0.2, duration=1)")
                    # print(f"  -> 含义: 向右平移（vy正值=右移），速度25000，持续0.2秒，之后等待1秒")
                    # print(f"  -> 目的: 让球进入中心对齐区域")
                    # if show_video:
                    #     cv2.putText(frame, "Action: TURN RIGHT", (w//2 - 120, h//2 + 50),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    #     cv2.arrowedLine(frame, (w//2, h//2), (w//2 + 80, h//2), (0, 255, 255), 3)

                # 前后距离判断
                elif ball_area < CONFIG['target_area_min']:
                    # 球太远，需要向前走
                    distance_percent = (ball_area / CONFIG['target_area_min']) * 100
                    print(f"\n[决策] 球太远 (面积={ball_area} < {CONFIG['target_area_min']})")
                    print(f"  当前距离约为目标距离的 {distance_percent:.1f}%")
                    # print(f"[动作] 应执行: robot.move(vx=20000, last_time=0.2, duration=1)")
                    # print(f"  -> 含义: 向前移动（vx正值=前进），速度20000，持续0.2秒，之后等待1秒")
                    # print(f"  -> 目的: 靠近球到合适的踢球距离")
                    # if show_video:
                    #     cv2.putText(frame, "Action: MOVE FORWARD", (w//2 - 140, h//2 + 50),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    #     cv2.arrowedLine(frame, (w//2, h//2 + 30), (w//2, h//2 - 50), (255, 255, 0), 3)

                elif ball_area > CONFIG['target_area_max']:
                    # 球太近，需要向后退
                    distance_percent = (ball_area / CONFIG['target_area_max']) * 100
                    print(f"\n[决策] 球太近 (面积={ball_area} > {CONFIG['target_area_max']})")
                    print(f"  当前距离约为目标距离的 {distance_percent:.1f}%")
                    # print(f"[动作] 应执行: robot.move(vx=-20000, last_time=0.2, duration=1)")
                    # print(f"  -> 含义: 向后移动（vx负值=后退），速度20000，持续0.2秒，之后等待1秒")
                    # print(f"  -> 目的: 后退到合适的踢球距离")
                    # if show_video:
                    #     cv2.putText(frame, "Action: MOVE BACK", (w//2 - 120, h//2 + 50),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    #     cv2.arrowedLine(frame, (w//2, h//2 - 30), (w//2, h//2 + 50), (255, 255, 0), 3)

                # 球在最佳位置，执行踢球
                else:
                    print(f"\n{'='*70}")
                    # print(f"[决策] 球在最佳位置！准备踢球！")
                    # print(f"  中心位置: X比例={ball_x_ratio:.3f} (目标: 0.5±{CONFIG['center_threshold']})")
                    # print(f"  距离合适: 面积={ball_area} (目标: {CONFIG['target_area_min']}~{CONFIG['target_area_max']})")
                    # print(f"\n[踢球序列] 以下是机器狗应该执行的完整动作序列：")
                    # print(f"  0. robot.start_continue()  # 开启持续运动模式")
                    # print(f"  1. robot.move(vx=40000, last_time=2)")
                    # print(f"     -> 向前冲刺2秒，速度40000")
                    # print(f"     -> 用右前腿碰球（踢球动作）")
                    # print(f"  2. time.sleep(1)  # 等待1秒让机器狗稳定")
                    # print(f"  3. robot.move(vx=-20000, last_time=1.1)")
                    # print(f"     -> 向后退1.1秒，速度20000")
                    # print(f"     -> 复位到初始位置")
                    # print(f"  4. time.sleep(1)  # 等待1秒让机器狗稳定")
                    # print(f"  5. robot.move(vy=-40000, last_time=0.4, duration=0.5)")
                    # print(f"     -> 向左侧移，速度40000，持续0.4秒，之后等待0.5秒")
                    # print(f"     -> 最终位置调整")
                    # print(f"  6. robot.close_continue()  # 关闭持续运动模式")
                    print(f"\n踢球任务结束！")
                    print(f"{'='*70}\n")

                    # if show_video:
                    #     cv2.putText(frame, "READY TO KICK!", (w//2 - 150, h//2),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    #     cv2.putText(frame, "(Press 'q' to exit)", (w//2 - 100, h//2 + 40),
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    #     cv2.imshow('Football Detection', frame)
                    #     cv2.waitKey(2000)  # 显示2秒

                    kicked = True
                    break

                print("-"*70)

            else:
                # 没有检测到球
                if show_video:
                    cv2.putText(frame, "Searching for ball...", (10, h - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                # 每隔一段时间打印一次
                if frame_cnt % 30 == 0:
                    print(f"[等待] 未检测到球... (已处理 {frame_cnt} 帧)")
                    print(f"  检查: 球是否在摄像头视野内?")
                    # print(f"        2) 球的颜色是否为 {color}?")
                    # print(f"        3) HSV范围是否正确? Lower={lower_color}, Upper={upper_color}")

            # 显示画面
            if show_video:
                cv2.imshow('Football Detection', frame)
                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[用户中断] 按下 'q' 键，测试终止")
                    break

    finally:
        # 清理资源
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        print("\n" + "="*70)
        print("视觉测试结束")
        print(f"总共处理帧数: {action_count}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # 直接运行此文件进行视觉测试
    print("\n欢迎使用足球视觉识别测试工具")
    print("此工具仅用于测试视觉识别，不会控制机器狗运动\n")

    # 默认禁用视频显示，避免在无显示器的系统上出现Qt/xcb错误
    # 如需显示视频，将 show_video 改为 True
    test_vision_football(color='orange', show_video=False)
