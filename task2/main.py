from dog_control_sxh_test import DogControl
import argparse
import time
from tasks.task2_new import task2_new
from tasks.task2_3 import task2_3
from tasks.dashboard_letter_detector import SimpleInfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="显示任务二检测画面")
    args = parser.parse_args()

    robot = DogControl()
    time.sleep(1)
    robot.close_continue()
    robot.stop()
    time.sleep(0.5) #起立之后是原地模式，同时自带了duration为3，也就是说这里的起立之后会停顿3秒左右
    robot.stand_up()
    time.sleep(0.5)

    detector = SimpleInfer(show_stream=args.stream) #初始化摄像头

    # ------------------------------------任务二---------------------------------
    task2_new(robot, detector=detector, show_stream=args.stream)

    # ------------------------------------任务二和任务三的衔接--------------------
    robot.move(last_time=7, vy=25000)
    time.sleep(0.5)
    robot.move(last_time=2.5, vx=-20000)
    time.sleep(0.5)
    robot.revolve_90_r()
    time.sleep(0.5)

    task2_3(robot, detector=detector)

    robot.revolve_180()
    time.sleep(0.5)
    robot.move(last_time=1.0, vx=20000)


    detector.close()    # 关闭摄像头

    robot.close_continue()
    robot.stop()
    print("stop")
    time.sleep(2)
